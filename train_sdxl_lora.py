#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig

import sdxl_lora_trainer_gpu_helper as helper

@dataclass
class TrainConfig:
    preset: str

    model_id: str
    dataset_root: str
    out_dir: str

    resolution: int
    batch_size: int
    grad_accum: int
    max_steps: int
    lr: float
    weight_decay: float
    mixed_precision: str

    lora_rank: int
    lora_alpha: int

    center_crop: bool
    num_workers: int
    save_every: int
    seed: int

    local_files_only: bool


def smokeTestCfg() -> TrainConfig:

    return TrainConfig(
        preset="smoke",
        model_id=os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"), #tiny-test model
        dataset_root=os.getenv("DATASET_ROOT", str(Path("dataset") / "small_test")),
        out_dir=os.getenv("OUT_DIR", str(Path("runs") / "smoke")),
        resolution=int(os.getenv("RESOLUTION", "256")),
        batch_size=int(os.getenv("BATCH_SIZE", "1")),
        grad_accum=int(os.getenv("GRAD_ACCUM", "1")),
        max_steps=int(os.getenv("MAX_STEPS", "30")),
        lr=float(os.getenv("LR", "1e-4")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "1e-2")),
        mixed_precision=os.getenv("MIXED_PRECISION", "bf16" if torch.cuda.is_available() else "no"),
        lora_rank=int(os.getenv("LORA_RANK", "4")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "4")),
        center_crop=(os.getenv("CENTER_CROP", "1") == "1"),
        num_workers=int(os.getenv("NUM_WORKERS", "0")),
        save_every=int(os.getenv("SAVE_EVERY", "0")),
        seed=int(os.getenv("SEED", "0")),
        local_files_only=(os.getenv("HF_HUB_OFFLINE", "") == "1"),
    )

def _select_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.device.type != "cuda":
        return torch.float32
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _save_checkpoint(
    accelerator: Accelerator,
    unet,
    out_dir: Path,
    step: int,
) -> None:
    if not accelerator.is_local_main_process:
        return
    ckpt_dir = out_dir / f"ckpt_step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    helper.save_unet_lora_peft(accelerator.unwrap_model(unet), str(ckpt_dir))
    print(f"[ckpt] saved: {ckpt_dir}")


def train(cfg: TrainConfig) -> None:

    if cfg.preset == "smoke":
        cfg = smokeTestCfg()

    out_dir = Path(cfg.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum,
        mixed_precision=cfg.mixed_precision,
    )

    if accelerator.is_local_main_process:
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = accelerator.device
    is_cuda = device.type == "cuda"
    weight_dtype = _select_weight_dtype(accelerator)

    if accelerator.is_local_main_process:
        print("----- config -----")
        print(json.dumps(asdict(cfg), indent=2))
        print("device:", device)
        print("torch:", torch.__version__)
        print("weight_dtype:", weight_dtype)
        print("------------------")

    set_seed(cfg.seed)

    # Load SDXL pipeline
    if accelerator.is_local_main_process:
        print("Loading SDXL pipeline…")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=weight_dtype,
        use_safetensors=True,
        local_files_only=cfg.local_files_only,
    )

    if is_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            if accelerator.is_local_main_process:
                print("xFormers enabled")
        except Exception as e:
            if accelerator.is_local_main_process:
                print("xFormers not enabled:", repr(e))

    # VAE tiling reduces memory
    pipe.enable_vae_tiling()

    unet = pipe.unet
    vae = pipe.vae
    te1 = pipe.text_encoder
    te2 = pipe.text_encoder_2
    tok1 = pipe.tokenizer
    tok2 = pipe.tokenizer_2

    # Freeze base weights
    vae.requires_grad_(False)
    te1.requires_grad_(False)
    te2.requires_grad_(False)
    unet.requires_grad_(False)

    # Put frozen modules on device
    vae.to(device, dtype=weight_dtype).eval()
    te1.to(device, dtype=weight_dtype).eval()
    te2.to(device, dtype=weight_dtype).eval()

    # Add LoRA to UNet attention
    unet_lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    lora_params = [p for n, p in unet.named_parameters() if "lora" in n and p.requires_grad]
    assert len(lora_params) > 0, "No LoRA params marked trainable."

    # IMPORTANT: fp16 + GradScaler expects trainable params in fp32
    if accelerator.mixed_precision == "fp16":
        for p in lora_params:
            p.data = p.data.float()
        if accelerator.is_local_main_process:
            print("LoRA params dtype after fp16 fix:", sorted({str(p.dtype) for p in lora_params}))

    if accelerator.is_local_main_process:
        print("trainable LoRA params:", sum(p.numel() for p in lora_params))

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Dataset
    ds = helper.ImageCaptionFolder(
        root=str(Path(cfg.dataset_root).expanduser()),
        resolution=cfg.resolution,
        center_crop=cfg.center_crop,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=is_cuda,
        drop_last=True,
    )

    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Prepare for (D)DP / mixed precision
    unet, opt, dl = accelerator.prepare(unet, opt, dl)
    unet.train()

    # Track timing
    start_time = time.time()
    global_step = 0

    if accelerator.is_local_main_process:
        print("Starting training…")

    while global_step < cfg.max_steps:
        for batch in dl:
            if global_step >= cfg.max_steps:
                break

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype, non_blocking=is_cuda)

                # Encode -> latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Noise + timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Tokenize captions
                captions = batch["caption"]

                enc1 = tok1(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=tok1.model_max_length,
                    return_tensors="pt",
                )
                enc2 = tok2(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=tok2.model_max_length,
                    return_tensors="pt",
                )

                input_ids_1 = enc1.input_ids.to(device, non_blocking=is_cuda)
                input_ids_2 = enc2.input_ids.to(device, non_blocking=is_cuda)

                # Text encoders (frozen)
                with torch.no_grad():
                    out1 = te1(input_ids_1, output_hidden_states=True)
                    out2 = te2(input_ids_2, output_hidden_states=True)

                    pooled = out2[0]
                    hs1 = out1.hidden_states[-2]
                    hs2 = out2.hidden_states[-2]
                    prompt_embeds = torch.cat([hs1, hs2], dim=-1)

                # SDXL time_ids
                add_time_ids = helper.make_sdxl_time_ids(
                    original_size=(cfg.resolution, cfg.resolution),
                    crop_coords=(0, 0),
                    target_size=(cfg.resolution, cfg.resolution),
                    device=device,
                    dtype=prompt_embeds.dtype,
                    batch_size=bsz,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
                    return_dict=False,
                )[0]

                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process and (global_step % 10 == 0 or global_step == cfg.max_steps - 1):
                elapsed = time.time() - start_time
                steps_per_sec = (global_step + 1) / max(elapsed, 1e-6)
                eta = (cfg.max_steps - global_step - 1) / max(steps_per_sec, 1e-6)
                print(f"step={global_step} loss={loss.item():.6f}  ({steps_per_sec:.2f} steps/s, ETA {eta/60:.1f} min)")

            global_step += 1

            #checkpoints
            if cfg.save_every and cfg.save_every > 0 and (global_step % cfg.save_every == 0):
                accelerator.wait_for_everyone()
                _save_checkpoint(accelerator, unet, out_dir, global_step)

    accelerator.wait_for_everyone()

    # saving final model
    if accelerator.is_local_main_process:
        final_dir = out_dir / "ckpt_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        helper.save_unet_lora_peft(accelerator.unwrap_model(unet), str(final_dir))
        print(f"[final] saved: {final_dir}")
        print("done")


