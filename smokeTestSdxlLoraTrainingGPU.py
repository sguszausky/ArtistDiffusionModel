#!/usr/bin/env python3
"""
SDXL LoRA smoke test (CPU/GPU friendly, cluster-safe).

Defaults:
- Tiny random SDXL model
- 1 training step
- 64x64 resolution
- batch size 1
- saves LoRA to ./smoke_out/ckpt and reloads into a fresh pipeline

Environment variables:
  SMOKE_MODEL           default: dg845/tiny-random-stable-diffusion-xl
  SMOKE_DEVICE          default: cuda if available else cpu
  SMOKE_STEPS           default: 1
  SMOKE_RES             default: 64
  SMOKE_BS              default: 1
  SMOKE_LR              default: 1e-4
  HF_HUB_OFFLINE        if "1", local_files_only=True
  MIXED_PRECISION       default: fp16 on cuda, no on cpu  (values: "no", "fp16", "bf16")
  OUT_DIR               default: ./smoke_out
  DATASET_ROOT          default: ./dataset/small_test
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig

import sdxl_lora_trainer_gpu_helper as helper


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, str(default))
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, str(default))
    try:
        return float(v)
    except Exception:
        return default


def main() -> None:

    model_id = os.getenv("SMOKE_MODEL", "dg845/tiny-random-stable-diffusion-xl")
    requested_device = os.getenv("SMOKE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    steps = _env_int("SMOKE_STEPS", 1)
    res = _env_int("SMOKE_RES", 64)
    batch_size = _env_int("SMOKE_BS", 1)
    lr = _env_float("SMOKE_LR", 1e-4)

    dataset_root = Path(os.getenv("DATASET_ROOT", str(Path("dataset") / "small_test"))).expanduser()
    out_root = Path(os.getenv("OUT_DIR", str(Path("smoke_out")))).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    local_files_only = os.getenv("HF_HUB_OFFLINE", "") == "1"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SMOKE_DEVICE=cuda requested but CUDA is not available.")

    mixed_precision = os.getenv("MIXED_PRECISION", "fp16" if requested_device == "cuda" else "no")

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=mixed_precision)
    device = accelerator.device
    is_cuda = device.type == "cuda"

    # dtype selection
    if is_cuda:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("MODEL_ID:", model_id)
    print("REQUESTED_DEVICE:", requested_device, "=> accelerator.device:", device)
    print("HF_HUB_OFFLINE:", os.getenv("HF_HUB_OFFLINE", ""), "=> local_files_only:", local_files_only)
    print("mixed_precision:", accelerator.mixed_precision)
    print("weight_dtype:", weight_dtype)
    print("STEPS:", steps, "RES:", res, "BATCH_SIZE:", batch_size, "LR:", lr)
    print("DATASET_ROOT:", dataset_root.resolve())
    print("OUT_DIR:", out_root.resolve())

    set_seed(0)

    # -------------------------
    # Load pipeline
    # -------------------------
    if accelerator.is_local_main_process:
        print("Loading pipeline…")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=weight_dtype,
        use_safetensors=True,
        local_files_only=local_files_only,
    )

    # xFormers is optional; avoid hard fail
    if is_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            if accelerator.is_local_main_process:
                print("xFormers enabled")
        except Exception as e:
            if accelerator.is_local_main_process:
                print("xFormers not enabled:", repr(e))

    pipe.enable_vae_tiling()

    if accelerator.is_local_main_process:
        print("Pipeline loaded")

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

    # Put frozen parts in eval on correct device/dtype
    vae.to(device, dtype=weight_dtype).eval()
    te1.to(device, dtype=weight_dtype).eval()
    te2.to(device, dtype=weight_dtype).eval()


    unet_lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    lora_params = [p for n, p in unet.named_parameters() if "lora" in n and p.requires_grad]
    if accelerator.is_local_main_process:
        print("trainable LoRA params:", sum(p.numel() for p in lora_params))
    assert len(lora_params) > 0, "No LoRA params marked trainable."

    if accelerator.mixed_precision == "fp16":
        for p in lora_params:
            p.data = p.data.float()
        if accelerator.is_local_main_process:
            dtypes = sorted({str(p.dtype) for p in lora_params})
            print("LoRA params dtype after fp16 fix:", ", ".join(dtypes))

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    ds = helper.ImageCaptionFolder(str(dataset_root), resolution=res, center_crop=True)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=is_cuda,
    )

    opt = torch.optim.AdamW(lora_params, lr=lr)

    unet, opt, dl = accelerator.prepare(unet, opt, dl)
    unet.train()


    if accelerator.is_local_main_process:
        print("Running train step…")

    global_step = 0
    for batch in dl:
        if global_step >= steps:
            break

        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype, non_blocking=is_cuda)

            # Encode images -> latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise + timesteps
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

            # Tokenize prompts (from dataset)
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
                prompt_embeds_1 = te1(input_ids_1, output_hidden_states=True)
                prompt_embeds_2 = te2(input_ids_2, output_hidden_states=True)

                pooled = prompt_embeds_2[0]
                hidden_states_1 = prompt_embeds_1.hidden_states[-2]
                hidden_states_2 = prompt_embeds_2.hidden_states[-2]
                prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

            # Add time ids
            add_time_ids = helper.make_sdxl_time_ids(
                original_size=(res, res),
                crop_coords=(0, 0),
                target_size=(res, res),
                device=device,
                dtype=prompt_embeds.dtype,
            )

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
                return_dict=False,
            )[0]

            # Simple epsilon prediction loss
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if accelerator.is_local_main_process:
            print(f"step={global_step} loss={loss.item():.6f}")

        global_step += 1

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        ckpt = out_root / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        print("Saving LoRA…")
        helper.save_unet_lora_peft(accelerator.unwrap_model(unet), str(ckpt))

        print("Reload LoRA into a fresh pipeline…")
        pipe2 = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=weight_dtype,
            use_safetensors=True,
            local_files_only=local_files_only,
        ).to(device)

        helper.load_lora_into_pipe(pipe2, str(ckpt))
        _ = pipe2(
            "a photo of a cat",
            num_inference_steps=2,
            height=res,
            width=res,
        ).images[0]
        print("Reload OK")

    if accelerator.is_local_main_process:
        print("done")


if __name__ == "__main__":
    main()
