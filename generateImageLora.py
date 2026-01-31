#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import StableDiffusionXLPipeline


def _slugify(text: str, max_len: int = 120) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "image"
    return text[:max_len]


def _format_metadata_txt(meta: dict) -> str:
    preferred_order = [
        "full_prompt",
        "prefix",
        "caption",
        "model_id",
        "lora_path",
        "lora_scale",
        "adapter_name_used",
        "steps",
        "guidance",
        "width",
        "height",
        "device",
        "local_files_only",
        "seed_base",
        "seed_used",
        "overwrite",
        "output_image",
        "output_txt",
        "out_dir",
    ]

    lines = []
    for k in preferred_order:
        if k in meta:
            lines.append(f"{k}: {meta[k]}")

    for k in sorted(meta.keys()):
        if k not in preferred_order:
            lines.append(f"{k}: {meta[k]}")

    return "\n".join(lines) + "\n"


def _pick_adapter_name_from_error(msg: str) -> Optional[str]:
    # Example msg: "Adapter name(s) {'default'} ... present adapters: {'default_0'}."
    m = re.search(r"present adapters:\s*\{([^}]*)\}", msg)
    if not m:
        return None
    inside = m.group(1).strip()
    names = [x.strip().strip("'").strip('"') for x in inside.split(",") if x.strip()]
    return names[0] if names else None


@dataclass
class SDXLConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    steps: int = 30
    guidance: float = 6.5
    width: int = 1024
    height: int = 1024
    device: str = "cuda"  # "cuda" or "cpu"
    local_files_only: bool = False
    dtype_cuda: torch.dtype = torch.bfloat16
    dtype_cpu: torch.dtype = torch.float32


def load_sdxl_pipe(cfg: SDXLConfig) -> StableDiffusionXLPipeline:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        device = "cpu"

    dtype = cfg.dtype_cuda if device == "cuda" else cfg.dtype_cpu

    print(f"Loading base model: {cfg.model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=cfg.local_files_only,
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled")
    except Exception as e:
        print("xFormers not enabled:", repr(e))

    pipe.enable_vae_tiling()
    pipe = pipe.to(device)
    return pipe


def _apply_lora(pipe: StableDiffusionXLPipeline, lora_path: Path, lora_scale: float) -> tuple[dict, str]:
    """
    Loads LoRA weights into the pipeline and sets scale.
    Returns (extra_kwargs_for_pipe_call, adapter_name_used).
    """
    print("Loading LoRA:", lora_path)
    pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    extra_kwargs: dict = {}
    adapter_name_used = ""

    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(["default"], adapter_weights=[lora_scale])
            adapter_name_used = "default"
            print(f"LoRA scale set via set_adapters('default'): {lora_scale}")
        except ValueError as e:
            name = _pick_adapter_name_from_error(str(e))
            if name:
                pipe.set_adapters([name], adapter_weights=[lora_scale])
                adapter_name_used = name
                print(f"LoRA scale set via set_adapters('{name}'): {lora_scale}")
            else:
                extra_kwargs = {"cross_attention_kwargs": {"scale": lora_scale}}
                adapter_name_used = "cross_attention_kwargs"
                print(f"LoRA scale via cross_attention_kwargs: {lora_scale}")
    else:
        extra_kwargs = {"cross_attention_kwargs": {"scale": lora_scale}}
        adapter_name_used = "cross_attention_kwargs"
        print(f"LoRA scale via cross_attention_kwargs: {lora_scale}")

    return extra_kwargs, adapter_name_used


def generate_images_for_captions(
    captions: List[str],
    out_dir: str | Path,
    *,
    prefix: str = "",
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora: str | Path,                 # REQUIRED (keyword-only)
    lora_scale: float = 0.8,
    seed: int = 123,
    steps: int = 30,
    guidance: float = 6.5,
    width: int = 1024,
    height: int = 1024,
    device: str = "cuda",
    local_files_only: bool = False,
    overwrite: bool = False,
) -> List[Path]:
    """
    For each caption:
      - full_prompt = "{prefix} {caption}".strip()
      - save <slug>.png and <slug>.txt (prompt + args + seed_used + lora info)
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_path = Path(lora).expanduser().resolve()
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    cfg = SDXLConfig(
        model_id=model_id,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        device=device,
        local_files_only=local_files_only,
    )

    pipe = load_sdxl_pipe(cfg)

    # Load/apply LoRA once
    extra_kwargs, adapter_name_used = _apply_lora(pipe, lora_path, lora_scale)

    saved: List[Path] = []
    base_gen = torch.Generator(device=pipe.device)

    common_args = {
        "prefix": prefix,
        "model_id": model_id,
        "lora_path": str(lora_path),
        "lora_scale": lora_scale,
        "adapter_name_used": adapter_name_used,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "device": str(pipe.device),
        "local_files_only": local_files_only,
        "seed_base": seed,
        "overwrite": overwrite,
        "out_dir": str(out_dir),
        "internal_config": {k: (str(v) if "dtype" in k else v) for k, v in asdict(cfg).items()},
    }

    for i, cap in enumerate(captions):
        cap = (cap or "").strip()
        if not cap:
            continue

        full_prompt = " ".join([prefix.strip(), cap]).strip()

        slug = _slugify(cap)
        img_path = out_dir / f"{slug}.png"
        txt_path = out_dir / f"{slug}.txt"

        if not overwrite and (img_path.exists() or txt_path.exists()):
            print(f"Skipping existing: {img_path.name}")
            saved.append(img_path)
            continue


        gen = base_gen.manual_seed(int(seed))

        print(f"[{i+1}/{len(captions)}] {img_path.name} | seed={seed}")
        with torch.inference_mode():
            img = pipe(
                prompt=full_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=gen,
                **extra_kwargs,
            ).images[0]

        img.save(img_path)

        meta = dict(common_args)
        meta.update(
            {
                "caption": cap,
                "full_prompt": full_prompt,
                "seed_used": int(seed),
                "output_image": str(img_path),
                "output_txt": str(txt_path),
            }
        )
        txt_path.write_text(_format_metadata_txt(meta), encoding="utf-8")

        saved.append(img_path)

    print(f"Done. Saved {len(saved)} images to: {out_dir}")
    return saved
