#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

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


def generate_images_for_captions(
    captions: List[str],
    out_dir: str | Path,
    *,
    prefix: str = "",
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
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
    For each caption in captions:
      - build prompt = "{prefix} {caption}".strip()
      - save: <slug>.png and <slug>.txt (prompt + all args + seed used)
    Returns list of saved image paths.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    saved: List[Path] = []
    base_gen = torch.Generator(device=pipe.device)

    common_args = {
        "prefix": prefix,
        "model_id": model_id,
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
