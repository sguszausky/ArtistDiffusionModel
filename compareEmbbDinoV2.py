#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from transformers import AutoImageProcessor, AutoModel


validEndings = {".jpg", ".jpeg", ".png"}


def listImages(folder: str | Path) -> list[str]:
    p = Path(folder).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {p}")
    paths = [str(x) for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in validEndings]
    if not paths:
        raise RuntimeError(f"No images found in: {p}")
    return paths


def loadImage(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def embedImagePaths(image_paths: list[str], model_name: str, device: str, batch_size: int = 8):

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        imgs = [loadImage(p) for p in batch_paths]

        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model(**inputs)

        # DINOv2: last_hidden_state [B, tokens, D], CLS token at index 0
        cls = out.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, p=2, dim=-1)  # normalize so cosine = dot product
        all_embs.append(cls.detach().cpu())

    return torch.cat(all_embs, dim=0)


def runRanking(artist_dir: str | Path, gen_dir: str | Path, model: str = "facebook/dinov2-base", batch_size: int = 8, topk: int = 20, csv_out: str | Path | None = None, quiet: bool = False):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    artist_paths = listImages(artist_dir)
    gen_paths = listImages(gen_dir)

    if not quiet:
        print(f"Device: {device}")
        print(f"Model:  {model}")
        print(f"Artist images (ref): {len(artist_paths)}  |  dir: {Path(artist_dir).resolve()}")
        print(f"Gen images (ranked): {len(gen_paths)}     |  dir: {Path(gen_dir).resolve()}\n")

    # Embeddings
    artist_embs = embedImagePaths(artist_paths, model, device, batch_size=batch_size)
    gen_embs = embedImagePaths(gen_paths, model, device, batch_size=batch_size)

    # Artist centroid (mean) + normalize
    centroid = F.normalize(artist_embs.mean(dim=0, keepdim=True), p=2, dim=-1)

    # Scores
    centroid_cos = (gen_embs @ centroid.T).squeeze(1)
    sim_matrix = gen_embs @ artist_embs.T
    max_cos, nn_idx = torch.max(sim_matrix, dim=1)

    rows = []
    for i, gp in enumerate(gen_paths):
        rows.append({
            "generated_path": gp,
            "centroid_cos": float(centroid_cos[i].item()),
            "max_cos": float(max_cos[i].item()),
            "nearest_artist_path": artist_paths[int(nn_idx[i].item())],
        })
    df = pd.DataFrame(rows)

    if not quiet:
        k = min(topk, len(df))

        print(f"Top {k} by centroid_cos (closest to overall artist centroid):\n")
        for r, row in enumerate(df.sort_values("centroid_cos", ascending=False).head(k).itertuples(index=False), 1):
            print(f"{r:02d}. centroid_cos={row.centroid_cos:.6f} | max_cos={row.max_cos:.6f}")
            print(f"    gen: {row.generated_path}")
            print(f"    nn : {row.nearest_artist_path}\n")

        print(f"\nTop {k} by max_cos (closest single match in artist set):\n")
        for r, row in enumerate(df.sort_values("max_cos", ascending=False).head(k).itertuples(index=False), 1):
            print(f"{r:02d}. max_cos={row.max_cos:.6f} | centroid_cos={row.centroid_cos:.6f}")
            print(f"    gen: {row.generated_path}")
            print(f"    nn : {row.nearest_artist_path}\n")

    # Optional CSV export
    if csv_out:
        out = Path(csv_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["centroid_cos", "max_cos"], ascending=False).to_csv(out, index=False)
        if not quiet:
            print(f"\nWrote CSV: {out}")

    return df


