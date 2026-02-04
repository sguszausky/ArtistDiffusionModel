##!/usr/bin/env python3
"""
SDXL LoRA training runner.
Edit the TrainConfig values below, then run:
    python runTraining.py
In dataset, each image must have a same-named .txt file
"""

from train_sdxl_lora import TrainConfig, train


def main() -> None:

    cfg = TrainConfig(
        preset="train",   # smoke = quick test if setup works, variables not used, existing dataset/small_test needed;
                          # train = full training run with cfg

        model_id="stabilityai/stable-diffusion-xl-base-1.0",  # Base SDXL model
        dataset_root="dataset/Edgar_Payne_all",  # Path to your training dataset folder
        out_dir="runs/train",  # Output folder for checkpoints/config

        resolution=1024,  # Train image size; good for 50–100 imgs: 768–1024;
                          # smaller imgs get upscaled, so imgs too small may look funky
        batch_size=1,  # Batch per GPU; good: 1–2 (SDXL commonly 1)
        grad_accum=4,  # Gradient accumulation; good: 2–8
        max_steps=1500,  # Total training steps; good: 1200–3000 for style for 50–100 imgs
        lr=1e-4,  # Learning rate for LoRA params; good: 5e-5 to 1e-4 (start 1e-4; lower if overfitting)
        weight_decay=1e-2,  # AdamW weight decay; good: 0–1e-2 (start 1e-2; reduce if learning too weak)
        mixed_precision="bf16",  # Precision; good: "bf16" (best) or "fp16" (common) or "no" (slow)

        lora_rank=16,  # LoRA capacity; good: 8–16 (start 16; use 8 if overfitting)
        lora_alpha=16,  # LoRA scaling; good: rank or 2×rank (start = rank, e.g. 16)

        center_crop=True,  # Images crop type; True = center crop, False = random crop
        num_workers=4,  # DataLoader workers; good: 2–8
        save_every=250,  # Checkpoint frequency; good: 100–250 for small sets for watching overfitting
        seed=0,  # Random seed
        local_files_only=False,  # If True: load model only from local cache
    )

    train(cfg)


if __name__ == "__main__":
    main()

