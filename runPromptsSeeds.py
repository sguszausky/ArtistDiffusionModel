#!/usr/bin/env python3
from __future__ import annotations

from generateImageBase import generate_images_for_captions as gen_base
from generateImageLora import generate_images_for_captions as gen_lora


def main():
    prefixBase = "oil painting landscape,"
    prefixLora = "<EdgarAPayne>, oil painting landscape,"
    loraPath = "./models/edgar_payne_lora/edgar_payne_lora.safetensors" # Path to LoRA model

    captions = [
        "riders at an alpine lake with snowy mountains in the background at sunset",
        "a lake inside a canyon with some pine trees",
        "rugged mountains, lake, scattered pine trees, pine forest, atmospheric perspective, golden hour lighting",
        "prairie with mesas in the morning",
        "inside a canyon, riders, midday sunlight",
        "New York",
        "Small village in a pine forest",
        "a beach on sunset with palm trees"
    ]

    seeds = [42, 1234, 2026, 7, 99, 555]
    
    # Common generation settings
    common = dict(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        steps=30,
        guidance=6.5,
        width=1024,
        height=1024,
        device="cuda",
        local_files_only=False,
        overwrite=False,
    )

    for seed in seeds:
        base_out = f"./outputs/base/seed_{seed}"
        lora_out = f"./outputs/lora/seed_{seed}"

        #Base SDXL outputs
        gen_base(
            captions,
            base_out,
            prefix=prefixBase,
            seed=seed,
            **common,
        )

        #LoRA outputs
        gen_lora(
            captions,
            lora_out,
            prefix=prefixLora,
            lora=loraPath,
            lora_scale=0.8,
            seed=seed,
            **common,
        )


if __name__ == "__main__":
    main()

