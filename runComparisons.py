#!/usr/bin/env python3
from pathlib import Path

from compareEmbbDinoV2 import runRanking


def main():

    original_dir = "outputs/toCompare/canyonRiders/original"
    base_dir = "outputs/toCompare/canyonRiders/base"
    lora_dir = "outputs/toCompare/canyonRiders/lora"

    out_dir = Path("outputs/toCompare/canyonRiders/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = "facebook/dinov2-base"
    batch_size = 8
    topk = 20

    print("\noriginal vs base\n")
    runRanking(
        artist_dir=original_dir,
        gen_dir=base_dir,
        model=model,
        batch_size=batch_size,
        topk=topk,
        csv_out=out_dir / "original_vs_base.csv",
        quiet=False,
    )

    print("\n original vs lora\n")
    runRanking(
        artist_dir=original_dir,
        gen_dir=lora_dir,
        model=model,
        batch_size=batch_size,
        topk=topk,
        csv_out=out_dir / "original_vs_lora.csv",
        quiet=False,
    )


if __name__ == "__main__":
    main()
