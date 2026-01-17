from PIL import Image
from pathlib import Path
import time

validEndings = {".jpg", ".jpeg", ".png", ".webp"}

def imageTooSmall(img_path: Path, minWidth: int, minHeight: int) -> tuple[bool, int, int]:
    with Image.open(img_path) as img:
        img.load()
        w, h = img.size
    return (w < minWidth or h < minHeight), w, h


def filterBySize(imageDir: str,minWidth: int,minHeight: int,):

    imageDir = Path(imageDir)
    to_delete = []

    for img_path in imageDir.iterdir():
        if img_path.suffix.lower() not in validEndings:
            continue
        try:
            too_small, w, h = imageTooSmall(img_path, minWidth, minHeight)
            if too_small:
                to_delete.append({"path": img_path, "width": w, "height": h})
        except Exception as e:
            print(f"Error occured at {img_path.name}: {e}")

    print(f"\nFound {len(to_delete)} images smaller as {minWidth}×{minHeight}")
    for item in to_delete:
        path = item["path"]
        width = item["width"]
        height = item["height"]
        print(f"{path.name}: {width}×{height}")
    return to_delete



def deleteImages(to_delete: list[dict],retries: int = 3,wait: float = 0.5,):
    deleted = 0
    locked = 0

    for item in to_delete:
        img_path: Path = item["path"]
        w = item["width"]
        h = item["height"]

        if not img_path.exists():
            continue

        for attempt in range(retries):
            try:
                img_path.unlink()
                deleted += 1
                print(f"Deleted: {img_path.name} ({w}×{h})")
                break
            except PermissionError:
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    locked += 1
                    print(f"Could not delete: {img_path.name}, skipped")

    print("\nResult")
    print("Deleted:", deleted)
    print("Skipped:", locked)


def cropBottomPercent(imageDir: str,outDir: str,cropPercent: float,overwrite: bool = False,quality: int = 95):
    """
    Crops a fixed percentage from the bottom of every image.

    Args:
        imageDir: input folder
        outDir: output folder (ignored if overwrite=True; then writes into imageDir)
        cropPercent: e.g. 12.0 means cut off 12% from bottom
        overwrite: if True, overwrite originals (NOT recommended)
        quality: JPEG quality
    """
    in_dir = Path(imageDir)
    out_dir = in_dir if overwrite else Path(outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cropPercent <= 0 or cropPercent >= 100:
        raise ValueError("cropPercent must be between 0 and 100 (exclusive).")

    processed = 0
    skipped = 0
    failed = 0

    for img_path in in_dir.iterdir():
        if img_path.suffix.lower() not in validEndings:
            continue

        try:
            with Image.open(img_path) as img:
                img.load()
                img = img.convert("RGB")
                w, h = img.size

                cut_px = int(h * (cropPercent / 100.0))
                new_h = h - cut_px

                if new_h < 1:
                    skipped += 1
                    print(f"Skipped (too small after crop): {img_path.name}")
                    continue

                cropped = img.crop((0, 0, w, new_h))
                out_path = out_dir / img_path.name
                cropped.save(out_path, quality=quality)

            processed += 1
            if processed <= 3 or processed % 25 == 0:
                print(f"Cropped: {img_path.name} ({w}×{h} -> {w}×{new_h})")

        except Exception as e:
            failed += 1
            print(f"Error occured at {img_path.name}: {e}")

    print("\nResult")
    print("Processed:", processed)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Output:", out_dir)
