import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from safetensors.torch import load_file



def make_sdxl_time_ids(
    original_size: tuple[int, int],
    crop_coords: tuple[int, int],
    target_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> torch.Tensor:
    add_time_ids = torch.tensor(
        [list(original_size) + list(crop_coords) + list(target_size)],
        device=device,
        dtype=dtype,
    )
    if batch_size != 1:
        add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids


class ImageCaptionFolder(Dataset):
    def __init__(self, root: str, resolution: int = 1024, center_crop: bool = True):
        self.root = Path(root)
        self.resolution = resolution
        self.center_crop = center_crop
        exts = {".jpg", ".jpeg", ".png"}
        self.images = sorted([p for p in self.root.iterdir() if p.suffix.lower() in exts])
        if not self.images:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.images)

    def _load_caption(self, img_path: Path) -> str:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            raise ValueError(f"Missing caption file for {img_path.name}: expected {txt_path.name}")
        return txt_path.read_text(encoding="utf-8").strip()

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        w, h = image.size
        scale = self.resolution / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

        if self.center_crop:
            left = (new_w - self.resolution) // 2
            top = (new_h - self.resolution) // 2
        else:
            left = random.randint(0, max(0, new_w - self.resolution))
            top = random.randint(0, max(0, new_h - self.resolution))

        image = image.crop((left, top, left + self.resolution, top + self.resolution))

        # Faster + cleaner: PIL -> numpy -> torch
        arr = np.array(image, dtype=np.uint8)  # (H,W,3), 0..255
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3,H,W), 0..1
        tensor = tensor * 2.0 - 1.0  # -1..1
        return tensor

    def __getitem__(self, idx):
        img_path = self.images[idx]
        caption = self._load_caption(img_path)
        with Image.open(img_path) as image:
            pixel_values = self._preprocess_image(image)
        return {"pixel_values": pixel_values, "caption": caption}


def save_unet_lora_peft(unet, save_dir: str, weight_name: str = "pytorch_lora_weights.safetensors"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

    from diffusers import StableDiffusionXLPipeline
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(save_dir),
        unet_lora_layers=lora_state_dict,
        weight_name=weight_name,
        safe_serialization=True,
    )


def load_lora_into_pipe(pipe, lora_dir: str, weight_name: str = "pytorch_lora_weights.safetensors"):
    lora_dir = Path(lora_dir)
    path = lora_dir / weight_name
    if not path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {path}")

    state = load_file(str(path))  # dict[str, Tensor]

    pipe.unet.load_attn_procs(state)
