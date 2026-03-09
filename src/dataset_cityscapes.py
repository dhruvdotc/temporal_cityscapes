# src/dataset_cityscapes.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.datasets import Cityscapes


# Cityscapes official "trainId" mapping (19 classes) + ignore=255.
# Maps labelId -> trainId. Anything not in map becomes 255.
_LABELID_TO_TRAINID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}

# Standard ImageNet normalization (common for torchvision backbones)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def labelid_to_trainid(mask_labelid: torch.Tensor) -> torch.Tensor:
    """
    mask_labelid: HxW, dtype uint8/long with Cityscapes labelIds.
    returns: HxW, dtype long with trainIds 0..18 and ignore=255
    """
    mask = mask_labelid.long()
    out = torch.full_like(mask, 255)
    for k, v in _LABELID_TO_TRAINID.items():
        out[mask == k] = v
    return out


class CityscapesTrainId(Dataset):
    """
    Cityscapes semantic segmentation dataset returning:
      image: FloatTensor [3,H,W] normalized
      mask:  LongTensor  [H,W] trainIds in 0..18, ignore=255
      meta:  dict with paths + city name etc (handy for debugging/visuals)
    """
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        size_hw: Tuple[int, int] = (512, 1024),
    ):
        self.root = Path(root)
        self.split = split
        self.size_hw = size_hw

        # torchvision Cityscapes expects root to contain leftImg8bit/ and gtFine/
        # mode="fine" gives fine annotations; target_type="semantic" returns labelIds
        self.ds = Cityscapes(
            root=str(self.root),
            split=split,
            mode="fine",
            target_type="semantic",
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img_pil, mask_pil = self.ds[idx]  # PIL image, PIL mask (labelIds)

        # Convert image to tensor [3,H,W] float in [0,1]
        x = TF.to_tensor(img_pil)
        # Resize image (bilinear)
        x = TF.resize(x, self.size_hw, interpolation=TF.InterpolationMode.BILINEAR)

        # Normalize
        x = TF.normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

        # Mask: convert to tensor HxW (labelIds)
        mask_np = np.array(mask_pil, dtype=np.uint8)
        y = torch.from_numpy(mask_np)
        # Resize mask (nearest)
        y = TF.resize(
            y.unsqueeze(0),
            self.size_hw,
            interpolation=TF.InterpolationMode.NEAREST
        ).squeeze(0)

        # Map labelIds -> trainIds
        y = labelid_to_trainid(y)

        meta: Dict[str, Any] = {
            "index": idx,
            "split": self.split,
            "size_hw": self.size_hw,
        }
        return x, y, meta