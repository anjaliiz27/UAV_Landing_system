"""PyTorch dataset over the prepared 800x600 images (served to the net at 256x256)."""
import os, random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing import PREP_DIR, to_net_size, normalize, mask_to_net_size

SEED = 42


def split_stems(val_fraction=0.2):
    """Deterministic train/val split by filename stem."""
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{PREP_DIR}/images"))
    random.Random(SEED).shuffle(stems)
    n_val = int(len(stems) * val_fraction)
    return stems[n_val:], stems[:n_val]


class DroneSeg(Dataset):
    def __init__(self, stems, augment=False):
        self.stems, self.augment = stems, augment

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        img = to_net_size(cv2.imread(f"{PREP_DIR}/images/{s}.png"))
        msk = mask_to_net_size(cv2.imread(f"{PREP_DIR}/masks/{s}.png", cv2.IMREAD_UNCHANGED))
        if self.augment:                                  # top-down view: flips/rotations are valid
            if random.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = random.randint(0, 3)
            img, msk = np.rot90(img, k), np.rot90(msk, k)
        x = torch.from_numpy(normalize(np.ascontiguousarray(img)))
        y = torch.from_numpy(np.ascontiguousarray(msk)).long()
        return x, y
