"""Inference: any image -> 800x600 main image -> 256x256 -> CNN -> mask upscaled to 800x600.

    python -m src.inference path/to/image.jpg
"""
import sys
import cv2
import numpy as np
import torch

from models.segmentation_pipeline import SegNet
from src.preprocessing import (NUM_CLASSES, MAIN_W, MAIN_H, to_main, to_net_size,
                               normalize, upscale_mask)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoints/best_model.pth"


def load_model(path=CKPT):
    model = SegNet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.eval()


@torch.no_grad()
def predict_mask(model, image_bgr):
    """Returns (main_image_800x600_bgr, class_mask_600x800_uint8)."""
    main = to_main(image_bgr)
    x = torch.from_numpy(normalize(to_net_size(main))).unsqueeze(0).to(DEVICE)
    pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)     # 256x256
    mask = upscale_mask(pred)                                       # 600x800
    assert mask.shape == (MAIN_H, MAIN_W)
    return main, mask


if __name__ == "__main__":
    main_img, mask = predict_mask(load_model(), cv2.imread(sys.argv[1]))
    color = cv2.applyColorMap((mask * (255 // max(NUM_CLASSES - 1, 1))).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite("pred_overlay.png", cv2.addWeighted(main_img, 0.5, color, 0.5, 0))
    np.save("pred_mask.npy", mask)
    print("saved pred_overlay.png and pred_mask.npy; mask shape", mask.shape)
