
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from pathlib import Path

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WEIGHTS = (
    Path(__file__).resolve().parent
    / "weights"
    / "best_unet_r34.pth"
)

NUM_CLASSES = 24
INPUT_SIZE = 256

_device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

_model = None


def _load():
    global _model

    if _model is None:

        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"Model weights not found: {WEIGHTS}"
            )

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            in_channels=3,
            classes=NUM_CLASSES
        )

        checkpoint = torch.load(
            WEIGHTS,
            map_location=_device
        )

        # Support either a raw state_dict or a checkpoint dictionary.
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint)
        _model = model.to(_device).eval()

        print(f"U-Net loaded on {_device}")

    return _model


def predict(image_rgb, clean=True, ksize=5):

    """
    Input:
        RGB image of shape H x W x 3.

    Returns:
        class_map: H x W predicted class IDs.
        probs: 24 x H x W class probabilities.
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    model = _load()

    h, w = image_rgb.shape[:2]

    image = cv2.resize(
        image_rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD

    tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).unsqueeze(0).to(_device)

    with torch.no_grad():

        logits = model(tensor)

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        probabilities = logits.softmax(dim=1)[0]

    class_map = (
        probabilities.argmax(dim=0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    if clean:
        if ksize % 2 == 0 or ksize < 3:
            raise ValueError("ksize must be an odd integer >= 3.")

        class_map = cv2.medianBlur(
            class_map,
            ksize
        )

    return (
        class_map.astype(np.int32),
        probabilities.cpu().numpy()
    )
