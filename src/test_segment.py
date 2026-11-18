import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from segment import predict

root = Path(__file__).resolve().parent

image_path = root / "test_images" / "570.jpg"

bgr = cv2.imread(str(image_path))

if bgr is None:
    raise FileNotFoundError(image_path)

img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

img = cv2.resize(
    img,
    (800, 600),
    interpolation=cv2.INTER_AREA
)

print("Running U-Net prediction...")

pred, probs = predict(img)

print("Prediction completed!")
print("Mask shape:", pred.shape)
print("Probability shape:", probs.shape)
print("Predicted classes:", np.unique(pred))
print("Class pixel counts:", np.bincount(pred.ravel(), minlength=24))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred, cmap="tab20", vmin=0, vmax=23)
plt.title("U-Net Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig(root / "prediction_570.png", dpi=120)
plt.show()