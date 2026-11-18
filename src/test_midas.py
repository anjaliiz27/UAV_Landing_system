
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from depth_module_v2 import DepthAnalysis

IMAGE_DIR = Path(__file__).resolve().parent / "test_images"
OUTPUT_DIR = Path(__file__).resolve().parent / "midas_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

if not image_paths:
    raise FileNotFoundError("No test images found!")

fig, axes = plt.subplots(
    len(image_paths), 4,
    figsize=(18, 5 * len(image_paths))
)

def visualize(arr):
    """Normalize only for visualization, not for calculations."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    normalized = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    return normalized

for i, path in enumerate(image_paths):

    print("\nProcessing:", path.name)

    bgr = cv2.imread(str(path))
    if bgr is None:
        print("Could not read:", path.name)
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Match the project's common image dimensions
    rgb = cv2.resize(rgb, (800, 600))

    # Run MiDaS and depth analysis
    da = DepthAnalysis(rgb, size=256)

    # Save numerical outputs for later integration
    stem = path.stem
    np.save(OUTPUT_DIR / f"{stem}_depth.npy", da.depth)
    np.save(OUTPUT_DIR / f"{stem}_relief.npy", da.local)
    np.save(OUTPUT_DIR / f"{stem}_gradient.npy", da.grad)

    print("Depth shape:", da.depth.shape)
    print("Depth range:", da.depth.min(), da.depth.max())
    print("Local relief std:", da.local.std())
    print("Gradient mean:", da.grad.mean())
    print("Gradient 95th percentile:", np.percentile(da.grad, 95))

    # Display results
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"Original: {stem}")

    axes[i, 1].imshow(visualize(da.depth), cmap="inferno")
    axes[i, 1].set_title("MiDaS depth")

    axes[i, 2].imshow(visualize(da.local), cmap="coolwarm")
    axes[i, 2].set_title("Local relief")

    axes[i, 3].imshow(visualize(da.grad), cmap="inferno")
    axes[i, 3].set_title("Depth gradient")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "midas_comparison.png", dpi=150)
plt.show()

print("\nAll outputs saved in:", OUTPUT_DIR)
