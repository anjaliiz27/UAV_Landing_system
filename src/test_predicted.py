
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from segment import predict
from depth_module_v2 import DepthAnalysis
from candidates import generate_candidates


# Project paths
ROOT = Path(__file__).resolve().parent

# Landing-site semantic policy
SAFE_CLASSES = {1, 2, 3}


def main():

    # Get image name from terminal
    name = sys.argv[1] if len(sys.argv) > 1 else "570"

    image_path = ROOT / "test_images" / f"{name}.jpg"
    mask_path = ROOT / "test_masks" / f"{name}.png"

    # Load image and ground-truth mask
    image_bgr = cv2.imread(str(image_path))
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if gt is None:
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

    # Resize both to the project resolution
    image_bgr = cv2.resize(image_bgr, (800, 600))
    gt = cv2.resize(
        gt,
        (800, 600),
        interpolation=cv2.INTER_NEAREST
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------
    # 1. U-Net prediction
    # --------------------------------------------------

    print("\nRunning U-Net prediction...")

    pred, probabilities = predict(image_rgb, clean=True)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Mask shape mismatch: GT={gt.shape}, Prediction={pred.shape}"
        )

    agreement = np.mean(pred == gt) * 100

    print(f"\nImage: {name}")
    print(f"Pixel agreement: {agreement:.2f}%")

    # --------------------------------------------------
    # 2. MiDaS depth analysis
    # --------------------------------------------------

    print("\nRunning MiDaS depth analysis...")

    da = DepthAnalysis(image_rgb)

    # --------------------------------------------------
    # 3. Candidate search on both masks
    # --------------------------------------------------

    target = (400, 300)

    common = dict(
        box_w=60,
        box_h=40,
        min_safe_frac=0.995,
        stride=16,
        max_depth_checks=300,
        top_k=50,
        nms_iou=0.0,
        allow_conditional=False
    )

    print("\nRunning candidate search on ground-truth mask...")

    gt_result = generate_candidates(
        gt,
        target,
        da,
        **common
    )

    print("\nRunning candidate search on predicted mask...")

    pred_result = generate_candidates(
        pred,
        target,
        da,
        **common
    )

    # Extract candidate lists from returned dictionaries
    gt_candidates = gt_result["candidates"]
    pred_candidates = pred_result["candidates"]

    print("\n========== CANDIDATE SEARCH RESULTS ==========")

    print(f"Ground-truth status: {gt_result['status']}")
    print(f"Predicted status: {pred_result['status']}")

    print(f"Ground-truth candidates: {len(gt_candidates)}")
    print(f"Predicted candidates: {len(pred_candidates)}")

    print(f"GT rejected: {gt_result['n_rejected']}")
    print(f"Predicted rejected: {pred_result['n_rejected']}")

    # --------------------------------------------------
    # 4. Evaluate predicted candidates using GT mask
    # --------------------------------------------------

    print("\n========== PREDICTED CANDIDATE SAFETY ==========")

    truly_safe = 0
    safety_scores = []

    for i, candidate in enumerate(pred_candidates):

        polygon = np.round(
            candidate["corners"]
        ).astype(np.int32)

        region = np.zeros(gt.shape, dtype=np.uint8)

        cv2.fillPoly(
            region,
            [polygon],
            1
        )

        pixels = gt[region == 1]

        if len(pixels) == 0:
            safe_fraction = 0.0
        else:
            safe_fraction = np.mean(
                np.isin(pixels, list(SAFE_CLASSES))
            )

        safety_scores.append(safe_fraction)

        if safe_fraction >= 0.995:
            truly_safe += 1

        print(
            f"Candidate {i + 1}: "
            f"GT safe fraction = {safe_fraction:.4f}"
        )

    total_predicted = len(pred_candidates)

    print("\n========== SAFETY SUMMARY ==========")

    print(
        f"Truly safe predicted candidates: "
        f"{truly_safe}/{total_predicted}"
    )

    if total_predicted > 0:

        safety_percentage = (
            truly_safe / total_predicted
        ) * 100

        print(
            f"Safety percentage: {safety_percentage:.2f}%"
        )

        print(
            f"Average GT safe fraction: "
            f"{np.mean(safety_scores):.4f}"
        )

    else:
        print("No predicted candidates were found.")

    # --------------------------------------------------
    # 5. Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Aerial Image")

    # Ground-truth mask
    axes[0, 1].imshow(
        gt,
        cmap="tab20",
        vmin=0,
        vmax=23
    )
    axes[0, 1].set_title("Ground-Truth Mask")

    # Predicted mask
    axes[1, 0].imshow(
        pred,
        cmap="tab20",
        vmin=0,
        vmax=23
    )
    axes[1, 0].set_title(
        f"U-Net Predicted Mask ({agreement:.2f}% agreement)"
    )

    # Candidate comparison
    axes[1, 1].imshow(image_rgb)
    axes[1, 1].set_title(
        "Candidate Search: GT (Green) vs Prediction (Yellow)"
    )

    # Draw ground-truth candidates
    for c in gt_candidates[:10]:

        polygon = np.array(
            c["corners"],
            dtype=np.int32
        )

        axes[1, 1].add_patch(
            plt.Polygon(
                polygon,
                fill=False,
                edgecolor="lime",
                linewidth=1.5
            )
        )

    # Draw predicted candidates
    for c in pred_candidates[:10]:

        polygon = np.array(
            c["corners"],
            dtype=np.int32
        )

        axes[1, 1].add_patch(
            plt.Polygon(
                polygon,
                fill=False,
                edgecolor="yellow",
                linewidth=1.5
            )
        )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    # Save comparison
    output_path = ROOT / f"pred_vs_gt_{name}.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"\nComparison image saved at: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
