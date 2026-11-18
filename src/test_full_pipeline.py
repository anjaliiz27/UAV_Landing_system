
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from segment import predict
from depth_module_v2 import DepthAnalysis
from candidates import generate_candidates
from semantic_brain import load_active_nodes
from cost import rank_candidates, explain, DEFAULT_K


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

W, H = 800, 600
TARGETS = [
    (200, 200),
    (400, 300),
    (600, 400)
]

IMAGE_NAMES = ["570", "489", "179", "202"]


PACKAGES = {
    "none": {},
    "fragile": {"fragile": True},
    "heavy": {"heavy": True},
    "valuable": {"valuable": True},
    "biohazard": {"biohazard": True},
    "fragile+valuable": {
        "fragile": True,
        "valuable": True
    }
}

COMMON = {
    "box_w": 60,
    "box_h": 40,
    "min_safe_frac": 0.995,
    "stride": 16,
    "max_depth_checks": 300,
    "top_k": 50,
    "nms_iou": 0.0,
    "allow_conditional": False
}


# ==================================================
# 2. PROCESS ONE IMAGE
# ==================================================

def process_image(name):

    print("\n" + "#" * 75)
    print(f"PROCESSING IMAGE {name}")
    print("#" * 75)

    # ----------------------------------------------
    # LOAD IMAGE AND GROUND-TRUTH MASK
    # ----------------------------------------------

    image_path = ROOT / "test_images" / f"{name}.jpg"
    mask_path = ROOT / "test_masks" / f"{name}.png"

    image_bgr = cv2.imread(str(image_path))
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if gt is None:
        raise FileNotFoundError(
            f"Ground-truth mask not found: {mask_path}"
        )

    image_bgr = cv2.resize(image_bgr, (W, H))

    gt = cv2.resize(
        gt,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # ----------------------------------------------
    # RUN U-NET PREDICTION
    # ----------------------------------------------

    print("\nRunning U-Net prediction...")

    pred, probabilities = predict(
        image_rgb,
        clean=True
    )

    if pred.shape != gt.shape:
        raise ValueError(
            f"Mask shape mismatch: GT={gt.shape}, "
            f"Prediction={pred.shape}"
        )

    agreement = np.mean(pred == gt) * 100

    print(f"Pixel agreement: {agreement:.2f}%")

    # ----------------------------------------------
    # RUN MIDAS DEPTH ANALYSIS
    # ----------------------------------------------

    print("\nRunning MiDaS depth analysis...")

    da = DepthAnalysis(image_rgb)

    # ----------------------------------------------
    # GENERATE CANDIDATES USING PREDICTED MASK
    # ----------------------------------------------

    print("\nGenerating candidates using predicted mask...")

    result = generate_candidates(
        pred,
        TARGET,
        da,
        **COMMON
    )

    candidates = result["candidates"]

    print("\n========== CANDIDATE SEARCH ==========")
    print("Status:", result["status"])
    print("Candidates generated:", len(candidates))
    print("Candidates rejected:", result["n_rejected"])

    if not candidates:
        print("No valid landing candidates found.")
        return None

    # ----------------------------------------------
    # PREPARE VISUALIZATION
    # ----------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11)
    )

    axes = axes.flatten()

    package_results = {}

    # ----------------------------------------------
    # KNOWLEDGE GRAPH + COST RANKING
    # ----------------------------------------------

    for index, (package_name, traits) in enumerate(
        PACKAGES.items()
    ):

        print("\n" + "=" * 65)
        print("PACKAGE CONFIGURATION:", package_name)
        print("=" * 65)

        # Activate KG source nodes
        active = load_active_nodes({
            "package": traits
        })

        # Rank candidates
        out = rank_candidates(
            candidates,
            pred,
            active,
            k=DEFAULT_K
        )

        if out["status"] != "ok":
            print("No valid landing candidate.")
            axes[index].axis("off")
            continue

        package_results[package_name] = out

        # Print explanation
        print(explain(out, top=3))

        # ------------------------------------------
        # PRINT TOP FIVE CANDIDATES
        # ------------------------------------------

        print("\nTop 5 candidates:")

        for item in out["ranked"][:5]:

            print(
                f"#{item['rank']} "
                f"Location=({item['x']}, {item['y']}) "
                f"Terrain={item['terrain']} "
                f"D={item['D_norm']:.4f} "
                f"R={item['R_norm']:.4f} "
                f"S={item['S']:.4f} "
                f"Cost={item['cost']:.4f}"
            )

        # ------------------------------------------
        # CANDIDATE TERRAIN DISTRIBUTION
        # ------------------------------------------

        terrain_counts = {}

        for item in out["ranked"]:

            terrain = item["terrain"]

            terrain_counts[terrain] = (
                terrain_counts.get(terrain, 0) + 1
            )

        print("\nCandidate terrain distribution:")

        for terrain, count in terrain_counts.items():
            print(f"{terrain}: {count}")

        # ------------------------------------------
        # VISUALIZATION
        # ------------------------------------------

        ax = axes[index]

        ax.imshow(image_rgb)

        # Target marker
        ax.scatter(
            TARGET[0],
            TARGET[1],
            color="red",
            marker="*",
            s=250,
            edgecolors="white",
            linewidths=1.2,
            label="Target",
            zorder=10
        )

        # Draw top three candidates
        for i, item in enumerate(out["ranked"][:3]):

            corners = np.asarray(
                item["corners"],
                dtype=np.int32
            )

            if i == 0:
                color = "lime"
            else:
                color = "yellow"

            ax.add_patch(
                plt.Polygon(
                    corners,
                    fill=False,
                    edgecolor=color,
                    linewidth=2
                )
            )

        winner = out["winner"]

        ax.set_title(
            f"{package_name}\n"
            f"Terrain: {winner['terrain']}\n"
            f"Cost: {winner['cost']:.4f}"
        )

        ax.legend(
            loc="upper right",
            fontsize=8
        )

        ax.axis("off")

    # ----------------------------------------------
    # SHOW IMAGE INFORMATION IN LAST PANEL
    # ----------------------------------------------

    axes[5].imshow(image_rgb)

    axes[5].set_title(
        f"Image {name}\n"
        f"Pixel agreement: {agreement:.2f}%\n"
        f"Candidates: {len(candidates)}"
    )

    axes[5].axis("off")

    # ----------------------------------------------
    # SAVE VISUALIZATION
    # ----------------------------------------------

    plt.suptitle(
        f"Full Pipeline Results - Image {name}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = ROOT / f"full_pipeline_{name}.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("\nVisualization saved:", output_path)

    plt.show()
    plt.close(fig)

    return {
        "image": name,
        "agreement": agreement,
        "candidate_count": len(candidates),
        "results": package_results
    }


# ==================================================
# 3. MAIN FUNCTION
# ==================================================

def main():

    all_results = []

    for name in IMAGE_NAMES:

        result = process_image(name)

        if result is not None:
            all_results.append(result)

    # ----------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------

    print("\n" + "#" * 75)
    print("FULL PIPELINE SUMMARY")
    print("#" * 75)

    print(
        f"{'Image':<12}"
        f"{'Agreement':<18}"
        f"{'Candidates':<15}"
    )

    for result in all_results:

        print(
            f"{result['image']:<12}"
            f"{result['agreement']:<18.2f}"
            f"{result['candidate_count']:<15}"
        )

    print("\nAll image tests completed.")


if __name__ == "__main__":
    main()
