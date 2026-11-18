
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from depth_module_v2 import DepthAnalysis
from candidates import generate_candidates
from semantic_brain import load_active_nodes
from cost import rank_candidates, explain, DEFAULT_K


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

W, H = 800, 600
TARGET = (400, 300)

NAME = sys.argv[1] if len(sys.argv) > 1 else "570"

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


def main():

    # --------------------------------------------------
    # 1. LOAD IMAGE AND GROUND-TRUTH MASK
    # --------------------------------------------------

    print(f"\nTesting cost function on image {NAME}")

    image_path = ROOT / "test_images" / f"{NAME}.jpg"
    mask_path = ROOT / "test_masks" / f"{NAME}.png"

    image_bgr = cv2.imread(str(image_path))
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

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

    # --------------------------------------------------
    # 2. INITIALIZE DEPTH ANALYSIS
    # --------------------------------------------------

    print("\nRunning MiDaS depth analysis...")

    da = DepthAnalysis(image_rgb)

    # --------------------------------------------------
    # 3. GENERATE LANDING CANDIDATES
    # --------------------------------------------------

    print("\nGenerating landing candidates...")

    common = dict(
        box_w=60,
        box_h=40,
        min_safe_frac=0.995,
        stride=16,
        max_depth_checks=300,
        top_k=50,
        nms_iou=0.0,
        allow_conditional=True
    )

    result = generate_candidates(
        gt,
        TARGET,
        da,
        **common
    )

    candidates = result["candidates"]

    print("\n========== CANDIDATE SEARCH ==========")
    print("Status:", result["status"])
    print("Candidates generated:", len(candidates))
    print("Candidates rejected:", result["n_rejected"])

    if not candidates:
        print("No valid landing candidates found.")
        return

    # --------------------------------------------------
    # 4. TEST EACH PACKAGE CONFIGURATION
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(PACKAGES),
        figsize=(21, 5)
    )

    for ax, (package_name, traits) in zip(
        axes, PACKAGES.items()
    ):

        print("\n" + "=" * 65)
        print("PACKAGE CONFIGURATION:", package_name)
        print("=" * 65)

        active = load_active_nodes({
            "package": traits
        })

        out = rank_candidates(
            candidates,
            gt,
            active,
            k=DEFAULT_K
        )

        if out["status"] != "ok":
            print("No valid candidate.")
            ax.axis("off")
            continue

        # Print winner explanation
        print(explain(out, top=3))

        # Print top five candidates
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

        # --------------------------------------------------
        # 5. VISUALIZATION
        # --------------------------------------------------

        ax.imshow(image_rgb)

        # Draw target location
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
                label = "Best candidate"
            else:
                color = "yellow"
                label = "Other candidates" if i == 1 else None

            polygon = plt.Polygon(
                corners,
                fill=False,
                edgecolor=color,
                linewidth=2,
                label=label,
                zorder=5
            )

            ax.add_patch(polygon)

        winner = out["winner"]

        ax.set_title(
            f"{package_name}\n"
            f"Winner: {winner['terrain']}\n"
            f"Cost: {winner['cost']:.4f}"
        )

        ax.legend(
            loc="upper right",
            fontsize=8
        )

        ax.axis("off")

    # --------------------------------------------------
    # 6. SAVE VISUALIZATION
    # --------------------------------------------------

    plt.tight_layout()

    output_path = ROOT / f"cost_{NAME}.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("\nSaved visualization:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
