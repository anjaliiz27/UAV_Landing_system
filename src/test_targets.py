
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
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

IMAGE_NAME = "202"

W, H = 800, 600

TARGETS = [
    (200, 200),
    (400, 300),
    (600, 400)
]

PACKAGE = {
    "fragile": True,
    "valuable": True
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
# MAIN
# ==================================================

def main():

    # ----------------------------------------------
    # 1. LOAD IMAGE
    # ----------------------------------------------

    print(f"\nTesting target positions on image {IMAGE_NAME}")

    image_path = ROOT / "test_images" / f"{IMAGE_NAME}.jpg"

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    image_bgr = cv2.resize(image_bgr, (W, H))

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # ----------------------------------------------
    # 2. RUN U-NET
    # ----------------------------------------------

    print("\nRunning U-Net prediction...")

    pred, probabilities = predict(
        image_rgb,
        clean=True
    )

    print("Prediction shape:", pred.shape)

    # ----------------------------------------------
    # 3. RUN MIDAS
    # ----------------------------------------------

    print("\nRunning MiDaS depth analysis...")

    da = DepthAnalysis(image_rgb)

    # ----------------------------------------------
    # 4. ACTIVATE PACKAGE PROPERTIES
    # ----------------------------------------------

    active = load_active_nodes({
        "package": PACKAGE
    })

    print("\nPackage configuration:", PACKAGE)

    # ----------------------------------------------
    # 5. PREPARE VISUALIZATION
    # ----------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(TARGETS),
        figsize=(18, 6)
    )

    results = []

    # ----------------------------------------------
    # 6. TEST EACH TARGET
    # ----------------------------------------------

    for i, target in enumerate(TARGETS):

        print("\n" + "=" * 65)
        print("TARGET:", target)
        print("=" * 65)

        # Generate candidates for this target
        result = generate_candidates(
            pred,
            target,
            da,
            **COMMON
        )

        candidates = result["candidates"]

        print("Search status:", result["status"])
        print("Candidates generated:", len(candidates))
        print("Candidates rejected:", result["n_rejected"])

        ax = axes[i]

        ax.imshow(image_rgb)

        # Draw target
        ax.scatter(
            target[0],
            target[1],
            color="red",
            marker="*",
            s=250,
            edgecolors="white",
            linewidths=1.2,
            label="Target",
            zorder=10
        )

        if not candidates:

            print("No valid candidates for this target.")

            ax.set_title(f"Target: {target}\nNo candidates")
            ax.axis("off")
            continue

        # Rank candidates using KG and cost function
        out = rank_candidates(
            candidates,
            pred,
            active,
            k=DEFAULT_K
        )

        if out["status"] != "ok":

            print("Ranking failed.")

            ax.set_title(f"Target: {target}\nNo valid result")
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

        winner = out["winner"]

        results.append({
            "target": target,
            "winner": winner,
            "candidate_count": len(candidates)
        })

        # ------------------------------------------
        # DRAW TOP 10 CANDIDATES
        # ------------------------------------------

        for j, item in enumerate(out["ranked"][:10]):

            corners = np.asarray(
                item["corners"],
                dtype=np.int32
            )

            color = "lime" if j == 0 else "yellow"

            ax.add_patch(
                plt.Polygon(
                    corners,
                    fill=False,
                    edgecolor=color,
                    linewidth=2
                )
            )

        ax.set_title(
            f"Target: {target}\n"
            f"Winner: {winner['terrain']}\n"
            f"Location: ({winner['x']}, {winner['y']})\n"
            f"Cost: {winner['cost']:.4f}"
        )

        ax.legend(
            loc="upper right",
            fontsize=8
        )

        ax.axis("off")

    # ----------------------------------------------
    # 7. SAVE VISUALIZATION
    # ----------------------------------------------

    plt.suptitle(
        f"Target Position Comparison - Image {IMAGE_NAME}\n"
        f"Package: Fragile + Valuable",
        fontsize=15
    )

    plt.tight_layout()

    output_path = ROOT / f"target_comparison_{IMAGE_NAME}.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("\nVisualization saved:", output_path)

    plt.show()
    plt.close(fig)

    # ----------------------------------------------
    # 8. FINAL COMPARISON
    # ----------------------------------------------

    print("\n" + "=" * 65)
    print("TARGET COMPARISON SUMMARY")
    print("=" * 65)

    for item in results:

        winner = item["winner"]

        print(
            f"Target={item['target']} | "
            f"Winner=({winner['x']}, {winner['y']}) | "
            f"Terrain={winner['terrain']} | "
            f"Cost={winner['cost']:.4f} | "
            f"Candidates={item['candidate_count']}"
        )


if __name__ == "__main__":
    main()
