
import cv2
import csv
import numpy as np
from pathlib import Path

from segment import predict
from depth_module_v2 import DepthAnalysis
from candidates import generate_candidates
from semantic_brain import load_active_nodes
from cost import rank_candidates, DEFAULT_K


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

IMAGE_NAMES = ["570", "489", "179", "202"]

TARGETS = [
    (200, 200),
    (400, 300),
    (600, 400)
]

PACKAGE = {
    "fragile": True,
    "valuable": True
}

W, H = 800, 600

SAFE_CLASSES = {1, 2, 3}
SAFETY_THRESHOLD = 0.995

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
# GROUND-TRUTH SAFETY EVALUATION
# ==================================================

def evaluate_candidate(candidate, gt):

    polygon = np.round(
        candidate["corners"]
    ).astype(np.int32)

    region = np.zeros(
        gt.shape,
        dtype=np.uint8
    )

    cv2.fillPoly(
        region,
        [polygon],
        1
    )

    pixels = gt[region == 1]

    if len(pixels) == 0:
        return 0.0, 0, 0

    safe_pixels = np.isin(
        pixels,
        list(SAFE_CLASSES)
    )

    safe_count = int(np.sum(safe_pixels))
    total_count = len(pixels)

    return safe_count / total_count, safe_count, total_count


# ==================================================
# PROCESS ONE IMAGE
# ==================================================


def process_image(name, active):

    print("\n" + "#" * 70)
    print(f"PROCESSING IMAGE {name}")
    print("#" * 70)

    image_path = ROOT / "test_images" / f"{name}.jpg"
    mask_path = ROOT / "test_masks" / f"{name}.png"

    image_bgr = cv2.imread(str(image_path))
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    if gt is None:
        raise FileNotFoundError(mask_path)

    image_bgr = cv2.resize(image_bgr, (W, H))
    gt = cv2.resize(
        gt, (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # U-NET PREDICTION
    print("\nRunning U-Net prediction...")

    pred, probabilities = predict(image_rgb, clean=True)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: prediction={pred.shape}, GT={gt.shape}"
        )

    agreement = np.mean(pred == gt) * 100
    print(f"Pixel agreement: {agreement:.2f}%")

    # MIDAS DEPTH ANALYSIS
    print("\nRunning MiDaS depth analysis...")
    da = DepthAnalysis(image_rgb)

    image_results = []

    # TEST EACH TARGET
    for target in TARGETS:

        print("\n" + "-" * 65)
        print("TARGET:", target)
        print("-" * 65)

        result = generate_candidates(
            pred,
            target,
            da,
            **COMMON
        )

        candidates = result["candidates"]

        print("Search status:", result["status"])
        print("Candidates generated:", len(candidates))

        if not candidates:
            print("No candidates found.")

            image_results.append({
                "image": name,
                "target": str(target),
                "winner": "None",
                "terrain": "None",
                "cost": "",
                "safe_fraction": 0.0,
                "safe": False,
                "agreement": agreement
            })
            continue

        # COST RANKING
        ranked = rank_candidates(
            candidates,
            pred,
            active,
            k=DEFAULT_K
        )

        if ranked["status"] != "ok":
            print("No valid ranked candidate.")

            image_results.append({
                "image": name,
                "target": str(target),
                "winner": "None",
                "terrain": "None",
                "cost": "",
                "safe_fraction": 0.0,
                "safe": False,
                "agreement": agreement
            })
            continue

        winner = ranked["winner"]

        # GROUND-TRUTH EVALUATION OF WINNER
        safe_fraction, safe_count, total_count = (
            evaluate_candidate(winner, gt)
        )

        passed = safe_fraction >= SAFETY_THRESHOLD

        print("\nSELECTED LANDING SITE")
        print("Location:", (winner["x"], winner["y"]))
        print("Terrain:", winner["terrain"])
        print(f"Cost: {winner['cost']:.4f}")
        print(f"GT safe pixels: {safe_count}/{total_count}")
        print(f"GT safe fraction: {safe_fraction:.4f}")
        print(f"Safety percentage: {safe_fraction * 100:.2f}%")
        print("Safety evaluation:", "PASS" if passed else "FAIL")

        # INVESTIGATE FAILED WINNER
        if not passed:

            print("\n" + "!" * 65)
            print("FAILED WINNER INVESTIGATION")
            print("!" * 65)

            polygon = np.round(
                winner["corners"]
            ).astype(np.int32)

            region = np.zeros(gt.shape, dtype=np.uint8)
            cv2.fillPoly(region, [polygon], 1)

            pred_pixels = pred[region == 1]
            gt_pixels = gt[region == 1]

            pred_safe = np.isin(
                pred_pixels, list(SAFE_CLASSES)
            )

            gt_safe = np.isin(
                gt_pixels, list(SAFE_CLASSES)
            )

            false_safe = np.sum(pred_safe & ~gt_safe)
            false_unsafe = np.sum(~pred_safe & gt_safe)

            print("Predicted-safe but GT-unsafe pixels:", int(false_safe))
            print("Predicted-unsafe but GT-safe pixels:", int(false_unsafe))
            print("Predicted safe fraction:", f"{np.mean(pred_safe):.4f}")
            print("Ground-truth safe fraction:", f"{np.mean(gt_safe):.4f}")

            # Evaluate alternative candidates.
            # Ground truth is used ONLY for offline evaluation,
            # never for actual candidate selection.
            print("\nALTERNATIVE CANDIDATE ANALYSIS")
            print("-" * 65)

            ranked_candidates = ranked["ranked"]

            safe_alternative_found = False

            for i, candidate in enumerate(ranked_candidates):

                alt_fraction, alt_safe, alt_total = (
                    evaluate_candidate(candidate, gt)
                )

                print(
                    f"Rank {i + 1}: "
                    f"Location=({candidate['x']}, {candidate['y']}), "
                    f"Terrain={candidate['terrain']}, "
                    f"Cost={candidate['cost']:.4f}, "
                    f"GT safe={alt_fraction * 100:.2f}%"
                )

                if (
                    not safe_alternative_found
                    and alt_fraction >= SAFETY_THRESHOLD
                ):
                    print("\nFirst ground-truth-safe alternative:")
                    print(
                        "Location:",
                        (candidate["x"], candidate["y"])
                    )
                    print("Terrain:", candidate["terrain"])
                    print(f"Cost: {candidate['cost']:.4f}")
                    print(f"GT safe fraction: {alt_fraction:.4f}")

                    safe_alternative_found = True

            if not safe_alternative_found:
                print(
                    "\nNo alternative among the ranked candidates "
                    "met the ground-truth safety threshold."
                )

        image_results.append({
            "image": name,
            "target": str(target),
            "winner": f"({winner['x']}, {winner['y']})",
            "terrain": winner["terrain"],
            "cost": winner["cost"],
            "safe_fraction": safe_fraction,
            "safe": passed,
            "agreement": agreement
        })

    return image_results


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 75)
    print("BATCH LANDING-SITE SAFETY EVALUATION")
    print("=" * 75)

    print("Images:", IMAGE_NAMES)
    print("Targets:", TARGETS)
    print("Package:", PACKAGE)
    print("Safety threshold:", SAFETY_THRESHOLD)

    active = load_active_nodes({
        "package": PACKAGE
    })

    print("Active KG nodes:", active)

    all_results = []

    # ----------------------------------------------
    # PROCESS ALL FOUR IMAGES
    # ----------------------------------------------

    for name in IMAGE_NAMES:

        results = process_image(name, active)

        all_results.extend(results)

    # ----------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------

    print("\n" + "=" * 95)
    print("COMPLETE SAFETY EVALUATION SUMMARY")
    print("=" * 95)

    print(
        f"{'Image':<10}"
        f"{'Target':<18}"
        f"{'Winner':<20}"
        f"{'Terrain':<12}"
        f"{'Safe %':<12}"
        f"{'Result'}"
    )

    for item in all_results:

        print(
            f"{item['image']:<10}"
            f"{item['target']:<18}"
            f"{item['winner']:<20}"
            f"{item['terrain']:<12}"
            f"{item['safe_fraction'] * 100:<12.2f}"
            f"{'PASS' if item['safe'] else 'FAIL'}"
        )

    # ----------------------------------------------
    # AGGREGATE METRICS
    # ----------------------------------------------

    total = len(all_results)

    passed = sum(
        item["safe"] for item in all_results
    )

    failed = total - passed

    if total > 0:

        pass_rate = passed / total * 100

        print("\n" + "-" * 75)
        print("AGGREGATE RESULTS")
        print("-" * 75)

        print("Total evaluations:", total)
        print("Passed:", passed)
        print("Failed:", failed)

        print(f"Safety pass rate: {pass_rate:.2f}%")

        mean_safety = np.mean([
            item["safe_fraction"]
            for item in all_results
        ])

        print(
            f"Mean ground-truth safe fraction: {mean_safety:.4f}"
        )

    # ----------------------------------------------
    # SAVE CSV RESULTS
    # ----------------------------------------------

    csv_path = ROOT / "batch_safety_results.csv"

    fieldnames = [
        "image",
        "target",
        "winner",
        "terrain",
        "cost",
        "safe_fraction",
        "safe",
        "agreement"
    ]

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nResults saved to:", csv_path)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
