import json, random, argparse, os
import numpy as np
from tqdm import tqdm

def main():
    # === CONFIG ===
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to folder containing index.json or index_split.json")
    ap.add_argument("--out", default=None, help="Output filename")
    ap.add_argument("--win", default=None, help="Smoothing wnidow size")
    args = ap.parse_args()

    if args.out is None:
        index_dir = os.path.dirname(args.index) or "."
        stem, ext = os.path.splitext(os.path.basename(args.index))
        args.out = os.path.join(index_dir, f"index_smooth{ext}")


    DATA_DIR = args.out   # folder with your index.json
    WINDOW = 5                            # smoothing half-window size (in frames)
    #
    # INPUT_JSON = os.path.join(DATA_DIR, "index_split.json")
    # OUTPUT_JSON = os.path.join(DATA_DIR, "index_smooth.json")
    INPUT_JSON = args.index
    OUTPUT_JSON = args.out
    print(INPUT_JSON)
    print(OUTPUT_JSON)
    # === LOAD ===
    with open(INPUT_JSON, "r") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {INPUT_JSON}")

    # Extract omegas
    omegas = np.array([r.get("label", {}).get("omega", 0.0) for r in records], dtype=float)

    # === SMOOTHING ===
    kernel = np.ones(2 * WINDOW + 1)
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(omegas, kernel, mode="same")

    # === ASSIGN BACK ===
    for i, r in enumerate(records):
        if "label" not in r:
            r["label"] = {}
        r["label"]["omega"] = float(smoothed[i])

    # === SAVE ===
    with open(OUTPUT_JSON, "w") as f:
        json.dump(records, f, indent=2)

    # === REPORT ===
    abs_diff = np.abs(smoothed - omegas)
    print(f"  Smoothed {len(records)} records")
    print(f"  mean |delta(omega)| = {abs_diff.mean():.4f}")
    print(f"  max  |delta(omega)| = {abs_diff.max():.4f}")
    print(f"  output saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
