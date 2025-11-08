#!/usr/bin/env python3
import json, random, argparse, os
from math import floor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to index.json")
    # ap.add_argument("--out", default="index_split.json", help="Output filename")
    ap.add_argument("--out", default=None, help="Output filename")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = ap.parse_args()

    with open(args.index, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input index must be a list of records.")

    if args.out is None:
        index_dir = os.path.dirname(args.index) or "."
        stem, ext = os.path.splitext(os.path.basename(args.index))
        args.out = os.path.join(index_dir, f"{stem}_split{ext}")

    rng = random.Random(args.seed)
    idxs = list(range(len(data)))
    rng.shuffle(idxs)

    n = len(idxs)
    n_train = round(0.70 * n)
    n_val   = round(0.20 * n)
    # ensure total matches n
    n_test  = n - n_train - n_val

    # edge-case guards to avoid empty splits on very small datasets
    if n >= 3:
        if n_train == 0: n_train = 1
        if n_val   == 0: n_val   = 1
        n_test = max(1, n - n_train - n_val)

    train_set = set(idxs[:n_train])
    val_set   = set(idxs[n_train:n_train+n_val])
    test_set  = set(idxs[n_train+n_val:])

    for i, rec in enumerate(data):
        if i in train_set:
            rec["split"] = "train"
        elif i in val_set:
            rec["split"] = "val"
        else:
            rec["split"] = "test"

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Total: {n}  -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
