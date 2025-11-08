#!/usr/bin/env python3
# scripts/merge_runs_keep_splits.py
"""
Merge any number of run folders into one merged dataset.

Assumptions:
- Each input RUN has:
    RUN/
      images/
      index_split.json   # records include "split": train|val|test

Behavior:
- Preserves the existing split labels.
- Copies or symlinks images into <out>/images/.
- Avoids filename collisions by prefixing each image with the run folder name:
    run_bag1__bag_1_<ts>.png
- Writes <out>/index_split.json with updated image_path values.

Usage:
  python3 merge_dataset.py --runs data/processed/run_bag1 data/processed/run_bag2 --out  data/processed/merged_dataset --copy
  # add --copy to copy instead of symlink
"""

import argparse, json, os, shutil, sys
from pathlib import Path

def merge_runs(run_dirs, out_dir, copy_files=False):
    out_dir = Path(out_dir)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    merged = []
    n_imgs = 0

    for run in run_dirs:
        run = Path(run).resolve()
        idx_path = run / "index_split.json"
        img_dir  = run / "images"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing {idx_path}")
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing images folder: {img_dir}")

        with open(idx_path, "r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"{idx_path} must contain a list of records")

        run_name = run.name  # e.g., run_bag1

        for rec in entries:
            # Resolve source image path
            p = rec.get("image_path")
            if p is None:
                continue
            src = (run / p) if not os.path.isabs(p) else Path(p)
            if not src.exists():
                # fallback if image_path is just a filename
                fallback = img_dir / Path(p).name
                if fallback.exists():
                    src = fallback
                else:
                    print(f"[WARN] Missing image, skipping: {src}", file=sys.stderr)
                    continue

            # Destination filename (namespaced by run)
            dst_name = f"{run_name}__{src.name}"
            dst = out_images / dst_name

            if not dst.exists():
                if copy_files:
                    shutil.copy2(src, dst)
                else:
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                n_imgs += 1

            # Build merged record with preserved split and new image_path
            new_rec = dict(rec)
            new_rec["image_path"] = f"images/{dst_name}"
            # ensure split exists (should already)
            new_rec["split"] = new_rec.get("split", "train")
            merged.append(new_rec)

    # Write merged index_split.json
    out_index = out_dir / "index_split.json"
    with open(out_index, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(run_dirs)} runs → {len(merged)} records")
    print(f"Images placed in: {out_images}  ({'copied' if copy_files else 'symlinked/copy-fallback'})")
    print(f"Wrote: {out_index}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run folders (each with images/ and index_split.json)")
    ap.add_argument("--out", required=True, help="Output merged dataset folder")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    args = ap.parse_args()

    merge_runs(args.runs, args.out, copy_files=args.copy)

if __name__ == "__main__":
    main()

