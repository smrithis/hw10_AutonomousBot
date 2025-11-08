#!/usr/bin/env python3
"""
Plots omega and linear_x using only the 'label' field in each record.
X-axis is the record index (no use of 't').
"""

import json
import gzip
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def _open_json(path: str):
    p = Path(path)
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def load_json_any(path: str):
    """Load JSON array or JSON Lines into a list of dicts."""
    with _open_json(path) as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            f.seek(0)
            records = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            records.extend(obj)
                        else:
                            records.append(obj)
                    except Exception:
                        continue
            return records
    return []


def extract_label_series(records):
    """Extract (omega, linear_x) per record from 'label'."""
    omegas, vels = [], []
    for r in records:
        if not isinstance(r, dict):
            continue
        label = r.get("label")
        if not isinstance(label, dict):
            continue

        omega = label.get("omega")
        vx = label.get("linear_x")

        # ensure numeric
        if not isinstance(omega, (int, float)) or not isinstance(vx, (int, float)):
            continue

        omegas.append(float(omega))
        vels.append(float(vx))

    print(f"Loaded {len(records)} records, plotting {len(omegas)} valid samples.")
    return omegas, vels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to JSON file (.json or .json.gz)")
    parser.add_argument("--title", default="Omega and Linear Velocity from Label")
    args = parser.parse_args()

    records = load_json_any(args.json_file)
    omegas, vels = extract_label_series(records)

    if not omegas:
        print("No valid label data found.")
        return

    x = list(range(len(omegas)))  # index as x-axis

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(x, omegas, label="omega")
    plt.ylabel("omega")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(x, vels, label="linear_x", color="orange")
    plt.xlabel("record index")
    plt.ylabel("linear_x")
    plt.grid(True)

    plt.suptitle(args.title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
