# eval_image_only.py
"""
Evaluate ImageOnlySteerNet (image → scalar ω), no history.

Usage:
  python eval_image_only.py \
    --index data/processed/run_YYYYMMDD/index_split.json \
    --root  data/processed/run_YYYYMMDD \
    --ckpt ckpt_best.pt --split val \
    --outdir eval_out_imageonly --save-csv --save-overlays \
    --short-side 224 --top-crop 0.2 --flip-sign
"""

import os, json, math, argparse, numpy as np, torch, cv2
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_loaders
from model import ImageOnlySteerNet

@torch.no_grad()
def evaluate(model, loader, device, mu, sigma):
    model.eval()
    y_true, y_pred = [], []
    for x, _, y_std, y_raw, _ in loader:
        x = x.to(device, non_blocking=True)
        y_std = y_std.to(device, non_blocking=True)
        yhat_std = model(x)                 # [B]
        yhat_raw = (yhat_std * sigma + mu).cpu().numpy()
        y_true.append(y_raw.numpy())
        y_pred.append(yhat_raw)
    y_true = np.concatenate(y_true, axis=0)   # [N]
    y_pred = np.concatenate(y_pred, axis=0)   # [N]

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    sign_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    mask = (np.abs(y_true) > 0.01) | (np.abs(y_pred) > 0.01)
    sign_acc_eps = float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))) if mask.any() else float("nan")

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2,
               "sign_acc": sign_acc, "sign_acc_eps@0.01": sign_acc_eps}
    return metrics, (y_true, y_pred)

def save_csv(path, image_paths, stamps, y_true, y_pred):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "stamp_img_bag", "omega_true", "omega_pred"])
        for p, t, yt, yp in zip(image_paths, stamps, y_true, y_pred):
            w.writerow([p, t if t is not None else "", f"{yt:.6f}", f"{yp:.6f}"])

def make_figures(outdir, y_true, y_pred):
    os.makedirs(outdir, exist_ok=True)
    # scatter
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    mn = float(min(y_true.min(), y_pred.min())); mx = float(max(y_true.max(), y_pred.max()))
    if mn == mx: mn -= 1e-3; mx += 1e-3
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("True ω (rad/s)"); plt.ylabel("Pred ω (rad/s)")
    plt.grid(True, lw=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_true_vs_pred.png")); plt.close()
    # error hist
    plt.figure(figsize=(6,4))
    plt.hist(y_pred - y_true, bins=50)
    plt.xlabel("Prediction error (rad/s)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_hist.png")); plt.close()
    
        # 3) Short time-series strip
    n_strip = 300
    n = min(n_strip, len(y_true))
    xs = np.arange(n)
    plt.figure(figsize=(10,3))
    plt.plot(xs, y_true[:n], label="True", linewidth=1.0)
    plt.plot(xs, y_pred[:n], label="Pred", linewidth=1.0)
    plt.xlabel("Sample index")
    plt.ylabel("ω (rad/s)")
    plt.title(f"Time Series (first {n} samples)")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "timeseries_firstN.png"))
    plt.close()
    

def gather_meta(loader):
    paths, stamps = [], []
    for _, _, _, _, meta in loader:
        paths += list(meta["image_path"])
        stamps += list(meta["stamp_img_bag"])
    return paths, stamps

def _draw_arrow(img_bgr, angle, color, thickness=3, tip_len=0.12, length_ratio=0.28):
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h - 1
    L = int(min(h, w) * length_ratio)
    dx, dy = L * math.sin(angle), -L * math.cos(angle)
    p0, p1 = (cx, cy), (int(round(cx + dx)), int(round(cy + dy)))
    cv2.arrowedLine(img_bgr, p0, p1, color, thickness=thickness, tipLength=tip_len)

def save_overlays(outdir, paths, root, y_true, y_pred):
    os.makedirs(outdir, exist_ok=True)
    for p, yt, yp in zip(paths, y_true, y_pred):
        in_path = p if (root is None or os.path.isabs(p)) else os.path.join(root, p)
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None: continue
        _draw_arrow(img, float(yt), (0,255,0), 3)
        _draw_arrow(img, float(yp), (0,0,255), 5, length_ratio=0.15)
        cv2.putText(img, f"T={float(yt):+.3f}  P={float(yp):+.3f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(os.path.join(outdir, os.path.basename(in_path)), img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--outdir", default="eval_out_imageonly")
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--short-side", type=int, default=224)
    ap.add_argument("--top-crop", type=float, default=0.2)
    ap.add_argument("--flip-sign", action="store_true",
                    help="Flip label sign at load time (must match training).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    omega_sign = -1.0 if args.flip_sign else 1.0
    train_dl, val_dl, test_dl, stats = make_loaders(
        args.index, root=args.root, bs=args.bs,
        hist_len=0, omega_sign=omega_sign,
        short_side=args.short_side, top_crop_frac=args.top_crop
    )
    mu, sigma = stats["mu"], stats["sigma"]

    # pick split
    loader = {"train": train_dl, "val": val_dl, "test": test_dl}[args.split]
    if loader is None:
        raise ValueError(f"No '{args.split}' split found in index file.")

    # model & ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ImageOnlySteerNet(out_len=1, pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    # Prefer ckpt mu/sigma if present (sign already baked into training stats)
    mu, sigma = ckpt.get("mu", mu), ckpt.get("sigma", sigma)
    if "omega_sign" in ckpt and omega_sign != ckpt["omega_sign"]:
        print(f"Note: dataset omega_sign={omega_sign:g} but ckpt omega_sign={ckpt['omega_sign']:g}. "
              f"Continue only if you intend to evaluate with a different sign convention.")

    metrics, (y_true, y_pred) = evaluate(model, loader, device, mu, sigma)
    print(json.dumps(metrics, indent=2))

    make_figures(args.outdir, y_true, y_pred)
    paths, stamps = gather_meta(loader)

    if args.save_csv:
        save_csv(os.path.join(args.outdir, f"preds_{args.split}.csv"), paths, stamps, y_true, y_pred)
        print(f"Saved CSV: {os.path.join(args.outdir, f'preds_{args.split}.csv')}")

    if args.save_overlays:
        overlay_dir = os.path.join(args.outdir, "overlay")
        save_overlays(overlay_dir, paths, args.root, y_true, y_pred)
        print(f"Saved overlays to: {overlay_dir}")

    with open(os.path.join(args.outdir, f"metrics_{args.split}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

