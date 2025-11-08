#!/usr/bin/env python3
"""
Extract 3-history + next-label steering dataset from a ROS 2 bag.

- Images: /image_raw/compressed  (sensor_msgs/CompressedImage)
- Commands: /cmd_vel             (geometry_msgs/Twist, no header)
- Output:
    <out_dir>/images/bag_1_<bag_timestamp_ns>.png
    <out_dir>/index.json  # list of records (see example at bottom)

History = [ω_{t-2}, ω_{t-1}, ω_t]  (oldest -> newest), label = ω_{t+1}
All timestamps are bag-time seconds; image header stamp is also stored if present.
"""

import os
import json
import argparse
import pathlib
import numpy as np
import cv2
import shutil
import datetime as dt

# ROS 2 bag reading
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def ns_to_sec(ns: int) -> float:
    return ns / 1e9


def sec_to_str(s: float) -> str:
    return dt.datetime.utcfromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def decode_compressed_image(msg):
    """sensor_msgs/msg/CompressedImage -> BGR uint8 ndarray"""
    np_arr = np.frombuffer(msg.data, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Failed to decode CompressedImage (cv2.imdecode returned None).")
    return img_bgr


def dir_size_bytes(path: pathlib.Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def read_bag_messages(bag_path, wanted_topics):
    """
    Returns:
      streams: dict(topic -> list of (t_bag_ns:int, msg:Any)), sorted by time
      type_map: dict(topic -> ROS msg type string)
      storage_id: storage backend used by rosbag2 (e.g., 'sqlite3')
    """
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    missing = [t for t in wanted_topics if t not in type_map]
    if missing:
        raise RuntimeError(f"Topics not found in bag: {missing}\nAvailable: {list(type_map.keys())}")

    # Prepare deserializers
    type_classes = {t: get_message(type_map[t]) for t in wanted_topics}

    out = {t: [] for t in wanted_topics}
    while reader.has_next():
        topic, data, t_bag_ns = reader.read_next()
        if topic not in wanted_topics:
            continue
        msg_type = type_classes[topic]
        msg = deserialize_message(data, msg_type)
        out[topic].append((t_bag_ns, msg))

    # Sort by time
    for t in out:
        out[t].sort(key=lambda x: x[0])

    # Try to detect storage_id (best-effort; we used sqlite3 above)
    storage_id = storage_options.storage_id or "unknown"
    return out, type_map, storage_id


def print_bag_info(bag_path: pathlib.Path, streams: dict, type_map: dict, image_topic: str, cmd_topic: str):
    print("=" * 72)
    print(f"Bag path        : {bag_path}")
    print(f"Bag size        : {dir_size_bytes(bag_path)/1e6:.2f} MB")
    print("-" * 72)

    # Topic summary
    print("Topics found:")
    for t, typ in type_map.items():
        count = len(streams.get(t, []))
        print(f"  • {t:<32} ({typ})  count={count}")

    # Time span per requested topic
    def topic_stats(topic):
        msgs = streams.get(topic, [])
        if not msgs:
            return None
        t0_ns = msgs[0][0]
        t1_ns = msgs[-1][0]
        dur_s = (t1_ns - t0_ns) / 1e9 if t1_ns > t0_ns else 0.0
        hz = (len(msgs) / dur_s) if dur_s > 0 else float('nan')
        return t0_ns, t1_ns, dur_s, hz, len(msgs)

    print("-" * 72)
    for t in [image_topic, cmd_topic]:
        stats = topic_stats(t)
        if stats is None:
            print(f"{t}: no messages.")
            continue
        t0, t1, dur_s, hz, n = stats
        print(f"{t}")
        print(f"    first stamp : {sec_to_str(ns_to_sec(t0))}  ({t0} ns)")
        print(f"    last stamp  : {sec_to_str(ns_to_sec(t1))}  ({t1} ns)")
        print(f"    duration    : {dur_s:.3f} s")
        print(f"    count       : {n}")
        print(f"    approx rate : {hz:.3f} Hz")
    print("=" * 72)
    input("Review the info above. Press <Enter> to start extraction... ")
    print()


def build_index(images, cmds, out_dir,
                hist_len=3,
                max_nearest_gap_s=0.10,
                max_oldest_span_s=0.50,
                max_next_gap_s=0.20):
    """
    Build dataset entries using:
      history = [ω_{i-2}, ω_{i-1}, ω_i] where t_i <= t_img < t_{i+1}
      label   = ω_{i+1}
    Filters:
      - nearest gap (|t_img - t_i|) <= max_nearest_gap_s
      - oldest span (t_img - t_{i-2}) <= max_oldest_span_s
      - next gap (t_{i+1} - t_img) <= max_next_gap_s
    Saves images (PNG) if not present.
    """
    cmd_times = np.array([c["t_bag_ns"] for c in cmds], dtype=np.int64)
    entries = []

    for im in images:
        t_img = im["t_bag_ns"]
        # find i_next: first cmd strictly after image time
        i_next = int(np.searchsorted(cmd_times, t_img, side='right'))
        i = i_next - 1  # last cmd at or before image

        # need i-2, i-1, i for history and i+1 for label
        if i < 2 or (i + 1) >= len(cmds):
            continue

        # quality checks
        nearest_gap_s = (t_img - cmd_times[i]) / 1e9
        if nearest_gap_s > max_nearest_gap_s:
            continue

        oldest_span_s = (t_img - cmd_times[i - 2]) / 1e9
        if oldest_span_s > max_oldest_span_s:
            continue

        next_gap_s = (cmd_times[i + 1] - t_img) / 1e9
        if next_gap_s > max_next_gap_s:
            continue

        # image save (PNG); filename already prepared by caller
        if not os.path.exists(im["save_path"]):
            ok = cv2.imwrite(im["save_path"], im["img_bgr"])
            if not ok:
                raise RuntimeError(f"Failed to write image: {im['save_path']}")

        # assemble record
        prev_idxs = [i - 2, i - 1, i]
        label_idx = i + 1

        entry = {
            "image_path": os.path.relpath(im["save_path"], out_dir),
            "stamp_img_bag": ns_to_sec(im["t_bag_ns"]),
            "stamp_img_header": (ns_to_sec(im["t_hdr_ns"]) if im["t_hdr_ns"] is not None else None),

            "history": [
                {
                    "t": ns_to_sec(cmds[k]["t_bag_ns"]),
                    "omega": float(cmds[k]["omega"]),
                    "linear_x": float(cmds[k]["vx"])
                } for k in prev_idxs
            ],

            "label": {
                "t": ns_to_sec(cmds[label_idx]["t_bag_ns"]),
                "omega": float(cmds[label_idx]["omega"]),
                "linear_x": float(cmds[label_idx]["vx"])
            },

            "audit": {
                "nearest_cmd_at_or_before_img": {
                    "t": ns_to_sec(cmds[i]["t_bag_ns"]),
                    "omega": float(cmds[i]["omega"]),
                    "gap_s": float(nearest_gap_s)
                },
                "oldest_history_span_s": float(oldest_span_s),
                "next_cmd_gap_s": float(next_gap_s)
            }
        }
        entries.append(entry)

    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to ros2 bag directory (folder containing metadata.yaml)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--image-topic", default="/image_raw/compressed")
    ap.add_argument("--cmd-topic", default="/cmd_vel")
    ap.add_argument("--hist-len", type=int, default=3)
    ap.add_argument("--max-nearest-gap", type=float, default=0.10,
                    help="Max |t_img - t_cmd_i| in seconds")
    ap.add_argument("--max-oldest-span", type=float, default=0.50,
                    help="Max (t_img - t_cmd_{i-2}) in seconds")
    ap.add_argument("--max-next-gap", type=float, default=0.20,
                    help="Max (t_cmd_{i+1} - t_img) in seconds")
    ap.add_argument("--subsample", type=int, default=1, help="Keep every Nth image (1=no subsampling)")
    args = ap.parse_args()

    if args.hist_len != 3:
        raise ValueError("This script is configured for hist_len=3 per your spec.")

    bag_path = pathlib.Path(args.bag)
    out_dir = ensure_dir(args.out)
    img_dir = ensure_dir(os.path.join(out_dir, "images"))

    wanted = [args.image_topic, args.cmd_topic]
    streams, type_map, storage_id = read_bag_messages(bag_path, wanted)

    # --- Print bag info and wait for confirmation ---
    print_bag_info(bag_path, streams, type_map, args.image_topic, args.cmd_topic)

    # /cmd_vel (geometry_msgs/Twist)
    cmd_msgs = []
    for (t_ns, m) in streams[args.cmd_topic]:
        omega = float(m.angular.z)
        vx = float(m.linear.x)
        cmd_msgs.append({
            "t_bag_ns": t_ns,
            "omega": omega,
            "vx": vx
        })
    if not cmd_msgs:
        raise RuntimeError("No /cmd_vel messages found.")
    cmd_msgs.sort(key=lambda d: d["t_bag_ns"])

    # /image_raw/compressed (sensor_msgs/CompressedImage)
    from sensor_msgs.msg import CompressedImage
    images = []
    keep_counter = 0
    for (t_ns, m) in streams[args.image_topic]:
        if not isinstance(m, CompressedImage):
            raise RuntimeError(f"Expected sensor_msgs/CompressedImage on {args.image_topic}, got {type(m)}")

        keep_counter += 1
        if args.subsample > 1 and (keep_counter % args.subsample != 0):
            continue

        # header (if present)
        t_hdr_ns = None
        try:
            t_hdr_ns = int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec)
        except Exception:
            t_hdr_ns = None

        img_bgr = decode_compressed_image(m)

        # Filename rule: bag_1_<bag_timestamp_ns>.png
        fname = f"bag_1_{t_ns}.png"
        save_path = os.path.join(img_dir, fname)

        images.append({
            "t_bag_ns": t_ns,
            "t_hdr_ns": t_hdr_ns,
            "img_bgr": img_bgr,
            "save_path": save_path
        })

    if not images:
        raise RuntimeError(f"No images found on {args.image_topic}.")

    # Build dataset
    entries = build_index(
        images=images,
        cmds=cmd_msgs,
        out_dir=out_dir,
        hist_len=args.hist_len,
        max_nearest_gap_s=args.max_nearest_gap,
        max_oldest_span_s=args.max_oldest_span,
        max_next_gap_s=args.max_next_gap
    )

    # Write index.json
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"[Done] Saved {len(entries)} samples")
    print(f"Images dir: {img_dir}")
    print(f"Index file: {index_path}")
    if len(entries) and "image_path" in entries[0]:
        print(f"Example image: {entries[0]['image_path']}")


if __name__ == "__main__":
    main()
