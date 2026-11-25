"""Compare scored HMB3 edits on the exact same source frames."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from hmb3_detect import RESULTS_DIR
from hmb3_suitability import decide, read_scores


LABELS = ("uygun", "uygunsuz", "inceleme")
METRICS = ("road_iou", "boundary_iou", "alignment")
ROAD_THRESHOLD = .90
BOUNDARY_THRESHOLD = .40


def index_scores(path):
    index = {}
    for row in read_scores(path):
        frame = row["frame"]
        timestamp = int(row["timestamp_ns"])
        if frame in index or row["otomatik_karar"] not in LABELS:
            raise ValueError(f"Invalid or duplicate score row: {path}, frame {frame}")
        if decide(row, ROAD_THRESHOLD, BOUNDARY_THRESHOLD)["otomatik_karar"] != row["otomatik_karar"]:
            raise ValueError(f"Decision disagrees with comparison thresholds: {path}, frame {frame}")
        index[frame] = {**row, "timestamp_ns": timestamp}
    if not index:
        raise ValueError(f"No scores found: {path}")
    return index


def metric_summary(rows, key):
    scores = [row[key] for row in rows if row[key] is not None]
    return {
        "count": len(scores),
        "median": float(np.median(scores)) if scores else None,
    }


def dataset_summary(rows, road_threshold):
    counts = Counter(row["otomatik_karar"] for row in rows)
    eligible = [row for row in rows if row["otomatik_karar"] != "inceleme"]
    road_pass = sum(row["road_iou"] is not None and row["road_iou"] >= road_threshold
                    for row in eligible)
    return {
        "counts": {label: counts[label] for label in LABELS},
        "rate_of_all": {label: counts[label] / len(rows) for label in LABELS},
        "reasons": dict(Counter(row["neden"] for row in rows)),
        "road_only_pass_among_evaluable": road_pass,
        "road_only_evaluable_count": len(eligible),
        "alignment_below_035": sum(row["alignment"] < .35 for row in rows),
        "metrics_all": {key: metric_summary(rows, key) for key in METRICS},
    }


def mask_decision(row, use_boundary):
    road = row["road_iou"]
    boundary = row["boundary_iou"]
    if road is None or not np.isfinite(road):
        label, reason = "inceleme", "road_iou_hesaplanamadi"
    elif road < ROAD_THRESHOLD:
        label, reason = "uygunsuz", "road_iou_esik_alti"
    elif use_boundary and (boundary is None or not np.isfinite(boundary)):
        label, reason = "inceleme", "boundary_iou_hesaplanamadi"
    elif use_boundary and boundary < BOUNDARY_THRESHOLD:
        label, reason = "uygunsuz", "boundary_iou_esik_alti"
    else:
        label, reason = "uygun", "iki_esik_gecti" if use_boundary else "road_iou_esigi_gecti"
    return {**row, "onceki_karar": row["otomatik_karar"], "otomatik_karar": label,
            "neden": reason}


def compare(ip2p_path, lucy_path, output, rule="road_boundary_and_quality_checks"):
    mask_only = rule != "road_boundary_and_quality_checks"
    road_only = rule == "road_iou_only"
    ip2p, lucy = index_scores(ip2p_path), index_scores(lucy_path)
    if ip2p.keys() != lucy.keys():
        raise ValueError("The two inputs must cover the same source frames.")
    pairs = [(ip2p[frame], lucy[frame]) for frame in sorted(ip2p)]
    if any(a["timestamp_ns"] != b["timestamp_ns"] for a, b in pairs):
        raise ValueError("Source timestamps do not match.")
    if mask_only:
        pairs = [(mask_decision(a, not road_only), mask_decision(b, not road_only))
                 for a, b in pairs]
    common = [(a, b) for a, b in pairs
              if a["otomatik_karar"] != "inceleme" and b["otomatik_karar"] != "inceleme"]
    matrix = Counter((a["otomatik_karar"], b["otomatik_karar"]) for a, b in pairs)
    summary = {
        "pair_count": len(pairs),
        "road_threshold": ROAD_THRESHOLD,
        "boundary_threshold": None if road_only else BOUNDARY_THRESHOLD,
        "decision_rule": rule,
        "instructpix2pix": dataset_summary([a for a, _ in pairs], ROAD_THRESHOLD),
        "lucyrestyle": dataset_summary([b for _, b in pairs], ROAD_THRESHOLD),
        "matrix_ip2p_then_lucy": {
            a: {b: matrix[a, b] for b in LABELS} for a in LABELS
        },
        "common_evaluable": {
            "count": len(common),
            "ip2p_pass": sum(a["otomatik_karar"] == "uygun" for a, _ in common),
            "lucy_pass": sum(b["otomatik_karar"] == "uygun" for _, b in common),
            "ip2p_road_only_pass": sum(a["road_iou"] >= ROAD_THRESHOLD for a, _ in common),
            "lucy_road_only_pass": sum(b["road_iou"] >= ROAD_THRESHOLD for _, b in common),
            "paired_median_lucy_minus_ip2p": {
                key: metric_summary([{key: b[key] - a[key]} for a, b in common
                                     if a[key] is not None and b[key] is not None], key)["median"]
                for key in METRICS
            },
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    fields = ("jpeg", *METRICS, "otomatik_karar", "neden")
    if mask_only:
        fields += ("onceki_karar",)
        for index, dataset in enumerate(("instructpix2pix", "lucyrestyle")):
            transitions = Counter((pair[index]["onceki_karar"], pair[index]["otomatik_karar"])
                                  for pair in pairs)
            transition_key = "previous_to_road_only" if road_only else "previous_to_mask_only"
            summary[dataset][transition_key] = {
                old: {new: transitions[old, new] for new in LABELS} for old in LABELS
            }
    with (output / "kare_bazinda_karsilastirma.csv").open(
            "w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("frame", "timestamp_ns", *(f"{prefix}_{key}"
                         for prefix in ("ip2p", "lucy") for key in fields),
                         "lucy_eksi_ip2p_road_iou", "lucy_eksi_ip2p_boundary_iou"))
        for a, b in pairs:
            delta = [b[key] - a[key] if a[key] is not None and b[key] is not None else None
                     for key in METRICS[:2]]
            writer.writerow((a["frame"], a["timestamp_ns"], *(row[key]
                             for row in (a, b) for key in fields), *delta))
    (output / "ozet.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip2p", type=Path, default=RESULTS_DIR / "suitability" / "uygunluk.csv")
    parser.add_argument("--lucy", type=Path, default=RESULTS_DIR / "lucyrestyle" / "uygunluk.csv")
    parser.add_argument("--output", type=Path)
    rules = parser.add_mutually_exclusive_group()
    rules.add_argument("--road-only", action="store_true", help="Decide using only Road IoU.")
    rules.add_argument("--mask-only", action="store_true", help="Decide using Road and Boundary IoU.")
    args = parser.parse_args()
    if args.mask_only:
        rule, folder = "road_and_boundary_iou", "comparison_mask_only"
    elif args.road_only:
        rule, folder = "road_iou_only", "comparison_road_only"
    else:
        rule, folder = "road_boundary_and_quality_checks", "comparison"
    compare(args.ip2p, args.lucy, args.output or RESULTS_DIR / folder, rule)


if __name__ == "__main__":
    main()
