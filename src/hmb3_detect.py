"""Reference-based edited-frame detection for HMB3."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "hmb3_driving_video.mp4"
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "hmb3_ip2p_frames.zip"
RESULTS_DIR = PROJECT_ROOT / "results"


def decode(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode a JPEG.")
    return image


def appearance_difference(reference, candidate):
    def prepare(image):
        small = cv2.resize(image, (128, 96), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (5, 5), 0)

    return cv2.absdiff(prepare(reference), prepare(candidate)).mean(axis=2) / 255


def alignment(reference, candidate):
    def gradient(image):
        gray = cv2.cvtColor(cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return np.stack([cv2.Sobel(gray, cv2.CV_32F, dx, dy)
                         for dx, dy in ((1, 0), (0, 1))])

    # Exclude the foreground where wet-road reflections are most pronounced.
    first, second = (gradient(image)[:, 4:78, 4:-4] for image in (reference, candidate))
    denominator = np.sqrt(np.sum(first ** 2) * np.sum(second ** 2))
    return float(np.sum(first * second) / denominator) if denominator else 0.0


class RoadModel:
    """Binary U-Net exported as ONNX, with one probability map as output."""

    def __init__(self, path, width, height):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size = (width, height)

    def __call__(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255, self.size, swapRB=True)
        self.net.setInput(blob)
        output = self.net.forward()
        if output.ndim != 4 or output.shape[:2] != (1, 1):
            raise ValueError(f"Expected U-Net output [1,1,H,W], got {output.shape}.")
        if not np.isfinite(output).all() or output.min() < 0 or output.max() > 1:
            raise ValueError("U-Net must return road probabilities in [0, 1].")
        return cv2.resize(output[0, 0], image.shape[1::-1]) >= 0.5


def road_similarity(first, second):
    union = np.count_nonzero(first | second)
    if not union:
        return None
    return float(np.count_nonzero(first & second) / union)


def preview(path, reference, candidate, masks):
    rows = [np.hstack((reference, candidate))]
    if masks is not None:
        rows.append(np.hstack([
            cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for mask in masks
        ]))
    ok, encoded = cv2.imencode(".jpg", np.vstack(rows))
    if not ok:
        raise ValueError("Could not encode preview.")
    path.write_bytes(encoded.tobytes())


def measure(frame, candidate, model, road_threshold):
    reference = cv2.resize(frame, candidate.shape[1::-1], interpolation=cv2.INTER_AREA)
    difference = appearance_difference(reference, candidate)
    masks = (model(reference), model(candidate)) if model else None
    iou = road_similarity(*masks) if masks is not None else None
    road_score = background_score = None
    if masks is not None:
        road = cv2.resize(masks[0].astype(np.uint8), difference.shape[::-1],
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        road_score = float(difference[road].mean()) if road.any() else None
        background_score = float(difference[~road].mean()) if not road.all() else None
    scores = {
        "alignment": alignment(reference, candidate),
        "appearance": float(difference.mean()), "road_appearance": road_score,
        "background_appearance": background_score, "oc_tss": iou,
        "road_validation": ("unavailable" if iou is None else
                            "preserved" if iou >= road_threshold else "changed"),
    }
    return scores, reference, masks


def compare_image(args, model):
    video = cv2.VideoCapture(str(args.video))
    try:
        if not video.isOpened() or not video.set(cv2.CAP_PROP_POS_FRAMES, args.frame):
            raise ValueError(f"Cannot seek to frame {args.frame} in {args.video}.")
        ok, frame = video.read()
        if not ok:
            raise ValueError(f"Cannot read video frame {args.frame}.")
    finally:
        video.release()
    candidate = decode(args.candidate.read_bytes())
    scores, reference, masks = measure(frame, candidate, model, args.road_threshold)
    if args.previews:
        args.previews.mkdir(parents=True, exist_ok=True)
        preview(args.previews / f"{args.frame:06}.jpg", reference, candidate, masks)
    scores["label"] = classify(scores, args.normal_max, args.edited_min, args.min_alignment)
    print(json.dumps({"frame": args.frame, "candidate": str(args.candidate), **scores}, indent=2))


def collect(args, model):
    rows = []
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    try:
        with ZipFile(args.archive) as archive:
            entries = sorted(
                (int(Path(entry.filename).stem), entry.filename)
                for entry in archive.infolist()
                if not entry.is_dir() and Path(entry.filename).suffix.lower() in (".jpg", ".jpeg")
            )
            if len(entries) < 3 or len({time for time, _ in entries}) != len(entries):
                raise ValueError("Expected at least three JPEGs with unique numeric timestamps.")
            if args.previews:
                args.previews.mkdir(parents=True, exist_ok=True)
            preview_indices = set(np.linspace(0, len(entries) - 1, 9, dtype=int))
            for index, (timestamp, name) in enumerate(entries):
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f"Video ended before ZIP at frame {index}.")
                candidate = decode(archive.read(name))
                scores, reference, masks = measure(frame, candidate, model, args.road_threshold)
                pair = {
                    "frame": index, "timestamp_ns": timestamp, "jpeg": name,
                    "split": "calibration" if index < len(entries) * 0.6 else
                             "gap" if index < len(entries) * 0.65 else "test",
                    **scores,
                }
                rows.append(pair)
                if args.previews and index in preview_indices:
                    preview(args.previews / f"{index:06}.jpg", reference, candidate, masks)
                if args.evaluate:
                    for quality in (60, 90):
                        ok, jpeg = cv2.imencode(".jpg", reference, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        if not ok:
                            raise ValueError("Could not encode normal control.")
                        normal = decode(jpeg.tobytes())
                        rows.append({
                            **pair, "jpeg": "",
                            "control_quality": quality, "alignment": alignment(reference, normal),
                            "appearance": float(appearance_difference(reference, normal).mean()),
                            "road_appearance": None, "background_appearance": None,
                            "oc_tss": None, "road_validation": "unavailable",
                        })
            if video.read()[0]:
                raise ValueError("Video has more frames than the ZIP.")
    finally:
        video.release()
    return rows


def classify(row, normal_max, edited_min, min_alignment):
    if row["alignment"] < min_alignment:
        return "belirsiz"
    if row["appearance"] <= normal_max:
        return "normal"
    if row["appearance"] >= edited_min:
        return "edited"
    return "belirsiz"


def evaluate(rows, min_alignment):
    normal = [row["appearance"] for row in rows if row["split"] == "calibration"
              and "control_quality" in row]
    if not normal or not any(row["split"] == "test" for row in rows):
        raise ValueError("Evaluation requires calibration and test frames.")
    normal_max = float(np.quantile(normal, 0.995))
    edited_min = 1.5 * normal_max
    counts = {label: dict.fromkeys(("normal", "edited", "belirsiz"), 0)
              for label in ("normal", "edited")}
    for row in rows:
        if row["split"] == "test":
            actual = "normal" if "control_quality" in row else "edited"
            counts[actual][classify(row, normal_max, edited_min, min_alignment)] += 1

    tp = counts["edited"]["edited"]
    fp = counts["normal"]["edited"]
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / sum(counts["edited"].values())
    total = sum(sum(group.values()) for group in counts.values())
    return {
        "normal_max": normal_max, "edited_min": edited_min, "min_alignment": min_alignment,
        "matrix_actual_then_predicted": counts, "edited_precision": precision,
        "edited_recall": recall, "edited_f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0,
        "uncertain_rate": sum(group["belirsiz"] for group in counts.values()) / total,
    }


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--candidate", type=Path, help="Compare one image with a video frame.")
    parser.add_argument("--frame", type=int, help="Zero-based reference frame for --candidate.")
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "appearance" / "hmb3_results.csv")
    parser.add_argument("--evaluate", type=Path, metavar="REPORT_JSON")
    parser.add_argument("--previews", type=Path)
    parser.add_argument("--road-model", type=Path, metavar="UNET_ONNX")
    parser.add_argument("--model-width", type=int, default=512)
    parser.add_argument("--model-height", type=int, default=384)
    parser.add_argument("--road-threshold", type=float, default=0.9)
    parser.add_argument("--min-alignment", type=float, default=0.35)
    parser.add_argument("--normal-max", type=float, default=0.0023150828907952067)
    parser.add_argument("--edited-min", type=float, default=0.0034726243361928102)
    args = parser.parse_args()
    if not 0 <= args.normal_max < args.edited_min <= 1:
        parser.error("Expected 0 <= normal-max < edited-min <= 1.")
    if not 0 <= args.road_threshold <= 1 or not -1 <= args.min_alignment <= 1:
        parser.error("Invalid road or alignment threshold.")
    if min(args.model_height, args.model_width) < 1:
        parser.error("Model dimensions must be positive.")
    if args.candidate and (args.frame is None or args.frame < 0 or args.evaluate):
        parser.error("--candidate requires a non-negative --frame and cannot use --evaluate.")
    if not args.candidate and args.frame is not None:
        parser.error("--frame requires --candidate.")
    try:
        model = (RoadModel(args.road_model, args.model_width, args.model_height)
                 if args.road_model else None)
        if args.candidate:
            compare_image(args, model)
            return
        rows = collect(args, model)
        normal_max, edited_min = args.normal_max, args.edited_min
        if args.evaluate:
            report = evaluate(rows, args.min_alignment)
            normal_max, edited_min = report["normal_max"], report["edited_min"]
            args.evaluate.parent.mkdir(parents=True, exist_ok=True)
            args.evaluate.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        results = [
            {**row, "label": classify(row, normal_max, edited_min, args.min_alignment)}
            for row in rows if "control_quality" not in row
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} frames to {args.output}")
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
