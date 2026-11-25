"""Classify HMB3 pairs by preservation of the segmented road."""
import argparse
import csv
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from hmb3_detect import DEFAULT_ARCHIVE, DEFAULT_VIDEO, PROJECT_ROOT, RESULTS_DIR
from hmb3_detect import alignment, decode, road_similarity


class SegFormer:
    def __init__(self, path, threads, directml):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if directml:
            if "DmlExecutionProvider" not in ort.get_available_providers():
                raise ValueError("--directml requires onnxruntime-directml.")
            options.enable_mem_pattern = False
            providers.insert(0, "DmlExecutionProvider")
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=providers)
        if directml and "DmlExecutionProvider" not in self.session.get_providers():
            raise ValueError("DirectML initialization failed.")
        self.input = self.session.get_inputs()[0]
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def mask(self, image):
        rgb = cv2.cvtColor(cv2.resize(image, (320, 256), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        tensor = np.ascontiguousarray(((rgb - self.mean) / self.std).transpose(2, 0, 1)[None])
        logits = self.session.run(None, {self.input.name: tensor})[0][0]
        # Bilinear interpolation before argmax preserves the class boundaries.
        resized = np.stack([cv2.resize(channel, image.shape[1::-1]) for channel in logits])
        return resized.argmax(axis=0) == 0  # Cityscapes road train ID


def inner_boundary(mask, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_REPLICATE)
    return mask & ~eroded.astype(bool)


def assess(reference, edited, original_mask, edited_mask, road_limit, boundary_limit):
    registration = alignment(reference, edited)
    road = road_similarity(original_mask, edited_mask)
    # Fixed 8 px on the 512x384 canvas (about 1.6% of its diagonal).
    original_boundary, edited_boundary = (inner_boundary(mask, 8)
                                          for mask in (original_mask, edited_mask))
    boundary = road_similarity(original_boundary, edited_boundary)
    areas = [float(mask.mean()) for mask in (original_mask, edited_mask)]
    scores = {
        "road_iou": road, "boundary_iou": boundary, "alignment": registration,
        "original_road_fraction": areas[0], "edited_road_fraction": areas[1],
    }
    return decide(scores, road_limit, boundary_limit), (original_boundary, edited_boundary)


def decide(scores, road_limit, boundary_limit):
    road, boundary = scores["road_iou"], scores["boundary_iou"]
    areas = scores["original_road_fraction"], scores["edited_road_fraction"]
    if scores["alignment"] < 0.35 or min(areas) < 0.03 or max(areas) > 0.85 or boundary is None:
        label, reason = "inceleme", "eslesme_veya_maske"
    elif road >= road_limit and boundary >= boundary_limit:
        label, reason = "uygun", "iki_esik_gecti"
    else:
        label = "uygunsuz"
        reason = ("alan_ve_kenar" if road < road_limit and boundary < boundary_limit else
                  "alan" if road < road_limit else "kenar")
    return {**scores, "otomatik_karar": label, "neden": reason}


def tint(image, mask, color):
    output = image.copy()
    output[mask] = (0.50 * output[mask] + 0.50 * np.asarray(color)).astype(np.uint8)
    return output


def example(reference, edited, masks, boundaries, row):
    def panel(image, title):
        image = cv2.resize(image, (384, 288), interpolation=cv2.INTER_AREA)
        return np.vstack((header(title, 384, 32), image))

    first, second = masks
    edge_first, edge_second = boundaries
    comparison = cv2.cvtColor(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    comparison = tint(comparison, first & second, (80, 185, 45))
    comparison = tint(comparison, first & ~second, (20, 135, 255))
    comparison = tint(comparison, second & ~first, (235, 130, 40))
    edge_view = np.full_like(reference, 34)
    edge_view[edge_first & ~edge_second] = (20, 135, 255)
    edge_view[edge_second & ~edge_first] = (235, 130, 40)
    edge_view[edge_first & edge_second] = (80, 185, 45)
    top = np.hstack((panel(reference, "Orijinal"), panel(edited, "AI duzenleme"),
                     panel(comparison, "Yol alanlari karsilastirmasi")))
    bottom = np.hstack((panel(tint(reference, first, (80, 185, 45)), "SegFormer yol maskesi"),
                        panel(tint(edited, second, (80, 185, 45)), "SegFormer yol maskesi"),
                        panel(edge_view, "Yol kenari bantlari (8 px)")))
    road, edge = (f"{row[key]:.3f}" if row[key] is not None else "-"
                  for key in ("road_iou", "boundary_iou"))
    title = f"Kare {row['frame']} | {row['otomatik_karar'].upper()} | Road IoU {road} | Boundary IoU {edge}"
    legend = "Yesil: ortak   Turuncu: yalniz orijinal   Mavi: yalniz duzenlenmis"
    return np.vstack((header(title, top.shape[1], 38), top, bottom,
                      header(legend, top.shape[1], 35)))


def header(text, width, height):
    image = np.full((height, width, 3), 29, np.uint8)
    cv2.putText(image, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, .59,
                (242, 242, 242), 1, cv2.LINE_AA)
    return image


def write_rows(args, rows):
    with (args.output / "uygunluk.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_scores(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            row["frame"] = int(row["frame"])
            for field in ("road_iou", "boundary_iou", "alignment",
                          "original_road_fraction", "edited_road_fraction"):
                row[field] = float(row[field]) if row[field] else None
            yield row


def rescore(args):
    rows = [decide(row, args.road_threshold, args.boundary_threshold)
            for row in read_scores(args.scores)]
    if not rows:
        raise ValueError("The score file has no rows.")
    write_rows(args, rows)
    for status in ("uygun", "uygunsuz", "inceleme"):
        print(f"{status}: {sum(row['otomatik_karar'] == status for row in rows)}")
    if not args.example_frames:
        return
    selected = {row["frame"]: row for row in rows if row["frame"] in args.example_frames}
    if len(selected) != len(set(args.example_frames)):
        raise ValueError("Some requested example frames are missing from the score file.")
    model = SegFormer(args.model, args.threads, args.directml)
    video = cv2.VideoCapture(str(args.video))
    try:
        with ZipFile(args.archive) as archive:
            for frame, row in selected.items():
                if not video.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise ValueError(f"Could not seek to frame {frame}.")
                ok, image = video.read()
                if not ok:
                    raise ValueError(f"Could not decode frame {frame}.")
                reference = cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA)
                edited = cv2.resize(decode(archive.read(row["jpeg"])), (512, 384),
                                    interpolation=cv2.INTER_AREA)
                masks = model.mask(reference), model.mask(edited)
                scores, boundaries = assess(reference, edited, *masks, args.road_threshold,
                                            args.boundary_threshold)
                for key in ("road_iou", "boundary_iou"):
                    if scores[key] is not None and row[key] is not None:
                        if abs(scores[key] - row[key]) > .005:
                            raise ValueError(f"Recomputed {key} differs at frame {frame}.")
                save_example(args.output, row, example(reference, edited, masks, boundaries, row))
    finally:
        video.release()


def save_example(output, row, figure):
    path = output / f"{row['otomatik_karar']}_{row['frame']:05d}.png"
    if not cv2.imwrite(str(path), figure):
        raise ValueError(f"Could not write {path}")


def archive_entries(archive, frame_prefix=""):
    entries = []
    for name in archive.namelist():
        path = Path(name)
        if path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        if not path.stem.startswith(frame_prefix):
            raise ValueError(f"Unexpected JPEG name: {name}")
        number = path.stem[len(frame_prefix):]
        if not number.isdecimal():
            raise ValueError(f"Expected a numeric frame key: {name}")
        entries.append((int(number), name))
    entries.sort()
    if not entries or len({key for key, _ in entries}) != len(entries):
        raise ValueError("Expected unique numeric JPEG keys.")
    return entries


def paired_entries(archive, frame_prefix):
    entries = archive_entries(archive, frame_prefix or "")
    if frame_prefix is None:
        return entries
    if [index for index, _ in entries] != list(range(len(entries))):
        raise ValueError("Indexed ZIP must contain every frame starting at index zero.")
    with ZipFile(DEFAULT_ARCHIVE) as reference:
        timestamps = archive_entries(reference)
    if len(timestamps) != len(entries):
        raise ValueError("Indexed ZIP frame count differs from the HMB3 timestamp archive.")
    return [(timestamp, name) for (timestamp, _), (_, name) in zip(timestamps, entries)]


def process(args):
    model = SegFormer(args.model, args.threads, args.directml)
    judge = partial(assess, road_limit=args.road_threshold, boundary_limit=args.boundary_threshold)
    rows, examples = [], {}
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise ValueError(f"Failed to open {args.video}")
    try:
        with ZipFile(args.archive) as archive:
            entries = paired_entries(archive, args.frame_prefix)
            for frame, (timestamp, name) in enumerate(entries):
                if not video.grab():
                    raise ValueError(f"Video ends before ZIP at frame {frame}.")
                if frame % args.stride:
                    continue
                ok, image = video.retrieve()
                if not ok:
                    raise ValueError(f"Could not decode frame {frame}.")
                reference = cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA)
                edited = cv2.resize(decode(archive.read(name)), (512, 384), interpolation=cv2.INTER_AREA)
                masks = model.mask(reference), model.mask(edited)
                scores, boundaries = judge(reference, edited, *masks)
                row = {"frame": frame, "timestamp_ns": timestamp, "jpeg": name, **scores}
                rows.append(row)
                # Retain the strongest examples of each class in separate time blocks.
                group = (row["otomatik_karar"], frame // 450)
                merit = row["boundary_iou"] or 0
                if row["otomatik_karar"] != "uygun":
                    merit = 1 - merit
                if group not in examples or merit > examples[group][0]:
                    examples[group] = merit, frame, example(reference, edited, masks, boundaries, row)
                if len(rows) % 100 == 0:
                    print(f"{len(rows)} pairs scored, last video frame {frame}", flush=True)
            if video.grab():
                raise ValueError("Video has more frames than the ZIP.")
    finally:
        video.release()

    write_rows(args, rows)
    for status in ("uygun", "uygunsuz", "inceleme"):
        chosen = sorted((entry for (label, _), entry in examples.items() if label == status),
                        key=lambda item: (-item[0], item[1]))[:8]
        for _, frame, figure in chosen:
            save_example(args.output, {"frame": frame, "otomatik_karar": status}, figure)
        print(f"{status}: {sum(row['otomatik_karar'] == status for row in rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--frame-prefix", help="Filename prefix for an indexed HMB3 ZIP.")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "models" / "segformer_cityscapes.onnx")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "suitability")
    parser.add_argument("--road-threshold", type=float, default=.90)
    parser.add_argument("--boundary-threshold", type=float, default=.40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--directml", action="store_true", help="Use the Windows DirectML provider.")
    parser.add_argument("--scores", type=Path, help="Reapply thresholds to an existing score CSV.")
    parser.add_argument("--example-frames", type=int, nargs="+",
                        help="Regenerate example panels for these frames when using --scores.")
    args = parser.parse_args()
    if min(args.stride, args.threads) < 1 or not all(
            0 <= threshold <= 1 for threshold in (args.road_threshold, args.boundary_threshold)):
        parser.error("Invalid stride, threads or threshold.")
    if args.example_frames and not args.scores:
        parser.error("--example-frames requires --scores.")
    if args.frame_prefix is not None and args.video.resolve() != DEFAULT_VIDEO:
        parser.error("--frame-prefix uses timestamps from the standard HMB3 video.")
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.scores:
        rescore(args)
    else:
        process(args)


if __name__ == "__main__":
    main()
