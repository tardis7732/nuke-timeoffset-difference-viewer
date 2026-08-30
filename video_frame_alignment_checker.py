#!/usr/bin/env python3
"""
Content-based video frame alignment checker.

This tool does not look for timestamp gaps. It decodes both videos, builds a
small motion/edge feature for every frame, then aligns the two frame sequences.
Unmatched frames and long offset runs are reported as hold/drop/sync candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


MATCH = 0
SKIP_REF = 1
SKIP_TARGET = 2


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass
class MatchStep:
    op: str
    ref_frame: Optional[int]
    target_frame: Optional[int]
    similarity: Optional[float]
    match_cost: Optional[float]


@dataclass
class OffsetRun:
    offset: int
    ref_start: int
    ref_end: int
    target_start: int
    target_end: int
    length: int


@dataclass
class Event:
    kind: str
    frames: List[int]
    start: int
    end: int
    time_start: float
    time_end: float
    time_basis: str
    note: str
    contact_sheet: Optional[str] = None


def parse_size(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError("size must look like 160x90")
    width, height = int(match.group(1)), int(match.group(2))
    if width < 16 or height < 16:
        raise argparse.ArgumentTypeError("size is too small")
    return width, height


def parse_crop_ratio(value: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop ratio must be x0,y0,x1,y1")
    vals = tuple(float(p) for p in parts)
    x0, y0, x1, y1 = vals
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise argparse.ArgumentTypeError("crop ratios must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return vals


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "video"


def video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(str(path), width, height, fps, frame_count, duration)


def roi_rect(
    width: int,
    height: int,
    roi: str,
    crop_ratio: Optional[Tuple[float, float, float, float]],
) -> Tuple[int, int, int, int]:
    if crop_ratio:
        x0, y0, x1, y1 = crop_ratio
        return (
            int(round(x0 * width)),
            int(round(y0 * height)),
            int(round(x1 * width)),
            int(round(y1 * height)),
        )
    if roi == "full":
        return 0, 0, width, height
    if roi == "lower75":
        return 0, int(round(height * 0.25)), width, height
    if roi == "lower60":
        return 0, int(round(height * 0.40)), width, height
    if roi == "center_lower":
        return int(round(width * 0.05)), int(round(height * 0.25)), int(round(width * 0.95)), height
    if roi == "center":
        return int(round(width * 0.10)), int(round(height * 0.10)), int(round(width * 0.90)), int(round(height * 0.90))
    if roi == "center_upper_body":
        return int(round(width * 0.12)), int(round(height * 0.05)), int(round(width * 0.78)), int(round(height * 0.82))
    raise ValueError(f"unknown roi: {roi}")


def normalize_vector(image: np.ndarray) -> np.ndarray:
    vector = image.reshape(-1).astype(np.float32)
    vector = vector - float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


def make_feature(gray: np.ndarray, feature: str) -> np.ndarray:
    if feature == "gray":
        return normalize_vector(gray.astype(np.float32))
    if feature == "sobel":
        sx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sx, sy)
        return normalize_vector(np.log1p(magnitude))
    if feature == "edges":
        blur = cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0)
        edges = cv2.Canny(blur, 45, 120).astype(np.float32)
        return normalize_vector(edges)
    raise ValueError(f"unknown feature: {feature}")


def load_features(
    path: Path,
    resize: Tuple[int, int],
    roi: str,
    crop_ratio: Optional[Tuple[float, float, float, float]],
    feature: str,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    features: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = roi_rect(width, height, roi, crop_ratio)
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)
        features.append(make_feature(small, feature))
    cap.release()
    if not features:
        raise RuntimeError(f"no frames decoded from video: {path}")
    return np.vstack(features)


def load_gray_frames(
    path: Path,
    resize: Tuple[int, int],
    roi: str,
    crop_ratio: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = roi_rect(width, height, roi, crop_ratio)
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)
        frames.append(small.astype(np.float32))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from video: {path}")
    return np.stack(frames)


def load_motion_features(
    path: Path,
    resize: Tuple[int, int],
    roi: str,
    crop_ratio: Optional[Tuple[float, float, float, float]],
    feature: str,
) -> np.ndarray:
    frames = load_gray_frames(path, resize, roi, crop_ratio)
    if len(frames) < 2:
        raise RuntimeError(f"motion mode requires at least 2 frames: {path}")
    features: List[np.ndarray] = []
    for idx in range(1, len(frames)):
        diff = np.abs(frames[idx] - frames[idx - 1])
        features.append(make_feature(diff, feature))
    return np.vstack(features)


def build_match_costs(ref: np.ndarray, target: np.ndarray, band: int, outside_cost: float) -> np.ndarray:
    ref_count, target_count = len(ref), len(target)
    costs = np.full((ref_count, target_count), outside_cost, dtype=np.float32)
    for i in range(ref_count):
        start = max(0, i - band)
        end = min(target_count, i + band + 1)
        if start >= end:
            continue
        sims = target[start:end] @ ref[i]
        costs[i, start:end] = 1.0 - sims
    return costs


def align_sequences(costs: np.ndarray, skip_penalty: float) -> List[MatchStep]:
    ref_count, target_count = costs.shape
    dp = np.full((ref_count + 1, target_count + 1), np.inf, dtype=np.float32)
    back = np.zeros((ref_count + 1, target_count + 1), dtype=np.uint8)
    dp[0, 0] = 0.0

    for i in range(1, ref_count + 1):
        dp[i, 0] = dp[i - 1, 0] + skip_penalty
        back[i, 0] = SKIP_REF
    for j in range(1, target_count + 1):
        dp[0, j] = dp[0, j - 1] + skip_penalty
        back[0, j] = SKIP_TARGET

    for i in range(1, ref_count + 1):
        row_costs = costs[i - 1]
        for j in range(1, target_count + 1):
            diag = dp[i - 1, j - 1] + row_costs[j - 1]
            skip_ref = dp[i - 1, j] + skip_penalty
            skip_target = dp[i, j - 1] + skip_penalty
            if diag <= skip_ref and diag <= skip_target:
                dp[i, j] = diag
                back[i, j] = MATCH
            elif skip_ref <= skip_target:
                dp[i, j] = skip_ref
                back[i, j] = SKIP_REF
            else:
                dp[i, j] = skip_target
                back[i, j] = SKIP_TARGET

    i, j = ref_count, target_count
    steps: List[MatchStep] = []
    while i > 0 or j > 0:
        op = int(back[i, j])
        if i > 0 and j > 0 and op == MATCH:
            cost = float(costs[i - 1, j - 1])
            steps.append(MatchStep("match", i - 1, j - 1, 1.0 - cost, cost))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or op == SKIP_REF):
            steps.append(MatchStep("skip_ref", i - 1, None, None, None))
            i -= 1
        else:
            steps.append(MatchStep("skip_target", None, j - 1, None, None))
            j -= 1
    steps.reverse()
    return steps


def group_ranges(frames: Iterable[int]) -> List[List[int]]:
    sorted_frames = sorted(frames)
    if not sorted_frames:
        return []
    groups: List[List[int]] = [[sorted_frames[0]]]
    for frame in sorted_frames[1:]:
        if frame == groups[-1][-1] + 1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def offset_runs(steps: Sequence[MatchStep]) -> List[OffsetRun]:
    matches = [s for s in steps if s.op == "match" and s.ref_frame is not None and s.target_frame is not None]
    if not matches:
        return []
    runs: List[OffsetRun] = []
    first = matches[0]
    current_offset = first.target_frame - first.ref_frame
    ref_start = ref_end = first.ref_frame
    target_start = target_end = first.target_frame
    length = 1
    previous_ref = first.ref_frame

    for step in matches[1:]:
        assert step.ref_frame is not None and step.target_frame is not None
        offset = step.target_frame - step.ref_frame
        if offset == current_offset and step.ref_frame == previous_ref + 1:
            ref_end = step.ref_frame
            target_end = step.target_frame
            length += 1
        else:
            runs.append(OffsetRun(current_offset, ref_start, ref_end, target_start, target_end, length))
            current_offset = offset
            ref_start = ref_end = step.ref_frame
            target_start = target_end = step.target_frame
            length = 1
        previous_ref = step.ref_frame
    runs.append(OffsetRun(current_offset, ref_start, ref_end, target_start, target_end, length))
    return runs


def seconds(frame: int, fps: float) -> float:
    return frame / fps if fps > 0 else 0.0


def build_events(steps: Sequence[MatchStep], ref_fps: float, target_fps: float) -> List[Event]:
    events: List[Event] = []
    target_skips = [s.target_frame for s in steps if s.op == "skip_target" and s.target_frame is not None]
    ref_skips = [s.ref_frame for s in steps if s.op == "skip_ref" and s.ref_frame is not None]

    for group in group_ranges(target_skips):
        start, end = group[0], group[-1]
        events.append(
            Event(
                "target_extra_or_hold",
                group,
                start,
                end,
                seconds(start, target_fps),
                seconds(end, target_fps),
                "target",
                "Target has frame(s) not aligned to reference. This often means a hold/duplicate caused later sync to shift.",
            )
        )
    for group in group_ranges(ref_skips):
        start, end = group[0], group[-1]
        events.append(
            Event(
                "reference_missing_in_target",
                group,
                start,
                end,
                seconds(start, ref_fps),
                seconds(end, ref_fps),
                "reference",
                "Reference frame(s) have no target match. This often means target dropped/caught up or ends early.",
            )
        )
    return sorted(events, key=lambda e: (e.time_start, e.kind))


def build_motion_events(steps: Sequence[MatchStep], ref_fps: float, target_fps: float) -> List[Event]:
    events: List[Event] = []
    target_skips = [s.target_frame for s in steps if s.op == "skip_target" and s.target_frame is not None]
    ref_skips = [s.ref_frame for s in steps if s.op == "skip_ref" and s.ref_frame is not None]

    for group in group_ranges(target_skips):
        start_interval, end_interval = group[0], group[-1]
        start_frame, end_frame = start_interval, end_interval + 1
        events.append(
            Event(
                "target_extra_motion_interval",
                list(range(start_frame, end_frame + 1)),
                start_frame,
                end_frame,
                seconds(start_frame, target_fps),
                seconds(end_frame, target_fps),
                "target",
                "Target transition(s) have no reference counterpart. At the tail this usually means target has extra final frame(s).",
            )
        )

    for group in group_ranges(ref_skips):
        start_interval, end_interval = group[0], group[-1]
        start_frame, end_frame = start_interval, end_interval + 1
        events.append(
            Event(
                "reference_extra_motion_interval",
                list(range(start_frame, end_frame + 1)),
                start_frame,
                end_frame,
                seconds(start_frame, ref_fps),
                seconds(end_frame, ref_fps),
                "reference",
                "Reference transition(s) have no target counterpart. This can mean target skipped/caught up in this area.",
            )
        )
    return sorted(events, key=lambda e: (e.time_start, e.kind))


def load_selected_frames(path: Path, indexes: Sequence[int], size: Tuple[int, int]) -> dict[int, np.ndarray]:
    wanted = set(i for i in indexes if i >= 0)
    cap = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames[idx] = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        idx += 1
    cap.release()
    return frames


def make_contact_sheet(
    ref_path: Path,
    target_path: Path,
    out_path: Path,
    center: int,
    context: int,
    ref_label: str,
    target_label: str,
    ref_fps: float,
    target_fps: float,
    frame_start: int,
    display_size: Tuple[int, int] = (360, 203),
) -> None:
    indexes = list(range(max(0, center - context), center + context + 1))
    ref_frames = load_selected_frames(ref_path, indexes, display_size)
    target_frames = load_selected_frames(target_path, indexes, display_size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows: List[np.ndarray] = []
    blank = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
    for idx in indexes:
        left = ref_frames.get(idx, blank.copy())
        right = target_frames.get(idx, blank.copy())
        row = np.hstack([left, right])
        label_idx = idx + frame_start
        cv2.rectangle(row, (0, 0), (display_size[0] * 2, 24), (0, 0, 0), -1)
        cv2.putText(
            row,
            f"{ref_label} frame {label_idx:03d} t={seconds(idx, ref_fps):.3f}s",
            (8, 17),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            row,
            f"{target_label} frame {label_idx:03d} t={seconds(idx, target_fps):.3f}s",
            (display_size[0] + 8, 17),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(row)
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def write_csv(path: Path, steps: Sequence[MatchStep], ref_fps: float, target_fps: float, frame_start: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "op",
                "ref_frame_index",
                "target_frame_index",
                "ref_frame_label",
                "target_frame_label",
                "ref_time",
                "target_time",
                "target_minus_ref_offset",
                "similarity",
                "match_cost",
            ]
        )
        for step in steps:
            ref_time = "" if step.ref_frame is None else f"{seconds(step.ref_frame, ref_fps):.6f}"
            target_time = "" if step.target_frame is None else f"{seconds(step.target_frame, target_fps):.6f}"
            offset = ""
            if step.ref_frame is not None and step.target_frame is not None:
                offset = step.target_frame - step.ref_frame
            writer.writerow(
                [
                    step.op,
                    "" if step.ref_frame is None else step.ref_frame,
                    "" if step.target_frame is None else step.target_frame,
                    "" if step.ref_frame is None else step.ref_frame + frame_start,
                    "" if step.target_frame is None else step.target_frame + frame_start,
                    ref_time,
                    target_time,
                    offset,
                    "" if step.similarity is None else f"{step.similarity:.9f}",
                    "" if step.match_cost is None else f"{step.match_cost:.9f}",
                ]
            )


def write_report(
    path: Path,
    ref_info: VideoInfo,
    target_info: VideoInfo,
    args: argparse.Namespace,
    events: Sequence[Event],
    runs: Sequence[OffsetRun],
    csv_name: str,
    json_name: str,
) -> None:
    lines: List[str] = []
    lines.append("# Content Frame Alignment Report")
    lines.append("")
    lines.append("This report is based on decoded frame content, not timestamp gaps.")
    lines.append("Event times are helper values from each event's own file FPS; frame numbers are the primary evidence.")
    lines.append(f"Frame labels in this report start at `{args.frame_start}`; internal decoded frame indexes start at `0`.")
    if args.mode == "motion":
        lines.append("Motion mode aligns frame-to-frame change intervals, so event frame ranges mark interval boundaries.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Reference: `{ref_info.path}`")
    lines.append(f"  - {ref_info.width}x{ref_info.height}, {ref_info.fps:.6f} fps, {ref_info.frame_count} frames, {ref_info.duration_seconds:.6f}s")
    lines.append(f"- Target: `{target_info.path}`")
    lines.append(f"  - {target_info.width}x{target_info.height}, {target_info.fps:.6f} fps, {target_info.frame_count} frames, {target_info.duration_seconds:.6f}s")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Mode: `{args.mode}`")
    lines.append(f"- ROI: `{args.roi}`" + (f", custom crop `{args.crop_ratio}`" if args.crop_ratio else ""))
    lines.append(f"- Feature: `{args.feature}` at `{args.resize[0]}x{args.resize[1]}`")
    lines.append(f"- Alignment band: `+/-{args.band}` frames")
    lines.append(f"- Skip penalty: `{args.skip_penalty}`")
    lines.append("")
    lines.append("## Events")
    lines.append("")
    if events:
        lines.append("| kind | frames | time | basis | note | contact |")
        lines.append("| --- | ---: | ---: | --- | --- | --- |")
        for event in events:
            display_start = event.start + args.frame_start
            display_end = event.end + args.frame_start
            frame_text = f"{display_start}" if event.start == event.end else f"{display_start}-{display_end}"
            time_text = f"{event.time_start:.3f}s" if event.start == event.end else f"{event.time_start:.3f}-{event.time_end:.3f}s"
            contact = "" if not event.contact_sheet else f"[image]({event.contact_sheet})"
            lines.append(f"| `{event.kind}` | {frame_text} | {time_text} | {event.time_basis} | {event.note} | {contact} |")
    else:
        lines.append("No unmatched frames were found by the sequence alignment.")
    lines.append("")
    lines.append("## Offset Runs")
    lines.append("")
    if runs:
        lines.append("| offset | reference frames | target frames | time | length |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for run in runs:
            if run.offset == 0 and run.length < args.min_run:
                continue
            if run.offset == 0:
                continue
            time_text = f"{seconds(run.ref_start, ref_info.fps):.3f}-{seconds(run.ref_end, ref_info.fps):.3f}s"
            lines.append(
                f"| {run.offset:+d} | {run.ref_start}-{run.ref_end} | {run.target_start}-{run.target_end} | {time_text} | {run.length} |"
            )
    else:
        lines.append("No match runs were produced.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- CSV path: `{csv_name}`")
    lines.append(f"- JSON path: `{json_name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find hold/drop/sync candidates by aligning decoded video frame content."
    )
    parser.add_argument("reference", type=Path, help="reference/source video")
    parser.add_argument("target", type=Path, help="target/video to check")
    parser.add_argument("--out-dir", type=Path, default=Path("frame_alignment_reports"), help="output directory")
    parser.add_argument("--ref-label", default="Reference", help="label shown in contact sheets")
    parser.add_argument("--target-label", default="Target", help="label shown in contact sheets")
    parser.add_argument(
        "--roi",
        choices=["full", "lower75", "lower60", "center_lower", "center", "center_upper_body"],
        default="lower75",
    )
    parser.add_argument("--crop-ratio", type=parse_crop_ratio, default=None, help="override ROI as x0,y0,x1,y1 ratios")
    parser.add_argument("--mode", choices=["frame", "motion"], default="frame", help="frame compares frames directly; motion compares frame-to-frame changes")
    parser.add_argument("--feature", choices=["sobel", "edges", "gray"], default="sobel")
    parser.add_argument("--resize", type=parse_size, default=(160, 90), help="feature size, e.g. 160x90")
    parser.add_argument("--band", type=int, default=6, help="maximum expected offset from the diagonal")
    parser.add_argument("--skip-penalty", type=float, default=None, help="cost for leaving one frame unmatched")
    parser.add_argument("--outside-cost", type=float, default=2.0, help="match cost outside the alignment band")
    parser.add_argument("--min-run", type=int, default=5, help="minimum non-zero offset run length to highlight")
    parser.add_argument("--context", type=int, default=6, help="frames before/after each event in contact sheets")
    parser.add_argument("--max-contact-sheets", type=int, default=12)
    parser.add_argument("--no-contact-sheets", action="store_true")
    parser.add_argument("--frame-start", type=int, default=0, help="frame label for decoded frame index 0")
    parser.add_argument("--max-cells", type=int, default=25_000_000, help="ref_frames * target_frames safety limit")
    args = parser.parse_args(argv)

    if args.band < 1:
        parser.error("--band must be >= 1")
    if args.skip_penalty is None:
        args.skip_penalty = 0.80 if args.mode == "motion" else 0.36
    if args.skip_penalty <= 0:
        parser.error("--skip-penalty must be > 0")

    reference = args.reference.resolve()
    target = args.target.resolve()
    if not reference.exists():
        parser.error(f"reference does not exist: {reference}")
    if not target.exists():
        parser.error(f"target does not exist: {target}")

    ref_info = video_info(reference)
    target_info = video_info(target)
    cells = ref_info.frame_count * target_info.frame_count
    if cells > args.max_cells:
        parser.error(
            f"alignment matrix is too large ({cells} cells). Raise --max-cells or compare a shorter shot."
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{safe_stem(reference)}__VS__{safe_stem(target)}"
    csv_path = out_dir / f"{prefix}.alignment.csv"
    json_path = out_dir / f"{prefix}.summary.json"
    md_path = out_dir / f"{prefix}.report.md"

    print(f"Loading {args.mode} features: {reference.name}")
    if args.mode == "motion":
        ref_features = load_motion_features(reference, args.resize, args.roi, args.crop_ratio, args.feature)
    else:
        ref_features = load_features(reference, args.resize, args.roi, args.crop_ratio, args.feature)
    print(f"Loading {args.mode} features: {target.name}")
    if args.mode == "motion":
        target_features = load_motion_features(target, args.resize, args.roi, args.crop_ratio, args.feature)
    else:
        target_features = load_features(target, args.resize, args.roi, args.crop_ratio, args.feature)
    print(f"Aligning {len(ref_features)} reference frames to {len(target_features)} target frames")
    costs = build_match_costs(ref_features, target_features, args.band, args.outside_cost)
    steps = align_sequences(costs, args.skip_penalty)
    runs = offset_runs(steps)
    if args.mode == "motion":
        events = build_motion_events(steps, ref_info.fps, target_info.fps)
    else:
        events = build_events(steps, ref_info.fps, target_info.fps)

    if not args.no_contact_sheets:
        for idx, event in enumerate(events[: args.max_contact_sheets], start=1):
            center = int(round((event.start + event.end) / 2))
            name = f"{prefix}.event_{idx:02d}.{event.kind}.{event.start:04d}_{event.end:04d}.jpg"
            make_contact_sheet(
                reference,
                target,
                out_dir / name,
                center,
                args.context,
                args.ref_label,
                args.target_label,
                ref_info.fps,
                target_info.fps,
                args.frame_start,
            )
            event.contact_sheet = name

    write_csv(csv_path, steps, ref_info.fps, target_info.fps, args.frame_start)

    summary = {
        "reference": asdict(ref_info),
        "target": asdict(target_info),
        "settings": {
            "roi": args.roi,
            "crop_ratio": args.crop_ratio,
            "mode": args.mode,
            "feature": args.feature,
            "resize": args.resize,
            "band": args.band,
            "skip_penalty": args.skip_penalty,
            "outside_cost": args.outside_cost,
            "frame_start": args.frame_start,
        },
        "events": [asdict(e) for e in events],
        "offset_runs": [asdict(r) for r in runs],
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "report": str(md_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(md_path, ref_info, target_info, args, events, runs, csv_path.name, json_path.name)

    print("")
    print(f"Report: {md_path}")
    print(f"CSV:    {csv_path}")
    print(f"JSON:   {json_path}")
    print("")
    if events:
        print("Events:")
        for event in events:
            display_start = event.start + args.frame_start
            display_end = event.end + args.frame_start
            frame_text = f"{display_start}" if event.start == event.end else f"{display_start}-{display_end}"
            time_text = f"{event.time_start:.3f}s" if event.start == event.end else f"{event.time_start:.3f}-{event.time_end:.3f}s"
            print(f"  {event.kind}: frame {frame_text}, time {time_text} ({event.time_basis})")
    else:
        print("No unmatched frames found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
