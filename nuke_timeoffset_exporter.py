#!/usr/bin/env python3
"""
Build a Nuke TimeOffset node from frame-alignment offsets.

Two workflows are supported:

1. Manual keys:
   py -3 nuke_timeoffset_exporter.py --keys "1:-1,61:0" --out fix.nk

2. Automatic analysis:
   py -3 nuke_timeoffset_exporter.py ref.mp4 target.mp4 --mode motion --roi center_upper_body

The generated curve uses explicit boundary keys so integer frame changes do not
accidentally interpolate across a long range.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from video_frame_alignment_checker import (
    align_sequences,
    build_match_costs,
    load_features,
    load_motion_features,
    parse_crop_ratio,
    parse_size,
    video_info,
)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not cleaned:
        return "AutoTimeOffset"
    if cleaned[0].isdigit():
        cleaned = "TimeOffset_" + cleaned
    return cleaned


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def parse_keys(value: str) -> List[Tuple[int, float]]:
    keys: List[Tuple[int, float]] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            frame_text, offset_text = item.split(":", 1)
        elif "=" in item:
            frame_text, offset_text = item.split("=", 1)
        else:
            raise argparse.ArgumentTypeError("keys must look like 1:-1,61:0")
        try:
            frame = int(frame_text.strip())
            offset = float(offset_text.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"bad key: {item}") from exc
        keys.append((frame, offset))
    if not keys:
        raise argparse.ArgumentTypeError("at least one key is required")
    keys.sort(key=lambda x: x[0])
    deduped: Dict[int, float] = {}
    for frame, offset in keys:
        deduped[frame] = offset
    return sorted(deduped.items())


def expand_step_keys(segments: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Convert segment starts into explicit Nuke curve keys.

    A segment list [(1, -1), (61, 0)] means -1 from frame 1 through 60, then
    0 from frame 61 onward. The returned keys become:
    [(1, -1), (60, -1), (61, 0)].
    """

    if not segments:
        return []
    ordered = sorted(segments, key=lambda x: x[0])
    keys: Dict[int, float] = {}
    for idx, (frame, value) in enumerate(ordered):
        keys[frame] = value
        if idx + 1 < len(ordered):
            next_frame = ordered[idx + 1][0]
            if next_frame > frame:
                keys[next_frame - 1] = value
    return sorted(keys.items())


def curve_text_from_segments(segments: Sequence[Tuple[int, float]]) -> str:
    keys = expand_step_keys(segments)
    tokens: List[str] = []
    for frame, value in keys:
        tokens.append(f"x{frame}")
        tokens.append(format_number(value))
    return "{{curve " + " ".join(tokens) + "}}"


def write_nuke_node(
    path: Path,
    curve_text: str,
    node_name: str,
    nuke_version: str,
    selected: bool = True,
) -> None:
    selected_text = "true" if selected else "false"
    text = (
        "set cut_paste_input [stack 0]\n"
        f"version {nuke_version}\n"
        "push $cut_paste_input\n"
        "TimeOffset {\n"
        f" time_offset {curve_text}\n"
        ' time ""\n'
        f" name {safe_name(node_name)}\n"
        f" selected {selected_text}\n"
        "}\n"
    )
    path.write_text(text, encoding="utf-8")


def compress_offsets_to_segments(
    offset_by_frame: Sequence[Tuple[int, int]],
    min_run: int,
    include_zero: bool,
) -> List[Tuple[int, float]]:
    if not offset_by_frame:
        return []

    runs: List[Tuple[int, int, int]] = []
    start, current = offset_by_frame[0]
    previous_frame = start
    length = 1

    for frame, value in offset_by_frame[1:]:
        if frame == previous_frame + 1 and value == current:
            length += 1
            previous_frame = frame
            continue
        runs.append((start, previous_frame, current))
        start, current = frame, value
        previous_frame = frame
        length = 1
    runs.append((start, previous_frame, current))

    # Merge very short runs into their left neighbor unless they are the only
    # non-zero signal. This keeps single-frame score noise from creating a node
    # that is impossible to inspect.
    merged: List[Tuple[int, int, int]] = []
    for run in runs:
        run_len = run[1] - run[0] + 1
        if merged and run_len < min_run:
            prev = merged[-1]
            merged[-1] = (prev[0], run[1], prev[2])
        else:
            merged.append(run)

    segments: List[Tuple[int, float]] = []
    leading_zero_end: Optional[int] = None
    for start_frame, _end_frame, offset in merged:
        if offset == 0 and not include_zero and not segments:
            leading_zero_end = _end_frame
            continue
        if offset != 0 and not segments and leading_zero_end is not None and leading_zero_end == start_frame - 1:
            segments.append((leading_zero_end, 0.0))
        if offset != 0 or include_zero or segments:
            if segments and segments[-1][1] == float(offset):
                continue
            segments.append((start_frame, float(offset)))
    return segments


def auto_segments(args: argparse.Namespace) -> Tuple[List[Tuple[int, float]], dict]:
    if args.reference is None or args.target is None:
        raise SystemExit("auto mode requires reference and target video paths")

    reference = args.reference.resolve()
    target = args.target.resolve()
    if not reference.exists():
        raise SystemExit(f"reference does not exist: {reference}")
    if not target.exists():
        raise SystemExit(f"target does not exist: {target}")

    ref_info = video_info(reference)
    target_info = video_info(target)
    if args.skip_penalty is None:
        args.skip_penalty = 0.80 if args.mode == "motion" else 0.36

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

    print(f"Aligning {len(ref_features)} reference features to {len(target_features)} target features")
    costs = build_match_costs(ref_features, target_features, args.band, args.outside_cost)
    steps = align_sequences(costs, args.skip_penalty)

    def signed_offset(target_minus_ref: int) -> int:
        if args.offset_sign == "target_minus_ref":
            return target_minus_ref
        return -target_minus_ref

    def next_match_offset(start_index: int) -> Optional[int]:
        for later in steps[start_index + 1 :]:
            if later.op == "match" and later.ref_frame is not None and later.target_frame is not None:
                return signed_offset(later.target_frame - later.ref_frame)
        return None

    offset_by_frame: List[Tuple[int, int]] = []
    for step_index, step in enumerate(steps):
        if step.op != "match" or step.ref_frame is None or step.target_frame is None:
            if (
                step.op == "skip_target"
                and args.timeline == "target"
                and step.target_frame is not None
            ):
                upcoming = next_match_offset(step_index)
                if upcoming is not None:
                    frame_label = step.target_frame + args.frame_start
                    offset_by_frame.append((frame_label, int(round(upcoming))))
            elif (
                step.op == "skip_ref"
                and args.timeline == "reference"
                and step.ref_frame is not None
            ):
                upcoming = next_match_offset(step_index)
                if upcoming is not None:
                    frame_label = step.ref_frame + args.frame_start
                    offset_by_frame.append((frame_label, int(round(upcoming))))
            continue
        if args.timeline == "target":
            feature_frame = step.target_frame
        else:
            feature_frame = step.ref_frame

        # Motion features represent interval N->N+1. For TimeOffset keys the
        # useful boundary is the interval index in the target/reference timeline;
        # adding one here shifts the Nuke key one frame late.
        decoded_frame = feature_frame
        frame_label = decoded_frame + args.frame_start

        target_minus_ref = step.target_frame - step.ref_frame
        offset = signed_offset(target_minus_ref)
        offset_by_frame.append((frame_label, int(round(offset))))

    offset_by_frame = sorted({frame: offset for frame, offset in offset_by_frame}.items(), key=lambda x: x[0])
    segments = compress_offsets_to_segments(offset_by_frame, args.min_run, args.include_zero)

    summary = {
        "reference": asdict(ref_info),
        "target": asdict(target_info),
        "settings": {
            "mode": args.mode,
            "roi": args.roi,
            "crop_ratio": args.crop_ratio,
            "feature": args.feature,
            "resize": args.resize,
            "band": args.band,
            "skip_penalty": args.skip_penalty,
            "offset_sign": args.offset_sign,
            "timeline": args.timeline,
            "frame_start": args.frame_start,
            "min_run": args.min_run,
        },
        "offset_by_frame": offset_by_frame,
        "segments": segments,
    }
    return segments, summary


def default_out_path(args: argparse.Namespace) -> Path:
    if args.out:
        return args.out
    if args.target:
        stem = args.target.stem
    elif args.name:
        stem = args.name
    else:
        stem = "timeoffset"
    return Path(f"{stem}.TimeOffset.nk")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export Nuke TimeOffset nodes from manual or detected offsets.")
    parser.add_argument("reference", type=Path, nargs="?", help="reference/correct video for auto mode")
    parser.add_argument("target", type=Path, nargs="?", help="target video to fix in auto mode")
    parser.add_argument("--keys", type=parse_keys, help='manual segment starts, e.g. "1:-1,61:0"')
    parser.add_argument("--out", type=Path, help="output .nk file")
    parser.add_argument("--json-out", type=Path, help="optional JSON summary path")
    parser.add_argument("--name", default=None, help="Nuke node name")
    parser.add_argument("--nuke-version", default="15.1 v3")
    parser.add_argument("--frame-start", type=int, default=1, help="Nuke label for decoded frame index 0")

    parser.add_argument("--mode", choices=["frame", "motion"], default="motion")
    parser.add_argument(
        "--roi",
        choices=["full", "lower75", "lower60", "center_lower", "center", "center_upper_body"],
        default="center_upper_body",
    )
    parser.add_argument("--crop-ratio", type=parse_crop_ratio, default=None)
    parser.add_argument("--feature", choices=["sobel", "edges", "gray"], default="sobel")
    parser.add_argument("--resize", type=parse_size, default=(160, 90))
    parser.add_argument("--band", type=int, default=6)
    parser.add_argument("--skip-penalty", type=float, default=None)
    parser.add_argument("--outside-cost", type=float, default=2.0)
    parser.add_argument("--min-run", type=int, default=3)
    parser.add_argument("--include-zero", action="store_true", help="keep zero-offset segments in auto output")
    parser.add_argument(
        "--offset-sign",
        choices=["ref_minus_target", "target_minus_ref"],
        default="target_minus_ref",
        help="target_minus_ref matches Nuke TimeOffset on the target: sample target frame current+offset",
    )
    parser.add_argument("--timeline", choices=["target", "reference"], default="target")
    args = parser.parse_args(argv)

    if args.keys:
        segments = args.keys
        summary = {
            "mode": "manual",
            "frame_start": args.frame_start,
            "segments": segments,
            "offset_sign": "manual",
        }
    else:
        segments, summary = auto_segments(args)

    if not segments:
        raise SystemExit("no offset segments were produced")

    curve_text = curve_text_from_segments(segments)
    out_path = default_out_path(args).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    node_name = args.name or (f"TimeOffset_{args.target.stem}" if args.target else "TimeOffset_Auto")
    write_nuke_node(out_path, curve_text, node_name, args.nuke_version)

    summary["curve"] = curve_text
    summary["nuke_node"] = str(out_path)
    summary["expanded_keys"] = expand_step_keys(segments)
    if args.json_out:
        json_path = args.json_out.resolve()
    else:
        json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Nuke node: {out_path}")
    print(f"JSON:      {json_path}")
    print("Segments:")
    for frame, offset in segments:
        print(f"  frame {frame}: {format_number(offset)}")
    print(f"Curve: {curve_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
