# Difference Viewer + Nuke TimeOffset Export

Frame-by-frame video comparison tool for checking editorial offset issues and exporting the confirmed result as a Nuke `TimeOffset` node.

## Problem

Seedream 2.0 style generated shots can drift by one frame, or only develop an offset inside a specific section of a shot. Checking this only by timestamp is unreliable, and manually drawing a Nuke `TimeOffset` curve is easy to get wrong.

This project focuses on two recurring failure points:

- Checking exact frame offsets while looking at a visual Difference view.
- Avoiding sign mistakes and frame-start mistakes when converting the offset into a Nuke `TimeOffset` curve.

## Implementation

### 1. Video Difference GUI

`nuke_timeoffset_gui.py` loads two videos, reformats both inputs to a shared working resolution, and previews them as:

- Reference
- Target
- Side by side
- Overlay
- Difference
- Difference over target

The default working resolution is `1920x1080`, matching the review resolution used for shot checking.

### 2. Frame Offset Analysis

`video_frame_alignment_checker.py` decodes both videos and compares actual image content instead of trusting container timestamps.

Supported feature modes:

- Gray
- Sobel
- Edge
- Motion-based alignment

Supported analysis regions:

- Full frame
- Lower frame regions
- Center regions
- Custom crop ratio

### 3. User Confirmed Keyframe Workflow

Automatic analysis can be ambiguous around low-motion or transition frames, so the GUI keeps a manual key workflow:

- Review the Difference view frame by frame.
- Adjust offset manually.
- Add user-confirmed keys.
- Preview the resulting offset curve before export.

### 4. Nuke TimeOffset Export

`nuke_timeoffset_exporter.py` writes a pasteable Nuke `.nk` node.

The internal analysis can keep frame-level offset samples, but the Nuke export compresses continuous equal-offset sections into boundary keys. This keeps the resulting curve inspectable while preserving step changes.

## Workflow

```text
Frame Decode
-> ROI / Full Frame selection
-> Sobel / Edge / Gray feature extraction
-> Frame or Motion based alignment
-> Offset sample generation
-> Same-offset run compression
-> Nuke TimeOffset .nk export
```

## Install

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`tkinter` is used for the GUI and is included with the standard Windows Python installer.

## Usage

Launch the GUI:

```powershell
.\.venv\Scripts\python.exe nuke_timeoffset_gui.py
```

Run content-based alignment and write a report:

```powershell
.\.venv\Scripts\python.exe video_frame_alignment_checker.py reference.mp4 target.mp4 --mode motion --roi center_upper_body --frame-start 1
```

Export a Nuke node from automatic analysis:

```powershell
.\.venv\Scripts\python.exe nuke_timeoffset_exporter.py reference.mp4 target.mp4 --mode motion --roi center_upper_body --frame-start 1 --out shot.TimeOffset.nk
```

Export a Nuke node from manual segment keys:

```powershell
.\.venv\Scripts\python.exe nuke_timeoffset_exporter.py --keys "1:-1,61:0" --out manual.TimeOffset.nk
```

## GUI Shortcuts

| Key | Action |
| --- | --- |
| Left / Right | Move 1 frame |
| Shift + Left / Right | Move 10 frames |
| Up | Offset +1 and create key |
| Down | Offset -1 and create key |

## Output

The exporter writes:

- `.nk`: Nuke `TimeOffset` node
- `.json`: analysis settings, offset samples, compressed segments, and expanded curve keys

The default offset sign is `target_minus_ref`, which matches applying `TimeOffset` to the target clip. Use the Difference view to confirm the sign before final comp use.

## Validation

The exporter and GUI include checks for:

- Missing input video paths
- Invalid frame size text
- Invalid ROI crop ratios
- Offset sign selection
- Frame-start labeling
- Step-curve boundary key generation

## Repository Notes

Raw review videos, contact sheets, alignment reports, and local backup folders are ignored by Git. Keep those files local unless they have been cleared for public sharing.

## Date

2026.06
