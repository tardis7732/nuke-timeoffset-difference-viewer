#!/usr/bin/env python3
"""
GUI viewer for comparing two videos and exporting a Nuke TimeOffset node.

The viewer reformats both inputs to the same working frame size, previews
reference/target/difference/overlay, lets the user build frame-offset keys, and
exports a pasteable Nuke TimeOffset node.
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from nuke_timeoffset_exporter import (
    auto_segments,
    curve_text_from_segments,
    parse_keys,
    safe_name,
    write_nuke_node,
)
from video_frame_alignment_checker import roi_rect


DEFAULT_SIZE = (1920, 1080)


class VideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []
        self._cache_limit = 96

    def close(self) -> None:
        self.cap.release()

    def _remember(self, index: int, frame: np.ndarray) -> None:
        if index in self._cache:
            return
        self._cache[index] = frame
        self._cache_order.append(index)
        while len(self._cache_order) > self._cache_limit:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def raw_frame(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= self.frame_count:
            return None
        cached = self._cache.get(index)
        if cached is not None:
            return cached.copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._remember(index, frame)
        return frame.copy()


def reformat_frame(
    frame: Optional[np.ndarray],
    size: Tuple[int, int],
    mode: str,
) -> np.ndarray:
    width, height = size
    if frame is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    src_h, src_w = frame.shape[:2]
    if mode == "resize":
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    if mode == "fit":
        scale = min(width / src_w, height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x = (width - new_w) // 2
        y = (height - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas

    if mode == "crop":
        scale = max(width / src_w, height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = max(0, (new_w - width) // 2)
        y = max(0, (new_h - height) // 2)
        return resized[y : y + height, x : x + width].copy()

    raise ValueError(f"Unknown reformat mode: {mode}")


def parse_size_text(value: str) -> Tuple[int, int]:
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError("Size must look like 1920x1080")
    width, height = int(parts[0]), int(parts[1])
    if width < 16 or height < 16:
        raise ValueError("Size is too small")
    return width, height


def bgr_to_photo(frame: np.ndarray, max_size: Tuple[int, int]) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    max_w, max_h = max_size
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def compose_view(
    ref: np.ndarray,
    target: np.ndarray,
    mode: str,
    diff_gain: float,
    overlay_alpha: float,
) -> np.ndarray:
    if mode == "Reference":
        return ref
    if mode == "Target":
        return target
    if mode == "Side by side":
        return np.hstack([ref, target])
    if mode == "Overlay":
        return cv2.addWeighted(ref, overlay_alpha, target, 1.0 - overlay_alpha, 0.0)
    if mode == "Difference":
        diff = cv2.absdiff(ref, target)
        diff = np.clip(diff.astype(np.float32) * diff_gain, 0, 255).astype(np.uint8)
        return diff
    if mode == "Difference over target":
        diff = cv2.absdiff(ref, target)
        heat = np.clip(diff.astype(np.float32) * diff_gain, 0, 255).astype(np.uint8)
        heat[:, :, 0] = 0
        return cv2.addWeighted(target, 0.55, heat, 0.85, 0.0)
    raise ValueError(f"Unknown view mode: {mode}")


class TimeOffsetGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Difference TimeOffset Builder")
        self.root.geometry("1320x900")
        self.root.minsize(1120, 760)
        self._configure_style()

        self.ref_video: Optional[VideoSource] = None
        self.target_video: Optional[VideoSource] = None
        self.keys: List[Tuple[int, float]] = []
        self.auto_offset_samples: List[Tuple[int, int]] = []
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.is_playing = False
        self.analyze_thread: Optional[threading.Thread] = None
        self._programmatic_offset_update = False

        self.ref_path_var = tk.StringVar()
        self.target_path_var = tk.StringVar()
        self.frame_start_var = tk.IntVar(value=1)
        self.frame_var = tk.IntVar(value=1)
        self.offset_var = tk.IntVar(value=0)
        self.size_var = tk.StringVar(value="1920x1080")
        self.reformat_var = tk.StringVar(value="resize")
        self.view_var = tk.StringVar(value="Difference")
        self.preview_curve_var = tk.BooleanVar(value=True)
        self.respect_keys_var = tk.BooleanVar(value=True)
        self.diff_gain_var = tk.DoubleVar(value=3.0)
        self.overlay_alpha_var = tk.DoubleVar(value=0.5)
        self.manual_keys_var = tk.StringVar(value="1:-1,61:0")
        self.nuke_version_var = tk.StringVar(value="15.1 v3")
        self.auto_mode_var = tk.StringVar(value="motion")
        self.auto_roi_var = tk.StringVar(value="center_upper_body")
        self.auto_feature_var = tk.StringVar(value="sobel")
        self.auto_sign_var = tk.StringVar(value="target_minus_ref")
        self.auto_skip_penalty_var = tk.StringVar(value="auto")
        self.auto_min_run_var = tk.IntVar(value=3)
        self.status_var = tk.StringVar(value="Load two videos to begin.")

        self._build_ui()
        self._bind_events()

    def _configure_style(self) -> None:
        self.root.configure(bg="#f3f4f6")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        base_bg = "#f3f4f6"
        panel_bg = "#ffffff"
        panel_bg_2 = "#e5e7eb"
        text = "#111827"
        muted = "#6b7280"
        accent = "#0369a1"
        style.configure(".", background=base_bg, foreground=text, font=("Segoe UI", 10))
        style.configure("TFrame", background=base_bg)
        style.configure("Panel.TFrame", background=panel_bg)
        style.configure("Header.TFrame", background="#ffffff")
        style.configure("TLabel", background=base_bg, foreground=text)
        style.configure("Muted.TLabel", background=base_bg, foreground=muted)
        style.configure("Header.TLabel", background="#ffffff", foreground=text, font=("Segoe UI Semibold", 15))
        style.configure("Subheader.TLabel", background="#ffffff", foreground=muted, font=("Segoe UI", 9))
        style.configure("Viewer.TLabel", background="#05070d", foreground=text)
        style.configure("Status.TLabel", background=base_bg, foreground=accent)
        style.configure("TLabelframe", background=panel_bg, foreground=text, bordercolor="#d1d5db", relief="solid")
        style.configure("TLabelframe.Label", background=panel_bg, foreground=text, font=("Segoe UI Semibold", 10))
        style.configure("TButton", background=panel_bg_2, foreground=text, bordercolor="#cbd5e1", padding=(8, 5))
        style.map("TButton", background=[("active", "#d1d5db")], foreground=[("disabled", "#9ca3af")])
        style.configure("Accent.TButton", background="#e0f2fe", foreground="#075985", bordercolor="#7dd3fc")
        style.map("Accent.TButton", background=[("active", "#bae6fd")])
        style.configure("Danger.TButton", background="#fee2e2", foreground="#991b1b", bordercolor="#fecaca")
        style.map("Danger.TButton", background=[("active", "#fecaca")])
        style.configure("TEntry", fieldbackground="#ffffff", foreground=text, insertcolor=text, bordercolor="#d1d5db")
        style.configure("TCombobox", fieldbackground="#ffffff", foreground=text, arrowcolor=text)
        style.configure("TCheckbutton", background=panel_bg, foreground=text)
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=text, rowheight=24)
        style.configure("Treeview.Heading", background="#e5e7eb", foreground=text, font=("Segoe UI Semibold", 9))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "#111827")])

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer, style="Header.TFrame", padding=(12, 10))
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="Video Difference TimeOffset Builder", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="  Difference / overlay review, anchored auto analysis, Nuke TimeOffset export",
            style="Subheader.TLabel",
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(header, text="←/→ frame  ·  ↑/↓ offset key", style="Subheader.TLabel").pack(side=tk.RIGHT)

        top = ttk.LabelFrame(outer, text="Inputs", padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Reference").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.ref_path_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_ref).grid(row=0, column=2, padx=2)
        ttk.Label(top, text="Target").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.target_path_var).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_target).grid(row=1, column=2, padx=2)
        ttk.Button(top, text="Load Pair", command=self.load_videos, style="Accent.TButton").grid(
            row=0, column=3, rowspan=2, sticky="nsew", padx=6
        )
        top.columnconfigure(1, weight=1)

        main = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, pady=(8, 4))

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=390)
        main.add(left, weight=4)
        main.add(right, weight=1)

        viewer_bar = ttk.LabelFrame(left, text="Viewer", padding=8)
        viewer_bar.pack(fill=tk.X)
        ttk.Label(viewer_bar, text="View").pack(side=tk.LEFT)
        ttk.Combobox(
            viewer_bar,
            textvariable=self.view_var,
            values=["Difference", "Difference over target", "Overlay", "Side by side", "Reference", "Target"],
            state="readonly",
            width=22,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(viewer_bar, text="Reformat").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Combobox(
            viewer_bar,
            textvariable=self.reformat_var,
            values=["resize", "fit", "crop"],
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(viewer_bar, text="Work size").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(viewer_bar, textvariable=self.size_var, width=10).pack(side=tk.LEFT, padx=4)

        image_frame = ttk.Frame(left, style="Panel.TFrame", padding=4)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.image_label = ttk.Label(image_frame, anchor=tk.CENTER, style="Viewer.TLabel")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        scrub = ttk.Frame(left)
        scrub.pack(fill=tk.X)
        ttk.Button(scrub, text="|<", width=4, command=lambda: self.step_frame(-999999)).pack(side=tk.LEFT)
        ttk.Button(scrub, text="<", width=4, command=lambda: self.step_frame(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(scrub, text="Play", width=7, command=self.toggle_play).pack(side=tk.LEFT, padx=2)
        ttk.Button(scrub, text=">", width=4, command=lambda: self.step_frame(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(scrub, text=">|", width=4, command=lambda: self.step_frame(999999)).pack(side=tk.LEFT)
        self.frame_scale = ttk.Scale(scrub, from_=1, to=100, variable=self.frame_var, command=lambda _v: self.on_frame_change())
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Label(scrub, text="Frame").pack(side=tk.LEFT)
        ttk.Spinbox(scrub, textvariable=self.frame_var, width=8, command=self.on_frame_change).pack(side=tk.LEFT, padx=4)

        info = ttk.Frame(left)
        info.pack(fill=tk.X)
        self.info_label = ttk.Label(info, text="No videos loaded.", style="Muted.TLabel")
        self.info_label.pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        options = ttk.LabelFrame(right, text="Offset Keys", padding=8)
        options.pack(fill=tk.X)
        row = ttk.Frame(options)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Frame start").pack(side=tk.LEFT)
        ttk.Spinbox(row, textvariable=self.frame_start_var, from_=-100000, to=100000, width=8, command=self.on_frame_start_change).pack(
            side=tk.RIGHT
        )
        row = ttk.Frame(options)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Offset").pack(side=tk.LEFT)
        ttk.Spinbox(row, textvariable=self.offset_var, from_=-1000, to=1000, width=8, command=self.refresh_view).pack(side=tk.RIGHT)
        ttk.Checkbutton(options, text="Preview curve keys", variable=self.preview_curve_var, command=self.refresh_view).pack(anchor="w")

        buttons = ttk.Frame(options)
        buttons.pack(fill=tk.X, pady=4)
        ttk.Button(buttons, text="Set Key", command=self.set_key, style="Accent.TButton").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1
        )
        ttk.Button(buttons, text="Delete", command=self.delete_selected_key).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(buttons, text="Clear All", command=self.clear_keys, style="Danger.TButton").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1
        )

        key_tools = ttk.Frame(options)
        key_tools.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(key_tools, text="Invert", command=self.invert_keys).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(key_tools, text="Simplify", command=self.simplify_keys).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        self.key_tree = ttk.Treeview(options, columns=("frame", "offset"), show="headings", height=8)
        self.key_tree.heading("frame", text="Frame")
        self.key_tree.heading("offset", text="Offset")
        self.key_tree.column("frame", width=70, anchor=tk.E)
        self.key_tree.column("offset", width=70, anchor=tk.E)
        self.key_tree.pack(fill=tk.X, pady=4)

        manual = ttk.LabelFrame(right, text="Manual / Export", padding=8)
        manual.pack(fill=tk.X, pady=8)
        ttk.Label(manual, text='Keys like "1:-1,61:0"').pack(anchor="w")
        ttk.Entry(manual, textvariable=self.manual_keys_var).pack(fill=tk.X, pady=2)
        version_row = ttk.Frame(manual)
        version_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(version_row, text="Nuke version").pack(side=tk.LEFT)
        ttk.Entry(version_row, textvariable=self.nuke_version_var, width=12).pack(side=tk.RIGHT)
        row = ttk.Frame(manual)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="Apply Keys", command=self.apply_manual_keys).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row, text="Copy NK", command=self.copy_nuke_node).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(manual, text="Export Nuke TimeOffset...", command=self.export_nuke, style="Accent.TButton").pack(
            fill=tk.X, pady=2
        )

        auto = ttk.LabelFrame(right, text="Auto Analyze", padding=8)
        auto.pack(fill=tk.X, pady=8)
        ttk.Label(auto, text="Auto fills candidates; checked keys are treated as confirmed anchors.").pack(anchor="w")
        ttk.Checkbutton(auto, text="Respect existing keys as anchors", variable=self.respect_keys_var).pack(anchor="w", pady=(2, 0))

        grid = ttk.Frame(auto)
        grid.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(grid, text="Mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.auto_mode_var, values=["motion", "frame"], state="readonly", width=14).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Label(grid, text="ROI").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            grid,
            textvariable=self.auto_roi_var,
            values=["center_upper_body", "center", "full", "lower75", "lower60", "center_lower"],
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(grid, text="Feature").grid(row=2, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.auto_feature_var, values=["sobel", "edges", "gray"], state="readonly", width=14).grid(
            row=2, column=1, sticky="ew", padx=4
        )
        ttk.Label(grid, text="Sign").grid(row=3, column=0, sticky="w")
        ttk.Combobox(
            grid,
            textvariable=self.auto_sign_var,
            values=["target_minus_ref", "ref_minus_target"],
            state="readonly",
            width=14,
        ).grid(row=3, column=1, sticky="ew", padx=4)
        ttk.Label(grid, text="Skip").grid(row=4, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.auto_skip_penalty_var, width=8).grid(row=4, column=1, sticky="ew", padx=4)
        ttk.Label(grid, text="Min run").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(grid, textvariable=self.auto_min_run_var, from_=1, to=24, width=8).grid(row=5, column=1, sticky="ew", padx=4)
        grid.columnconfigure(1, weight=1)

        ttk.Button(auto, text="Run Auto", command=self.run_auto, style="Accent.TButton").pack(fill=tk.X, pady=4)
        ttk.Label(auto, text="Diff gain").pack(anchor="w", pady=(6, 0))
        ttk.Scale(auto, from_=1.0, to=12.0, variable=self.diff_gain_var, command=lambda _v: self.refresh_view()).pack(fill=tk.X)
        ttk.Label(auto, text="Overlay alpha").pack(anchor="w", pady=(6, 0))
        ttk.Scale(auto, from_=0.0, to=1.0, variable=self.overlay_alpha_var, command=lambda _v: self.refresh_view()).pack(fill=tk.X)

        log_frame = ttk.LabelFrame(right, text="Log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(
            log_frame,
            height=10,
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self) -> None:
        for var in [self.view_var, self.reformat_var, self.size_var]:
            var.trace_add("write", lambda *_args: self.refresh_view())
        self.frame_var.trace_add("write", lambda *_args: self.on_frame_change())
        self.offset_var.trace_add("write", lambda *_args: self.on_offset_change())
        self.key_tree.bind("<<TreeviewSelect>>", self.on_key_select)
        self.root.bind_all("<Left>", lambda event: self.handle_frame_key(event, -1))
        self.root.bind_all("<Right>", lambda event: self.handle_frame_key(event, 1))
        self.root.bind_all("<Shift-Left>", lambda event: self.handle_frame_key(event, -10))
        self.root.bind_all("<Shift-Right>", lambda event: self.handle_frame_key(event, 10))
        self.root.bind_all("<Up>", lambda event: self.handle_offset_key(event, 1))
        self.root.bind_all("<Down>", lambda event: self.handle_offset_key(event, -1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def handle_frame_key(self, event: tk.Event, delta: int) -> str:
        widget = event.widget
        if isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return ""
        self.step_frame(delta)
        return "break"

    def handle_offset_key(self, event: tk.Event, delta: int) -> str:
        widget = event.widget
        if isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return ""
        current = self.current_offset()
        new_offset = current + delta
        self.set_offset_display(new_offset)
        self.set_key()
        return "break"

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, message.rstrip() + "\n")
        self.log_text.see(tk.END)
        self.status_var.set(message.rstrip())

    def browse_ref(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.mxf *.avi *.mkv"), ("All files", "*.*")])
        if path:
            self.ref_path_var.set(path)

    def browse_target(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.mxf *.avi *.mkv"), ("All files", "*.*")])
        if path:
            self.target_path_var.set(path)

    def load_videos(self) -> None:
        try:
            if self.ref_video:
                self.ref_video.close()
            if self.target_video:
                self.target_video.close()
            ref_path = Path(self.ref_path_var.get()).expanduser()
            target_path = Path(self.target_path_var.get()).expanduser()
            self.ref_video = VideoSource(ref_path)
            self.target_video = VideoSource(target_path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        start = self.frame_start_var.get()
        max_frames = max(self.ref_video.frame_count, self.target_video.frame_count)
        end = start + max_frames - 1
        self.frame_scale.configure(from_=start, to=end)
        self.frame_var.set(start)
        self.info_label.configure(
            text=(
                f"Reference: {self.ref_video.width}x{self.ref_video.height}, {self.ref_video.fps:.3f} fps, "
                f"{self.ref_video.frame_count}f | Target: {self.target_video.width}x{self.target_video.height}, "
                f"{self.target_video.fps:.3f} fps, {self.target_video.frame_count}f"
            )
        )
        self.log(f"Loaded {ref_path.name} and {target_path.name}")
        self.refresh_view()

    def on_frame_start_change(self) -> None:
        if self.ref_video and self.target_video:
            max_frames = max(self.ref_video.frame_count, self.target_video.frame_count)
            start = self.frame_start_var.get()
            self.frame_scale.configure(from_=start, to=start + max_frames - 1)
        self.refresh_view()

    def timeline_frame(self) -> int:
        return int(round(float(self.frame_var.get())))

    def decoded_index(self) -> int:
        return self.timeline_frame() - self.frame_start_var.get()

    def current_offset(self) -> int:
        if self.preview_curve_var.get() and self.keys:
            frame = self.timeline_frame()
            offset = self.offset_for_frame(frame)
            self.set_offset_display(offset)
            return offset
        return int(self.offset_var.get())

    def set_offset_display(self, offset: int) -> None:
        if self.offset_var.get() == offset:
            return
        self._programmatic_offset_update = True
        try:
            self.offset_var.set(offset)
        finally:
            self._programmatic_offset_update = False

    def on_offset_change(self) -> None:
        if self._programmatic_offset_update:
            return
        if not self._programmatic_offset_update and self.preview_curve_var.get():
            # User is trying a manual offset. Stop the curve preview from
            # snapping the spinbox back to the keyed value.
            self.preview_curve_var.set(False)
        self.refresh_view()

    def offset_for_frame(self, frame: int) -> int:
        if not self.keys:
            return int(self.offset_var.get())
        result = int(round(self.keys[0][1]))
        for key_frame, key_offset in sorted(self.keys):
            if frame >= key_frame:
                result = int(round(key_offset))
            else:
                break
        return result

    def on_frame_change(self) -> None:
        self.refresh_view()

    def step_frame(self, delta: int) -> None:
        start = int(float(self.frame_scale.cget("from")))
        end = int(float(self.frame_scale.cget("to")))
        if delta < -1000:
            frame = start
        elif delta > 1000:
            frame = end
        else:
            frame = max(start, min(end, self.timeline_frame() + delta))
        self.frame_var.set(frame)
        self.refresh_view()

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        if self.is_playing:
            self._play_tick()

    def _play_tick(self) -> None:
        if not self.is_playing:
            return
        self.step_frame(1)
        fps = 24.0
        if self.ref_video and self.ref_video.fps > 0:
            fps = self.ref_video.fps
        self.root.after(max(1, int(1000 / fps)), self._play_tick)

    def work_size(self) -> Tuple[int, int]:
        try:
            return parse_size_text(self.size_var.get())
        except Exception:
            return DEFAULT_SIZE

    def refresh_view(self) -> None:
        if not self.ref_video or not self.target_video:
            return
        try:
            size = self.work_size()
            mode = self.reformat_var.get()
            ref_idx = self.decoded_index()
            offset = self.current_offset()
            target_idx = ref_idx + offset
            ref = reformat_frame(self.ref_video.raw_frame(ref_idx), size, mode)
            target = reformat_frame(self.target_video.raw_frame(target_idx), size, mode)
            composed = compose_view(
                ref,
                target,
                self.view_var.get(),
                float(self.diff_gain_var.get()),
                float(self.overlay_alpha_var.get()),
            )
            max_w = max(320, self.image_label.winfo_width() - 12)
            max_h = max(240, self.image_label.winfo_height() - 12)
            self.photo = bgr_to_photo(composed, (max_w, max_h))
            self.image_label.configure(image=self.photo)
            self.status_var.set(
                f"timeline {self.timeline_frame()} | ref index {ref_idx} | target index {target_idx} | offset {offset}"
            )
        except Exception as exc:
            self.status_var.set(str(exc))

    def update_key_tree(self) -> None:
        self.key_tree.delete(*self.key_tree.get_children())
        for frame, offset in sorted(self.keys):
            self.key_tree.insert("", tk.END, values=(frame, int(round(offset))))
        self.manual_keys_var.set(",".join(f"{frame}:{int(round(offset))}" for frame, offset in sorted(self.keys)))

    def set_key(self) -> None:
        frame = self.timeline_frame()
        offset = int(self.offset_var.get())
        data = {f: o for f, o in self.keys}
        data[frame] = offset
        self.keys = sorted(data.items())
        self.update_key_tree()
        self.preview_curve_var.set(True)
        self.log(f"Set confirmed key frame {frame}: {offset}")
        self.refresh_view()

    def delete_selected_key(self) -> None:
        selected = self.key_tree.selection()
        if not selected:
            return
        values = self.key_tree.item(selected[0], "values")
        frame = int(values[0])
        self.keys = [(f, o) for f, o in self.keys if f != frame]
        self.update_key_tree()
        self.log(f"Deleted key frame {frame}")
        self.refresh_view()

    def clear_keys(self) -> None:
        if not self.keys and not self.auto_offset_samples:
            return
        self.keys = []
        self.auto_offset_samples = []
        self.update_key_tree()
        self.manual_keys_var.set("")
        self.log("Cleared all keys")
        self.refresh_view()

    def invert_keys(self) -> None:
        if not self.keys:
            return
        self.keys = [(frame, -offset) for frame, offset in self.keys]
        self.update_key_tree()
        self.log("Inverted all key offsets")
        self.refresh_view()

    def simplify_keys(self) -> None:
        if not self.keys:
            return
        simplified: List[Tuple[int, float]] = []
        for frame, offset in sorted(self.keys):
            if simplified and simplified[-1][1] == offset:
                continue
            simplified.append((frame, offset))
        self.keys = simplified
        self.update_key_tree()
        self.log("Simplified repeated adjacent offset keys")
        self.refresh_view()

    def on_key_select(self, _event: object) -> None:
        selected = self.key_tree.selection()
        if not selected:
            return
        values = self.key_tree.item(selected[0], "values")
        self.frame_var.set(int(values[0]))
        self.set_offset_display(int(values[1]))
        self.preview_curve_var.set(True)
        self.refresh_view()

    def apply_manual_keys(self) -> None:
        try:
            self.keys = parse_keys(self.manual_keys_var.get())
        except Exception as exc:
            messagebox.showerror("Bad keys", str(exc))
            return
        self.update_key_tree()
        self.log(f"Applied {len(self.keys)} manual key(s)")
        self.refresh_view()

    def curve_text(self) -> str:
        if not self.keys:
            raise RuntimeError("No keys to export")
        return curve_text_from_segments(self.keys)

    def nuke_node_text(self) -> str:
        curve = self.curve_text()
        name = "TimeOffset_GUI"
        if self.target_video:
            name = safe_name("TimeOffset_" + self.target_video.path.stem)
        return (
            "set cut_paste_input [stack 0]\n"
            f"version {self.nuke_version_var.get().strip() or '15.1 v3'}\n"
            "push $cut_paste_input\n"
            "TimeOffset {\n"
            f" time_offset {curve}\n"
            ' time ""\n'
            f" name {name}\n"
            " selected true\n"
            "}\n"
        )

    def copy_nuke_node(self) -> None:
        try:
            text = self.nuke_node_text()
        except Exception as exc:
            messagebox.showerror("Copy failed", str(exc))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.log("Copied Nuke node to clipboard")

    def export_nuke(self) -> None:
        if not self.keys:
            messagebox.showwarning("No keys", "Set or apply offset keys first.")
            return
        default = "TimeOffset.nk"
        if self.target_video:
            default = f"{self.target_video.path.stem}.TimeOffset.nk"
        path = filedialog.asksaveasfilename(
            defaultextension=".nk",
            initialfile=default,
            filetypes=[("Nuke script", "*.nk"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            name = "TimeOffset_GUI"
            if self.target_video:
                name = safe_name("TimeOffset_" + self.target_video.path.stem)
            write_nuke_node(Path(path), self.curve_text(), name, self.nuke_version_var.get().strip() or "15.1 v3")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.log(f"Exported {path}")

    def run_auto(self) -> None:
        if not self.ref_video or not self.target_video:
            messagebox.showwarning("No videos", "Load two videos first.")
            return
        if self.analyze_thread and self.analyze_thread.is_alive():
            messagebox.showinfo("Busy", "Auto analysis is already running.")
            return

        skip_text = self.auto_skip_penalty_var.get().strip().lower()
        try:
            skip_penalty = None if skip_text in ("", "auto") else float(skip_text)
            min_run = int(self.auto_min_run_var.get())
        except ValueError as exc:
            messagebox.showerror("Bad auto settings", str(exc))
            return

        def worker() -> None:
            try:
                ns = argparse.Namespace(
                    reference=self.ref_video.path,
                    target=self.target_video.path,
                    mode=self.auto_mode_var.get(),
                    roi=self.auto_roi_var.get(),
                    crop_ratio=None,
                    feature=self.auto_feature_var.get(),
                    resize=(160, 90),
                    band=6,
                    skip_penalty=skip_penalty,
                    outside_cost=2.0,
                    min_run=min_run,
                    include_zero=False,
                    offset_sign=self.auto_sign_var.get(),
                    timeline="target",
                    frame_start=self.frame_start_var.get(),
                )
                segments, summary = auto_segments(ns)
                anchors = list(self.keys)
                respect = bool(self.respect_keys_var.get())
                self.root.after(0, lambda: self.finish_auto(segments, summary, anchors, respect))
            except Exception as exc:
                error = str(exc)
                self.root.after(0, lambda: messagebox.showerror("Auto failed", error))

        self.log(
            "Running auto analysis... "
            f"mode={self.auto_mode_var.get()} roi={self.auto_roi_var.get()} sign={self.auto_sign_var.get()} skip={skip_text or 'auto'}"
        )
        self.analyze_thread = threading.Thread(target=worker, daemon=True)
        self.analyze_thread.start()

    def finish_auto(
        self,
        segments: List[Tuple[int, float]],
        summary: dict,
        anchors: Optional[List[Tuple[int, float]]] = None,
        respect_anchors: bool = False,
    ) -> None:
        if respect_anchors and anchors:
            auto_keys = self.keys_from_auto_summary(summary, segments)
            self.keys = self.merge_auto_with_anchors(auto_keys, anchors)
            mode_text = f"Auto analysis finished; preserved {len(anchors)} confirmed key(s)"
        else:
            self.keys = self.keys_from_auto_summary(summary, segments)
            mode_text = "Auto analysis finished"
        self.update_key_tree()
        self.preview_curve_var.set(True)
        self.log(mode_text)
        self.log(f"Auto segments: {self.manual_keys_var.get()}")
        self.log(f"Auto curve: {curve_text_from_segments(self.keys)}")
        self.refresh_view()

    def keys_from_auto_summary(self, summary: dict, fallback_segments: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
        samples = [
            (int(frame), int(round(offset)))
            for frame, offset in summary.get("offset_by_frame", [])
        ]
        self.auto_offset_samples = samples
        if not samples:
            return sorted((int(frame), float(offset)) for frame, offset in fallback_segments)
        self.warn_if_opposite_sign_scores_better(samples)
        keys = self.compress_offset_samples(samples)
        self.log(f"Auto raw offset samples kept internally: {len(samples)} frame(s)")
        self.log(f"Auto export keys compressed from samples: {','.join(f'{f}:{int(o)}' for f, o in keys)}")
        return keys or sorted((int(frame), float(offset)) for frame, offset in fallback_segments)

    def compress_offset_samples(self, samples: Sequence[Tuple[int, int]]) -> List[Tuple[int, float]]:
        """Turn per-frame offsets like 1,1,1,0,0 into run-start keys."""
        ordered = sorted(samples, key=lambda x: x[0])
        if not ordered:
            return []
        runs: List[Tuple[int, int, int]] = []
        run_start, current_offset = ordered[0]
        prev_frame = run_start
        for frame, offset in ordered[1:]:
            if frame == prev_frame + 1 and offset == current_offset:
                prev_frame = frame
                continue
            runs.append((run_start, prev_frame, current_offset))
            run_start, current_offset = frame, offset
            prev_frame = frame
        runs.append((run_start, prev_frame, current_offset))

        keys: List[Tuple[int, float]] = []
        leading_zero_end: Optional[int] = None
        for start_frame, end_frame, offset in runs:
            if offset == 0 and not keys:
                leading_zero_end = end_frame
                continue
            if offset != 0 and not keys and leading_zero_end is not None and leading_zero_end == start_frame - 1:
                keys.append((leading_zero_end, 0.0))
            if offset != 0 or keys:
                if keys and keys[-1][1] == float(offset):
                    continue
                keys.append((start_frame, float(offset)))

        compact: List[Tuple[int, float]] = []
        for frame, offset in keys:
            if compact and compact[-1][1] == offset:
                continue
            compact.append((frame, offset))
        return compact

    def warn_if_opposite_sign_scores_better(self, samples: Sequence[Tuple[int, int]]) -> None:
        current_score, current_count = self.score_offset_samples(samples, sign=1)
        opposite_score, opposite_count = self.score_offset_samples(samples, sign=-1)
        if current_count == 0 or opposite_count == 0:
            self.log("Auto sign check: not enough non-zero samples to compare.")
            return
        self.log(
            "Auto sign check: "
            f"current={current_score:.5f} over {current_count}f, "
            f"opposite={opposite_score:.5f} over {opposite_count}f"
        )
        if opposite_score + 0.01 < current_score:
            self.log("WARNING: opposite offset sign scores better. Inspect/invert the auto keys before export.")

    def score_offset_samples(self, samples: Sequence[Tuple[int, int]], sign: int) -> Tuple[float, int]:
        if not self.ref_video or not self.target_video:
            return 0.0, 0
        costs: List[float] = []
        for frame_label, offset in samples:
            if offset == 0:
                continue
            ref_index = frame_label - self.frame_start_var.get()
            target_index = ref_index + (offset * sign)
            ref_feature = self.frame_feature(self.ref_video, ref_index)
            target_feature = self.frame_feature(self.target_video, target_index)
            if ref_feature is None or target_feature is None:
                continue
            similarity = float(np.dot(ref_feature, target_feature))
            costs.append(1.0 - similarity)
        if not costs:
            return 0.0, 0
        return float(np.mean(costs)), len(costs)

    def frame_feature(self, source: VideoSource, index: int) -> Optional[np.ndarray]:
        frame = source.raw_frame(index)
        if frame is None:
            return None
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = roi_rect(width, height, self.auto_roi_var.get(), None)
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        sx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.log1p(cv2.magnitude(sx, sy))
        vector = mag.reshape(-1).astype(np.float32)
        vector -= float(vector.mean())
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return None
        return vector / norm

    def merge_auto_with_anchors(
        self,
        auto_segments: Sequence[Tuple[int, float]],
        anchors: Sequence[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        locked = sorted((int(frame), float(offset)) for frame, offset in anchors)
        auto = sorted((int(frame), float(offset)) for frame, offset in auto_segments)

        merged: Dict[int, float] = {frame: offset for frame, offset in locked}
        for frame, offset in auto:
            # User keys are exact confirmed frames, not interval locks. Auto is
            # allowed to add corrections between them, but never overwrite the
            # keyed frame itself.
            if frame in merged:
                continue
            merged[frame] = offset

        result = sorted(merged.items())
        compact: List[Tuple[int, float]] = []
        for frame, offset in result:
            if compact and compact[-1][1] == offset:
                # Keep user anchors even when the value equals the previous
                # segment. Artists may have placed it as an inspection marker.
                if any(frame == locked_frame for locked_frame, _ in locked):
                    compact.append((frame, offset))
                continue
            compact.append((frame, offset))
        return compact

    def on_close(self) -> None:
        self.is_playing = False
        if self.ref_video:
            self.ref_video.close()
        if self.target_video:
            self.target_video.close()
        self.root.destroy()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the video TimeOffset GUI.")
    parser.add_argument("reference", nargs="?", help="optional reference video")
    parser.add_argument("target", nargs="?", help="optional target video")
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = TimeOffsetGui(root)
    if args.reference:
        app.ref_path_var.set(str(Path(args.reference).resolve()))
    if args.target:
        app.target_path_var.set(str(Path(args.target).resolve()))
    if args.reference and args.target:
        root.after(100, app.load_videos)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
