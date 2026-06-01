#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


CANVAS_BG = "#111111"
LINK_COLOR = "#5f6670"
SELECTED_LINK_COLOR = "#ffb000"
HOVER_LINK_COLOR = "#4dc3ff"
ACTIVE_GROUP_COLOR = "#42d392"
INACTIVE_GROUP_COLOR = "#b286ff"
PARTIAL_LINK_COLOR = "#ff5e8a"
CENTER_LINK_COLOR = "#ff4d8d"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_selection_data(path):
    if not path:
        return {"selected_link_ids": set(), "center_link_ids": set(), "fill_groups": [], "partial_links": {}}

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"selection file not found: {path}")

    if path.lower().endswith(".json"):
        data = load_json(path)

        if isinstance(data, list):
            return {
                "selected_link_ids": {str(item).strip() for item in data if str(item).strip()},
                "center_link_ids": set(),
                "fill_groups": [],
                "partial_links": {},
            }

        if isinstance(data, dict):
            return {
                "selected_link_ids": {str(item).strip() for item in data.get("selected_link_ids", []) if str(item).strip()},
                "center_link_ids": {str(item).strip() for item in data.get("center_link_ids", []) if str(item).strip()},
                "fill_groups": [
                    {
                        "name": str(group.get("name") or f"group_{index + 1:02d}"),
                        "link_ids": {str(item).strip() for item in group.get("link_ids", []) if str(item).strip()},
                    }
                    for index, group in enumerate(data.get("fill_groups", []))
                ],
                "partial_links": {
                    str(item.get("link_id")).strip(): {
                        "start_ratio": float(item.get("start_ratio", 0.0)),
                        "end_ratio": float(item.get("end_ratio", 1.0)),
                    }
                    for item in data.get("partial_links", [])
                    if str(item.get("link_id", "")).strip()
                },
            }

        raise ValueError("selection json must be a list or an object")

    with open(path, "r", encoding="utf-8") as file:
        raw = file.read()

    tokens = []
    for line in raw.splitlines():
        tokens.extend(line.replace(",", " ").split())

    return {
        "selected_link_ids": {token.strip() for token in tokens if token.strip()},
        "center_link_ids": set(),
        "fill_groups": [],
        "partial_links": {},
    }


def save_selection_data(path, selected_link_ids, center_link_ids, fill_groups, partial_links):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ordered_selected = sorted(selected_link_ids)
    ordered_center = sorted(center_link_ids)
    normalized_groups = [
        {
            "name": str(group.get("name") or f"group_{index + 1:02d}"),
            "link_ids": sorted(group.get("link_ids", set())),
        }
        for index, group in enumerate(fill_groups)
    ]
    normalized_partial = [
        {
            "link_id": link_id,
            "start_ratio": float(spec["start_ratio"]),
            "end_ratio": float(spec["end_ratio"]),
        }
        for link_id, spec in sorted(partial_links.items())
    ]

    payload = {
        "selected_link_ids": ordered_selected,
        "center_link_ids": ordered_center,
        "fill_groups": normalized_groups,
        "partial_links": normalized_partial,
    }

    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        return path, None

    with open(path, "w", encoding="utf-8") as file:
        for link_id in ordered_selected:
            file.write(link_id + "\n")

    sidecar_path = None
    if ordered_center or normalized_groups or normalized_partial:
        sidecar_path = os.path.splitext(path)[0] + "_selection.json"
        with open(sidecar_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    return path, sidecar_path


def compute_bounds(link_set):
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for link in link_set:
        for point in link.get("points", []):
            min_x = min(min_x, float(point[0]))
            min_y = min(min_y, float(point[1]))
            max_x = max(max_x, float(point[0]))
            max_y = max(max_y, float(point[1]))

    if not math.isfinite(min_x):
        raise ValueError("Could not compute bounds from link_set.")

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
    }


def point_to_segment_distance_px(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def distance_2d(p0, p1):
    return math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))


def interpolate_point(p0, p1, t):
    z0 = float(p0[2]) if len(p0) > 2 else 0.0
    z1 = float(p1[2]) if len(p1) > 2 else 0.0
    return [
        float(p0[0]) + t * (float(p1[0]) - float(p0[0])),
        float(p0[1]) + t * (float(p1[1]) - float(p0[1])),
        z0 + t * (z1 - z0),
    ]


def trim_polyline_by_ratio(points, start_ratio, end_ratio):
    if len(points) < 2:
        return points

    start_ratio = max(0.0, min(1.0, float(start_ratio)))
    end_ratio = max(0.0, min(1.0, float(end_ratio)))
    if end_ratio <= start_ratio:
        return []
    if start_ratio <= 0.0 and end_ratio >= 1.0:
        return [list(point) for point in points]

    cumulative = [0.0]
    total_length = 0.0
    for index in range(1, len(points)):
        seg_len = distance_2d(points[index - 1], points[index])
        total_length += seg_len
        cumulative.append(total_length)

    if total_length <= 1e-9:
        return []

    start_distance = start_ratio * total_length
    end_distance = end_ratio * total_length
    trimmed = []

    for index in range(1, len(points)):
        seg_start = cumulative[index - 1]
        seg_end = cumulative[index]
        if seg_end <= start_distance or seg_start >= end_distance:
            continue

        p0 = points[index - 1]
        p1 = points[index]
        seg_len = seg_end - seg_start
        if seg_len <= 1e-9:
            continue

        local_start = max(start_distance, seg_start)
        local_end = min(end_distance, seg_end)
        t0 = (local_start - seg_start) / seg_len
        t1 = (local_end - seg_start) / seg_len
        start_point = interpolate_point(p0, p1, t0)
        end_point = interpolate_point(p0, p1, t1)

        if not trimmed or distance_2d(trimmed[-1], start_point) > 1e-9:
            trimmed.append(start_point)
        if distance_2d(trimmed[-1], end_point) > 1e-9:
            trimmed.append(end_point)

    return trimmed


class LinkSelectorApp(tk.Tk):
    def __init__(
        self,
        mgeo_dir,
        output_file,
        initial_selected_link_ids=None,
        initial_center_link_ids=None,
        initial_fill_groups=None,
        initial_partial_links=None,
    ):
        super().__init__()
        self.title("MGeo Link Selector")
        self.geometry("1560x940")

        self.mgeo_dir = os.path.abspath(mgeo_dir)
        self.output_file = os.path.abspath(output_file)
        self.link_set = load_json(os.path.join(self.mgeo_dir, "link_set.json"))
        self.link_by_id = {link["idx"]: link for link in self.link_set}

        self.selected_link_ids = set(initial_selected_link_ids or set())
        self.center_link_ids = {
            link_id for link_id in (initial_center_link_ids or set()) if link_id in self.selected_link_ids
        }
        self.fill_groups = []
        for index, group in enumerate(initial_fill_groups or []):
            self.fill_groups.append(
                {
                    "name": str(group.get("name") or f"group_{index + 1:02d}"),
                    "link_ids": set(group.get("link_ids", set())),
                }
            )
        self.partial_links = {}
        for link_id, spec in (initial_partial_links or {}).items():
            self.partial_links[str(link_id)] = {
                "start_ratio": max(0.0, min(1.0, float(spec.get("start_ratio", 0.0)))),
                "end_ratio": max(0.0, min(1.0, float(spec.get("end_ratio", 1.0)))),
            }

        self.bounds = compute_bounds(self.link_set)
        self.hover_link_id = None
        self.current_mode = tk.StringVar(value="select")
        self.active_group_index = 0 if self.fill_groups else -1
        self.active_partial_link_id = None

        self.canvas_width = 1000
        self.canvas_height = 900

        world_width = max(self.bounds["max_x"] - self.bounds["min_x"], 1.0)
        world_height = max(self.bounds["max_y"] - self.bounds["min_y"], 1.0)
        fit_scale_x = self.canvas_width / world_width
        fit_scale_y = self.canvas_height / world_height
        self.pixels_per_meter = min(fit_scale_x, fit_scale_y) * 0.9
        self.view_center_x = 0.5 * (self.bounds["min_x"] + self.bounds["max_x"])
        self.view_center_y = 0.5 * (self.bounds["min_y"] + self.bounds["max_y"])

        self.pan_last = None
        self.link_segments_world = self._precompute_segments()

        self._build_ui()
        self._bind_events()
        self._refresh_selection_list()
        self._refresh_center_list()
        self._refresh_fill_groups_list()
        self._refresh_partial_links_list()
        self._update_mode_status()
        self._redraw()

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(root, width=470)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        toolbar = ttk.Frame(left)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Save", command=self._save_selection).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Save As", command=self._save_selection_as).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Clear Links", command=self._clear_selection).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Reset View", command=self._reset_view).pack(side=tk.LEFT, padx=4, pady=4)

        ttk.Label(toolbar, text="Mode").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Radiobutton(toolbar, text="Select Links", variable=self.current_mode, value="select", command=self._on_mode_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(toolbar, text="Center Links", variable=self.current_mode, value="center", command=self._on_mode_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(toolbar, text="Edit Fill Groups", variable=self.current_mode, value="fill", command=self._on_mode_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(toolbar, text="Partial Links", variable=self.current_mode, value="partial", command=self._on_mode_changed).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(left, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        right_canvas = tk.Canvas(right, highlightthickness=0)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scrollbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=right_canvas.yview)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        self.right_scroll_frame = ttk.Frame(right_canvas)
        self.right_scroll_window = right_canvas.create_window((0, 0), window=self.right_scroll_frame, anchor="nw")
        self.right_canvas = right_canvas

        self.right_scroll_frame.bind("<Configure>", self._on_right_panel_configure)
        self.right_canvas.bind("<Configure>", self._on_right_canvas_resize)

        selected_frame = ttk.LabelFrame(self.right_scroll_frame, text="Selected Links")
        selected_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 6))

        self.selected_count_var = tk.StringVar(value="0 links")
        ttk.Label(selected_frame, textvariable=self.selected_count_var).pack(anchor=tk.W, padx=8, pady=(6, 4))

        search_row = ttk.Frame(selected_frame)
        search_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(search_row, text="Toggle ID", command=self._toggle_link_from_entry).pack(side=tk.LEFT, padx=(6, 0))

        self.selected_listbox = tk.Listbox(selected_frame, exportselection=False, height=14)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        center_frame = ttk.LabelFrame(self.right_scroll_frame, text="Center Links")
        center_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        self.center_count_var = tk.StringVar(value="0 center links")
        ttk.Label(center_frame, textvariable=self.center_count_var).pack(anchor=tk.W, padx=8, pady=(6, 4))
        ttk.Label(
            center_frame,
            text="Use these links as the progress backbone. They should follow the center route you want to race on.",
            wraplength=420,
        ).pack(anchor=tk.W, padx=8, pady=(0, 6))
        self.center_listbox = tk.Listbox(center_frame, exportselection=False, height=8)
        self.center_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        groups_frame = ttk.LabelFrame(self.right_scroll_frame, text="Fill Groups")
        groups_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        self.group_summary_var = tk.StringVar(value="0 groups")
        ttk.Label(groups_frame, textvariable=self.group_summary_var).pack(anchor=tk.W, padx=8, pady=(6, 4))

        group_buttons = ttk.Frame(groups_frame)
        group_buttons.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(group_buttons, text="New Group", command=self._create_fill_group).pack(side=tk.LEFT)
        ttk.Button(group_buttons, text="Rename", command=self._rename_active_group).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(group_buttons, text="Delete", command=self._delete_active_group).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(group_buttons, text="Clear Group", command=self._clear_active_group).pack(side=tk.LEFT, padx=(6, 0))

        self.fill_groups_listbox = tk.Listbox(groups_frame, exportselection=False, height=8)
        self.fill_groups_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        group_name_row = ttk.Frame(groups_frame)
        group_name_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.group_name_var = tk.StringVar()
        self.group_name_entry = ttk.Entry(group_name_row, textvariable=self.group_name_var)
        self.group_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(group_name_row, text="Apply Name", command=self._apply_group_name_from_entry).pack(side=tk.LEFT, padx=(6, 0))

        partial_frame = ttk.LabelFrame(self.right_scroll_frame, text="Partial Links")
        partial_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.partial_summary_var = tk.StringVar(value="0 partial links")
        ttk.Label(partial_frame, textvariable=self.partial_summary_var).pack(anchor=tk.W, padx=8, pady=(6, 4))

        partial_buttons = ttk.Frame(partial_frame)
        partial_buttons.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(partial_buttons, text="Remove Partial", command=self._remove_active_partial_link).pack(side=tk.LEFT)
        ttk.Button(partial_buttons, text="Reset Range", command=self._reset_active_partial_link).pack(side=tk.LEFT, padx=(6, 0))

        self.partial_listbox = tk.Listbox(partial_frame, exportselection=False, height=8)
        self.partial_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.partial_link_var = tk.StringVar(value="No partial link selected")
        ttk.Label(partial_frame, textvariable=self.partial_link_var).pack(anchor=tk.W, padx=8, pady=(0, 6))

        start_row = ttk.Frame(partial_frame)
        start_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(start_row, text="Start").pack(side=tk.LEFT)
        self.partial_start_var = tk.DoubleVar(value=0.0)
        self.partial_start_scale = ttk.Scale(start_row, from_=0.0, to=100.0, variable=self.partial_start_var, command=self._on_partial_scale_changed)
        self.partial_start_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.partial_start_label = ttk.Label(start_row, text="0.00")
        self.partial_start_label.pack(side=tk.LEFT)

        end_row = ttk.Frame(partial_frame)
        end_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(end_row, text="End").pack(side=tk.LEFT)
        self.partial_end_var = tk.DoubleVar(value=100.0)
        self.partial_end_scale = ttk.Scale(end_row, from_=0.0, to=100.0, variable=self.partial_end_var, command=self._on_partial_scale_changed)
        self.partial_end_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.partial_end_label = ttk.Label(end_row, text="1.00")
        self.partial_end_label.pack(side=tk.LEFT)

        help_frame = ttk.LabelFrame(self.right_scroll_frame, text="Help")
        help_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        help_lines = [
            "Select Links mode: click to add/remove racing links",
            "Center Links mode: click selected links to mark/unmark progress backbone links",
            "Edit Fill Groups mode: click selected links to add/remove from active fill group",
            "Partial Links mode: click a selected link, then adjust start/end range",
            "Save .json to keep selected links, center links, fill groups, and partial links together",
            "Save .txt writes selected links and creates a sidecar json when needed",
        ]
        for line in help_lines:
            ttk.Label(help_frame, text=line).pack(anchor=tk.W, padx=8, pady=2)

    def _on_right_panel_configure(self, _event):
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def _on_right_canvas_resize(self, event):
        self.right_canvas.itemconfigure(self.right_scroll_window, width=event.width)

    def _bind_events(self):
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel_linux)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel_linux)
        self.selected_listbox.bind("<Double-Button-1>", self._remove_selected_from_listbox)
        self.center_listbox.bind("<Double-Button-1>", self._remove_center_from_listbox)
        self.fill_groups_listbox.bind("<<ListboxSelect>>", self._on_fill_group_selected)
        self.partial_listbox.bind("<<ListboxSelect>>", self._on_partial_link_selected)
        self.search_entry.bind("<Return>", lambda _event: self._toggle_link_from_entry())
        self.group_name_entry.bind("<Return>", lambda _event: self._apply_group_name_from_entry())

    def _precompute_segments(self):
        segments = {}
        for link in self.link_set:
            points = link.get("points", [])
            world_segments = []
            for idx in range(len(points) - 1):
                ax, ay = float(points[idx][0]), float(points[idx][1])
                bx, by = float(points[idx + 1][0]), float(points[idx + 1][1])
                world_segments.append((ax, ay, bx, by))
            segments[link["idx"]] = world_segments
        return segments

    def _get_active_group(self):
        if 0 <= self.active_group_index < len(self.fill_groups):
            return self.fill_groups[self.active_group_index]
        return None

    def _group_color(self, group_index):
        return ACTIVE_GROUP_COLOR if group_index == self.active_group_index else INACTIVE_GROUP_COLOR

    def _all_fill_group_link_ids(self):
        all_ids = set()
        for group in self.fill_groups:
            all_ids.update(group["link_ids"])
        return all_ids

    def _reset_view(self):
        self.view_center_x = 0.5 * (self.bounds["min_x"] + self.bounds["max_x"])
        self.view_center_y = 0.5 * (self.bounds["min_y"] + self.bounds["max_y"])

        world_width = max(self.bounds["max_x"] - self.bounds["min_x"], 1.0)
        world_height = max(self.bounds["max_y"] - self.bounds["min_y"], 1.0)
        fit_scale_x = max(self.canvas_width, 1) / world_width
        fit_scale_y = max(self.canvas_height, 1) / world_height
        self.pixels_per_meter = min(fit_scale_x, fit_scale_y) * 0.9
        self._redraw()

    def _on_canvas_resize(self, event):
        self.canvas_width = max(event.width, 1)
        self.canvas_height = max(event.height, 1)
        self._redraw()

    def _world_to_canvas(self, x, y):
        cx = (float(x) - self.view_center_x) * self.pixels_per_meter + 0.5 * self.canvas_width
        cy = (self.view_center_y - float(y)) * self.pixels_per_meter + 0.5 * self.canvas_height
        return cx, cy

    def _canvas_to_world(self, x, y):
        wx = (x - 0.5 * self.canvas_width) / self.pixels_per_meter + self.view_center_x
        wy = self.view_center_y - (y - 0.5 * self.canvas_height) / self.pixels_per_meter
        return wx, wy

    def _draw_polyline(self, points, color, width):
        if len(points) < 2:
            return

        coords = []
        for point in points:
            px, py = self._world_to_canvas(point[0], point[1])
            coords.extend([px, py])

        self.canvas.create_line(
            *coords,
            fill=color,
            width=width,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )

    def _draw_link(self, link_id, color, width, partial_spec=None):
        points = self.link_by_id[link_id].get("points", [])
        if partial_spec is not None:
            points = trim_polyline_by_ratio(
                points,
                partial_spec.get("start_ratio", 0.0),
                partial_spec.get("end_ratio", 1.0),
            )
        self._draw_polyline(points, color, width)

    def _redraw(self):
        self.canvas.delete("all")

        for link in self.link_set:
            self._draw_link(link["idx"], LINK_COLOR, 1)

        for group_index, group in enumerate(self.fill_groups):
            for link_id in sorted(group["link_ids"]):
                if link_id in self.link_by_id:
                    self._draw_link(link_id, self._group_color(group_index), 4 if group_index == self.active_group_index else 2)

        for link_id in sorted(self.partial_links):
            if link_id in self.link_by_id:
                self._draw_link(link_id, SELECTED_LINK_COLOR, 2)
                self._draw_link(
                    link_id,
                    PARTIAL_LINK_COLOR,
                    5 if link_id == self.active_partial_link_id else 3,
                    partial_spec=self.partial_links[link_id],
                )

        for link_id in sorted(self.selected_link_ids):
            if link_id not in self._all_fill_group_link_ids() and link_id not in self.partial_links:
                self._draw_link(link_id, SELECTED_LINK_COLOR, 3)

        for link_id in sorted(self.center_link_ids):
            if link_id in self.link_by_id:
                self._draw_link(link_id, CENTER_LINK_COLOR, 4)

        if self.hover_link_id and self.hover_link_id not in self.selected_link_ids:
            self._draw_link(self.hover_link_id, HOVER_LINK_COLOR, 3)

    def _find_nearest_link(self, canvas_x, canvas_y, threshold_px=12.0):
        best_link_id = None
        best_distance = threshold_px

        for link_id, segments in self.link_segments_world.items():
            if not segments:
                continue

            for ax_w, ay_w, bx_w, by_w in segments:
                ax, ay = self._world_to_canvas(ax_w, ay_w)
                bx, by = self._world_to_canvas(bx_w, by_w)
                distance = point_to_segment_distance_px(canvas_x, canvas_y, ax, ay, bx, by)
                if distance < best_distance:
                    best_distance = distance
                    best_link_id = link_id

        return best_link_id

    def _toggle_link_selection(self, link_id):
        if not link_id:
            return

        if link_id in self.selected_link_ids:
            self.selected_link_ids.remove(link_id)
            self.center_link_ids.discard(link_id)
            for group in self.fill_groups:
                group["link_ids"].discard(link_id)
            self.partial_links.pop(link_id, None)
            if self.active_partial_link_id == link_id:
                self.active_partial_link_id = None
        else:
            self.selected_link_ids.add(link_id)

        self._refresh_selection_list()
        self._refresh_center_list()
        self._refresh_fill_groups_list()
        self._refresh_partial_links_list()
        self._redraw()

    def _toggle_center_link(self, link_id):
        if not link_id:
            return
        if link_id not in self.selected_link_ids:
            messagebox.showwarning("Link Not Selected", "Add the link to selected links before marking it as a center link.")
            return
        if link_id in self.center_link_ids:
            self.center_link_ids.remove(link_id)
            self.status_var.set(f"Removed center link: {link_id}")
        else:
            self.center_link_ids.add(link_id)
            self.status_var.set(f"Added center link: {link_id}")
        self._refresh_center_list()
        self._redraw()

    def _toggle_link_in_active_group(self, link_id):
        group = self._get_active_group()
        if group is None:
            messagebox.showwarning("No Active Group", "Create or select a fill group first.")
            return

        if link_id not in self.selected_link_ids:
            messagebox.showwarning("Link Not Selected", "Add the link to selected links before assigning it to a fill group.")
            return

        if link_id in group["link_ids"]:
            group["link_ids"].remove(link_id)
            self.status_var.set(f"Removed from {group['name']}: {link_id}")
        else:
            group["link_ids"].add(link_id)
            self.status_var.set(f"Added to {group['name']}: {link_id}")

        self._refresh_fill_groups_list()
        self._redraw()

    def _select_partial_link(self, link_id):
        if link_id not in self.selected_link_ids:
            messagebox.showwarning("Link Not Selected", "Add the link to selected links before editing its partial range.")
            return

        if link_id not in self.partial_links:
            self.partial_links[link_id] = {"start_ratio": 0.0, "end_ratio": 1.0}

        self.active_partial_link_id = link_id
        self._refresh_partial_links_list()
        self._redraw()
        self.status_var.set(f"Editing partial range for {link_id}")

    def _refresh_selection_list(self):
        ordered = sorted(self.selected_link_ids)
        self.selected_listbox.delete(0, tk.END)
        for link_id in ordered:
            self.selected_listbox.insert(tk.END, link_id)
        self.selected_count_var.set(f"{len(ordered)} links")

    def _refresh_center_list(self):
        ordered = sorted(self.center_link_ids)
        self.center_listbox.delete(0, tk.END)
        for link_id in ordered:
            self.center_listbox.insert(tk.END, link_id)
        self.center_count_var.set(f"{len(ordered)} center links")

    def _refresh_fill_groups_list(self):
        self.fill_groups_listbox.delete(0, tk.END)
        for group in self.fill_groups:
            label = f"{group['name']} ({len(group['link_ids'])})"
            self.fill_groups_listbox.insert(tk.END, label)

        if self.fill_groups:
            if self.active_group_index < 0 or self.active_group_index >= len(self.fill_groups):
                self.active_group_index = 0
            self.fill_groups_listbox.selection_clear(0, tk.END)
            self.fill_groups_listbox.selection_set(self.active_group_index)
            self.group_name_var.set(self.fill_groups[self.active_group_index]["name"])
        else:
            self.active_group_index = -1
            self.group_name_var.set("")

        self.group_summary_var.set(f"{len(self.fill_groups)} groups")

    def _refresh_partial_links_list(self):
        ordered_ids = sorted(self.partial_links)
        self.partial_listbox.delete(0, tk.END)
        for link_id in ordered_ids:
            spec = self.partial_links[link_id]
            self.partial_listbox.insert(tk.END, f"{link_id} [{spec['start_ratio']:.2f}, {spec['end_ratio']:.2f}]")

        if self.active_partial_link_id not in self.partial_links:
            self.active_partial_link_id = ordered_ids[0] if ordered_ids else None

        if self.active_partial_link_id in self.partial_links:
            active_index = ordered_ids.index(self.active_partial_link_id)
            self.partial_listbox.selection_clear(0, tk.END)
            self.partial_listbox.selection_set(active_index)
            self._sync_partial_controls()
        else:
            self.partial_link_var.set("No partial link selected")
            self.partial_start_var.set(0.0)
            self.partial_end_var.set(100.0)
            self.partial_start_label.config(text="0.00")
            self.partial_end_label.config(text="1.00")

        self.partial_summary_var.set(f"{len(ordered_ids)} partial links")

    def _sync_partial_controls(self):
        if self.active_partial_link_id not in self.partial_links:
            self.partial_link_var.set("No partial link selected")
            return

        spec = self.partial_links[self.active_partial_link_id]
        self.partial_link_var.set(f"Editing: {self.active_partial_link_id}")
        self.partial_start_var.set(spec["start_ratio"] * 100.0)
        self.partial_end_var.set(spec["end_ratio"] * 100.0)
        self.partial_start_label.config(text=f"{spec['start_ratio']:.2f}")
        self.partial_end_label.config(text=f"{spec['end_ratio']:.2f}")

    def _on_partial_scale_changed(self, _value=None):
        if self.active_partial_link_id not in self.partial_links:
            return

        start_ratio = self.partial_start_var.get() / 100.0
        end_ratio = self.partial_end_var.get() / 100.0
        start_ratio = max(0.0, min(1.0, start_ratio))
        end_ratio = max(0.0, min(1.0, end_ratio))

        if end_ratio < start_ratio:
            if self.focus_get() == self.partial_start_scale:
                end_ratio = start_ratio
                self.partial_end_var.set(end_ratio * 100.0)
            else:
                start_ratio = end_ratio
                self.partial_start_var.set(start_ratio * 100.0)

        self.partial_links[self.active_partial_link_id] = {
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
        }
        self.partial_start_label.config(text=f"{start_ratio:.2f}")
        self.partial_end_label.config(text=f"{end_ratio:.2f}")
        self._refresh_partial_links_list()
        self._redraw()

    def _remove_active_partial_link(self):
        if self.active_partial_link_id is None:
            return
        self.partial_links.pop(self.active_partial_link_id, None)
        self.active_partial_link_id = None
        self._refresh_partial_links_list()
        self._redraw()

    def _reset_active_partial_link(self):
        if self.active_partial_link_id is None:
            return
        self.partial_links[self.active_partial_link_id] = {"start_ratio": 0.0, "end_ratio": 1.0}
        self._refresh_partial_links_list()
        self._redraw()

    def _clear_selection(self):
        self.selected_link_ids.clear()
        self.center_link_ids.clear()
        for group in self.fill_groups:
            group["link_ids"].clear()
        self.partial_links.clear()
        self.active_partial_link_id = None
        self._refresh_selection_list()
        self._refresh_center_list()
        self._refresh_fill_groups_list()
        self._refresh_partial_links_list()
        self._redraw()
        self.status_var.set("Cleared selected links")

    def _create_fill_group(self):
        group_name = f"group_{len(self.fill_groups) + 1:02d}"
        self.fill_groups.append({"name": group_name, "link_ids": set()})
        self.active_group_index = len(self.fill_groups) - 1
        self.current_mode.set("fill")
        self._refresh_fill_groups_list()
        self._update_mode_status()
        self._redraw()

    def _rename_active_group(self):
        group = self._get_active_group()
        if group is None:
            return
        new_name = self.group_name_var.get().strip()
        if not new_name:
            messagebox.showwarning("Empty Name", "Enter a group name first.")
            return
        group["name"] = new_name
        self._refresh_fill_groups_list()
        self.status_var.set(f"Renamed active group to {new_name}")

    def _apply_group_name_from_entry(self):
        self._rename_active_group()

    def _delete_active_group(self):
        group = self._get_active_group()
        if group is None:
            return
        if not messagebox.askyesno("Delete Group", f"Delete fill group '{group['name']}'?"):
            return
        del self.fill_groups[self.active_group_index]
        if self.active_group_index >= len(self.fill_groups):
            self.active_group_index = len(self.fill_groups) - 1
        self._refresh_fill_groups_list()
        self._redraw()
        self.status_var.set("Deleted fill group")

    def _clear_active_group(self):
        group = self._get_active_group()
        if group is None:
            return
        group["link_ids"].clear()
        self._refresh_fill_groups_list()
        self._redraw()
        self.status_var.set(f"Cleared {group['name']}")

    def _on_fill_group_selected(self, _event):
        selection = self.fill_groups_listbox.curselection()
        if not selection:
            return
        self.active_group_index = selection[0]
        group = self._get_active_group()
        if group is not None:
            self.group_name_var.set(group["name"])
        self._redraw()
        self._update_mode_status()

    def _on_partial_link_selected(self, _event):
        selection = self.partial_listbox.curselection()
        if not selection:
            return
        ordered_ids = sorted(self.partial_links)
        if selection[0] < len(ordered_ids):
            self.active_partial_link_id = ordered_ids[selection[0]]
            self._sync_partial_controls()
            self._redraw()

    def _on_mode_changed(self):
        if self.current_mode.get() == "fill" and not self.fill_groups:
            self._create_fill_group()
            return
        self._update_mode_status()
        self._redraw()

    def _update_mode_status(self):
        mode = self.current_mode.get()
        if mode == "select":
            self.status_var.set("Select Links mode: left click toggles nearest racing link")
            return

        if mode == "center":
            self.status_var.set("Center Links mode: left click toggles nearest selected link as a progress backbone link")
            return

        if mode == "fill":
            group = self._get_active_group()
            if group is None:
                self.status_var.set("Edit Fill Groups mode: create or select a fill group first")
            else:
                self.status_var.set(f"Edit Fill Groups mode: left click toggles links in '{group['name']}'")
            return

        self.status_var.set("Partial Links mode: left click a selected link, then adjust start/end range")

    def _on_left_click(self, event):
        link_id = self._find_nearest_link(event.x, event.y)
        if not link_id:
            return

        mode = self.current_mode.get()
        if mode == "fill":
            self._toggle_link_in_active_group(link_id)
        elif mode == "partial":
            self._select_partial_link(link_id)
        elif mode == "center":
            self._toggle_center_link(link_id)
        else:
            self._toggle_link_selection(link_id)
            self.status_var.set(f"Toggled selected link: {link_id}")

    def _on_mouse_move(self, event):
        link_id = self._find_nearest_link(event.x, event.y)
        if link_id != self.hover_link_id:
            self.hover_link_id = link_id
            self._redraw()

    def _on_pan_start(self, event):
        self.pan_last = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self.pan_last is None:
            self.pan_last = (event.x, event.y)
            return

        dx = event.x - self.pan_last[0]
        dy = event.y - self.pan_last[1]
        self.pan_last = (event.x, event.y)

        self.view_center_x -= dx / self.pixels_per_meter
        self.view_center_y += dy / self.pixels_per_meter
        self._redraw()

    def _zoom_at_canvas_point(self, canvas_x, canvas_y, zoom_factor):
        before_x, before_y = self._canvas_to_world(canvas_x, canvas_y)
        self.pixels_per_meter = max(0.05, min(500.0, self.pixels_per_meter * zoom_factor))
        after_x, after_y = self._canvas_to_world(canvas_x, canvas_y)
        self.view_center_x += before_x - after_x
        self.view_center_y += before_y - after_y
        self._redraw()

    def _on_mouse_wheel(self, event):
        zoom_factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        self._zoom_at_canvas_point(event.x, event.y, zoom_factor)

    def _on_mouse_wheel_linux(self, event):
        zoom_factor = 1.15 if event.num == 4 else 1.0 / 1.15
        self._zoom_at_canvas_point(event.x, event.y, zoom_factor)

    def _toggle_link_from_entry(self):
        link_id = self.search_var.get().strip()
        if not link_id:
            return

        if link_id not in self.link_by_id:
            messagebox.showerror("Unknown Link", f"Link id not found: {link_id}")
            return

        mode = self.current_mode.get()
        if mode == "fill":
            self._toggle_link_in_active_group(link_id)
        elif mode == "partial":
            self._select_partial_link(link_id)
        elif mode == "center":
            self._toggle_center_link(link_id)
        else:
            self._toggle_link_selection(link_id)
            self.status_var.set(f"Toggled by id: {link_id}")

    def _remove_selected_from_listbox(self, _event):
        selection = self.selected_listbox.curselection()
        if not selection:
            return
        link_id = self.selected_listbox.get(selection[0])
        self._toggle_link_selection(link_id)

    def _remove_center_from_listbox(self, _event):
        selection = self.center_listbox.curselection()
        if not selection:
            return
        link_id = self.center_listbox.get(selection[0])
        self._toggle_center_link(link_id)

    def _save_selection(self):
        saved_path, sidecar_path = save_selection_data(
            self.output_file,
            self.selected_link_ids,
            self.center_link_ids,
            self.fill_groups,
            self.partial_links,
        )

        if sidecar_path:
            message = (
                f"Saved {len(self.selected_link_ids)} links to:\n{saved_path}\n\n"
                f"Saved center/fill/partial sidecar to:\n{sidecar_path}"
            )
        else:
            message = f"Saved {len(self.selected_link_ids)} links/centers/groups/partials to:\n{saved_path}"

        self.status_var.set(message.replace("\n", " "))
        messagebox.showinfo("Saved", message)

    def _save_selection_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Selection",
            initialdir=os.path.dirname(self.output_file),
            initialfile=os.path.basename(self.output_file),
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        self.output_file = os.path.abspath(path)
        self._save_selection()


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive GUI for selecting MGeo links, center links, fill groups, and partial links.")
    parser.add_argument(
        "--mgeo-dir",
        required=True,
        help="Path to the MGeo folder that contains link_set.json.",
    )
    parser.add_argument(
        "--output-file",
        default=os.path.abspath("./selected_links.json"),
        help="Path to save selected links, center links, fill groups, and partial links (.json recommended, .txt also supported).",
    )
    parser.add_argument(
        "--initial-selection",
        help="Optional existing txt/json file to preload selected links, center links, fill groups, and partial links.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    initial_data = load_selection_data(args.initial_selection) if args.initial_selection else {"selected_link_ids": set(), "center_link_ids": set(), "fill_groups": [], "partial_links": {}}
    app = LinkSelectorApp(
        mgeo_dir=args.mgeo_dir,
        output_file=args.output_file,
        initial_selected_link_ids=initial_data["selected_link_ids"],
        initial_center_link_ids=initial_data["center_link_ids"],
        initial_fill_groups=initial_data["fill_groups"],
        initial_partial_links=initial_data["partial_links"],
    )
    app.mainloop()


if __name__ == "__main__":
    main()
