from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

from morai_rl.core.types import PathProjection, VehicleState, normalize_angle_rad


@dataclass
class PathPoint:
    x: float
    y: float
    yaw_rad: float
    cumulative_s_m: float


class ReferencePath:
    def __init__(self, points: list[PathPoint]) -> None:
        if len(points) < 2:
            raise ValueError("reference path requires at least two points")
        self.points = points
        self.total_length_m = points[-1].cumulative_s_m
        first = points[0]
        last = points[-1]
        self.is_closed_loop = math.hypot(last.x - first.x, last.y - first.y) <= 1.0

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "ReferencePath":
        path = Path(csv_path)
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if not rows:
            raise ValueError(f"reference path is empty: {path}")

        raw_xy: list[tuple[float, float]] = []
        raw_yaw: list[float | None] = []
        for row in rows:
            raw_xy.append((float(row["x"]), float(row["y"])))
            yaw_deg = row.get("yaw_deg")
            raw_yaw.append(None if yaw_deg in (None, "") else math.radians(float(yaw_deg)))

        cumulative_s = 0.0
        points: list[PathPoint] = []
        for index, (x, y) in enumerate(raw_xy):
            if index > 0:
                prev_x, prev_y = raw_xy[index - 1]
                cumulative_s += math.hypot(x - prev_x, y - prev_y)

            yaw_rad = raw_yaw[index]
            if yaw_rad is None:
                if index < len(raw_xy) - 1:
                    next_x, next_y = raw_xy[index + 1]
                    yaw_rad = math.atan2(next_y - y, next_x - x)
                else:
                    prev_x, prev_y = raw_xy[index - 1]
                    yaw_rad = math.atan2(y - prev_y, x - prev_x)
            points.append(PathPoint(x=x, y=y, yaw_rad=yaw_rad, cumulative_s_m=cumulative_s))

        return cls(points=points)

    def project(
        self,
        state: VehicleState,
        hint_index: int | None = None,
        search_window: int | None = None,
    ) -> PathProjection:
        segment_count = len(self.points) - 1
        if segment_count <= 0:
            raise ValueError("reference path requires at least one segment")

        if hint_index is None or search_window is None:
            start_index = 0
            end_index = segment_count
            candidate_indices = range(start_index, end_index)
        elif self.is_closed_loop:
            hint_index = hint_index % segment_count
            candidate_indices = (
                index % segment_count
                for index in range(hint_index - search_window, hint_index + search_window + 1)
            )
        else:
            start_index = max(0, min(segment_count - 1, hint_index) - search_window)
            end_index = min(segment_count, hint_index + search_window + 1)
            candidate_indices = range(start_index, end_index)

        nearest_index = 0
        nearest_progress_m = 0.0
        nearest_heading_rad = self.points[0].yaw_rad
        nearest_lateral_error = 0.0
        min_dist_sq = float("inf")
        for index in candidate_indices:
            p0 = self.points[index]
            p1 = self.points[index + 1]
            vx = p1.x - p0.x
            vy = p1.y - p0.y
            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq <= 1e-12:
                continue

            wx = state.x - p0.x
            wy = state.y - p0.y
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
            qx = p0.x + t * vx
            qy = p0.y + t * vy
            dx = state.x - qx
            dy = state.y - qy
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_index = index if t < 0.5 else index + 1
                nearest_progress_m = p0.cumulative_s_m + t * (p1.cumulative_s_m - p0.cumulative_s_m)
                nearest_heading_rad = math.atan2(vy, vx)
                nearest_lateral_error = -math.sin(nearest_heading_rad) * dx + math.cos(nearest_heading_rad) * dy

        heading_error = normalize_angle_rad(state.yaw_rad - nearest_heading_rad)
        progress_ratio = 0.0
        if self.total_length_m > 0.0:
            progress_ratio = nearest_progress_m / self.total_length_m

        return PathProjection(
            nearest_index=nearest_index,
            distance_m=math.sqrt(min_dist_sq),
            progress_m=nearest_progress_m,
            progress_ratio=progress_ratio,
            path_heading_rad=nearest_heading_rad,
            heading_error_rad=heading_error,
            lateral_error_m=nearest_lateral_error,
            lookahead_heading_error_5m=self.lookahead_heading_error(state, nearest_index, 5.0),
            lookahead_heading_error_10m=self.lookahead_heading_error(state, nearest_index, 10.0),
        )

    def lookahead_heading_error(
        self, state: VehicleState, start_index: int, lookahead_m: float
    ) -> float:
        target_index = self.lookahead_index(start_index, lookahead_m)
        target_heading = self.points[target_index].yaw_rad
        return normalize_angle_rad(state.yaw_rad - target_heading)

    def lookahead_index(self, start_index: int, lookahead_m: float) -> int:
        start_s = self.points[start_index].cumulative_s_m
        target_s = start_s + lookahead_m
        if self.is_closed_loop and self.total_length_m > 0.0:
            target_s = target_s % self.total_length_m
            if target_s < start_s:
                for index, point in enumerate(self.points):
                    if point.cumulative_s_m >= target_s:
                        return index
        for index in range(start_index, len(self.points)):
            if self.points[index].cumulative_s_m >= target_s:
                return index
        return len(self.points) - 1
