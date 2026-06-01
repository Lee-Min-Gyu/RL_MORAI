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
        if hint_index is None or search_window is None:
            start_index = 0
            end_index = len(self.points)
        else:
            start_index = max(0, hint_index - search_window)
            end_index = min(len(self.points), hint_index + search_window + 1)

        nearest_index = start_index
        min_dist_sq = float("inf")
        for index in range(start_index, end_index):
            point = self.points[index]
            dx = state.x - point.x
            dy = state.y - point.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_index = index

        point = self.points[nearest_index]
        dx = state.x - point.x
        dy = state.y - point.y
        lateral_error = -math.sin(point.yaw_rad) * dx + math.cos(point.yaw_rad) * dy
        heading_error = normalize_angle_rad(state.yaw_rad - point.yaw_rad)
        progress_ratio = 0.0
        if self.total_length_m > 0.0:
            progress_ratio = point.cumulative_s_m / self.total_length_m

        return PathProjection(
            nearest_index=nearest_index,
            distance_m=math.sqrt(min_dist_sq),
            progress_m=point.cumulative_s_m,
            progress_ratio=progress_ratio,
            path_heading_rad=point.yaw_rad,
            heading_error_rad=heading_error,
            lateral_error_m=lateral_error,
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
        for index in range(start_index, len(self.points)):
            if self.points[index].cumulative_s_m >= target_s:
                return index
        return len(self.points) - 1
