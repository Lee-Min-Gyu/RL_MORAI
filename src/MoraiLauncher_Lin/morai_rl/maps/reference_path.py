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
    left_width_m: float | None = None
    right_width_m: float | None = None


class ReferencePath:
    def __init__(self, points: list[PathPoint]) -> None:
        if len(points) < 2:
            raise ValueError("reference path requires at least two points")
        self.points = points
        self.total_length_m = points[-1].cumulative_s_m
        first = points[0]
        last = points[-1]
        self.is_closed_loop = math.hypot(last.x - first.x, last.y - first.y) <= 1.0
        self._max_track_width_m = self._compute_max_track_width_m()

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "ReferencePath":
        path = Path(csv_path)
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if not rows:
            raise ValueError(f"reference path is empty: {path}")

        raw_xy: list[tuple[float, float]] = []
        raw_yaw: list[float | None] = []
        raw_left_width: list[float | None] = []
        raw_right_width: list[float | None] = []
        for row in rows:
            raw_xy.append((float(row["x"]), float(row["y"])))
            yaw_deg = row.get("yaw_deg")
            raw_yaw.append(None if yaw_deg in (None, "") else math.radians(float(yaw_deg)))
            left_width = row.get("left_width_m") or row.get("left_width") or row.get("left")
            right_width = row.get("right_width_m") or row.get("right_width") or row.get("right")
            raw_left_width.append(None if left_width in (None, "") else float(left_width))
            raw_right_width.append(None if right_width in (None, "") else float(right_width))

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
            points.append(
                PathPoint(
                    x=x,
                    y=y,
                    yaw_rad=yaw_rad,
                    cumulative_s_m=cumulative_s,
                    left_width_m=raw_left_width[index],
                    right_width_m=raw_right_width[index],
                )
            )

        return cls(points=points)

    def attach_widths_from_csv(self, csv_path: str | Path) -> None:
        entries = _load_width_entries(csv_path)
        if not entries:
            raise ValueError(f"track width csv is empty: {csv_path}")

        lookup = {
            _xy_key(x, y): (left_width_m, right_width_m)
            for x, y, left_width_m, right_width_m in entries
        }
        missing_count = 0
        for point in self.points:
            widths = lookup.get(_xy_key(point.x, point.y))
            if widths is None:
                missing_count += 1
                continue
            point.left_width_m, point.right_width_m = widths
        if missing_count > 0:
            raise ValueError(
                f"track width csv does not match reference path: "
                f"{missing_count}/{len(self.points)} points missing"
            )
        self._max_track_width_m = self._compute_max_track_width_m()

    def track_width_at(self, index: int) -> float:
        point = self.points[max(0, min(len(self.points) - 1, int(index)))]
        if point.left_width_m is None or point.right_width_m is None:
            return 0.0
        return max(0.0, float(point.left_width_m) + float(point.right_width_m))

    def max_track_width_m(self) -> float:
        return self._max_track_width_m

    def _compute_max_track_width_m(self) -> float:
        max_width_m = 0.0
        for point in self.points:
            if point.left_width_m is None or point.right_width_m is None:
                continue
            max_width_m = max(
                max_width_m,
                float(point.left_width_m) + float(point.right_width_m),
            )
        return max_width_m

    def boundary_margins_at(
        self,
        index: int,
        lateral_error_m: float,
        vehicle_width_m: float,
    ) -> tuple[float, float, float]:
        point = self.points[max(0, min(len(self.points) - 1, int(index)))]
        if point.left_width_m is None or point.right_width_m is None:
            return 0.0, 0.0, 0.0
        half_vehicle_width_m = 0.5 * max(0.0, float(vehicle_width_m))
        left_width_m = float(point.left_width_m)
        right_width_m = float(point.right_width_m)
        lateral_error_m = float(lateral_error_m)
        left_margin_m = left_width_m - lateral_error_m - half_vehicle_width_m
        right_margin_m = right_width_m + lateral_error_m - half_vehicle_width_m
        return left_margin_m, right_margin_m, left_width_m + right_width_m

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


def _load_width_entries(csv_path: str | Path) -> list[tuple[float, float, float, float]]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        first_row = next(csv.reader([sample]))
        has_header = any(not _is_float(value) for value in first_row)
        entries: list[tuple[float, float, float, float]] = []
        if has_header:
            for row in csv.DictReader(handle):
                entries.append(
                    (
                        float(row["x"]),
                        float(row["y"]),
                        float(row.get("left_width_m") or row.get("left_width") or row.get("left")),
                        float(row.get("right_width_m") or row.get("right_width") or row.get("right")),
                    )
                )
        else:
            for row in csv.reader(handle):
                if not row:
                    continue
                if len(row) < 4:
                    raise ValueError(f"track width csv requires at least 4 columns: {path}")
                x, y, left_width_m, right_width_m = map(float, row[:4])
                entries.append((x, y, left_width_m, right_width_m))
    return entries


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _xy_key(x: float, y: float) -> tuple[float, float]:
    return round(float(x), 3), round(float(y), 3)
