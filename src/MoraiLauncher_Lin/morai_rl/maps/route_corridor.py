from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from morai_rl.core.types import VehicleState
from morai_rl.maps.selection_io import load_link_ids


@dataclass
class CorridorProjection:
    nearest_link_id: str | None
    centerline_distance_m: float
    corridor_distance_m: float
    inside: bool

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "nearest_link_id": self.nearest_link_id,
            "centerline_distance_m": self.centerline_distance_m,
            "corridor_distance_m": self.corridor_distance_m,
            "inside": self.inside,
        }


def _point_to_segment_distance(point_x: float, point_y: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(point_x - ax, point_y - ay)
    t = ((point_x - ax) * dx + (point_y - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px = ax + t * dx
    py = ay + t * dy
    return math.hypot(point_x - px, point_y - py)


class RouteCorridor:
    def __init__(self, segments: list[tuple[str, float, float, float, float, float]]) -> None:
        if not segments:
            raise ValueError("route corridor requires at least one segment")
        self.segments = segments

    @classmethod
    def from_files(
        cls,
        link_set_path: str | Path,
        selection_path: str | Path,
        selection_key: str = "selected_link_ids",
        margin_m: float = 0.5,
    ) -> "RouteCorridor":
        selected_ids = set(load_link_ids(selection_path, selection_key=selection_key))
        if not selected_ids:
            raise ValueError("route corridor selection is empty")

        with Path(link_set_path).open("r", encoding="utf-8") as handle:
            raw_links = json.load(handle)

        segments: list[tuple[str, float, float, float, float, float]] = []
        for item in raw_links:
            link_id = item.get("idx")
            if link_id not in selected_ids:
                continue
            points = item.get("points", [])
            if len(points) < 2:
                continue
            width_start = float(item.get("width_start") or 3.5)
            width_end = float(item.get("width_end") or width_start)
            half_width = 0.5 * max(width_start, width_end) + float(margin_m)
            for index in range(len(points) - 1):
                ax = float(points[index][0])
                ay = float(points[index][1])
                bx = float(points[index + 1][0])
                by = float(points[index + 1][1])
                segments.append((str(link_id), ax, ay, bx, by, half_width))

        return cls(segments=segments)

    def project(self, state: VehicleState) -> CorridorProjection:
        best_link_id: str | None = None
        best_centerline_distance = float("inf")
        best_corridor_distance = float("inf")

        for link_id, ax, ay, bx, by, half_width in self.segments:
            centerline_distance = _point_to_segment_distance(state.x, state.y, ax, ay, bx, by)
            corridor_distance = centerline_distance - half_width
            if corridor_distance < best_corridor_distance:
                best_corridor_distance = corridor_distance
                best_centerline_distance = centerline_distance
                best_link_id = link_id

        return CorridorProjection(
            nearest_link_id=best_link_id,
            centerline_distance_m=best_centerline_distance,
            corridor_distance_m=best_corridor_distance,
            inside=best_corridor_distance <= 0.0,
        )
