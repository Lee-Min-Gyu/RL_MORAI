from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from morai_rl.core.types import VehicleState
from morai_rl.maps.reference_path import ReferencePath


class LocalBeVRenderer:
    def __init__(
        self,
        reference_path: ReferencePath,
        route_corridor,
        link_set_path: str | Path = "",
        lane_marking_path: str | Path = "",
        width_px: int = 96,
        height_px: int = 96,
        front_range_m: float = 30.0,
        rear_range_m: float = 10.0,
        left_range_m: float = 15.0,
        right_range_m: float = 15.0,
        include_lane_marking: bool = False,
        static_bev_npz_path: str | Path = "",
        static_bev_metadata_path: str | Path = "",
        corridor_boundary_width_m: float = 0.3,
        centerline_width_m: float = 0.3,
        lane_marking_min_width_m: float = 0.15,
        ego_vehicle_length_m: float = 4.845,
        ego_vehicle_width_m: float = 1.835,
        ego_vehicle_offset_forward_m: float = 0.0,
    ) -> None:
        del route_corridor, link_set_path, lane_marking_path, lane_marking_min_width_m

        self.reference_path = reference_path
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.front_range_m = float(front_range_m)
        self.rear_range_m = float(rear_range_m)
        self.left_range_m = float(left_range_m)
        self.right_range_m = float(right_range_m)
        self.include_lane_marking = bool(include_lane_marking)
        self.corridor_boundary_width_m = float(corridor_boundary_width_m)
        self.centerline_width_m = float(centerline_width_m)
        self.ego_vehicle_length_m = float(ego_vehicle_length_m)
        self.ego_vehicle_width_m = float(ego_vehicle_width_m)
        self.ego_vehicle_offset_forward_m = float(ego_vehicle_offset_forward_m)
        self.channel_names = self._build_channel_names()

        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("local BeV size must be positive")

        npz_path = Path(static_bev_npz_path)
        meta_path = Path(static_bev_metadata_path)
        if not npz_path.is_file():
            raise FileNotFoundError(f"static bev npz not found: {npz_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"static bev metadata not found: {meta_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            self.global_bev = np.asarray(data["bev"], dtype=np.uint8)
            raw_channel_names = data["channel_names"]
            self.global_channel_names = tuple(str(name) for name in raw_channel_names.tolist())

        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.global_resolution_m_per_px = float(metadata["resolution_m_per_px"])
        pixel_transform = metadata["pixel_transform"]
        self.global_min_x = float(pixel_transform["min_x"])
        self.global_max_y = float(pixel_transform["max_y"])
        self.global_height_px = int(self.global_bev.shape[1])
        self.global_width_px = int(self.global_bev.shape[2])
        self.drivable_channel_index = self._require_global_channel("drivable_area")
        self.lane_marking_channel_index = self._find_global_channel("lane_marking")
        self.reference_segments = self._build_reference_segments()

        self.grid_x, self.grid_y = self._build_local_grid()
        self.boundary_half_width_px = max(
            1,
            int(round((self.corridor_boundary_width_m * 0.5) / self.global_resolution_m_per_px)),
        )

    def render(self, state: VehicleState) -> np.ndarray:
        world_x, world_y = self._local_grid_to_world(state)
        drivable = self._sample_global_channel(self.drivable_channel_index, world_x, world_y)

        channels: list[np.ndarray] = []
        channels.append((drivable * 255).astype(np.uint8))
        channels.append((self._compute_boundary_mask(drivable) * 255).astype(np.uint8))

        if self.include_lane_marking:
            if self.lane_marking_channel_index is None:
                lane_marking = np.zeros_like(drivable, dtype=np.uint8)
            else:
                lane_marking = self._sample_global_channel(self.lane_marking_channel_index, world_x, world_y)
            channels.append((lane_marking * 255).astype(np.uint8))

        channels.append((self._render_reference_centerline(state) * 255).astype(np.uint8))
        channels.append((self._build_ego_mask(state) * 255).astype(np.uint8))
        return np.stack(channels, axis=0)

    def _build_channel_names(self) -> tuple[str, ...]:
        names = ["corridor_area", "corridor_boundary"]
        if self.include_lane_marking:
            names.append("lane_marking")
        names.extend(["reference_centerline", "ego_footprint"])
        return tuple(names)

    def _build_local_grid(self) -> tuple[np.ndarray, np.ndarray]:
        x_step = (self.front_range_m + self.rear_range_m) / float(self.height_px)
        y_step = (self.left_range_m + self.right_range_m) / float(self.width_px)
        local_x = np.linspace(
            self.front_range_m - 0.5 * x_step,
            -self.rear_range_m + 0.5 * x_step,
            self.height_px,
            dtype=np.float32,
        )
        local_y = np.linspace(
            self.left_range_m - 0.5 * y_step,
            -self.right_range_m + 0.5 * y_step,
            self.width_px,
            dtype=np.float32,
        )
        return np.meshgrid(local_x, local_y, indexing="ij")

    def _local_grid_to_world(self, state: VehicleState) -> tuple[np.ndarray, np.ndarray]:
        cos_yaw = math.cos(state.yaw_rad)
        sin_yaw = math.sin(state.yaw_rad)
        world_x = state.x + cos_yaw * self.grid_x - sin_yaw * self.grid_y
        world_y = state.y + sin_yaw * self.grid_x + cos_yaw * self.grid_y
        return world_x, world_y

    def _sample_global_channel(
        self,
        channel_index: int,
        world_x: np.ndarray,
        world_y: np.ndarray,
    ) -> np.ndarray:
        col = np.rint((world_x - self.global_min_x) / self.global_resolution_m_per_px).astype(np.int32)
        row = np.rint((self.global_max_y - world_y) / self.global_resolution_m_per_px).astype(np.int32)
        valid = (
            (row >= 0)
            & (row < self.global_height_px)
            & (col >= 0)
            & (col < self.global_width_px)
        )
        sampled = np.zeros((self.height_px, self.width_px), dtype=np.uint8)
        if np.any(valid):
            sampled[valid] = self.global_bev[channel_index, row[valid], col[valid]]
        return sampled

    def _compute_boundary_mask(self, drivable: np.ndarray) -> np.ndarray:
        if not np.any(drivable):
            return np.zeros_like(drivable, dtype=np.uint8)

        boundary = np.zeros_like(drivable, dtype=bool)
        radius = self.boundary_half_width_px
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.roll(np.roll(drivable, dx, axis=0), dy, axis=1)
                if dx > 0:
                    shifted[:dx, :] = 0
                elif dx < 0:
                    shifted[dx:, :] = 0
                if dy > 0:
                    shifted[:, :dy] = 0
                elif dy < 0:
                    shifted[:, dy:] = 0
                boundary |= (drivable > 0) & (shifted == 0)
        return boundary.astype(np.uint8)

    def _build_reference_segments(self) -> np.ndarray:
        segments = []
        for index in range(len(self.reference_path.points) - 1):
            p0 = self.reference_path.points[index]
            p1 = self.reference_path.points[index + 1]
            segments.append((float(p0.x), float(p0.y), float(p1.x), float(p1.y)))
        if not segments:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(segments, dtype=np.float32)

    def _render_reference_centerline(self, state: VehicleState) -> np.ndarray:
        if self.reference_segments.size == 0:
            return np.zeros((self.height_px, self.width_px), dtype=np.uint8)

        ax = self.reference_segments[:, 0] - state.x
        ay = self.reference_segments[:, 1] - state.y
        bx = self.reference_segments[:, 2] - state.x
        by = self.reference_segments[:, 3] - state.y

        cos_yaw = math.cos(state.yaw_rad)
        sin_yaw = math.sin(state.yaw_rad)

        local_ax = cos_yaw * ax + sin_yaw * ay
        local_ay = -sin_yaw * ax + cos_yaw * ay
        local_bx = cos_yaw * bx + sin_yaw * by
        local_by = -sin_yaw * bx + cos_yaw * by

        mask = np.zeros((self.height_px, self.width_px), dtype=bool)
        radius_m = max(0.05, 0.5 * self.centerline_width_m)
        for index in range(local_ax.shape[0]):
            seg_min_x = min(local_ax[index], local_bx[index]) - radius_m
            seg_max_x = max(local_ax[index], local_bx[index]) + radius_m
            seg_min_y = min(local_ay[index], local_by[index]) - radius_m
            seg_max_y = max(local_ay[index], local_by[index]) + radius_m
            if seg_max_x < -self.rear_range_m or seg_min_x > self.front_range_m:
                continue
            if seg_max_y < -self.right_range_m or seg_min_y > self.left_range_m:
                continue
            distance = self._distance_to_segment(
                ax=float(local_ax[index]),
                ay=float(local_ay[index]),
                bx=float(local_bx[index]),
                by=float(local_by[index]),
            )
            mask |= distance <= radius_m
        return mask.astype(np.uint8)

    def _build_ego_mask(self, state: VehicleState) -> np.ndarray:
        length_m = float(state.length_m) if state.length_m and state.length_m > 0.0 else self.ego_vehicle_length_m
        width_m = float(state.width_m) if state.width_m and state.width_m > 0.0 else self.ego_vehicle_width_m
        half_length_m = 0.5 * max(0.1, length_m)
        half_width_m = 0.5 * max(0.1, width_m)
        center_x_m = self.ego_vehicle_offset_forward_m
        mask = (
            (self.grid_x >= center_x_m - half_length_m)
            & (self.grid_x <= center_x_m + half_length_m)
            & (self.grid_y >= -half_width_m)
            & (self.grid_y <= half_width_m)
        )
        return mask.astype(np.uint8)

    def _distance_to_segment(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> np.ndarray:
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return np.sqrt((self.grid_x - ax) ** 2 + (self.grid_y - ay) ** 2)
        t = ((self.grid_x - ax) * dx + (self.grid_y - ay) * dy) / denom
        t = np.clip(t, 0.0, 1.0)
        px = ax + t * dx
        py = ay + t * dy
        return np.sqrt((self.grid_x - px) ** 2 + (self.grid_y - py) ** 2)

    def _require_global_channel(self, name: str) -> int:
        index = self._find_global_channel(name)
        if index is None:
            raise ValueError(f"global static bev channel not found: {name}")
        return index

    def _find_global_channel(self, name: str) -> int | None:
        for index, channel_name in enumerate(self.global_channel_names):
            if channel_name == name:
                return index
        return None
