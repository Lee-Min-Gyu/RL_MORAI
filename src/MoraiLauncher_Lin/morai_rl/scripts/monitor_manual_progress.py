from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    pygame = None

from morai_rl.envs.observation import build_observation
from morai_rl.envs.morai_env import MoraiRLEnv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"

CHANNEL_COLORS = {
    "corridor_area": np.array([90, 90, 90], dtype=np.uint8),
    "corridor_boundary": np.array([230, 70, 70], dtype=np.uint8),
    "lane_marking": np.array([240, 220, 80], dtype=np.uint8),
    "reference_centerline": np.array([70, 170, 255], dtype=np.uint8),
    "ego_footprint": np.array([255, 220, 0], dtype=np.uint8),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print manual-driving progress and optionally show local BeV.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--show-bev", action="store_true")
    parser.add_argument("--bev-scale", type=int, default=4)
    args = parser.parse_args()

    env = MoraiRLEnv.from_toml(args.config)
    viewer = None
    if args.show_bev:
        if pygame is None:
            raise ModuleNotFoundError("pygame is required for --show-bev")
        viewer = BeVViewer(scale=args.bev_scale)

    last_progress = None
    first_progress = None
    hint_index = None
    period = 1.0 / max(0.1, float(args.rate_hz))

    try:
        env._ensure_receivers()
        print("manual progress monitor started", flush=True)
        while True:
            loop_start = time.monotonic()
            state = env.vehicle_receiver.wait_for_state(timeout_sec=env.config.env.state_timeout_sec)
            projection = env.reference_path.project(state, hint_index=hint_index, search_window=500)
            corridor = env.route_corridor.project(state) if env.route_corridor is not None else None
            hint_index = projection.nearest_index
            progress = projection.progress_m
            if first_progress is None:
                first_progress = progress
            delta = 0.0 if last_progress is None else progress - last_progress
            last_progress = progress
            inside = corridor.inside if corridor is not None else projection.distance_m <= env.config.env.off_track_distance_m
            pose_projection_off_track = not inside
            observation = build_observation(
                state=state,
                projection=projection,
                corridor_projection=corridor,
                previous_action=env.previous_action,
                target_speed_mps=env.config.env.target_speed_mps,
                episode_progress_m=progress - first_progress,
                progress_delta_m=delta,
                observation_mode=env.config.observation.mode,
                bev_renderer=env.local_bev_renderer,
                vector_profile=env.config.observation.vector_profile,
                guide_dropout_prob=env.config.observation.guide_dropout_prob,
                lookahead_distances_m=env.config.observation.lookahead_distances_m,
                reference_path=env.reference_path,
                ego_vehicle_width_m=env.config.bev.ego_vehicle_width_m,
            )
            bev_contact = env._compute_bev_contact_metrics(observation)
            bev_available = bool(bev_contact["available"])
            footprint_off_track = bev_available and int(bev_contact["outside_pixels"]) > 0
            boundary_overlap_off_track = (
                bev_available and int(bev_contact["boundary_overlap_pixels"]) > 0
            )
            projection_off_track = False if bev_available else pose_projection_off_track
            off_track = footprint_off_track if bev_available else pose_projection_off_track
            off_reasons = []
            if footprint_off_track:
                off_reasons.append("footprint")
            elif projection_off_track:
                off_reasons.append("projection_fallback")
            off_reason = ",".join(off_reasons) if off_reasons else "-"
            left_boundary_margin = float(observation.named.get("left_boundary_margin_m", 0.0))
            right_boundary_margin = float(observation.named.get("right_boundary_margin_m", 0.0))
            track_width = float(observation.named.get("track_width_m", 0.0))
            print(
                "progress "
                f"path={progress:8.2f}m "
                f"delta={delta:+6.3f}m "
                f"episode={progress - first_progress:8.2f}m "
                f"idx={projection.nearest_index:05d} "
                f"speed={state.speed_mps:5.2f}m/s "
                f"yaw={state.yaw_deg:7.2f}deg "
                f"lat={projection.lateral_error_m:+6.2f}m "
                f"head={projection.heading_error_rad:+6.3f}rad "
                f"left_margin={left_boundary_margin:+5.2f}m "
                f"right_margin={right_boundary_margin:+5.2f}m "
                f"track_width={track_width:5.2f}m "
                f"off_track={'Y' if off_track else 'N'} "
                f"off_reason={off_reason} "
                f"boundary_touch={'Y' if boundary_overlap_off_track else 'N'} "
                f"bev_out={int(bev_contact['outside_pixels'])}px "
                f"bev_boundary={int(bev_contact['boundary_overlap_pixels'])}px "
                f"pos=({state.x:.2f},{state.y:.2f})",
                flush=True,
            )
            if viewer is not None and env.local_bev_renderer is not None:
                bev = observation.bev
                if not viewer.draw(bev, env.local_bev_renderer.channel_names):
                    viewer = None
            sleep_sec = period - (time.monotonic() - loop_start)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


class BeVViewer:
    def __init__(self, scale: int = 4) -> None:
        self.scale = max(1, int(scale))
        self.screen = None
        pygame.init()

    def draw(self, bev: np.ndarray, channel_names: tuple[str, ...]) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        height, width = int(bev.shape[1]), int(bev.shape[2])
        if self.screen is None:
            self.screen = pygame.display.set_mode((width * self.scale, height * self.scale))
            pygame.display.set_caption("MORAI Manual Progress BeV")
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for channel_index, channel_name in enumerate(channel_names):
            mask = bev[channel_index] > 0
            if not np.any(mask):
                continue
            color = CHANNEL_COLORS.get(channel_name, np.array([200, 200, 200], dtype=np.uint8))
            rgb[mask] = np.maximum(rgb[mask], color)
        upscaled = np.kron(rgb, np.ones((self.scale, self.scale, 1), dtype=np.uint8))
        surface = pygame.surfarray.make_surface(np.transpose(upscaled, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        return True

    def close(self) -> None:
        pygame.quit()


if __name__ == "__main__":
    main()
