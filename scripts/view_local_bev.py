from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    pygame = None
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None

from morai_rl.config.runtime import load_config
from morai_rl.io.vehicle_status_udp import VehicleStatusReceiver
from morai_rl.maps.local_bev import LocalBeVRenderer
from morai_rl.maps.reference_path import ReferencePath
from morai_rl.maps.route_corridor import RouteCorridor


CHANNEL_COLORS = {
    "corridor_area": np.array([110, 110, 110], dtype=np.uint8),
    "corridor_boundary": np.array([240, 80, 80], dtype=np.uint8),
    "lane_marking": np.array([250, 220, 90], dtype=np.uint8),
    "reference_centerline": np.array([80, 170, 255], dtype=np.uint8),
    "ego_footprint": np.array([255, 220, 0], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the ego-centered local BeV used by MORAI RL.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument("--scale", type=int, default=6, help="Nearest-neighbor upscale factor for display.")
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Viewer refresh rate. MORAI UDP reception stays asynchronous.",
    )
    return parser.parse_args()


def build_renderer(config_path: str) -> tuple[VehicleStatusReceiver, LocalBeVRenderer]:
    config = load_config(config_path)
    reference_path = ReferencePath.from_csv(config.path.csv_path)
    corridor = None
    if config.route.enabled:
        corridor = RouteCorridor.from_files(
            link_set_path=config.route.link_set_path,
            selection_path=config.route.corridor_selection_path,
            selection_key=config.route.corridor_selection_key,
            margin_m=config.route.corridor_margin_m,
        )

    receiver = VehicleStatusReceiver(
        host=config.udp.host,
        port=config.udp.vehicle_status_port,
    )
    renderer = LocalBeVRenderer(
        reference_path=reference_path,
        route_corridor=corridor,
        link_set_path=config.route.link_set_path,
        lane_marking_path=config.bev.lane_marking_path,
        width_px=config.bev.width_px,
        height_px=config.bev.height_px,
        front_range_m=config.bev.front_range_m,
        rear_range_m=config.bev.rear_range_m,
        left_range_m=config.bev.left_range_m,
        right_range_m=config.bev.right_range_m,
        include_lane_marking=config.bev.include_lane_marking,
        static_bev_npz_path=config.bev.static_bev_npz_path,
        static_bev_metadata_path=config.bev.static_bev_metadata_path,
        corridor_boundary_width_m=config.bev.corridor_boundary_width_m,
        centerline_width_m=config.bev.centerline_width_m,
        lane_marking_min_width_m=config.bev.lane_marking_min_width_m,
        ego_vehicle_length_m=config.bev.ego_vehicle_length_m,
        ego_vehicle_width_m=config.bev.ego_vehicle_width_m,
        ego_vehicle_offset_forward_m=config.bev.ego_vehicle_offset_forward_m,
    )
    return receiver, renderer


def make_overlay(bev: np.ndarray, channel_names: tuple[str, ...]) -> np.ndarray:
    rgb = np.zeros((bev.shape[1], bev.shape[2], 3), dtype=np.uint8)
    for channel_index, channel_name in enumerate(channel_names):
        mask = bev[channel_index] > 0
        if not np.any(mask):
            continue
        color = CHANNEL_COLORS.get(channel_name, np.array([200, 200, 200], dtype=np.uint8))
        rgb[mask] = np.maximum(rgb[mask], color)
    return rgb


def make_single_channel_rgb(bev: np.ndarray, channel_index: int) -> np.ndarray:
    channel = bev[channel_index]
    return np.repeat(channel[:, :, None], 3, axis=2)


def upscale_rgb(rgb: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8))


def rgb_to_surface(rgb: np.ndarray) -> pygame.Surface:
    # pygame.surfarray expects (width, height, channels)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def main() -> None:
    if pygame is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError(
            "pygame is required for the local BeV viewer. Install it with `pip install pygame`."
        ) from _PYGAME_IMPORT_ERROR

    args = parse_args()
    receiver, renderer = build_renderer(args.config)
    receiver.start()

    pygame.init()
    pygame.font.init()

    scale = max(1, int(args.scale))
    window_width = renderer.width_px * scale
    window_height = renderer.height_px * scale + 84
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("MORAI RL Local BeV Viewer")
    font = pygame.font.SysFont("Consolas", 18)
    small_font = pygame.font.SysFont("Consolas", 14)
    clock = pygame.time.Clock()

    view_mode = "overlay"
    selected_channel = 0
    last_state_time = 0.0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        if view_mode == "overlay":
                            view_mode = "channel"
                            selected_channel = 0
                        else:
                            view_mode = "overlay"
                    elif event.key == pygame.K_RIGHT and view_mode == "channel":
                        selected_channel = (selected_channel + 1) % len(renderer.channel_names)
                    elif event.key == pygame.K_LEFT and view_mode == "channel":
                        selected_channel = (selected_channel - 1) % len(renderer.channel_names)

            state = receiver.get_latest()
            screen.fill((18, 18, 18))

            if state is None:
                waiting = font.render("Waiting for MORAI vehicle status UDP...", True, (230, 230, 230))
                hint = small_font.render(
                    f"Listening on {receiver.host}:{receiver.port}",
                    True,
                    (180, 180, 180),
                )
                screen.blit(waiting, (18, 18))
                screen.blit(hint, (18, 48))
            else:
                bev = renderer.render(state)
                if view_mode == "overlay":
                    rgb = make_overlay(bev, renderer.channel_names)
                    title = "overlay"
                else:
                    rgb = make_single_channel_rgb(bev, selected_channel)
                    title = renderer.channel_names[selected_channel]

                surface = rgb_to_surface(upscale_rgb(rgb, scale))
                screen.blit(surface, (0, 0))

                last_state_time = time.monotonic()
                line_1 = (
                    f"view={title}  mode={view_mode}  speed={state.speed_mps:.2f} m/s  "
                    f"x={state.x:.2f} y={state.y:.2f} yaw={state.yaw_deg:.1f}"
                )
                line_2 = (
                    "channels="
                    + ", ".join(
                        f"[{index}] {name}" for index, name in enumerate(renderer.channel_names)
                    )
                )
                line_3 = "TAB overlay/channel  LEFT/RIGHT next channel  ESC quit"
                screen.blit(font.render(line_1, True, (240, 240, 240)), (12, renderer.height_px * scale + 10))
                screen.blit(small_font.render(line_2, True, (190, 190, 190)), (12, renderer.height_px * scale + 38))
                screen.blit(small_font.render(line_3, True, (160, 160, 160)), (12, renderer.height_px * scale + 60))

            if state is None and last_state_time > 0.0:
                stale_text = small_font.render("Last packet timed out.", True, (255, 160, 160))
                screen.blit(stale_text, (18, 72))

            pygame.display.flip()
            clock.tick(max(1, int(args.fps)))
    finally:
        receiver.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
