from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from morai_rl.config.runtime import load_config
from morai_rl.maps.reference_path import ReferencePath

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a clean global BeV map with world-coordinate grid labels.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default="runs/analysis/global_bev_grid.png")
    parser.add_argument("--grid-step-m", type=float, default=25.0)
    parser.add_argument("--minor-grid-step-m", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--no-reference", dest="show_reference", action="store_false", default=True)
    parser.add_argument("--title", default="Global BeV Coordinate Grid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("matplotlib is required for plot_global_bev_grid.py") from exc

    config = load_config(args.config)
    bev, channel_names = _load_static_bev(config.bev.static_bev_npz_path)
    metadata = _load_metadata(config.bev.static_bev_metadata_path)
    extent = _world_extent(metadata, width_px=bev.shape[2], height_px=bev.shape[1])

    rgb = _make_rgb(bev, channel_names)
    aspect = (extent["max_y"] - extent["min_y"]) / max(1e-6, extent["max_x"] - extent["min_x"])
    fig_height = max(4.0, float(args.fig_width) * aspect)
    fig, ax = plt.subplots(figsize=(float(args.fig_width), fig_height), dpi=int(args.dpi))
    ax.imshow(
        rgb,
        extent=[extent["min_x"], extent["max_x"], extent["min_y"], extent["max_y"]],
        origin="upper",
    )

    if args.show_reference:
        reference_path = ReferencePath.from_csv(config.path.csv_path)
        ax.plot(
            [point.x for point in reference_path.points],
            [point.y for point in reference_path.points],
            color="#3da5ff",
            linewidth=0.8,
            alpha=0.95,
            label="reference centerline",
        )

    _configure_grid(
        ax,
        min_x=extent["min_x"],
        max_x=extent["max_x"],
        min_y=extent["min_y"],
        max_y=extent["max_y"],
        grid_step_m=float(args.grid_step_m),
        minor_grid_step_m=float(args.minor_grid_step_m),
    )
    ax.set_title(args.title)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_aspect("equal", adjustable="box")
    if args.show_reference:
        ax.legend(loc="upper right")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"saved_global_bev_grid={output}", flush=True)


def _load_static_bev(path: str | Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as data:
        bev = np.asarray(data["bev"], dtype=np.uint8)
        channel_names = tuple(str(name) for name in data["channel_names"].tolist())
    if bev.ndim != 3:
        raise ValueError(f"expected static BeV shape (C,H,W), got {bev.shape}")
    return bev, channel_names


def _load_metadata(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _world_extent(metadata: dict, *, width_px: int, height_px: int) -> dict[str, float]:
    resolution = float(metadata["resolution_m_per_px"])
    transform = metadata["pixel_transform"]
    min_x = float(transform["min_x"])
    max_y = float(transform["max_y"])
    return {
        "min_x": min_x,
        "max_x": min_x + float(width_px - 1) * resolution,
        "max_y": max_y,
        "min_y": max_y - float(height_px - 1) * resolution,
    }


def _make_rgb(bev: np.ndarray, channel_names: tuple[str, ...]) -> np.ndarray:
    height, width = bev.shape[1], bev.shape[2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    colors = {
        "drivable_area": np.array([68, 68, 68], dtype=np.uint8),
        "lane_marking": np.array([230, 210, 90], dtype=np.uint8),
        "corridor_area": np.array([80, 80, 80], dtype=np.uint8),
    }
    for index, name in enumerate(channel_names):
        color = colors.get(name, np.array([120, 120, 120], dtype=np.uint8))
        mask = bev[index] > 0
        rgb[mask] = np.maximum(rgb[mask], color)
    return rgb


def _configure_grid(
    ax,
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    grid_step_m: float,
    minor_grid_step_m: float,
) -> None:
    major = max(1.0, float(grid_step_m))
    minor = max(1.0, float(minor_grid_step_m))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xticks(_ticks(min_x, max_x, major))
    ax.set_yticks(_ticks(min_y, max_y, major))
    if minor < major:
        ax.set_xticks(_ticks(min_x, max_x, minor), minor=True)
        ax.set_yticks(_ticks(min_y, max_y, minor), minor=True)
    ax.grid(which="major", color="white", alpha=0.45, linewidth=0.8)
    ax.grid(which="minor", color="white", alpha=0.16, linewidth=0.4)


def _ticks(start: float, end: float, step: float) -> np.ndarray:
    first = np.ceil(start / step) * step
    last = np.floor(end / step) * step
    if last < first:
        return np.asarray([], dtype=np.float32)
    return np.arange(first, last + 0.5 * step, step)


if __name__ == "__main__":
    main()
