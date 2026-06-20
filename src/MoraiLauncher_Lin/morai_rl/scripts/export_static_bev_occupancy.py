from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from morai_rl.config.runtime import load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export static BeV drivable_area as a pure occupancy PNG and ROS-style YAML."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default="output/kcity_2025_occupancy")
    parser.add_argument("--map-name", default="kcity_2025_drivable")
    parser.add_argument("--channel", default="drivable_area")
    parser.add_argument(
        "--occupied-black",
        action="store_true",
        default=True,
        help="Write drivable/free area as white and non-drivable/occupied as black.",
    )
    parser.add_argument("--occupied-thresh", type=float, default=0.65)
    parser.add_argument("--free-thresh", type=float, default=0.196)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bev, channel_names = _load_static_bev(config.bev.static_bev_npz_path)
    metadata = _load_metadata(config.bev.static_bev_metadata_path)

    channel_index = _channel_index(channel_names, args.channel)
    mask = bev[channel_index] > 0
    image = np.where(mask, 255, 0).astype(np.uint8)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.map_name}.png"
    yaml_path = output_dir / f"{args.map_name}.yaml"

    _save_png(image_path, image)
    origin = _ros_map_origin(metadata, width_px=image.shape[1], height_px=image.shape[0])
    yaml_text = _build_yaml(
        image_path=image_path.name,
        resolution=float(metadata["resolution_m_per_px"]),
        origin=origin,
        occupied_thresh=float(args.occupied_thresh),
        free_thresh=float(args.free_thresh),
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    free_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    print(f"saved_occupancy_png={image_path}", flush=True)
    print(f"saved_occupancy_yaml={yaml_path}", flush=True)
    print(
        "occupancy_metadata "
        f"resolution={float(metadata['resolution_m_per_px']):.6f} "
        f"origin=[{origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f}] "
        f"width_px={image.shape[1]} height_px={image.shape[0]} "
        f"free_pixels={free_pixels} occupied_pixels={total_pixels - free_pixels}",
        flush=True,
    )


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


def _channel_index(channel_names: tuple[str, ...], channel: str) -> int:
    try:
        return channel_names.index(channel)
    except ValueError as exc:
        raise ValueError(f"channel {channel!r} not found in static BeV channels {channel_names}") from exc


def _save_png(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("Pillow is required to save occupancy PNG") from exc
    Image.fromarray(image, mode="L").save(path)


def _ros_map_origin(metadata: dict, *, width_px: int, height_px: int) -> tuple[float, float, float]:
    resolution = float(metadata["resolution_m_per_px"])
    transform = metadata["pixel_transform"]
    min_x = float(transform["min_x"])
    max_y = float(transform["max_y"])
    min_y = max_y - float(height_px - 1) * resolution
    return min_x, min_y, 0.0


def _build_yaml(
    *,
    image_path: str,
    resolution: float,
    origin: tuple[float, float, float],
    occupied_thresh: float,
    free_thresh: float,
) -> str:
    return (
        f"image: {image_path}\n"
        f"resolution: {resolution:.10f}\n"
        f"origin: [{origin[0]:.10f}, {origin[1]:.10f}, {origin[2]:.10f}]\n"
        "negate: 0\n"
        f"occupied_thresh: {occupied_thresh:.6f}\n"
        f"free_thresh: {free_thresh:.6f}\n"
    )


if __name__ == "__main__":
    main()
