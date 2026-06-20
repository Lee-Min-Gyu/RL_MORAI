from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

try:
    import rospy
    from morai_msgs.msg import MultiEgoSetting
except ModuleNotFoundError as exc:  # pragma: no cover - ROS runtime guard
    rospy = None
    MultiEgoSetting = None
    _ROS_IMPORT_ERROR = exc
else:
    _ROS_IMPORT_ERROR = None


DEFAULT_CENTERLINE_PATH = (
    Path(__file__).resolve().parents[2] / "output" / "map_centerline_reference.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a MORAI MultiEgosetting message from a centerline CSV pose.",
    )
    parser.add_argument("--csv", default=str(DEFAULT_CENTERLINE_PATH))
    parser.add_argument("--index", type=int, default=0, help="0-based row index in the centerline CSV.")
    parser.add_argument("--topic", default="/ego_setting")
    parser.add_argument("--node-name", default="morai_set_ego_from_centerline")
    parser.add_argument("--ego-index", type=int, default=0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--velocity", type=float, default=0.0)
    parser.add_argument("--gear", type=int, default=4, help="1 Parking, 2 Reverse, 3 Neutral, 4 Drive.")
    parser.add_argument("--ctrl-mode", type=int, default=16, help="1 keyboard mode, 16 automode.")
    parser.add_argument("--x", type=float, default=None, help="Override CSV x.")
    parser.add_argument("--y", type=float, default=None, help="Override CSV y.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Override CSV yaw_deg.")
    parser.add_argument("--publish-count", type=int, default=5)
    parser.add_argument("--publish-rate-hz", type=float, default=10.0)
    parser.add_argument("--latch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if rospy is None or MultiEgoSetting is None:  # pragma: no cover - ROS runtime guard
        raise ModuleNotFoundError(
            "rospy and morai_msgs are required. Source ROS/catkin setup before running."
        ) from _ROS_IMPORT_ERROR

    if args.x is not None and args.y is not None and args.yaw_deg is not None:
        x = float(args.x)
        y = float(args.y)
        yaw_deg = float(args.yaw_deg)
    else:
        pose = _load_pose(Path(args.csv), args.index)
        x = float(args.x) if args.x is not None else pose["x"]
        y = float(args.y) if args.y is not None else pose["y"]
        yaw_deg = float(args.yaw_deg) if args.yaw_deg is not None else pose["yaw_deg"]

    rospy.init_node(args.node_name, anonymous=True)
    publisher = rospy.Publisher(args.topic, MultiEgoSetting, queue_size=1, latch=args.latch)
    message = _build_message(
        ego_index=int(args.ego_index),
        camera_index=int(args.camera_index),
        x=x,
        y=y,
        z=float(args.z),
        roll=float(args.roll),
        pitch=float(args.pitch),
        yaw_deg=yaw_deg,
        velocity=float(args.velocity),
        gear=int(args.gear),
        ctrl_mode=int(args.ctrl_mode),
    )
    time.sleep(0.5)
    count = max(1, int(args.publish_count))
    rate = rospy.Rate(max(0.1, float(args.publish_rate_hz)))
    for publish_index in range(count):
        publisher.publish(message)
        print(
            "ego_setting_published "
            f"topic={args.topic} publish={publish_index + 1}/{count} "
            f"ego_index={args.ego_index} x={x:.3f} y={y:.3f} z={float(args.z):.3f} "
            f"yaw_deg={yaw_deg:.3f} gear={args.gear} ctrl_mode={args.ctrl_mode}",
            flush=True,
        )
        rate.sleep()


def _load_pose(csv_path: Path, index: int) -> dict[str, float]:
    if index < 0:
        raise ValueError("--index must be non-negative")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index == index:
                return {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "yaw_deg": float(row["yaw_deg"]),
                }
    raise IndexError(f"centerline index {index} out of range: {csv_path}")


def _build_message(
    *,
    ego_index: int,
    camera_index: int,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw_deg: float,
    velocity: float,
    gear: int,
    ctrl_mode: int,
):
    message = MultiEgoSetting()
    message.number_of_ego_vehicle = 1
    message.camera_index = int(camera_index)
    message.ego_index = [int(ego_index)]
    message.global_position_x = [float(x)]
    message.global_position_y = [float(y)]
    message.global_position_z = [float(z)]
    message.global_roll = [float(roll)]
    message.global_pitch = [float(pitch)]
    message.global_yaw = [float(yaw_deg)]
    message.velocity = [float(velocity)]
    message.gear = [int(gear)]
    message.ctrl_mode = [int(ctrl_mode)]
    message.steering_angle = [0.0]
    message.vehicle_speed = [float(velocity)]
    message.turn_signal = [0]
    message.brake_light = [False]
    return message


if __name__ == "__main__":
    main()
