from __future__ import annotations

import threading
import time
from typing import Any

from morai_rl.core.types import (
    CollisionObjectState,
    CollisionState,
    ControlCommand,
    ObjectState,
    VehicleState,
)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
except ModuleNotFoundError:  # pragma: no cover - exercised only without ROS2 sourced.
    rclpy = None
    MultiThreadedExecutor = None

try:
    from morai_ros2_msgs.msg import (
        CollisionData,
        CtrlCmd,
        EgoVehicleStatus,
        MultiEgoSetting,
        ObjectStatus,
        ObjectStatusList,
        ScenarioLoad,
    )
except ModuleNotFoundError:  # MORAI docs sometimes call the package morai_msgs.
    try:
        from morai_msgs.msg import (
            CollisionData,
            CtrlCmd,
            EgoVehicleStatus,
            MultiEgoSetting,
            ObjectStatus,
            ObjectStatusList,
            ScenarioLoad,
        )
    except ModuleNotFoundError:
        CollisionData = None
        CtrlCmd = None
        EgoVehicleStatus = None
        MultiEgoSetting = None
        ObjectStatus = None
        ObjectStatusList = None
        ScenarioLoad = None


_NODE = None
_EXECUTOR = None
_EXECUTOR_THREAD: threading.Thread | None = None
_NODE_LOCK = threading.Lock()


def _require_ros2() -> None:
    if rclpy is None or MultiThreadedExecutor is None:
        raise RuntimeError("rclpy is not available. Source ROS2 before running morai_rl.")
    if CtrlCmd is None:
        raise RuntimeError(
            "MORAI ROS2 messages are not available. Build/source morai_ros2_msgs first."
        )


def get_ros2_node(node_name: str = "morai_rl"):
    global _NODE, _EXECUTOR, _EXECUTOR_THREAD
    _require_ros2()
    with _NODE_LOCK:
        if not rclpy.ok():
            rclpy.init(args=None)
        if _NODE is None:
            _NODE = rclpy.create_node(node_name)
            _EXECUTOR = MultiThreadedExecutor()
            _EXECUTOR.add_node(_NODE)
            _EXECUTOR_THREAD = threading.Thread(target=_EXECUTOR.spin, daemon=True)
            _EXECUTOR_THREAD.start()
        return _NODE


def _stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _stamp_parts(stamp: Any) -> tuple[int, int]:
    return int(stamp.sec), int(stamp.nanosec)


def _ros_heading_to_yaw_deg(heading: float) -> float:
    return float(heading)


class Ros2ControlClient:
    def __init__(self, topic: str, node_name: str = "morai_rl", qos_depth: int = 10) -> None:
        self.node = get_ros2_node(node_name)
        self.topic = topic
        self.publisher = self.node.create_publisher(CtrlCmd, topic, qos_depth)
        self.socket = object()

    def send(self, command: ControlCommand) -> None:
        command = command.clipped()
        msg = CtrlCmd()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.longl_cmd_type = int(command.long_cmd_type)
        msg.accel = float(command.throttle)
        msg.brake = float(command.brake)
        msg.front_steer = float(command.steering)
        msg.rear_steer = 0.0
        msg.velocity = float(command.velocity_kph)
        msg.acceleration = float(command.acceleration_mps2)
        self.publisher.publish(msg)

    def close(self) -> None:
        self.socket = None


class Ros2VehicleStatusReceiver:
    def __init__(
        self,
        topic: str,
        node_name: str = "morai_rl",
        qos_depth: int = 10,
        entity_id: str = "EGO",
    ) -> None:
        self.node = get_ros2_node(node_name)
        self.topic = topic
        self.qos_depth = qos_depth
        self.entity_id = entity_id
        self.subscription = None
        self._lock = threading.Lock()
        self._latest: VehicleState | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self.subscription is not None:
            return
        self.subscription = self.node.create_subscription(
            EgoVehicleStatus,
            self.topic,
            self._callback,
            self.qos_depth,
        )

    def stop(self) -> None:
        if self.subscription is not None:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

    def clear_latest(self) -> None:
        with self._lock:
            self._latest = None

    def drain_socket(self) -> None:
        self.clear_latest()

    def get_latest(self) -> VehicleState | None:
        with self._lock:
            return self._latest

    def get_debug_snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            return {"last_packet_len": None, "last_payload_len": None, "last_error": self._last_error}

    def wait_for_state(
        self,
        timeout_sec: float,
        min_timestamp_sec: float | None = None,
    ) -> VehicleState:
        sleeper = threading.Event()
        start = time.monotonic()
        while True:
            latest = self.get_latest()
            if latest is not None and (
                min_timestamp_sec is None or latest.timestamp_sec > min_timestamp_sec
            ):
                return latest
            if timeout_sec is not None and time.monotonic() - start >= timeout_sec:
                raise TimeoutError("vehicle state timeout")
            sleeper.wait(0.02)

    def _callback(self, msg: Any) -> None:
        try:
            state = VehicleState(
                timestamp_sec=_stamp_to_sec(msg.header.stamp),
                entity_id=f"{self.entity_id}[{int(msg.unique_id)}]",
                x=float(msg.position.x),
                y=float(msg.position.y),
                z=float(msg.position.z),
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=_ros_heading_to_yaw_deg(msg.heading),
                vx=float(msg.velocity.x),
                vy=float(msg.velocity.y),
                vz=float(msg.velocity.z),
                ax=float(msg.acceleration.x),
                ay=float(msg.acceleration.y),
                az=float(msg.acceleration.z),
                wx=float(msg.angular_velocity.x),
                wy=float(msg.angular_velocity.y),
                wz=float(msg.angular_velocity.z),
                throttle=float(msg.accel),
                brake=float(msg.brake),
                steer_angle=float(
                    getattr(msg, "front_steer_angle", getattr(msg, "front_steer", 0.0))
                ),
            )
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return
        with self._lock:
            self._latest = state
            self._last_error = None


class Ros2ObjectStatusReceiver:
    def __init__(self, topic: str, node_name: str = "morai_rl", qos_depth: int = 10) -> None:
        self.node = get_ros2_node(node_name)
        self.topic = topic
        self.qos_depth = qos_depth
        self.subscription = None
        self._lock = threading.Lock()
        self._latest: list[ObjectState] = []

    def start(self) -> None:
        if self.subscription is not None:
            return
        self.subscription = self.node.create_subscription(
            ObjectStatusList,
            self.topic,
            self._callback,
            self.qos_depth,
        )

    def stop(self) -> None:
        if self.subscription is not None:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

    def get_latest(self) -> list[ObjectState]:
        with self._lock:
            return list(self._latest)

    def _callback(self, msg: Any) -> None:
        objects = []
        for item in list(msg.npc_list) + list(msg.pedestrian_list) + list(msg.obstacle_list):
            objects.append(_object_status_to_state(item))
        with self._lock:
            self._latest = objects


class Ros2CollisionStatusReceiver:
    def __init__(self, topic: str, node_name: str = "morai_rl", qos_depth: int = 10) -> None:
        self.node = get_ros2_node(node_name)
        self.topic = topic
        self.qos_depth = qos_depth
        self.subscription = None
        self._lock = threading.Lock()
        self._latest: CollisionState | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self.subscription is not None:
            return
        self.subscription = self.node.create_subscription(
            CollisionData,
            self.topic,
            self._callback,
            self.qos_depth,
        )

    def stop(self) -> None:
        if self.subscription is not None:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

    def clear_latest(self) -> None:
        with self._lock:
            self._latest = None

    def get_latest(self) -> CollisionState | None:
        with self._lock:
            return self._latest

    def get_debug_snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "last_packet_len": None,
                "last_payload_len": None,
                "last_header_name": "ROS2 CollisionData",
                "last_error": self._last_error,
            }

    def _callback(self, msg: Any) -> None:
        try:
            seconds, nanos = _stamp_parts(msg.header.stamp)
            collisions = [
                CollisionObjectState(
                    object_type=int(item.type),
                    object_id=int(item.unique_id),
                    x=float(item.position.x),
                    y=float(item.position.y),
                    z=float(item.position.z),
                    global_offset_x=float(msg.global_offset_x),
                    global_offset_y=float(msg.global_offset_y),
                    global_offset_z=float(msg.global_offset_z),
                )
                for item in msg.collision_object
            ]
            state = CollisionState(
                timestamp_sec=float(seconds) + float(nanos) * 1e-9,
                seconds=seconds,
                nanos=nanos,
                collisions=collisions,
                header_name="ROS2 CollisionData",
            )
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return
        with self._lock:
            self._latest = state
            self._last_error = None


class Ros2ScenarioLoadClient:
    def __init__(
        self,
        topic: str,
        node_name: str = "morai_rl",
        qos_depth: int = 10,
        *,
        file_name: str = "",
        delete_all: bool = True,
        load_network_connection_data: bool = True,
        load_ego_vehicle_data: bool = True,
        load_surrounding_vehicle_data: bool = True,
        load_pedestrian_data: bool = True,
        load_object_data: bool = True,
        set_pause: bool = False,
    ) -> None:
        self.node = get_ros2_node(node_name)
        self.publisher = self.node.create_publisher(ScenarioLoad, topic, qos_depth)
        self.file_name = file_name
        self.delete_all = delete_all
        self.load_network_connection_data = load_network_connection_data
        self.load_ego_vehicle_data = load_ego_vehicle_data
        self.load_surrounding_vehicle_data = load_surrounding_vehicle_data
        self.load_pedestrian_data = load_pedestrian_data
        self.load_object_data = load_object_data
        self.set_pause = set_pause

    def send(self, file_name: str | None = None, **overrides: Any) -> None:
        resolved_file_name = file_name if file_name is not None else self.file_name
        if not resolved_file_name.strip():
            raise ValueError("scenario file name is empty")
        msg = ScenarioLoad()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.file_name = resolved_file_name
        msg.delete_all = bool(overrides.get("delete_all", self.delete_all))
        msg.load_network_connection_data = bool(
            overrides.get("load_network_connection_data", self.load_network_connection_data)
        )
        msg.load_ego_vehicle_data = bool(
            overrides.get("load_ego_vehicle_data", self.load_ego_vehicle_data)
        )
        msg.load_surrounding_vehicle_data = bool(
            overrides.get("load_surrounding_vehicle_data", self.load_surrounding_vehicle_data)
        )
        msg.load_pedestrian_data = bool(
            overrides.get("load_pedestrian_data", self.load_pedestrian_data)
        )
        msg.load_obstacle_data = bool(overrides.get("load_object_data", self.load_object_data))
        msg.set_pause = bool(overrides.get("set_pause", self.set_pause))
        self.publisher.publish(msg)

    def close(self) -> None:
        pass


class Ros2MultiEgoSettingClient:
    def __init__(
        self,
        topic: str,
        node_name: str = "morai_rl",
        qos_depth: int = 10,
        *,
        ego_index: int = 0,
        camera_index: int = 0,
        gear: int = 4,
        ctrl_mode: int = 2,
        send_repeats: int = 3,
        send_interval_sec: float = 0.05,
    ) -> None:
        self.node = get_ros2_node(node_name)
        self.publisher = self.node.create_publisher(MultiEgoSetting, topic, qos_depth)
        self.ego_index = int(ego_index)
        self.camera_index = int(camera_index)
        self.gear = int(gear)
        self.ctrl_mode = int(ctrl_mode)
        self.send_repeats = max(1, int(send_repeats))
        self.send_interval_sec = max(0.0, float(send_interval_sec))

    def send_state(
        self,
        state: VehicleState,
        *,
        gear: int | None = None,
        ctrl_mode: int | None = None,
    ) -> None:
        msg = MultiEgoSetting()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.number_of_ego_vehicle = 1
        msg.camera_index = int(self.camera_index)
        msg.ego_index = [int(self.ego_index)]
        msg.global_position_x = [float(state.x)]
        msg.global_position_y = [float(state.y)]
        msg.global_position_z = [float(state.z)]
        msg.global_roll = [float(state.roll_deg)]
        msg.global_pitch = [float(state.pitch_deg)]
        msg.global_yaw = [float(state.yaw_deg)]
        msg.velocity = [float(state.speed_mps) * 3.6]
        msg.gear = [self.gear if gear is None else int(gear)]
        msg.ctrl_mode = [self.ctrl_mode if ctrl_mode is None else int(ctrl_mode)]
        for repeat_index in range(self.send_repeats):
            self.publisher.publish(msg)
            if repeat_index + 1 < self.send_repeats and self.send_interval_sec > 0.0:
                time.sleep(self.send_interval_sec)

    def close(self) -> None:
        pass


def _object_status_to_state(item: Any) -> ObjectState:
    return ObjectState(
        entity_id=str(getattr(item, "name", "") or getattr(item, "unique_id", "")),
        object_type=int(item.type),
        x=float(item.position.x),
        y=float(item.position.y),
        z=float(item.position.z),
        yaw_deg=_ros_heading_to_yaw_deg(item.heading),
        length=float(item.size.x),
        width=float(item.size.y),
        height=float(item.size.z),
        vx=float(item.velocity.x),
        vy=float(item.velocity.y),
        vz=float(item.velocity.z),
    )
