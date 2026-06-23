from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any

from morai_rl.core.types import ControlCommand, ObjectState, VehicleState


def _import_ros() -> tuple[Any, Any, Any, Any]:
    try:
        import rospy
        from morai_msgs.msg import (
            CtrlCmd,
            EgoVehicleStatus,
            ObjectStatusList,
            SyncModeCmd,
            SyncModeCtrlCmd,
            SyncModeInfo,
            SyncModeScenarioLoad,
            SyncModeSetGear,
            WaitForTick,
        )
        from sensor_msgs.msg import Imu
        from morai_msgs.srv import (
            MoraiSyncModeCmdSrv,
            MoraiSyncModeCtrlCmdSrv,
            MoraiSyncModeSLSrv,
            MoraiSyncModeSetGearSrv,
            MoraiWaitForTickSrv,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ROS transport requires rospy and morai_msgs. "
            "Source the catkin workspace before running this environment."
        ) from exc
    messages = {
        "CtrlCmd": CtrlCmd,
        "EgoVehicleStatus": EgoVehicleStatus,
        "Imu": Imu,
        "ObjectStatusList": ObjectStatusList,
        "SyncModeCmd": SyncModeCmd,
        "SyncModeCtrlCmd": SyncModeCtrlCmd,
        "SyncModeInfo": SyncModeInfo,
        "SyncModeScenarioLoad": SyncModeScenarioLoad,
        "SyncModeSetGear": SyncModeSetGear,
        "WaitForTick": WaitForTick,
    }
    services = {
        "MoraiSyncModeCmdSrv": MoraiSyncModeCmdSrv,
        "MoraiSyncModeCtrlCmdSrv": MoraiSyncModeCtrlCmdSrv,
        "MoraiSyncModeSLSrv": MoraiSyncModeSLSrv,
        "MoraiSyncModeSetGearSrv": MoraiSyncModeSetGearSrv,
        "MoraiWaitForTickSrv": MoraiWaitForTickSrv,
    }
    return rospy, messages, services, EgoVehicleStatus


def ensure_ros_node(node_name: str, anonymous: bool = True) -> None:
    rospy, _, _, _ = _import_ros()
    if rospy.core.is_initialized():
        return
    rospy.init_node(node_name, anonymous=anonymous, disable_signals=True)


def _call_ros_service_with_timeout(
    proxy: Any,
    request: Any,
    timeout_sec: float,
    service_name: str,
) -> Any:
    timeout_sec = float(timeout_sec)
    if timeout_sec <= 0.0:
        return proxy(request)

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def call_service() -> None:
        try:
            result_queue.put(("result", proxy(request)))
        except BaseException as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=call_service, daemon=True)
    thread.start()
    try:
        kind, value = result_queue.get(timeout=timeout_sec)
    except queue.Empty as exc:
        raise RuntimeError(
            f"ROS service call timed out: {service_name} after {timeout_sec:.1f}s"
        ) from exc
    if kind == "error":
        raise RuntimeError(f"ROS service call failed: {service_name}: {value}") from value
    return value


def vehicle_state_from_msg(
    message: Any,
    entity_id: str = "EGO",
    yaw_rate_rps: float = 0.0,
) -> VehicleState:
    header_stamp = getattr(getattr(message, "header", None), "stamp", None)
    timestamp_sec = time.monotonic()
    if header_stamp is not None:
        try:
            stamped_sec = float(header_stamp.to_sec())
            if stamped_sec > 0.0:
                timestamp_sec = stamped_sec
        except Exception:
            pass
    return VehicleState(
        timestamp_sec=timestamp_sec,
        entity_id=str(getattr(message, "unique_id", entity_id)),
        x=float(message.position.x),
        y=float(message.position.y),
        z=float(message.position.z),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=float(message.heading),
        vx=float(message.velocity.x),
        vy=float(message.velocity.y),
        vz=float(message.velocity.z),
        ax=float(message.acceleration.x),
        ay=float(message.acceleration.y),
        az=float(message.acceleration.z),
        wx=0.0,
        wy=0.0,
        wz=float(yaw_rate_rps),
        throttle=float(getattr(message, "accel", 0.0)),
        brake=float(getattr(message, "brake", 0.0)),
        steer_angle=float(getattr(message, "front_steer_angle", 0.0)),
    )


class RosVehicleStatusReceiver:
    def __init__(
        self,
        topic: str,
        entity_id: str = "EGO",
        imu_topic: str = "",
        node_name: str = "morai_rl",
        anonymous: bool = True,
    ) -> None:
        self.topic = topic
        self.entity_id = entity_id
        self.imu_topic = imu_topic.strip()
        self.node_name = node_name
        self.anonymous = anonymous
        self._latest: VehicleState | None = None
        self._version = 0
        self._latest_yaw_rate_rps = 0.0
        self._condition = threading.Condition()
        self._subscriber = None
        self._imu_subscriber = None

    def start(self) -> None:
        if self._subscriber is not None:
            return
        ensure_ros_node(self.node_name, self.anonymous)
        rospy, messages, _, _ = _import_ros()
        self._subscriber = rospy.Subscriber(
            self.topic,
            messages["EgoVehicleStatus"],
            self._callback,
            queue_size=1,
        )
        if self.imu_topic:
            self._imu_subscriber = rospy.Subscriber(
                self.imu_topic,
                messages["Imu"],
                self._imu_callback,
                queue_size=1,
            )

    def stop(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None
        if self._imu_subscriber is not None:
            self._imu_subscriber.unregister()
            self._imu_subscriber = None

    def close(self) -> None:
        self.stop()

    def clear_latest(self) -> None:
        with self._condition:
            self._latest = None
            self._version += 1

    def drain_socket(self) -> None:
        return

    def get_latest(self) -> VehicleState | None:
        with self._condition:
            return self._latest

    def get_version(self) -> int:
        with self._condition:
            return self._version

    def wait_for_state(
        self,
        timeout_sec: float,
        min_timestamp_sec: float | None = None,
        min_version: int | None = None,
    ) -> VehicleState:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                if self._latest is not None and (
                    min_timestamp_sec is None
                    or self._latest.timestamp_sec > min_timestamp_sec
                ) and (
                    min_version is None
                    or self._version > min_version
                ):
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for ROS vehicle status")
                self._condition.wait(remaining)

    def ingest_message(self, message: Any) -> VehicleState:
        with self._condition:
            yaw_rate_rps = self._latest_yaw_rate_rps
        state = vehicle_state_from_msg(message, self.entity_id, yaw_rate_rps)
        with self._condition:
            self._latest = state
            self._version += 1
            self._condition.notify_all()
        return state

    def _callback(self, message: Any) -> None:
        self.ingest_message(message)

    def _imu_callback(self, message: Any) -> None:
        yaw_rate_rps = float(getattr(message.angular_velocity, "z", 0.0))
        with self._condition:
            self._latest_yaw_rate_rps = yaw_rate_rps


class RosObjectStatusReceiver:
    def __init__(
        self,
        topic: str,
        node_name: str = "morai_rl",
        anonymous: bool = True,
    ) -> None:
        self.topic = topic
        self.node_name = node_name
        self.anonymous = anonymous
        self._latest: list[ObjectState] = []
        self._subscriber = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._subscriber is not None:
            return
        ensure_ros_node(self.node_name, self.anonymous)
        rospy, messages, _, _ = _import_ros()
        self._subscriber = rospy.Subscriber(
            self.topic,
            messages["ObjectStatusList"],
            self._callback,
            queue_size=1,
        )

    def stop(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None

    def get_latest(self) -> list[ObjectState]:
        with self._lock:
            return list(self._latest)

    def _callback(self, message: Any) -> None:
        objects = []
        for item in list(message.npc_list) + list(message.pedestrian_list) + list(message.obstacle_list):
            objects.append(
                ObjectState(
                    entity_id=str(item.unique_id),
                    object_type=int(item.type),
                    x=float(item.position.x),
                    y=float(item.position.y),
                    z=float(item.position.z),
                    yaw_deg=float(item.heading),
                    length=float(item.size.x),
                    width=float(item.size.y),
                    height=float(item.size.z),
                    vx=float(item.velocity.x),
                    vy=float(item.velocity.y),
                    vz=float(item.velocity.z),
                )
            )
        with self._lock:
            self._latest = objects


class RosControlClient:
    def __init__(
        self,
        ctrl_topic: str,
        use_sync_mode: bool = True,
        user_id: str = "morai_rl",
        time_step: int = 20,
        sensor_capture: bool = False,
        sync_info_topic: str = "/SyncModeInfo",
        sync_mode_cmd_service: str = "/SyncModeCmd",
        sync_ctrl_cmd_service: str = "/SyncModeCtrlCmd",
        sync_set_gear_service: str = "/SyncModeSetGear",
        wait_for_tick_service: str = "/WaitForTick",
        service_timeout_sec: float = 5.0,
        wait_for_tick_timeout_sec: float = 5.0,
        start_sync_on_start: bool = True,
        stop_sync_on_close: bool = False,
        front_steer_command_scale: float = 1.0,
        front_steer_command_sign: float = 1.0,
        node_name: str = "morai_rl",
        anonymous: bool = True,
    ) -> None:
        self.ctrl_topic = ctrl_topic
        self.use_sync_mode = use_sync_mode
        self.user_id = user_id
        self.time_step = int(time_step)
        self.sensor_capture = bool(sensor_capture)
        self.sync_info_topic = sync_info_topic
        self.sync_mode_cmd_service = sync_mode_cmd_service
        self.sync_ctrl_cmd_service = sync_ctrl_cmd_service
        self.sync_set_gear_service = sync_set_gear_service
        self.wait_for_tick_service = wait_for_tick_service
        self.service_timeout_sec = service_timeout_sec
        self.wait_for_tick_timeout_sec = wait_for_tick_timeout_sec
        self.start_sync_on_start = start_sync_on_start
        self.stop_sync_on_close = stop_sync_on_close
        self.front_steer_command_scale = float(front_steer_command_scale)
        self.front_steer_command_sign = -1.0 if float(front_steer_command_sign) < 0.0 else 1.0
        self.node_name = node_name
        self.anonymous = anonymous
        self.frame = 0
        self.socket = None
        self._publisher = None
        self._sync_info_subscriber = None
        self._sync_mode_proxy = None
        self._sync_ctrl_proxy = None
        self._sync_set_gear_proxy = None
        self._wait_for_tick_proxy = None
        self._vehicle_receiver: RosVehicleStatusReceiver | None = None
        self._started = False
        # Sync frame ids advance one command at a time. The requested simulation
        # delta belongs in SyncModeCmd.time_step, not in the frame id stride.
        self._frame_step = 1
        self._last_gear: int | None = None
        self._latest_sync_frame: int | None = None
        self._latest_sync_master_id = ""
        self._latest_sync_status = False
        self._can_send_tick = False

    def attach_vehicle_receiver(self, receiver: RosVehicleStatusReceiver) -> None:
        self._vehicle_receiver = receiver

    def send(self, command: ControlCommand, set_gear: bool = True) -> None:
        self._ensure_started()
        ros_command = self._to_ctrl_cmd(command.clipped())
        if self.use_sync_mode:
            if set_gear and self._last_gear != int(command.gear):
                self._set_sync_gear(command.gear)
                self.wait_for_tick_only()
            self._send_sync_command(ros_command)
            return
        self._publisher.publish(ros_command)

    def close(self) -> None:
        if self.stop_sync_on_close and self._started and self._sync_mode_proxy is not None:
            self.release_sync_mode()
        if self._sync_info_subscriber is not None:
            try:
                self._sync_info_subscriber.unregister()
            except Exception:
                pass
            self._sync_info_subscriber = None
        self._started = False

    def release_sync_mode(self, retries: int = 3, wait_sec: float = 0.2) -> bool:
        if self._sync_mode_proxy is None:
            return False
        original_user_id = self.user_id
        candidates: list[str] = []
        for candidate in (self.user_id, self._latest_sync_master_id):
            candidate = str(candidate).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            candidates.append(original_user_id)
        for _ in range(max(1, int(retries))):
            for candidate in candidates:
                self.user_id = candidate
                try:
                    self._set_sync_mode(False)
                    if self._wait_for_sync_status(False, timeout_sec=wait_sec):
                        self.user_id = original_user_id
                        return True
                except Exception:
                    pass
            time.sleep(max(0.0, float(wait_sec)))
        self.user_id = original_user_id
        return False

    def _ensure_started(self) -> None:
        if self._started:
            return
        ensure_ros_node(self.node_name, self.anonymous)
        rospy, messages, services, _ = _import_ros()
        self._publisher = rospy.Publisher(self.ctrl_topic, messages["CtrlCmd"], queue_size=1)
        if self.use_sync_mode:
            self._sync_info_subscriber = rospy.Subscriber(
                self.sync_info_topic,
                messages["SyncModeInfo"],
                self._sync_info_callback,
                queue_size=1,
            )
            self._sync_mode_proxy = self._service_proxy(
                self.sync_mode_cmd_service,
                services["MoraiSyncModeCmdSrv"],
            )
            self._sync_ctrl_proxy = self._service_proxy(
                self.sync_ctrl_cmd_service,
                services["MoraiSyncModeCtrlCmdSrv"],
            )
            self._sync_set_gear_proxy = self._service_proxy(
                self.sync_set_gear_service,
                services["MoraiSyncModeSetGearSrv"],
            )
            self._wait_for_tick_proxy = self._service_proxy(
                self.wait_for_tick_service,
                services["MoraiWaitForTickSrv"],
            )
            if self.start_sync_on_start:
                self._set_sync_mode(True)
        self.socket = True
        self._started = True

    def _sync_info_callback(self, message: Any) -> None:
        self._latest_sync_master_id = str(getattr(message, "master_id", ""))
        self._latest_sync_frame = int(message.frame)
        self._latest_sync_status = bool(message.status)
        self._can_send_tick = bool(message.can_send_tick)

    def _sync_frame_from_info(
        self,
        timeout_sec: float = 0.5,
        min_frame: int | None = None,
        step: int | None = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if (
                self._latest_sync_frame is not None
                and self._latest_sync_status
                and (min_frame is None or self._latest_sync_frame >= min_frame)
            ):
                break
            time.sleep(0.01)
        if (
            self._latest_sync_frame is None
            or not self._latest_sync_status
            or (min_frame is not None and self._latest_sync_frame < min_frame)
        ):
            return False
        self.frame = int(self._latest_sync_frame) + (self._frame_step if step is None else int(step))
        return True

    def _wait_for_sync_status(self, enabled: bool, timeout_sec: float = 0.5) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if bool(self._latest_sync_status) == bool(enabled):
                return True
            time.sleep(0.01)
        return bool(self._latest_sync_status) == bool(enabled)

    def _service_proxy(self, service_name: str, service_type: Any) -> Any:
        rospy, _, _, _ = _import_ros()
        rospy.wait_for_service(service_name, timeout=self.service_timeout_sec)
        return rospy.ServiceProxy(service_name, service_type)

    def _call_service(self, proxy: Any, request: Any, service_name: str, timeout_sec: float | None = None) -> Any:
        if timeout_sec is None:
            timeout_sec = self.service_timeout_sec
        return _call_ros_service_with_timeout(proxy, request, timeout_sec, service_name)

    def _set_sync_mode(self, enabled: bool) -> None:
        _, messages, _, _ = _import_ros()
        request = messages["SyncModeCmd"](
            user_id=self.user_id,
            start_sync_mode=bool(enabled),
            time_step=self.time_step,
        )
        response = self._call_service(self._sync_mode_proxy, request, self.sync_mode_cmd_service)
        if not getattr(response.response, "result", False):
            if enabled and self._attach_to_existing_sync_master(timeout_sec=1.0):
                return
            raise RuntimeError(f"failed to set MORAI sync mode to {enabled}")
        response_user_id = str(getattr(response.response, "user_id", self.user_id))
        if response_user_id:
            self.user_id = response_user_id
        response_frame = int(getattr(response.response, "frame", self.frame))
        self.frame = response_frame + 1
        self._sync_frame_from_info(timeout_sec=0.2, min_frame=response_frame, step=1)

    def _attach_to_existing_sync_master(self, timeout_sec: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if (
                self._latest_sync_status
                and self._latest_sync_master_id == self.user_id
                and self._latest_sync_frame is not None
            ):
                self.frame = int(self._latest_sync_frame) + 1
                return True
            time.sleep(0.01)
        return False

    def _send_sync_command(self, ros_command: Any) -> None:
        _, messages, _, _ = _import_ros()
        request = messages["SyncModeCtrlCmd"](
            frame=self.frame,
            command=ros_command,
            sensor_capture=self.sensor_capture,
        )
        response = self._call_service(self._sync_ctrl_proxy, request, self.sync_ctrl_cmd_service)
        if not getattr(response.response, "result", False):
            failed_frame = self.frame
            tried_frames = [failed_frame]
            candidate_frames: list[int] = []
            if self._sync_frame_from_info(timeout_sec=0.5, min_frame=max(0, failed_frame - 1), step=1):
                candidate_frames.append(self.frame)
            candidate_frames.extend(
                [
                    failed_frame - self._frame_step,
                    failed_frame - 1,
                    failed_frame + 1,
                    failed_frame + self._frame_step,
                ]
            )
            for candidate_frame in candidate_frames:
                candidate_frame = int(candidate_frame)
                if candidate_frame < 0 or candidate_frame in tried_frames:
                    continue
                self.frame = candidate_frame
                tried_frames.append(candidate_frame)
                request = messages["SyncModeCtrlCmd"](
                    frame=self.frame,
                    command=ros_command,
                    sensor_capture=self.sensor_capture,
                )
                response = self._call_service(self._sync_ctrl_proxy, request, self.sync_ctrl_cmd_service)
                if getattr(response.response, "result", False):
                    break
            if not getattr(response.response, "result", False):
                raise RuntimeError(
                    "sync control command failed "
                    f"at frame {failed_frame}; tried_frames={tried_frames}"
                )
        tick_request = messages["WaitForTick"](
            user_id=self.user_id,
            frame=self.frame,
        )
        tick_response = self._call_service(
            self._wait_for_tick_proxy,
            tick_request,
            self.wait_for_tick_service,
            self.wait_for_tick_timeout_sec,
        )
        if not getattr(tick_response.response, "tick_status", False):
            failed_frame = self.frame
            if self._sync_frame_from_info(timeout_sec=0.5, min_frame=max(0, failed_frame - 1), step=1):
                tick_request = messages["WaitForTick"](
                    user_id=self.user_id,
                    frame=self.frame,
                )
                tick_response = self._call_service(
                    self._wait_for_tick_proxy,
                    tick_request,
                    self.wait_for_tick_service,
                    self.wait_for_tick_timeout_sec,
                )
            if not getattr(tick_response.response, "tick_status", False):
                raise RuntimeError(
                    f"wait for tick failed at frame {failed_frame}; retry_frame={self.frame}"
                )
        self.frame = int(getattr(tick_response.response, "frame", self.frame)) + self._frame_step
        if self._vehicle_receiver is not None:
            self._vehicle_receiver.ingest_message(tick_response.response.vehicle_status)

    def _set_sync_gear(self, gear: int) -> None:
        _, messages, _, _ = _import_ros()
        request = messages["SyncModeSetGear"](
            gear=int(gear),
            frame=self.frame,
        )
        response = self._call_service(self._sync_set_gear_proxy, request, self.sync_set_gear_service)
        if not getattr(response.response, "result", False):
            raise RuntimeError(f"sync set gear failed: gear={gear} frame={self.frame}")
        self._last_gear = int(gear)

    def wait_for_tick_only(self) -> None:
        self._ensure_started()
        _, messages, _, _ = _import_ros()
        tick_request = messages["WaitForTick"](
            user_id=self.user_id,
            frame=self.frame,
        )
        tick_response = self._call_service(
            self._wait_for_tick_proxy,
            tick_request,
            self.wait_for_tick_service,
            self.wait_for_tick_timeout_sec,
        )
        if not getattr(tick_response.response, "tick_status", False):
            raise RuntimeError(f"wait for tick failed at frame {self.frame}")
        self.frame = int(getattr(tick_response.response, "frame", self.frame)) + self._frame_step
        if self._vehicle_receiver is not None:
            self._vehicle_receiver.ingest_message(tick_response.response.vehicle_status)

    def _to_ctrl_cmd(self, command: ControlCommand) -> Any:
        _, messages, _, _ = _import_ros()
        steering = (
            float(command.steering)
            * self.front_steer_command_sign
            * self.front_steer_command_scale
        )
        steering = max(-1.0, min(1.0, steering))
        return messages["CtrlCmd"](
            longlCmdType=int(command.long_cmd_type),
            accel=float(command.throttle),
            brake=float(command.brake),
            front_steer=steering,
            rear_steer=0.0,
            velocity=float(command.velocity_kph),
            acceleration=float(command.acceleration_mps2),
        )


class RosSyncScenarioLoadClient:
    def __init__(
        self,
        service_name: str,
        user_id: str = "morai_rl",
        file_name: str = "",
        delete_all: bool = True,
        load_network_connection_data: bool = True,
        load_ego_vehicle_data: bool = True,
        load_surrounding_vehicle_data: bool = True,
        load_pedestrian_data: bool = True,
        load_object_data: bool = True,
        set_pause: bool = False,
        service_timeout_sec: float = 5.0,
        control_client: RosControlClient | None = None,
        post_load_ticks: int = 10,
        post_load_gear: int = 1,
    ) -> None:
        self.service_name = service_name
        self.user_id = user_id
        self.file_name = file_name
        self.delete_all = bool(delete_all)
        self.load_network_connection_data = bool(load_network_connection_data)
        self.load_ego_vehicle_data = bool(load_ego_vehicle_data)
        self.load_surrounding_vehicle_data = bool(load_surrounding_vehicle_data)
        self.load_pedestrian_data = bool(load_pedestrian_data)
        self.load_object_data = bool(load_object_data)
        self.set_pause = bool(set_pause)
        self.service_timeout_sec = service_timeout_sec
        self.control_client = control_client
        self.post_load_ticks = max(0, int(post_load_ticks))
        self.post_load_gear = int(post_load_gear)
        self._proxy = None
        self._frame = 0

    def send(self, file_name: str | None = None) -> None:
        rospy, messages, services, _ = _import_ros()
        if self._proxy is None:
            rospy.wait_for_service(self.service_name, timeout=self.service_timeout_sec)
            self._proxy = rospy.ServiceProxy(
                self.service_name,
                services["MoraiSyncModeSLSrv"],
            )
        frame = self._frame
        if self.control_client is not None:
            self.control_client._ensure_started()
            frame = int(self.control_client.frame)
        request = messages["SyncModeScenarioLoad"](
            frame=frame,
            file_name=file_name or self.file_name,
            load_network_connection_data=self.load_network_connection_data,
            delete_all=self.delete_all,
            load_ego_vehicle_data=self.load_ego_vehicle_data,
            load_surrounding_vehicle_data=self.load_surrounding_vehicle_data,
            load_pedestrian_data=self.load_pedestrian_data,
            load_obstacle_data=self.load_object_data,
            set_pause=self.set_pause,
        )
        response = _call_ros_service_with_timeout(
            self._proxy,
            request,
            self.service_timeout_sec,
            self.service_name,
        )
        if not getattr(response.response, "result", False):
            raise RuntimeError(f"sync scenario load failed: {request.file_name}")
        self._frame = frame + 1
        if self.control_client is not None:
            self.control_client._last_gear = None
            self.control_client.frame = frame + 1
            self.control_client._sync_frame_from_info(timeout_sec=1.0, min_frame=frame, step=1)
            for _ in range(self.post_load_ticks):
                self.control_client.wait_for_tick_only()

    def close(self) -> None:
        self._proxy = None
