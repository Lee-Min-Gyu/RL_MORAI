from __future__ import annotations

import argparse
import time

from morai_rl.config.runtime import load_config
from morai_rl.core.reset_manager import ScenarioResetManager
from morai_rl.io.multi_ego_setting_udp import MultiEgoSettingClient
from morai_rl.io.scenario_load_udp import ScenarioLoadClient
from morai_rl.io.vehicle_status_udp import VehicleStatusReceiver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Scenario Load reset check.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    receiver = VehicleStatusReceiver(config.udp.host, config.udp.vehicle_status_port)
    receiver.start()
    scenario_loader = None
    if config.reset.scenario_load_enabled:
        scenario_loader = ScenarioLoadClient(
            bind_host=config.reset.scenario_load_bind_host,
            bind_port=config.reset.scenario_load_bind_port,
            destination_host=config.reset.scenario_load_destination_host,
            destination_port=config.reset.scenario_load_destination_port,
            file_name=config.reset.scenario_load_file_name,
            delete_all=config.reset.scenario_delete_all,
            load_network_connection_data=config.reset.scenario_load_network_connection_data,
            load_ego_vehicle_data=config.reset.scenario_load_ego_vehicle_data,
            load_surrounding_vehicle_data=config.reset.scenario_load_surrounding_vehicle_data,
            load_pedestrian_data=config.reset.scenario_load_pedestrian_data,
            load_object_data=config.reset.scenario_load_object_data,
            set_pause=config.reset.scenario_set_pause,
        )
    multi_ego_client = None
    if config.reset.multi_ego_setting_enabled:
        multi_ego_client = MultiEgoSettingClient(
            bind_host=config.reset.multi_ego_setting_bind_host,
            bind_port=config.reset.multi_ego_setting_bind_port,
            destination_host=config.reset.multi_ego_setting_destination_host,
            destination_port=config.reset.multi_ego_setting_destination_port,
            ego_index=config.reset.multi_ego_setting_ego_index,
            camera_index=config.reset.multi_ego_setting_camera_index,
            gear=config.reset.multi_ego_setting_gear,
            ctrl_mode=config.reset.multi_ego_setting_ctrl_mode,
            send_repeats=config.reset.multi_ego_setting_send_repeats,
            send_interval_sec=config.reset.multi_ego_setting_send_interval_sec,
        )

    manager = ScenarioResetManager(
        vehicle_receiver=receiver,
        command=config.reset.command,
        scenario_loader=scenario_loader,
        multi_ego_client=multi_ego_client,
        reset_mode=config.reset.reset_mode,
        full_reload_interval=config.reset.full_reload_interval,
        scenario_file_names=config.reset.scenario_load_file_names,
        scenario_selection_mode=config.reset.scenario_selection_mode,
        command_timeout_sec=config.reset.command_timeout_sec,
        min_reset_interval_sec=config.reset.min_reset_interval_sec,
        post_command_wait_sec=config.reset.post_command_wait_sec,
        multi_ego_post_command_wait_sec=config.reset.multi_ego_setting_post_command_wait_sec,
        multi_ego_position_tolerance_m=config.reset.multi_ego_setting_position_tolerance_m,
        multi_ego_yaw_tolerance_deg=config.reset.multi_ego_setting_yaw_tolerance_deg,
        stable_speed_tolerance_mps=config.reset.stable_speed_tolerance_mps,
        stable_position_tolerance_m=config.reset.stable_position_tolerance_m,
        stable_frames_required=config.reset.stable_frames_required,
    )

    try:
        for repeat_index in range(max(1, int(args.repeats))):
            if repeat_index > 0 and args.sleep_sec > 0.0:
                time.sleep(args.sleep_sec)
            outcome = manager.reset(timeout_sec=config.reset.reset_timeout_sec)
            state = outcome.initial_state
            print(f"reset success ({repeat_index + 1}/{max(1, int(args.repeats))})")
            if outcome.scenario_name is not None:
                print(f"scenario={outcome.scenario_name}")
            print(f"reset_strategy={outcome.reset_strategy}")
            print(f"stable_frames={outcome.stable_frames}")
            print(
                f"id={state.entity_id} pos=({state.x:.2f}, {state.y:.2f}) "
                f"yaw={state.yaw_deg:.2f} speed={state.speed_mps:.2f}"
            )
    finally:
        if scenario_loader is not None:
            scenario_loader.close()
        if multi_ego_client is not None:
            multi_ego_client.close()
        receiver.stop()


if __name__ == "__main__":
    main()
