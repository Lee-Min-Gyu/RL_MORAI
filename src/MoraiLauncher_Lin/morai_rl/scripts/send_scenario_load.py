from __future__ import annotations

import argparse

from morai_rl.config.runtime import load_config
from morai_rl.io.scenario_load_udp import ScenarioLoadClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Send MORAI Scenario Load UDP packet.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument("--file-name", required=True, help="Scenario file name without .json")
    parser.add_argument("--bind-host", default=None)
    parser.add_argument("--bind-port", type=int, default=None)
    parser.add_argument("--destination-host", default=None)
    parser.add_argument("--destination-port", type=int, default=None)
    parser.add_argument("--delete-all", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-ego", action="store_true")
    parser.add_argument("--skip-npc", action="store_true")
    parser.add_argument("--skip-pedestrian", action="store_true")
    parser.add_argument("--skip-object", action="store_true")
    parser.add_argument("--pause", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    bind_host = config.reset.scenario_load_bind_host if args.bind_host is None else args.bind_host
    bind_port = config.reset.scenario_load_bind_port if args.bind_port is None else args.bind_port
    destination_host = (
        config.reset.scenario_load_destination_host
        if args.destination_host is None
        else args.destination_host
    )
    destination_port = (
        config.reset.scenario_load_destination_port
        if args.destination_port is None
        else args.destination_port
    )
    client = ScenarioLoadClient(
        bind_host=bind_host,
        bind_port=bind_port,
        destination_host=destination_host,
        destination_port=destination_port,
    )

    delete_all = True
    if args.keep_existing:
        delete_all = False
    elif args.delete_all:
        delete_all = True

    try:
        client.send(
            file_name=args.file_name,
            delete_all=delete_all,
            load_network_connection_data=not args.skip_network,
            load_ego_vehicle_data=not args.skip_ego,
            load_surrounding_vehicle_data=not args.skip_npc,
            load_pedestrian_data=not args.skip_pedestrian,
            load_object_data=not args.skip_object,
            set_pause=args.pause,
        )
        print("scenario load packet sent")
        print(
            f"from {bind_host}:{bind_port} "
            f"to {destination_host}:{destination_port}"
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
