from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import socket
import time

from morai_rl.distributed.model import build_distributed_ppo, dump_policy_state
from morai_rl.distributed.protocol import recv_message, send_message
from morai_rl.distributed.rollout_buffer import fill_rollout_buffer

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous distributed PPO learner for MORAI RL.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--save-dir", default="runs/ppo_morai_distributed")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--features-extractor", choices=["auto", "roach", "default"], default="auto")
    parser.add_argument("--std-init", type=float, default=0.1)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--sb3-verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_distributed_ppo(
        config_path=args.config,
        n_envs=args.workers,
        n_steps=args.rollout_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=args.device,
        policy=args.policy,
        features_extractor=args.features_extractor,
        std_init=args.std_init,
        log_std_init=args.log_std_init,
        verbose=args.sb3_verbose,
        tensorboard_log=str(save_dir / "tb"),
    )
    model._setup_learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=True,
        tb_log_name=args.run_name,
        progress_bar=False,
    )

    with socket.create_server((args.host, args.port), reuse_port=False) as server:
        server.listen(args.workers)
        print(f"learner_listening host={args.host} port={args.port} workers={args.workers}", flush=True)
        clients = _accept_workers(server, args.workers)
        policy_version = 0
        _broadcast_policy(clients, model, policy_version)

        update_count = 0
        while model.num_timesteps < args.timesteps:
            started_at = time.monotonic()
            rollouts = _recv_rollouts(clients, expected_version=policy_version)
            fill_rollout_buffer(model, rollouts)
            model._update_current_progress_remaining(model.num_timesteps, args.timesteps)
            model.train()
            update_count += 1
            policy_version += 1
            model.save(str(save_dir / "ppo_model"))
            elapsed = time.monotonic() - started_at
            print(
                "learner_update "
                f"update={update_count} policy_version={policy_version} "
                f"timesteps={model.num_timesteps} elapsed_sec={elapsed:.2f}",
                flush=True,
            )
            _broadcast_policy(clients, model, policy_version)

        model.save(str(save_dir / "ppo_model_final"))
        print(f"learner_done saved_model={save_dir / 'ppo_model_final'}", flush=True)


def _accept_workers(server: socket.socket, expected_workers: int) -> dict[str, socket.socket]:
    clients: dict[str, socket.socket] = {}
    while len(clients) < expected_workers:
        sock, address = server.accept()
        hello = recv_message(sock)
        if hello.get("type") != "hello":
            sock.close()
            raise RuntimeError(f"expected hello message, got {hello.get('type')!r}")
        worker_id = str(hello["worker_id"])
        if worker_id in clients:
            sock.close()
            raise RuntimeError(f"duplicate worker_id: {worker_id}")
        clients[worker_id] = sock
        print(f"worker_connected worker_id={worker_id} address={address}", flush=True)
    return clients


def _broadcast_policy(clients: dict[str, socket.socket], model, policy_version: int) -> None:
    payload, checksum = dump_policy_state(model)
    for worker_id, sock in clients.items():
        send_message(
            sock,
            {
                "type": "policy",
                "policy_version": int(policy_version),
                "policy_state": payload,
                "policy_checksum": checksum,
            },
        )
        print(
            f"policy_sent worker_id={worker_id} policy_version={policy_version} checksum={checksum[:12]}",
            flush=True,
        )


def _recv_rollouts(clients: dict[str, socket.socket], expected_version: int) -> list[dict]:
    rollouts: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = {
            executor.submit(recv_message, sock): worker_id
            for worker_id, sock in clients.items()
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            message = future.result()
            if message.get("type") != "rollout":
                raise RuntimeError(f"worker {worker_id} sent unexpected message type {message.get('type')!r}")
            if int(message["policy_version"]) != int(expected_version):
                raise RuntimeError(
                    f"worker {worker_id} sent policy_version={message['policy_version']}, "
                    f"expected={expected_version}"
                )
            rollout = dict(message["rollout"])
            rollout["worker_id"] = worker_id
            rollouts.append(rollout)
            print(
                "rollout_received "
                f"worker_id={worker_id} policy_version={expected_version} steps={rollout['steps']}",
                flush=True,
            )
    rollouts.sort(key=lambda rollout: str(rollout["worker_id"]))
    return rollouts


if __name__ == "__main__":
    main()
