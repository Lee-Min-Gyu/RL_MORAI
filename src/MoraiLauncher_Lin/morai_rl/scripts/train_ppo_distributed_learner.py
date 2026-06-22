from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import socket
import time

try:
    import torch as th
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    th = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional progress UI
    tqdm = None

from morai_rl.distributed.model import build_distributed_ppo, dump_policy_state, load_distributed_ppo
from morai_rl.distributed.protocol import recv_message, send_message
from morai_rl.distributed.rollout_buffer import fill_rollout_buffer
from morai_rl.core.episode_stats import ScenarioStatsAccumulator

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
    parser.add_argument("--action-dist", choices=["gaussian", "tanh_squashed"], default="gaussian")
    parser.add_argument("--std-init", type=float, default=0.1)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--set-log-std", type=float, default=None)
    parser.add_argument("--set-throttle-brake-mean", type=float, default=None)
    parser.add_argument("--set-throttle-brake-std", type=float, default=None)
    parser.add_argument("--set-steering-std", type=float, default=None)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--sb3-verbose", type=int, default=0)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=6_144)
    parser.add_argument("--disable-penalty-curriculum", action="store_true")
    parser.add_argument("--penalty-curriculum-window", type=int, default=39)
    parser.add_argument("--penalty-curriculum-required", type=int, default=34)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    penalty_curriculum = None
    if not args.disable_penalty_curriculum:
        penalty_curriculum = DistributedPenaltyCurriculum(
            state_path=save_dir / "penalty_curriculum_state.json",
            window_episodes=args.penalty_curriculum_window,
            required_episodes=args.penalty_curriculum_required,
        )

    model = _build_or_load_model(args, save_dir)
    _apply_policy_overrides(model, args)
    completed_timesteps = int(getattr(model, "num_timesteps", 0))
    remaining_timesteps = max(0, int(args.timesteps) - max(0, completed_timesteps))
    print(
        "training_budget "
        f"target={args.timesteps} completed={completed_timesteps} remaining={remaining_timesteps}",
        flush=True,
    )
    model._setup_learn(
        total_timesteps=remaining_timesteps,
        reset_num_timesteps=not bool(args.resume_from),
        tb_log_name=args.run_name,
        progress_bar=False,
    )

    with socket.create_server((args.host, args.port), reuse_port=False) as server:
        server.listen(args.workers)
        print(f"learner_listening host={args.host} port={args.port} workers={args.workers}", flush=True)
        clients = _accept_workers(server, args.workers)
        policy_version = 0
        _broadcast_policy(clients, model, policy_version, penalty_curriculum)

        update_count = 0
        last_checkpoint_step = int(model.num_timesteps)
        progress_bar = _make_progress_bar(args, initial_timesteps=int(model.num_timesteps))
        try:
            while model.num_timesteps < args.timesteps:
                before_timesteps = int(model.num_timesteps)
                started_at = time.monotonic()
                rollouts = _recv_rollouts(clients, expected_version=policy_version)
                _log_rollout_stats(rollouts, policy_version)
                if penalty_curriculum is not None:
                    penalty_curriculum.update_from_rollouts(rollouts)
                fill_rollout_buffer(model, rollouts)
                model._update_current_progress_remaining(model.num_timesteps, args.timesteps)
                model.train()
                update_count += 1
                policy_version += 1
                model.save(str(save_dir / "ppo_model"))
                last_checkpoint_step = _save_checkpoint_if_due(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_freq=args.checkpoint_freq,
                    last_checkpoint_step=last_checkpoint_step,
                )
                elapsed = time.monotonic() - started_at
                _update_progress_bar(progress_bar, int(model.num_timesteps) - before_timesteps)
                print(
                    "learner_update "
                    f"update={update_count} policy_version={policy_version} "
                    f"timesteps={model.num_timesteps} elapsed_sec={elapsed:.2f}",
                    flush=True,
                )
                if model.num_timesteps >= args.timesteps:
                    break
                _broadcast_policy(clients, model, policy_version, penalty_curriculum)
        except (EOFError, OSError, RuntimeError) as exc:
            interrupted_path = save_dir / "ppo_model_interrupted"
            model.save(str(interrupted_path))
            print(
                "learner_interrupted "
                f"timesteps={model.num_timesteps} policy_version={policy_version} "
                f"saved_model={interrupted_path}.zip error={exc}",
                flush=True,
            )
            _broadcast_shutdown(clients)
            raise
        finally:
            if progress_bar is not None:
                progress_bar.close()

        model.save(str(save_dir / "ppo_model_final"))
        _broadcast_shutdown(clients)
        print(f"learner_done saved_model={save_dir / 'ppo_model_final'}", flush=True)


class DistributedPenaltyCurriculum:
    STAGES = [
        {"threshold_m": None, "off_track_penalty": 800.0, "stalled_penalty": 800.0},
        {"threshold_m": 500.0, "off_track_penalty": 1200.0, "stalled_penalty": 1200.0},
        {"threshold_m": 900.0, "off_track_penalty": 1600.0, "stalled_penalty": 1600.0},
        {"threshold_m": 1300.0, "off_track_penalty": 2000.0, "stalled_penalty": 2000.0},
    ]

    def __init__(
        self,
        state_path: str | Path,
        window_episodes: int = 39,
        required_episodes: int = 34,
    ) -> None:
        self.state_path = Path(state_path)
        self.window_episodes = max(1, int(window_episodes))
        self.required_episodes = max(1, min(int(required_episodes), self.window_episodes))
        self.stage_index = 0
        self.recent_progress_m: deque[float] = deque(maxlen=self.window_episodes)
        self._load_state()
        self._print_state(reason="learner_start")

    def update_from_rollouts(self, rollouts: list[dict]) -> None:
        added = 0
        for rollout in rollouts:
            for summary in rollout.get("episode_summaries", []):
                if not isinstance(summary, dict):
                    continue
                try:
                    progress_m = float(summary.get("episode_progress_m", 0.0))
                except (TypeError, ValueError):
                    progress_m = 0.0
                self.recent_progress_m.append(progress_m)
                added += 1
        if added <= 0:
            return
        advanced = False
        while self._maybe_advance_stage():
            advanced = True
            self._print_state(reason="threshold_met")
        if not advanced:
            self._print_state(reason="window_update")
        self._save_state()

    def payload(self) -> dict:
        stage = self.STAGES[self.stage_index]
        return {
            "stage_index": int(self.stage_index),
            "off_track_penalty": float(stage["off_track_penalty"]),
            "stalled_penalty": float(stage["stalled_penalty"]),
            "window_episodes": int(self.window_episodes),
            "required_episodes": int(self.required_episodes),
            "recent_count": int(len(self.recent_progress_m)),
        }

    def _maybe_advance_stage(self) -> bool:
        next_stage_index = self.stage_index + 1
        if next_stage_index >= len(self.STAGES):
            return False
        if len(self.recent_progress_m) < self.window_episodes:
            return False
        threshold_m = self.STAGES[next_stage_index]["threshold_m"]
        if threshold_m is None:
            return False
        passed = sum(1 for progress_m in self.recent_progress_m if progress_m >= float(threshold_m))
        if passed < self.required_episodes:
            return False
        self.stage_index = next_stage_index
        return True

    def _print_state(self, reason: str) -> None:
        stage = self.STAGES[self.stage_index]
        next_threshold = None
        if self.stage_index + 1 < len(self.STAGES):
            next_threshold = self.STAGES[self.stage_index + 1]["threshold_m"]
        print(
            "distributed_penalty_curriculum "
            f"reason={reason} "
            f"stage={self.stage_index} "
            f"off_track_penalty={float(stage['off_track_penalty']):.1f} "
            f"stalled_penalty={float(stage['stalled_penalty']):.1f} "
            f"window={len(self.recent_progress_m)}/{self.window_episodes} "
            f"required={self.required_episodes} "
            f"next_threshold_m={next_threshold}",
            flush=True,
        )

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"distributed_penalty_curriculum_state_load_failed path={self.state_path} error={exc}", flush=True)
            return
        try:
            stage_index = int(data.get("stage_index", 0))
        except (TypeError, ValueError):
            stage_index = 0
        self.stage_index = max(0, min(stage_index, len(self.STAGES) - 1))
        progress_values = data.get("recent_progress_m", [])
        if isinstance(progress_values, list):
            self.recent_progress_m.clear()
            for value in progress_values[-self.window_episodes :]:
                try:
                    self.recent_progress_m.append(float(value))
                except (TypeError, ValueError):
                    continue

    def _save_state(self) -> None:
        data = {
            "stage_index": self.stage_index,
            "window_episodes": self.window_episodes,
            "required_episodes": self.required_episodes,
            "recent_progress_m": list(self.recent_progress_m),
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"distributed_penalty_curriculum_state_save_failed path={self.state_path} error={exc}", flush=True)


def _resolve_resume_path(resume_from: str) -> Path:
    resume_path = Path(resume_from)
    if resume_path.is_file():
        return resume_path
    if resume_path.suffix != ".zip" and resume_path.with_suffix(".zip").is_file():
        return resume_path.with_suffix(".zip")
    raise FileNotFoundError(f"resume model not found: {resume_from}")


def _build_or_load_model(args: argparse.Namespace, save_dir: Path):
    if args.resume_from:
        resume_path = _resolve_resume_path(args.resume_from)
        print(f"resuming_from={resume_path}", flush=True)
        if args.log_std_init is not None or args.std_init is not None:
            print("std-init/log-std-init ignored when resuming from a saved model", flush=True)
        return load_distributed_ppo(
            checkpoint_path=str(resume_path),
            config_path=args.config,
            n_envs=args.workers,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            device=args.device,
            action_dist=args.action_dist,
            verbose=args.sb3_verbose,
            tensorboard_log=str(save_dir / "tb"),
        )

    return build_distributed_ppo(
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
        action_dist=args.action_dist,
        std_init=args.std_init,
        log_std_init=args.log_std_init,
        verbose=args.sb3_verbose,
        tensorboard_log=str(save_dir / "tb"),
    )


def _action_net_row(model, index: int):
    policy = model.policy
    action_net = getattr(policy, "action_net", None)
    if action_net is None or not hasattr(action_net, "weight") or not hasattr(action_net, "bias"):
        raise RuntimeError("policy does not expose an action_net with weight/bias")
    if index < 0 or index >= int(action_net.bias.shape[0]):
        raise IndexError(f"action index {index} out of range for action dim {int(action_net.bias.shape[0])}")
    return action_net, index


def _set_throttle_brake_mean(model, mean: float) -> None:
    if th is None:
        raise ModuleNotFoundError("torch is required") from _TORCH_IMPORT_ERROR
    action_net, index = _action_net_row(model, 0)
    with th.no_grad():
        before_bias = float(action_net.bias[index].detach().cpu().item())
        before_weight_norm = float(th.linalg.vector_norm(action_net.weight[index]).detach().cpu().item())
        action_net.weight[index].zero_()
        action_net.bias[index].fill_(float(mean))
        after_bias = float(action_net.bias[index].detach().cpu().item())
        after_weight_norm = float(th.linalg.vector_norm(action_net.weight[index]).detach().cpu().item())
    print(
        "set_throttle_brake_mean "
        f"before_bias={before_bias:+.3f} before_weight_norm={before_weight_norm:.3f} "
        f"after_bias={after_bias:+.3f} after_weight_norm={after_weight_norm:.3f}",
        flush=True,
    )


def _set_action_std(model, index: int, std: float, label: str) -> None:
    if th is None:
        raise ModuleNotFoundError("torch is required") from _TORCH_IMPORT_ERROR
    if std <= 0.0:
        raise ValueError(f"{label} std must be positive")
    log_std = getattr(model.policy, "log_std", None)
    if log_std is None:
        raise RuntimeError("policy does not expose log_std")
    if index < 0 or index >= int(log_std.shape[-1]):
        raise IndexError(f"action index {index} out of range for log_std shape {tuple(log_std.shape)}")
    with th.no_grad():
        before_log_std = float(log_std[index].detach().cpu().item())
        before_std = math.exp(before_log_std)
        log_std[index].fill_(math.log(float(std)))
        after_log_std = float(log_std[index].detach().cpu().item())
        after_std = math.exp(after_log_std)
    print(
        f"set_{label}_std index={index} "
        f"before_log_std={before_log_std:+.3f} before_std={before_std:.3f} "
        f"after_log_std={after_log_std:+.3f} after_std={after_std:.3f}",
        flush=True,
    )


def _set_log_std(model, log_std_value: float) -> None:
    if th is None:
        raise ModuleNotFoundError("torch is required") from _TORCH_IMPORT_ERROR
    log_std = getattr(model.policy, "log_std", None)
    if log_std is None:
        raise RuntimeError("policy does not expose log_std")
    with th.no_grad():
        before_mean = float(log_std.detach().cpu().mean().item())
        before_std_mean = math.exp(before_mean)
        log_std.fill_(float(log_std_value))
        after_mean = float(log_std.detach().cpu().mean().item())
        after_std_mean = math.exp(after_mean)
    print(
        "set_log_std "
        f"before_mean={before_mean:+.3f} before_std_mean={before_std_mean:.3f} "
        f"after_mean={after_mean:+.3f} after_std_mean={after_std_mean:.3f}",
        flush=True,
    )


def _apply_policy_overrides(model, args: argparse.Namespace) -> None:
    if args.set_log_std is not None:
        _set_log_std(model, float(args.set_log_std))
    if args.set_throttle_brake_mean is not None:
        _set_throttle_brake_mean(model, float(args.set_throttle_brake_mean))
    if args.set_throttle_brake_std is not None:
        _set_action_std(model, 0, float(args.set_throttle_brake_std), "throttle_brake")
    if args.set_steering_std is not None:
        _set_action_std(model, 1, float(args.set_steering_std), "steering")


def _make_progress_bar(args: argparse.Namespace, initial_timesteps: int):
    if not args.progress_bar:
        return None
    if tqdm is None:
        print("progress_bar unavailable: install tqdm to enable it", flush=True)
        return None
    return tqdm(
        total=int(args.timesteps),
        initial=max(0, int(initial_timesteps)),
        unit="step",
        dynamic_ncols=True,
    )


def _update_progress_bar(progress_bar, delta_timesteps: int) -> None:
    if progress_bar is not None and delta_timesteps > 0:
        progress_bar.update(int(delta_timesteps))


def _save_checkpoint_if_due(
    *,
    model,
    checkpoint_dir: Path,
    checkpoint_freq: int,
    last_checkpoint_step: int,
) -> int:
    checkpoint_freq = int(checkpoint_freq)
    current_step = int(model.num_timesteps)
    if checkpoint_freq <= 0:
        return int(last_checkpoint_step)
    if current_step - int(last_checkpoint_step) < checkpoint_freq:
        return int(last_checkpoint_step)
    checkpoint_path = checkpoint_dir / f"ppo_checkpoint_{current_step}_steps"
    model.save(str(checkpoint_path))
    print(f"saved_checkpoint={checkpoint_path}.zip", flush=True)
    return current_step


def _log_rollout_stats(rollouts: list[dict], policy_version: int) -> None:
    episode_summaries = [
        episode
        for rollout in rollouts
        for episode in rollout.get("episode_summaries", [])
        if isinstance(episode, dict)
    ]
    total_reward = sum(float(rollout_reward) for rollout in rollouts for rollout_reward in rollout.get("rewards", []))
    total_steps = sum(int(rollout.get("steps", 0)) for rollout in rollouts)
    mean_step_reward = total_reward / total_steps if total_steps > 0 else 0.0
    print(
        "rollout_stats "
        f"policy_version={policy_version} workers={len(rollouts)} "
        f"steps={total_steps} mean_step_reward={mean_step_reward:+.4f} "
        f"completed_episodes={len(episode_summaries)}",
        flush=True,
    )
    scenario_stats = ScenarioStatsAccumulator()
    for summary in episode_summaries:
        scenario_stats.add(summary)
    scenario_stats.print_summary(label=f"scenario_stats policy_version={policy_version}")
    for rollout in rollouts:
        summaries = [episode for episode in rollout.get("episode_summaries", []) if isinstance(episode, dict)]
        if not summaries:
            continue
        worker_id = rollout.get("worker_id")
        rewards = [float(summary.get("episode_reward", 0.0)) for summary in summaries]
        progresses = [float(summary.get("episode_progress_m", 0.0)) for summary in summaries]
        steps = [int(summary.get("step_count", 0) or 0) for summary in summaries]
        print(
            "worker_episode_stats "
            f"worker_id={worker_id} episodes={len(summaries)} "
            f"reward_mean={sum(rewards) / len(rewards):+.3f} "
            f"progress_mean={sum(progresses) / len(progresses):+.2f} "
            f"steps_mean={sum(steps) / len(steps):.1f} "
            f"last_reason={summaries[-1].get('termination_reason')}",
            flush=True,
        )
        _print_reward_terms(worker_id, summaries)


def _print_reward_terms(worker_id, summaries: list[dict]) -> None:
    term_sums: dict[str, float] = {}
    term_counts: dict[str, int] = {}
    for summary in summaries:
        reward_terms = summary.get("reward_terms")
        if not isinstance(reward_terms, dict):
            continue
        for key, value in reward_terms.items():
            if isinstance(value, (int, float)):
                term_sums[key] = term_sums.get(key, 0.0) + float(value)
                term_counts[key] = term_counts.get(key, 0) + 1
    if not term_sums:
        return
    compact_terms = " ".join(
        f"{key}={term_sums[key] / max(1, term_counts[key]):+.3f}"
        for key in sorted(term_sums)
    )
    print(f"worker_reward_terms worker_id={worker_id} {compact_terms}", flush=True)


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


def _broadcast_policy(
    clients: dict[str, socket.socket],
    model,
    policy_version: int,
    penalty_curriculum: DistributedPenaltyCurriculum | None = None,
) -> None:
    payload, checksum = dump_policy_state(model)
    curriculum_payload = penalty_curriculum.payload() if penalty_curriculum is not None else None
    for worker_id, sock in clients.items():
        message = {
            "type": "policy",
            "policy_version": int(policy_version),
            "policy_state": payload,
            "policy_checksum": checksum,
        }
        if curriculum_payload is not None:
            message["penalty_curriculum"] = curriculum_payload
        send_message(sock, message)
        print(
            f"policy_sent worker_id={worker_id} policy_version={policy_version} checksum={checksum[:12]} "
            f"penalty_stage={curriculum_payload.get('stage_index') if curriculum_payload else '-'}",
            flush=True,
        )


def _broadcast_shutdown(clients: dict[str, socket.socket]) -> None:
    for worker_id, sock in clients.items():
        try:
            send_message(sock, {"type": "shutdown"})
        except OSError as exc:
            print(f"shutdown_send_failed worker_id={worker_id} error={exc}", flush=True)
            continue
        print(f"shutdown_sent worker_id={worker_id}", flush=True)


def _recv_rollouts(clients: dict[str, socket.socket], expected_version: int) -> list[dict]:
    rollouts: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = {
            executor.submit(recv_message, sock): worker_id
            for worker_id, sock in clients.items()
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                message = future.result()
            except (EOFError, OSError) as exc:
                raise RuntimeError(
                    f"worker {worker_id} disconnected while waiting for policy_version={expected_version}"
                ) from exc
            if message.get("type") == "worker_error":
                raise RuntimeError(
                    f"worker {worker_id} reported error at policy_version={message.get('policy_version')}: "
                    f"{message.get('error')}"
                )
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
