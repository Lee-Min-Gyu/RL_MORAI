# Distributed PPO Handoff

이 문서는 MORAI RL 분산 PPO 실험을 서버 쪽 Codex/작업자가 이어받기 위한 인수인계 메모입니다.

## 목표

- MORAI 시뮬레이터는 PC당 1개 인스턴스만 실행 가능하므로 여러 PC에서 rollout을 수집한다.
- 3090 4장 서버에 learner를 두고 synchronous PPO update를 수행한다.
- 각 worker는 동일한 `policy_version`을 받아 1024 step rollout을 수집한다.
- learner는 모든 worker rollout을 받은 뒤 한 번에 GAE/advantage/PPO update를 수행하고 새 policy를 다시 배포한다.

## 현재 추가된 파일

기존 단일 학습 스크립트는 덮어쓰지 않았다.

- `morai_rl/scripts/train_ppo.py`
  - 기존 단일 PPO 학습 코드.
  - 수정하지 않음.
- `morai_rl/scripts/train_ppo_distributed_learner.py`
  - 서버 learner entrypoint.
  - worker 연결을 기다리고, rollout을 모아 PPO update 후 policy를 broadcast한다.
- `morai_rl/scripts/train_ppo_distributed_worker.py`
  - 각 MORAI PC에서 실행할 worker entrypoint.
  - learner로부터 policy를 받고, MORAI env에서 rollout을 수집해 learner로 전송한다.
- `morai_rl/distributed/`
  - socket protocol, space-only env, policy state serialization, rollout buffer fill helper.

## 기본 설계

현재 synchronous barrier 방식이다.

```text
learner publishes policy_version = k
all workers load policy k
each worker collects rollout_steps
learner waits for all workers
learner concatenates rollouts into SB3 rollout buffer
learner computes returns/advantages
learner runs PPO train()
learner saves model
learner broadcasts policy_version = k + 1
```

느린 actor 예외 처리는 아직 넣지 않는다. 일단 모든 worker를 기다리는 방식으로 테스트한다.

## 권장 첫 테스트

처음에는 2 worker, 짧은 rollout으로 smoke test를 권장한다.

서버 learner:

```bash
cd /home/mglee/ros1_ws/src/MoraiLauncher_Lin

CUDA_VISIBLE_DEVICES=0 python3 -m morai_rl.scripts.train_ppo_distributed_learner \
  --host 0.0.0.0 \
  --port 50051 \
  --workers 2 \
  --rollout-steps 128 \
  --timesteps 512 \
  --batch-size 128 \
  --device cuda \
  --run-name dist_smoke_test
```

서버 내부 worker를 같이 띄우는 경우:

```bash
cd /home/mglee/ros1_ws/src/MoraiLauncher_Lin

CUDA_VISIBLE_DEVICES=1 python3 -m morai_rl.scripts.train_ppo_distributed_worker \
  --server-host 127.0.0.1 \
  --server-port 50051 \
  --worker-id worker_3090_server \
  --rollout-steps 128 \
  --device cuda
```

노트북 worker:

```bash
cd /home/mglee/ros1_ws/src/MoraiLauncher_Lin

python3 -m morai_rl.scripts.train_ppo_distributed_worker \
  --server-host <SERVER_LAN_IP> \
  --server-port 50051 \
  --worker-id worker_4070 \
  --rollout-steps 128 \
  --device cuda
```

2 worker smoke test가 통과하면 `--workers 3`, `--rollout-steps 1024`, `--batch-size 512`로 키운다.

## 3 worker 권장 실행 예시

서버 learner:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m morai_rl.scripts.train_ppo_distributed_learner \
  --host 0.0.0.0 \
  --port 50051 \
  --workers 3 \
  --rollout-steps 1024 \
  --batch-size 512 \
  --device cuda \
  --run-name dist_3worker
```

서버 worker:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m morai_rl.scripts.train_ppo_distributed_worker \
  --server-host 127.0.0.1 \
  --server-port 50051 \
  --worker-id worker_3090_server \
  --rollout-steps 1024 \
  --device cuda
```

4070 노트북 worker:

```bash
python3 -m morai_rl.scripts.train_ppo_distributed_worker \
  --server-host <SERVER_LAN_IP> \
  --server-port 50051 \
  --worker-id worker_4070 \
  --rollout-steps 1024 \
  --device cuda
```

3070 노트북 worker:

```bash
python3 -m morai_rl.scripts.train_ppo_distributed_worker \
  --server-host <SERVER_LAN_IP> \
  --server-port 50051 \
  --worker-id worker_3070 \
  --rollout-steps 1024 \
  --device cuda
```

## GPU 분리

서버에서 learner와 worker를 동시에 실행할 경우 `CUDA_VISIBLE_DEVICES`로 GPU를 분리한다.

- learner: `CUDA_VISIBLE_DEVICES=0`
- server-local worker: `CUDA_VISIBLE_DEVICES=1`

주의: `CUDA_VISIBLE_DEVICES=1`로 실행한 프로세스 내부에서는 물리 GPU 1장이 `cuda:0`처럼 보인다. 따라서 스크립트 인자는 `--device cuda`로 두면 된다.

## 서버 IP 확인

서버에서:

```bash
hostname -I
```

여러 IP가 나오면 Docker/VPN/가상 인터페이스 IP가 섞일 수 있다. 노트북과 같은 LAN에서 접근 가능한 IP를 사용한다.

더 정확히 보려면:

```bash
ip route get 8.8.8.8
```

출력의 `src` 뒤 IP가 보통 worker들이 접속할 서버 LAN IP다.

예:

```text
8.8.8.8 via 192.168.0.1 dev enp3s0 src 192.168.0.25
```

이 경우 노트북 worker는 `--server-host 192.168.0.25`를 사용한다.

## 현재 observation/action 전제

현재 기본 config는 `morai_rl/stage1_ros_sync_config.toml`이다.

- observation mode: `hybrid`
- vector profile: `racing_guide`
- BeV: `192 x 192`
- action: 2차원 `[-1, 1]`
  - `[accel_brake, steering]`

worker rollout payload에는 다음 정보가 들어간다.

- `policy_version`
- `obs`
- `actions`
- `rewards`
- `episode_starts`
- `terminateds`
- `truncateds`
- `values`
- `log_probs`
- `last_obs`
- `last_value`
- `last_episode_start`
- compact `infos`

## 현재 한계

현재 분산 코드는 1차 골격이다. 학습 실행은 가능하도록 만들었지만, 기존 단일 학습 코드의 모든 recovery가 이식된 상태는 아니다.

아직 미이식:

- 기존 `train_ppo.py`의 outer runtime recovery loop
- RuntimeError 발생 시 `ppo_model_crash` 저장 후 재시작
- simulator process kill/relaunch
- ROS service ready 대기
- worker crash 후 learner 재접속 수용
- learner resume 옵션

살아있는 부분:

- worker는 기존 `GymMoraiEnv`를 그대로 사용한다.
- env 내부 reset/timeout transition/config 기반 동작은 그대로 탄다.
- learner는 update마다 `ppo_model`을 저장한다.
- learner는 모든 worker의 `policy_version`이 같은지 검사한다.

## 다음에 붙이면 좋은 것

1. worker에 기존 `train_ppo.py`의 runtime recovery 옵션 이식
2. worker crash/reconnect 처리
3. learner resume 옵션 추가
4. learner가 checkpoint 저장 후 worker 실패를 명확히 출력하도록 개선
5. long rollout에서 socket payload 크기/전송 시간 측정

## 확인한 것

로컬에서 다음 검증은 통과했다.

```bash
python3 -m compileall \
  /home/mglee/ros1_ws/src/MoraiLauncher_Lin/morai_rl/distributed \
  /home/mglee/ros1_ws/src/MoraiLauncher_Lin/morai_rl/scripts/train_ppo_distributed_learner.py \
  /home/mglee/ros1_ws/src/MoraiLauncher_Lin/morai_rl/scripts/train_ppo_distributed_worker.py
```

또한 두 entrypoint의 `--help` 실행은 확인했다.

현재 작업 환경에는 `gymnasium` 등 학습 의존성이 없을 수 있으므로 실제 실행 검증은 서버/worker 환경에서 수행해야 한다.
