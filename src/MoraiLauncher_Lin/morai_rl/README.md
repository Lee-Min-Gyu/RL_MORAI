# MORAI ROS Sync RL

MORAI Simulator를 ROS Noetic synchronous mode로 제어하며 Stable-Baselines3 PPO를 학습하는 강화학습 환경입니다.
현재 `morai_rl`은 ROS service/topic 기반 경로만 사용합니다.

## 주요 기능

- MORAI synchronous mode 제어: `/SyncModeCtrlCmd`, `/SyncModeWaitForTick`
- 시나리오 리셋: `/SyncModeScenarioLoad`
- 차량 상태, 객체 상태, IMU yaw-rate 구독
- Gymnasium wrapper + Stable-Baselines3 PPO 학습
- ROACH-style `192x192` BeV observation
- racing guide vector observation
- action space: `[throttle_brake, steering]`
- checkpoint 저장, resume, action 통계 출력
- 시뮬레이터 무응답 `RuntimeError` 발생 시 자동 복구

## 디렉터리 구조

```text
morai_rl/
  config/      runtime TOML 설정 로더
  core/        reset manager, step clock, simulator recovery helper
  envs/        MORAI RL env, Gym wrapper, reward, termination, observation
  io/          ROS synchronous transport
  maps/        reference path, route corridor, local BeV utilities
  policies/    ROACH-style feature extractor
  scripts/     training, diagnostics, monitoring scripts
```

관련 ROS launcher는 `src/MSC/msc_ros`에 있습니다.

## 실행 환경

Docker/ROS Noetic 컨테이너 안에서 아래 환경을 먼저 로드합니다.

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /opt/rl_venv/bin/activate
export PYTHONPATH=/root/catkin_ws/src/MoraiLauncher_Lin:$PYTHONPATH
```

컨테이너가 host의 MORAI 프로세스를 종료할 수 있어야 자동 복구가 동작합니다.
Docker 실행 시 `--pid=host` 옵션이 필요합니다.

## MORAI 실행

`msc_ros.launch`로 MORAI Launcher API를 실행합니다.

```bash
roslaunch msc_ros msc_ros.launch
```

실제 실행 노드는 아래 파일에서 관리됩니다.

- `src/MSC/msc_ros/launch/msc_ros.launch`
- `src/MSC/msc_ros/scripts/api.py`
- `src/MSC/msc_ros/scripts/params.txt`

`params.txt`의 `network_file`에는 경로가 아니라 네트워크 설정 파일 이름만 적습니다.
예를 들어 실제 파일이 아래에 있어도:

```text
MoraiLauncher_Lin_Data/SaveFile/Network/26.R1.0/ROS_Network0601
```

`params.txt`에는 이렇게 적습니다.

```text
network_file : ROS_Network0601
```

MORAI Launcher가 내부 SaveFile 경로에서 해당 이름을 찾아 사용합니다.

## 필수 ROS 서비스 확인

학습 전 MORAI/rosbridge가 sync service를 advertise했는지 확인합니다.

```bash
rosservice list | grep -E "SyncMode|Scenario"
```

필수 서비스는 다음과 같습니다.

```text
/SyncModeCmd
/SyncModeCtrlCmd
/SyncModeSetGear
/SyncModeWaitForTick
/SyncModeScenarioLoad
```

`/SyncModeScenarioLoad`가 없으면 MORAI network 설정을 다시 연결하거나 MORAI/rosbridge를 재시작합니다.

## 메인 설정

기본 학습 설정 파일:

```text
/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml
```

중요 설정:

- ROS sync mode enabled
- `time_step = 20`
- `step_hz = 25.0`
- `action_repeat = 2`
- reset mode: `/SyncModeScenarioLoad`
- scenario round-robin: `ROS_RL`, `ROS_RL2`, `ROS_RL3`, `ROS_RL4`
- BeV size: `192x192`
- action mode: `throttle_brake_steering`
- recovery relaunch command: `roslaunch msc_ros msc_ros.launch`

## 진단 명령

리셋 확인:

```bash
python -m morai_rl.scripts.check_env_reset_loop \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --repeats 1
```

짧은 sync I/O 확인:

```bash
python -m morai_rl.scripts.check_ros_sync_io \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --steps 50
```

수동 주행 진행률/BeV 모니터:

```bash
python -m morai_rl.scripts.monitor_manual_progress \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --show-bev
```

## 학습 실행

새 학습:

```bash
python -m morai_rl.scripts.train_ppo \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --timesteps 2000000 \
  --save-dir runs/ppo_morai \
  --run-name ros_sync_racing_accel_steer \
  --checkpoint-freq 5000 \
  --max-restarts 3 \
  --progress-bar \
  --n-steps 1024 \
  --batch-size 128 \
  --action-log-freq 1000
```

기존 checkpoint에서 이어 학습:

```bash
python -m morai_rl.scripts.train_ppo \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --timesteps 2000000 \
  --save-dir runs/ppo_morai \
  --run-name ros_sync_racing_accel_steer_resume \
  --resume-from /root/catkin_ws/runs/ppo_morai/ros_sync_racing_steer2d_roach_1m/checkpoints/ppo_checkpoint_1010641_steps.zip \
  --checkpoint-freq 5000 \
  --max-restarts 3 \
  --progress-bar \
  --n-steps 1024 \
  --batch-size 128 \
  --action-log-freq 1000 \
  --set-throttle-brake-mean 0.3 \
  --set-throttle-brake-std 0.2 \
  --set-steering-std 0.15
```

`policy_kwargs ignored when resuming from a saved model` 메시지는 SB3 checkpoint resume 시 정상적으로 발생할 수 있습니다.

## 자동 복구

시뮬레이터가 무응답이 되어 ROS service call timeout 또는 sync tick timeout이 발생하면 학습 코드는 `RuntimeError`를 복구 대상으로 처리합니다.

복구 흐름:

```text
RuntimeError 발생
-> 현재 모델을 ppo_model_crash.zip으로 저장
-> Gym/MORAI env close
-> Simulator.x86_64 종료
-> 기존 roslaunch msc_ros / api.py 정리
-> roslaunch msc_ros msc_ros.launch 재실행
-> ROS sync services 준비 대기
-> ppo_model_crash.zip에서 이어 학습
```

복구 설정은 `stage1_ros_sync_config.toml`의 `[recovery]` 섹션에서 관리합니다.

```toml
[recovery]
enabled = true
simulator_process_names = ["Simulator.x86_64"]
relaunch_cleanup_cmdline_substrings = [
  "roslaunch msc_ros msc_ros.launch",
  "msc_ros/scripts/api.py",
]
terminate_timeout_sec = 5.0
relaunch_command = "roslaunch msc_ros msc_ros.launch"
relaunch_wait_sec = 60.0
wait_for_services = true
wait_services_timeout_sec = 300.0
wait_services_poll_sec = 2.0
post_services_wait_sec = 15.0
```

정상 복구 시 주요 로그:

```text
runtime_recovery killed_pids=[...]
runtime_recovery cleanup_pids=[...]
runtime_recovery launched_pid=...
runtime_recovery services_ready ...
```

## 복구 테스트

실제 MORAI 무응답을 기다리지 않고 복구 루프만 테스트하려면 `--inject-runtime-error-after-steps`를 사용합니다.
이 옵션은 지정한 absolute total timestep에서 한 번만 `RuntimeError`를 발생시킵니다.

예를 들어 `1010641` step checkpoint에서 재개 후 30 step 뒤에 에러를 내고 싶으면:

```bash
python -m morai_rl.scripts.train_ppo \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --timesteps 1010941 \
  --save-dir runs/ppo_morai \
  --run-name recovery_injected_test \
  --resume-from /root/catkin_ws/runs/ppo_morai/ros_sync_racing_steer2d_roach_1m/checkpoints/ppo_checkpoint_1010641_steps.zip \
  --checkpoint-freq 100 \
  --max-restarts 2 \
  --progress-bar \
  --n-steps 1024 \
  --batch-size 128 \
  --inject-runtime-error-after-steps 1010671
```

일반 학습에서는 `--inject-runtime-error-after-steps` 옵션을 사용하지 않습니다.

## 개발 메모

- ROS sync `send()`는 `/SyncModeCtrlCmd` 전송 후 `/SyncModeWaitForTick`으로 MORAI tick을 진행합니다.
- ROS sync mode에서는 환경이 별도의 wall-clock sleep을 추가하지 않습니다.
- `RuntimeError`만 자동 복구 대상으로 처리합니다.
- 일반 `Exception`은 crash model을 저장한 뒤 다시 raise합니다.
- 복구 중 stale `msc_ros`를 정리하지 않으면 새 `api.py`가 바로 종료될 수 있습니다.
- `--max-restarts` 횟수를 초과하면 학습을 중단합니다.
