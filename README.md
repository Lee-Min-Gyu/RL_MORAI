# MORAI ROS Sync RL Workspace

Last updated: 2026-06-04

MORAI Simulator를 ROS Noetic synchronous mode로 제어하며 PPO 강화학습을 수행하기 위한 catkin workspace입니다. 현재 학습 코드는 UDP 기반 흐름을 사용하지 않고, ROS service/topic 기반 sync control, scenario reset, checkpoint resume, runtime recovery를 중심으로 구성되어 있습니다.

## Repository Overview

- `src/MoraiLauncher_Lin/morai_rl`
  - PPO 학습 환경, Gym wrapper, ROS sync I/O, BeV/vector observation, 자동 복구 로직
- `src/MSC/msc_ros`
  - MORAI Launcher API 실행 패키지
  - `roslaunch msc_ros msc_ros.launch`로 Simulator를 실행/연결
- `src/MSC/ros_drive`
  - MORAI ROS 메시지 및 기존 autonomous driving 관련 코드
- `runs/`
  - 학습 checkpoint와 실행 결과 저장 경로

자세한 학습 실행법과 복구 테스트 명령은 아래 문서를 참고합니다.

```text
src/MoraiLauncher_Lin/morai_rl/README.md
src/MoraiLauncher_Lin/morai_rl/AGENT_NOTES.md
```

## Current Features

- MORAI synchronous mode 제어
- `/SyncModeCtrlCmd`, `/SyncModeWaitForTick` 기반 step 진행
- `/SyncModeScenarioLoad` 기반 scenario reset
- Stable-Baselines3 PPO 학습 및 checkpoint resume
- ROACH-style `192x192` BeV observation
- racing guide vector observation
- action space: `[accel_brake, steering]`
- 시뮬레이터 무응답 `RuntimeError` 발생 시 자동 재실행 및 재학습

## Quick Start

컨테이너 안에서 ROS와 RL 환경을 로드합니다.

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /opt/rl_venv/bin/activate
export PYTHONPATH=/root/catkin_ws/src/MoraiLauncher_Lin:$PYTHONPATH
```

MORAI Launcher API를 실행합니다.

```bash
roslaunch msc_ros msc_ros.launch
```

필수 ROS service를 확인합니다.

```bash
rosservice list | grep -E "SyncMode|Scenario"
```

학습은 `morai_rl` 패키지에서 실행합니다.

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
  --batch-size 128
```

## 2026-06-04 Update Summary

- `morai_rl/README.md`를 GitHub용 사용 문서로 정리했습니다.
- `AGENT_NOTES.md`에 다음 agent를 위한 handoff 내용을 최신화했습니다.
- 학습 중 `RuntimeError` 발생 시 자동 복구하도록 구현했습니다.
- 복구 시 `Simulator.x86_64`를 종료한 뒤 `msc_ros`를 다시 실행합니다.
- stale `roslaunch msc_ros` / `msc_ros/scripts/api.py` 프로세스를 정리하도록 수정했습니다.
- 복구 후 ROS sync service 준비를 기다린 뒤 `ppo_model_crash.zip`에서 학습을 재개합니다.
- injected `RuntimeError` 옵션으로 kill/relaunch/resume 흐름을 검증했습니다.
- startup readiness에서 `can_send_tick` 강제 대기는 제거했습니다.
- `network_file`은 경로가 아니라 MORAI SaveFile 내부 네트워크 파일 이름만 적어야 함을 문서화했습니다.
- 불필요한 `velocity_profile` debug 출력은 제거했습니다.

## Notes

- 자동 복구가 host의 Simulator 프로세스를 종료하려면 Docker `--pid=host`가 필요합니다.
- `src/MSC/msc_ros/scripts/params.txt`에는 계정 정보가 포함될 수 있으므로 GitHub 업로드 전 반드시 확인해야 합니다.
- 일반 학습에서는 테스트용 `--inject-runtime-error-after-steps` 옵션을 사용하지 않습니다.
