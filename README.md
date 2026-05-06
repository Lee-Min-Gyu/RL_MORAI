# MORAI RL

MORAI Simulator UDP I/O를 사용해 강화학습 실험을 돌리기 위한 RL 환경입니다.  
현재 구현은 차량 상태 수신, 제어 명령 송신, 기준 경로 기반 보상/종료 판정, BeV 관측, PPO 학습, 그리고 `MultiEgoSetting` 기반 soft reset까지 포함합니다.

## Current Setup

- MORAI UDP 기반 ego control / vehicle status 수발신
- Gym-like RL environment
- 기준 경로 투영 기반 reward / termination
- vector / BeV / hybrid observation
- Stable-Baselines3 PPO 학습 스크립트
- reset 진단 스크립트
- `MultiEgoSetting` 기반 ego pose reset

## Why MultiEgoSetting Reset

기존에는 episode가 끝날 때마다 `Scenario Load`로 전체 scenario를 다시 불러오는 방식이었습니다.  
Linux 환경에서 MORAI/Unity/Mono 조합에서 이 full reload가 반복될 때 시뮬레이터가 `SIGABRT`로 종료되는 문제가 있었고, 현재는 이를 피하기 위해 reset 시 scenario 전체를 다시 로드하지 않고 ego 차량만 지정 위치로 되돌리는 구조를 사용합니다.

현재 `stage1_rl8_config.toml` 기준 reset 흐름은 다음과 같습니다.

1. episode 종료
2. `MultiEgoSetting` UDP 전송
3. ego 차량을 지정 pose로 이동
4. `Parking` 상태로 4초 대기
5. 다음 episode 시작
6. 학습 step이 시작되면 control command에서 다시 `Drive` 제어가 들어감

## Repository Layout

- `config/`
  runtime config dataclass / TOML loader
- `core/`
  reset manager, sync manager, common types
- `envs/`
  MORAI RL env, gym wrapper, reward, termination, observation
- `io/`
  UDP sender / receiver implementations
- `maps/`
  reference path, route corridor, BeV utilities
- `baselines/`
  simple lane follower baseline
- `scripts/`
  UDP 점검, reset 점검, PPO 학습, rule-based 주행 스크립트

## Requirements

기본적으로 아래 Python 패키지가 필요합니다.

```bash
pip install numpy gymnasium stable-baselines3 torch
```

선택적으로 BeV viewer를 보려면:

```bash
pip install pygame
```

## Main Configs

- `example_config.toml`
  기본 예제 설정
- `safe_config.toml`
  보다 보수적인 reset/load 검사용 설정
- `stage1_rl5_config.toml`
  RL5용 설정
- `stage1_rl8_config.toml`
  현재 주로 사용하는 RL8 설정

현재 `stage1_rl8_config.toml`은 다음 전제를 둡니다.

- scenario는 사용자가 이미 MORAI에서 직접 세팅해둠
- 학습 코드가 시작된 뒤에는 `Scenario Load`를 사용하지 않음
- reset은 `MultiEgoSetting`으로만 수행
- reset target pose:
  - `x = 140.71`
  - `y = 1406.79`
  - `z = -0.53`
  - `roll = 0.354`
  - `pitch = 0.319`
  - `yaw = 91.494`
- reset gear:
  - `Parking(1)`

## Quick Start

가상환경을 활성화합니다.

```bash
source .venv/bin/activate
```

### 1. UDP 입출력 확인

```bash
python -m morai_rl.scripts.check_udp_io \
  --config morai_rl/stage1_rl8_config.toml
```

직접 제어 명령을 보내보려면:

```bash
python -m morai_rl.scripts.check_udp_io \
  --config morai_rl/stage1_rl8_config.toml \
  --send-command \
  --throttle 0.2 \
  --brake 0.0 \
  --steering 0.0
```

### 2. MultiEgoSetting reset 확인

```bash
python -m morai_rl.scripts.check_reset \
  --config morai_rl/stage1_rl8_config.toml \
  --repeats 3 \
  --sleep-sec 1
```

정상 동작 시 `reset_strategy=multi_ego_setting` 이 출력되고, 차량이 같은 위치로 복귀해야 합니다.

### 3. 짧은 step loop 확인

```bash
python -m morai_rl.scripts.check_step_loop \
  --config morai_rl/stage1_rl8_config.toml \
  --steps 50
```

### 4. PPO 학습 시작

```bash
python -m morai_rl.scripts.train_ppo \
  --config morai_rl/stage1_rl8_config.toml \
  --timesteps 50000 \
  --save-dir runs/ppo_morai \
  --run-name rl8_multi_ego_reset
```

짧은 smoke test는:

```bash
python -m morai_rl.scripts.train_ppo \
  --config morai_rl/stage1_rl8_config.toml \
  --timesteps 5000 \
  --run-name smoke_test
```

## Useful Scripts

### Send one MultiEgoSetting packet

차량을 지정 좌표로 한 번 보내보는 테스트입니다.

```bash
python -m morai_rl.scripts.send_multi_ego_setting \
  --config morai_rl/stage1_rl8_config.toml \
  --x 140.71 --y 1406.79 --z -0.53 \
  --roll-deg 0.354 --pitch-deg 0.319 --yaw-deg 91.494 \
  --speed-kph 0 \
  --gear 1
```

### Rule-based baseline

```bash
python -m morai_rl.scripts.run_simple_driver \
  --config morai_rl/stage1_rl8_config.toml \
  --episodes 1
```

### Scenario Load packet test

현재 RL8 설정은 scenario load를 reset에 사용하지 않지만, 별도 테스트는 가능합니다.

```bash
python -m morai_rl.scripts.send_scenario_load \
  --config morai_rl/example_config.toml \
  --file-name RL_9 \
  --delete-all
```

## Observation and Learning

현재 환경은 아래 요소를 사용합니다.

- reference path projection
- route corridor
- progress-based reward
- off-track / no-progress / reverse-progress termination
- steering-only action mode
- local BeV observation

`train_ppo.py`는 기본적으로:

- PPO
- checkpoint 저장
- crash 후 resume 시도
- optional BeV viewer

를 지원합니다.

## Known Notes

- MORAI Linux 환경에서 full `Scenario Load` 반복은 시뮬레이터 abort를 유발할 수 있습니다.
- 현재 RL8 설정은 이를 피하기 위해 `MultiEgoSetting` reset을 사용합니다.
- reset 안정화를 위해 reset 직후 `4초` 대기합니다.
- reset packet은 `Parking(1)`으로 보내고, 실제 주행은 학습 step의 control command가 시작하면서 다시 `Drive`로 넘어갑니다.

## Next Improvements

- reset target pose를 시나리오별로 관리
- collision 신호를 reward / termination에 더 적극 반영
- SAC / TD3 등 다른 알고리즘 지원
- training/evaluation config 분리
- reset diagnostics logging 강화
