# MORAI RL Starter

MORAI UDP 기반 강화학습 1단계를 바로 시작할 수 있도록 만든 최소 골격입니다.

## 포함된 것

- UDP 제어 송신기
- 차량 상태 수신기
- GT 객체 수신기
- 충돌 상태 수신기
- Scenario Load용 reset 래퍼
- `reset()` / `step()` 구조의 Gym-like 환경
- simple state 생성기
- simple reward / done 조건
- rule-based sanity-check 드라이버

## 빠른 시작

1. `morai_rl/example_config.toml`에서 포트와 경로를 맞춥니다.
2. `path.csv_path`를 실제 주행할 기준 경로 CSV로 바꿉니다.
3. `reset.command`에 Scenario Load를 호출하는 스크립트나 명령을 넣거나, UDP Scenario Load 설정을 채웁니다.
4. 아래 순서로 확인합니다.

```bash
python -m morai_rl.scripts.check_udp_io --config morai_rl/example_config.toml
python -m morai_rl.scripts.check_collision_udp --config morai_rl/example_config.toml
python -m morai_rl.scripts.check_reset --config morai_rl/example_config.toml
python -m morai_rl.scripts.check_step_loop --config morai_rl/example_config.toml
python -m morai_rl.scripts.run_simple_driver --config morai_rl/example_config.toml
```

직접 제어 명령을 보내보려면:

```bash
python -m morai_rl.scripts.check_udp_io \
  --config morai_rl/example_config.toml \
  --send-command \
  --throttle 0.2 \
  --brake 0.0 \
  --steering 0.0
```

기본 예제 설정은 MORAI `Ego Ctrl Cmd` UDP 형식에 맞춰
`127.0.0.1:9095 -> 127.0.0.1:9096`으로 제어를 송신합니다.

충돌 센서 수신 테스트:

```bash
python -m morai_rl.scripts.check_collision_udp \
  --config morai_rl/example_config.toml
```

기본 예제 설정은 충돌 패킷을 `127.0.0.1:9092`에서 수신하도록 되어 있습니다.
MORAI가 다른 PC에서 송신 중이라면 `--host 0.0.0.0`로 바꿔서 바인딩하면 됩니다.

UDP Scenario Load 테스트:

```bash
python -m morai_rl.scripts.send_scenario_load \
  --config morai_rl/example_config.toml \
  --file-name RL_0319 \
  --delete-all
```

기본 reset 예제 설정은 `127.0.0.1:9103 -> 127.0.0.1:9104` 포트와
`LoadScenario` 헤더 이름을 사용합니다.

## 기준 경로 CSV 형식

아래 헤더를 권장합니다.

```csv
x,y,yaw_deg
123.4,567.8,90.0
124.2,568.1,90.5
```

`yaw_deg`가 없으면 인접 점으로 자동 계산합니다.

## reset.command 예시

```toml
[reset]
command = "python /absolute/path/to/load_scenario.py"
```

지금 버전은 MORAI Scenario Load를 직접 구현하지 않고, 외부 명령을 연결하는 방식입니다.
이미 시나리오 복원 스크립트가 있다면 그대로 붙일 수 있습니다.

## 첫 체크리스트

- `check_udp_io`에서 ego 상태가 안정적으로 들어온다.
- `check_reset`이 연속 10번 성공한다.
- `check_step_loop`가 에러 없이 돈다.
- `run_simple_driver`가 짧게라도 차선을 따라간다.

## 다음 단계

- reward 튜닝
- action을 steering only에서 throttle까지 확장
- 충돌/트랙 이탈 센서 반영
- PPO/SAC 같은 RL 알고리즘 연결
