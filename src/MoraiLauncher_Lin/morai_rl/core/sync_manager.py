from __future__ import annotations

import time


class StepClock:
    """Small helper to keep a fixed control loop frequency."""

    def __init__(self, hz: float):
        if hz <= 0:
            raise ValueError("step hz must be positive")
        self.period_sec = 1.0 / hz
        self._next_tick = None

    def reset(self) -> None:
        self._next_tick = time.perf_counter() + self.period_sec

    def sleep(self) -> None:
        now = time.perf_counter()
        if self._next_tick is None:
            self._next_tick = now + self.period_sec
            return

        sleep_sec = self._next_tick - now
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        self._next_tick += self.period_sec
