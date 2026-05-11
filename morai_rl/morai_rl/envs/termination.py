from __future__ import annotations

def evaluate_termination(
    step_count: int,
    max_steps: int,
    off_track: bool,
    blocked_collision: bool,
    reverse_progress: bool,
    stalled: bool,
) -> tuple[bool, bool, str | None]:
    if off_track:
        return True, False, "off_track"
    if blocked_collision:
        return True, False, "blocked_collision"
    if reverse_progress:
        return True, False, "reverse_progress"
    if stalled:
        return True, False, "stalled"
    if step_count >= max_steps:
        return False, True, "max_steps"
    return False, False, None
