from __future__ import annotations

import os
import signal
import shlex
import subprocess
import time


def terminate_processes_by_name(
    process_names: list[str],
    terminate_timeout_sec: float = 5.0,
) -> list[int]:
    """Terminate simulator processes whose executable name exactly matches."""
    names = {name.strip() for name in process_names if name.strip()}
    if not names:
        return []
    matched_pids = _find_pids_by_name(names)
    if not matched_pids:
        return []
    current_pid = os.getpid()
    for pid in matched_pids:
        if pid == current_pid:
            continue
        _signal_process(pid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, terminate_timeout_sec)
    while time.monotonic() < deadline:
        remaining_pids = [pid for pid in matched_pids if pid != current_pid and _pid_exists(pid)]
        if not remaining_pids:
            break
        time.sleep(0.1)
    for pid in matched_pids:
        if pid == current_pid or not _pid_exists(pid):
            continue
        _signal_process(pid, signal.SIGKILL)
    return matched_pids


def terminate_processes_by_cmdline_substrings(
    patterns: list[str],
    terminate_timeout_sec: float = 5.0,
) -> list[int]:
    """Terminate processes whose full command line contains any configured substring."""
    needles = [pattern.strip() for pattern in patterns if pattern.strip()]
    if not needles:
        return []
    matched_pids = _find_pids_by_cmdline_substrings(needles)
    if not matched_pids:
        return []
    current_pid = os.getpid()
    for pid in matched_pids:
        if pid == current_pid:
            continue
        _signal_process(pid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, terminate_timeout_sec)
    while time.monotonic() < deadline:
        remaining_pids = [pid for pid in matched_pids if pid != current_pid and _pid_exists(pid)]
        if not remaining_pids:
            break
        time.sleep(0.1)
    for pid in matched_pids:
        if pid == current_pid or not _pid_exists(pid):
            continue
        _signal_process(pid, signal.SIGKILL)
    return matched_pids


def launch_process(command: str) -> int | None:
    command = command.strip()
    if not command:
        return None
    process = subprocess.Popen(
        shlex.split(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def _find_pids_by_name(names: set[str]) -> list[int]:
    pids: list[int] = []
    for pid_text in os.listdir("/proc"):
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        command_names = _process_command_names(pid)
        if names.intersection(command_names):
            pids.append(pid)
    return pids


def _find_pids_by_cmdline_substrings(patterns: list[str]) -> list[int]:
    pids: list[int] = []
    for pid_text in os.listdir("/proc"):
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        cmdline = _process_cmdline(pid)
        if cmdline and any(pattern in cmdline for pattern in patterns):
            pids.append(pid)
    return pids


def _process_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw_cmdline = handle.read()
    except OSError:
        return ""
    args = [arg.decode(errors="ignore") for arg in raw_cmdline.split(b"\0") if arg]
    return " ".join(args)


def _process_command_names(pid: int) -> set[str]:
    names: set[str] = set()
    try:
        with open(f"/proc/{pid}/comm", "rt", encoding="utf-8") as handle:
            names.add(handle.read().strip())
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw_cmdline = handle.read()
    except OSError:
        raw_cmdline = b""
    if raw_cmdline:
        for raw_arg in raw_cmdline.split(b"\0"):
            arg = raw_arg.decode(errors="ignore")
            if arg:
                names.add(os.path.basename(arg))
    return {name for name in names if name}


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process(pid: int, sig: signal.Signals) -> bool:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True
