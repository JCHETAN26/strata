"""Metadata recorded with every benchmark result: commit, hardware, OS, timestamp."""

from __future__ import annotations

import datetime as dt
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_info() -> dict[str, Any]:
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    dirty = bool(
        _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])
    )
    return {"commit": commit, "dirty": dirty}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "logical_cpus": os.cpu_count(),
        # Background load skews results, especially on the fanless dev laptop.
        "load_average_1m_5m_15m": [round(x, 2) for x in os.getloadavg()],
    }
    if platform.system() == "Darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_gib"] = round(int(mem) / 2**30, 1) if mem else None
        info["os_version"] = _run(["sw_vers", "-productVersion"])
    elif platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo").read_text()
        match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        info["cpu"] = match.group(1).strip() if match else platform.processor()
        meminfo = Path("/proc/meminfo").read_text()
        match = re.search(r"MemTotal:\s*(\d+) kB", meminfo)
        info["memory_gib"] = round(int(match.group(1)) / 2**20, 1) if match else None
        info["cpu_flags_avx2"] = " avx2 " in cpuinfo
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        info["cpu_governor"] = governor.read_text().strip() if governor.exists() else None
    return info


def metadata() -> dict[str, Any]:
    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "hardware": hardware_info(),
    }


def timestamp_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")
