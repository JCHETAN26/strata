"""Metadata recorded with every benchmark result: commit, hardware, OS, timestamp."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
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
    # Benchmark outputs under results/ don't affect what was measured; ignore them.
    status_cmd = ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"]
    dirty = bool(_run([*status_cmd, "--", ".", ":!results"]))
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


# Run in a separate interpreter: importing torch here would load its bundled OpenMP runtime into
# the benchmark process, and a process that already loaded FAISS's copy aborts on macOS
# (OMP Error #15). It also keeps torch's import time and memory out of the measured process.
_TORCH_CUDA_PROBE = """
import json
try:  # torch is optional; only the embed group installs it
    import torch
    available = torch.cuda.is_available()
    print(json.dumps({
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }))
except Exception:  # torch missing or a driver/runtime mismatch; record nothing
    print("null")
"""


def _torch_cuda_info() -> dict[str, Any] | None:
    out = _run([sys.executable, "-c", _TORCH_CUDA_PROBE])
    try:
        return json.loads(out) if out else None
    except json.JSONDecodeError:
        return None


def accelerator_info() -> dict[str, Any]:
    """Best-effort GPU and hosted-environment capture, for results produced on a GPU box (e.g.
    Kaggle). Never raises and adds no import dependency: it reads nvidia-smi if present, torch's
    CUDA view (from a subprocess) only if torch is importable, and a few well-known Kaggle env
    vars. Absent hardware or tools just yield empty fields, so it is safe to record on every
    result."""
    info: dict[str, Any] = {"gpus": [], "cuda": None, "hosted": None}

    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in (ln.strip() for ln in smi.splitlines() if ln.strip()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            name, driver, mem_mib, cap = parts
            info["gpus"].append(
                {
                    "name": name,
                    "driver_version": driver,
                    "memory_mib": int(mem_mib) if mem_mib.isdigit() else mem_mib,
                    "compute_capability": cap,
                }
            )

    info["cuda"] = _torch_cuda_info()

    # Kaggle sets these in kernel sessions; harmless (and empty) elsewhere.
    kaggle = {
        var: os.environ[var]
        for var in (
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_DOCKER_IMAGE",
            "KAGGLE_URL_BASE",
            "KAGGLE_DATA_PROXY_TOKEN",
        )
        if var in os.environ and var != "KAGGLE_DATA_PROXY_TOKEN"  # never record the proxy token
    }
    if os.path.isdir("/kaggle"):
        kaggle["kaggle_dirs"] = sorted(
            d for d in ("/kaggle/input", "/kaggle/working", "/kaggle/temp") if os.path.isdir(d)
        )
    info["hosted"] = kaggle or None
    return info


def metadata() -> dict[str, Any]:
    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="microseconds"),
        "git": git_info(),
        "hardware": hardware_info(),
        "accelerator": accelerator_info(),
    }


def is_development_machine(hardware: dict[str, Any]) -> bool:
    """The fanless Apple-silicon development laptop, whose QPS is indicative only."""
    return (hardware.get("cpu") or hardware["machine"]).startswith("Apple")


def hardware_note(hardware: dict[str, Any]) -> str:
    """How to read speed numbers from this machine. The fanless M2 development laptop throttles
    under sustained load, so its QPS is indicative only; recall is unaffected."""
    cpu = hardware.get("cpu") or hardware["machine"]
    if is_development_machine(hardware):
        return (
            f"{cpu} (fanless development machine): recall is valid; QPS is indicative only. "
            "Final speed comparisons run on dedicated hardware (Phase 9)."
        )
    return cpu


def timestamp_slug() -> str:
    # Microseconds: short runs finish within the same second and must not share a filename.
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def write_new(path: Path, text: str) -> None:
    """Write a result file, refusing to overwrite an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(text)
