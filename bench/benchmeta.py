"""Metadata recorded with every benchmark result: commit, hardware, OS, timestamp."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
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
        info["kernel"] = platform.release()
        info["topology"] = _lscpu_topology()
        info["ec2"] = ec2_info()
    return info


def _lscpu_topology() -> dict[str, Any] | None:
    """Sockets, physical cores, and threads per core, from `lscpu`: on a cloud VM the logical CPU
    count includes SMT siblings, which matters for thread-scaling results."""
    out = _run(["lscpu"])
    if not out:
        return None
    fields = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}

    def number(key: str) -> int | None:
        value = fields.get(key, "")
        return int(value) if value.isdigit() else None

    sockets = number("Socket(s)")
    cores = number("Core(s) per socket")
    return {
        "sockets": sockets,
        "cores_per_socket": cores,
        "threads_per_core": number("Thread(s) per core"),
        "physical_cores": sockets * cores if sockets and cores else None,
        "l3_cache": fields.get("L3 cache"),
        "flags_avx512f": " avx512f " in f" {fields.get('Flags', '')} ",
    }


def ec2_info(timeout: float = 0.5) -> dict[str, str] | None:
    """Instance type, AZ, and AMI from the EC2 instance metadata service (IMDSv2); None when not
    on EC2. The instance type is part of every AWS result's record."""
    import urllib.request

    base = "http://169.254.169.254/latest"
    try:
        token_request = urllib.request.Request(
            f"{base}/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_request, timeout=timeout) as response:
            token = response.read().decode()
        info = {}
        for key, path in [
            ("instance_type", "meta-data/instance-type"),
            ("availability_zone", "meta-data/placement/availability-zone"),
            ("ami_id", "meta-data/ami-id"),
            ("instance_id", "meta-data/instance-id"),
        ]:
            request = urllib.request.Request(
                f"{base}/{path}", headers={"X-aws-ec2-metadata-token": token}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                info[key] = response.read().decode()
        return info
    except OSError:
        return None


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


def thermal_warnings() -> str:
    """Thermal or performance warnings from `pmset -g therm` (macOS); empty when there are none
    or on other systems."""
    if platform.system() != "Darwin":
        return ""
    out = _run(["pmset", "-g", "therm"])
    return "\n".join(ln for ln in out.splitlines() if ln.strip() and not ln.startswith("Note: No"))


def busy_processes(threshold: float = 50.0) -> list[str]:
    """Other processes using more than `threshold` percent of a core (ps's recent average).
    Excludes this process and its parent, and the coding agent driving the run."""
    mine = {os.getpid(), os.getppid()}
    busy = []
    for line in _run(["ps", "-Ao", "pid=,pcpu=,comm="]).splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1].replace(".", "", 1).isdigit():
            pid, cpu, comm = int(parts[0]), float(parts[1]), parts[2]
            if cpu > threshold and pid not in mine and "claude" not in comm:
                busy.append(f"{pid} {cpu:.0f}% {comm}")
    return busy


def preflight(what: str, max_wait: float = 300, quiet_checks: int = 3, interval: float = 5) -> None:
    """Wait for a cool, quiet machine before a heavy benchmark step. The M2 development machine
    is fanless and has shut down under sustained load; contention also corrupts timings.

    Stops at once on a thermal warning. Otherwise waits until no other process is busy on
    `quiet_checks` consecutive checks `interval` seconds apart, so a brief UI spike does not abort
    a run but a sustained job (e.g. another project's test suite) does, after `max_wait` seconds.
    """
    deadline = time.monotonic() + max_wait
    quiet = 0
    while True:
        if warning := thermal_warnings():
            raise SystemExit(f"thermal warning before {what}; stopping:\n{warning}")
        busy = busy_processes()
        quiet = 0 if busy else quiet + 1
        if quiet >= quiet_checks:
            return
        if time.monotonic() > deadline:
            raise SystemExit(
                f"other heavy processes still running after {max_wait:.0f} s, before {what}; "
                "stopping:\n" + "\n".join(busy)
            )
        time.sleep(interval)


def hardware_note(hardware: dict[str, Any]) -> str:
    """How to read speed numbers from this machine. The fanless M2 development laptop throttles
    under sustained load, so its QPS is indicative only; recall is unaffected."""
    cpu = hardware.get("cpu") or hardware["machine"]
    if is_development_machine(hardware):
        return (
            f"{cpu} (fanless development machine): recall is valid; QPS is indicative only. "
            "Final speed comparisons run on dedicated hardware (Phase 9)."
        )
    ec2 = hardware.get("ec2")
    if ec2:
        topology = hardware.get("topology") or {}
        cores = topology.get("physical_cores")
        smt = topology.get("threads_per_core")
        shape = f", {cores} physical cores x {smt} threads" if cores and smt else ""
        return (
            f"AWS {ec2['instance_type']} ({cpu}{shape}, {hardware.get('memory_gib')} GiB, "
            f"{ec2['availability_zone']}), kernel {hardware.get('kernel')}"
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
