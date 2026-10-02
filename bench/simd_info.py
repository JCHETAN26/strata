"""Which SIMD instruction set each library's distance code actually runs with on this machine.

On a CPU with AVX-512 (e.g. the AWS c7i), the three libraries need not use the same width:
- Strata selects its kernels at compile time: AVX2 + FMA on x86 (-mavx2 -mfma), NEON on ARM.
  It has no AVX-512 kernels.
- FAISS (faiss-cpu 1.15 wheels) is one build with dynamic dispatch: it detects the CPU at run
  time and picks AVX2, AVX-512, or AVX-512 for Sapphire Rapids (AVX512_SPR) on x86, NEON on ARM.
- hnswlib selects at run time among the paths its compile flags enabled (SSE, AVX, AVX-512 on
  x86; it has no NEON code).

So the answer is measured, not assumed. Strata's is read from the kernel the harness reports,
FAISS's from FAISS itself (SIMDConfig: the level it dispatched to). hnswlib's and Strata's are
cross-checked by disassembling the binary and counting instructions on 512-bit (zmm) and 256-bit
(ymm) registers. bench/run_hnsw_curves.py prints the result next to each library's numbers and
flags any difference in vector width.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

WIDTH = {"AVX-512": 512, "AVX2": 256, "AVX": 256, "NEON": 128, "SSE": 128, "SVE": None, "scalar": 0}


def cpu_simd() -> dict[str, Any]:
    """The instruction sets this CPU offers (the relevant ones)."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return {"arch": machine, "neon": True}
    flags: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"^flags\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        flags = set(match.group(1).split()) if match else set()
    return {
        "arch": machine,
        "avx2": "avx2" in flags,
        "fma": "fma" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_extras": sorted(f for f in flags if f.startswith("avx512") and f != "avx512f"),
    }


def binary_vector_registers(path: str | Path) -> dict[str, Any] | None:
    """Counts disassembled x86 instructions that use zmm (AVX-512) and ymm (AVX/AVX2) registers.
    None when not x86 or no disassembler is available. A count of zero zmm instructions proves a
    binary cannot run AVX-512 code; a nonzero count means it contains an AVX-512 path, which runs
    only if the CPU supports it and the library's dispatch picks it."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return None
    objdump = shutil.which("objdump") or shutil.which("llvm-objdump")
    if not objdump:
        return None
    out = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", str(path)], capture_output=True, text=True
    ).stdout
    return {
        "binary": str(path),
        "zmm_instructions": len(re.findall(r"%zmm\d|\bzmm\d", out)),
        "ymm_instructions": len(re.findall(r"%ymm\d|\bymm\d", out)),
    }


def _label(isa: str, how: str) -> dict[str, Any]:
    return {"isa": isa, "width_bits": WIDTH[isa], "how": how}


def strata_simd(kernel: str, binary: str | Path) -> dict[str, Any]:
    """Strata: the kernel the harness reported, cross-checked against the binary."""
    isa = {"avx2": "AVX2", "neon": "NEON", "scalar": "scalar"}.get(kernel, kernel)
    info = _label(isa, "compile time (-mavx2 -mfma on x86, NEON on ARM)")
    info["disassembly"] = binary_vector_registers(binary)
    return info


def faiss_simd() -> dict[str, Any]:
    """FAISS: the SIMD level it dispatched to on this CPU (call after `import faiss`)."""
    import faiss

    options = faiss.get_compile_options()
    config = getattr(faiss, "SIMDConfig", None)
    if config is not None:
        level = config.get_level_name()
        how = "runtime dispatch" if config.has_dynamic_dispatch() else "compile time"
    else:  # older FAISS: separate builds per level, chosen at import
        level = next((x for x in ("AVX512_SPR", "AVX512", "AVX2", "NEON") if x in options), "NONE")
        how = "build variant loaded at import"
    isa = {"AVX512_SPR": "AVX-512", "AVX512": "AVX-512", "AVX512_VPOPCNT": "AVX-512",
           "AVX2": "AVX2", "ARM_NEON": "NEON", "NEON": "NEON", "ARM_SVE": "SVE",
           "NONE": "scalar"}.get(level, level)  # fmt: skip
    info = _label(isa, how) if isa in WIDTH else {"isa": isa, "width_bits": None, "how": how}
    info.update({"faiss_level": level, "compile_options": options})
    if config is not None:
        info["dispatched_level"] = int(config.get_dispatched_level())
        info["auto_detected_level"] = int(config.auto_detect_simd_level())
        info["level"] = int(config.get_level())
    return info


def restrict_faiss_to_avx2() -> None:
    """Holds FAISS to AVX2. Call before `import faiss` (sets the environment) and again after
    (sets the level). Both are needed for faiss-cpu 1.15: its runtime-dispatching build reads
    FAISS_SIMD_LEVEL at load time; FAISS_OPT_LEVEL only chooses among separate per-level builds,
    which that wheel does not ship, so on its own it would change nothing. An unsupported level
    in FAISS_SIMD_LEVEL silently becomes NONE (scalar), and set_level() raises, so the result is
    always checked afterwards with verify()."""
    import os

    os.environ["FAISS_SIMD_LEVEL"] = "AVX2"
    os.environ["FAISS_OPT_LEVEL"] = "AVX2"  # older multi-build wheels
    if "faiss" in sys.modules:
        import faiss

        if hasattr(faiss, "SIMDConfig"):
            try:
                faiss.SIMDConfig.set_level(faiss.SIMDLevel_AVX2)
            except RuntimeError as e:
                raise SystemExit(f"error: FAISS cannot be held to AVX2 here: {e}") from None


def hnswlib_simd(cpu: dict[str, Any]) -> dict[str, Any]:
    """hnswlib: which paths its extension contains, and which the CPU lets it pick."""
    import hnswlib

    disassembly = binary_vector_registers(hnswlib.__file__)
    if cpu.get("arch") in ("arm64", "aarch64"):
        info = _label("scalar", "no NEON path in hnswlib (SSE/AVX intrinsics only)")
    elif disassembly is None:
        info = {"isa": "unknown", "width_bits": None, "how": "no disassembler available"}
    elif disassembly["zmm_instructions"] and cpu.get("avx512f"):
        info = _label("AVX-512", "runtime dispatch; the extension has an AVX-512 path")
    elif disassembly["ymm_instructions"]:
        info = _label("AVX", "runtime dispatch; the extension's widest path is 256-bit")
    else:
        info = _label("SSE", "the extension has no AVX path")
    info["disassembly"] = disassembly
    return info


def library_simd(library: str, requested: str = "native") -> dict[str, Any]:
    """What `library` runs with, the configuration that was requested ("native": the widest the
    library and CPU support; "avx2": held to AVX2), and whether the two agree."""
    cpu = cpu_simd()
    info = faiss_simd() if library == "faiss" else hnswlib_simd(cpu)
    info["cpu"] = cpu
    info["requested"] = requested
    info["problems"] = verify(library, requested, info)
    return info


def verify(library: str, requested: str, info: dict[str, Any]) -> list[str]:
    """Checks that a library really runs the requested configuration. Empty list: it does.

    avx2:   FAISS dispatched (and reports) AVX2; hnswlib's extension has 256-bit code and no
            512-bit (zmm) instructions at all, so its dispatch cannot pick AVX-512.
    native: FAISS runs the level it auto-detects for this CPU; hnswlib, on a CPU with AVX-512,
            has an AVX-512 path (built with -march=native), which its dispatch then picks.
    """
    cpu = info.get("cpu") or cpu_simd()
    x86 = cpu.get("arch") in ("x86_64", "amd64")
    problems = []
    if requested == "avx2":
        if not x86:
            return [f"AVX2 restriction requested on {cpu.get('arch')}: x86 only"]
        if library == "faiss":
            if info.get("faiss_level") != "AVX2":
                problems.append(f"FAISS runs {info.get('faiss_level')}, not AVX2")
            if "dispatched_level" in info and info["dispatched_level"] != info.get("level"):
                problems.append("FAISS's dispatched level differs from its configured level")
        else:
            d = info.get("disassembly")
            if d is None:
                problems.append("cannot disassemble hnswlib to check it")
            elif d["zmm_instructions"]:
                problems.append(f"hnswlib has {d['zmm_instructions']} AVX-512 instructions")
            elif not d["ymm_instructions"]:
                problems.append("hnswlib has no 256-bit code: built without AVX (SSE only)")
    elif requested == "native":
        if library == "faiss":
            if "auto_detected_level" in info and info["level"] != info["auto_detected_level"]:
                problems.append("FAISS is not running the level it detects for this CPU")
        elif x86 and cpu.get("avx512f"):
            d = info.get("disassembly") or {}
            if not d.get("zmm_instructions"):
                problems.append(
                    "hnswlib has no AVX-512 path on an AVX-512 CPU (not -march=native?)"
                )
    else:
        problems.append(f"unknown SIMD configuration {requested!r}")
    return problems


def verify_strata(info: dict[str, Any]) -> list[str]:
    """Strata on x86 must run its AVX2 kernels and contain no AVX-512 code."""
    cpu = cpu_simd()
    if cpu.get("arch") not in ("x86_64", "amd64"):
        return []
    problems = []
    if info.get("isa") != "AVX2":
        problems.append(f"Strata runs {info.get('isa')} kernels, not AVX2 (built without -mavx2?)")
    d = info.get("disassembly") or {}
    if d.get("zmm_instructions"):
        problems.append(f"Strata's binary has {d['zmm_instructions']} AVX-512 instructions")
    return problems


def describe(info: dict[str, Any] | None) -> str:
    """Short label for a table cell, e.g. 'AVX-512 (512-bit)'."""
    if not info:
        return "not recorded"
    width = info.get("width_bits")
    return f"{info['isa']} ({width}-bit)" if width else str(info["isa"])
