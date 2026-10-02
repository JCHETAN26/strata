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
    return info


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


def library_simd(library: str) -> dict[str, Any]:
    cpu = cpu_simd()
    info = faiss_simd() if library == "faiss" else hnswlib_simd(cpu)
    info["cpu"] = cpu
    return info


def describe(info: dict[str, Any] | None) -> str:
    """Short label for a table cell, e.g. 'AVX-512 (512-bit)'."""
    if not info:
        return "not recorded"
    width = info.get("width_bits")
    return f"{info['isa']} ({width}-bit)" if width else str(info["isa"])
