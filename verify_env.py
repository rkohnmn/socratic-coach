#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 TA Phase 1 - Environment Setup & Diagnostics (offline local Teaching Assistant)
==============================================================================

Target host : Windows 11
              AMD Ryzen 7 5800U (8C/16T, 16 GB system RAM)
              NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM, reserved for
              Ollama LLM inference only)

Stages (sequential, colored terminal feedback):

  STEP 1/5  Python runtime & wheel provisioning
            * enforce Python 3.10 - 3.12
            * bootstrap pip; install CPU-only PyTorch FIRST so `pip install
              kokoro` can never drag in a CUDA build (4GB VRAM guardrail)
            * numpy / sounddevice / soundfile / requests / faster-whisper /
              kokoro; pyaudio with prebuilt-wheel fallback + diagnostics
  STEP 2/5  Ollama daemon & GPU LLM verification
            * ping http://localhost:11434/api/tags
            * start `ollama serve` detached (logs/ollama_serve.log) if needed
            * pull llama3.2:3b via subprocess when missing
            * 1-token generation + latency + VRAM-offload proof via /api/ps
  STEP 3/5  Audio hardware enumeration (sounddevice + pyaudio)
            * host APIs, default mic/speaker names + sample rates, mic probe
  STEP 4/5  CPU STT/TTS smoke tests
            * faster-whisper "base.en" (device="cpu", compute_type="int8",
              cpu_threads=4) weight download + tiny inference pass
            * kokoro on CPU: synthesizes "System initialized." and plays it
              through the default output device (kokoro-onnx fallback)
  STEP 5/5  Readiness report + actionable fixes + exit code

Usage:
  python verify_env.py [--skip-install] [--skip-models] [--skip-playback]
                       [--keep-cuda-torch] [--host http://localhost:11434]

Exit codes:
  0    environment is 100% operational (all critical checks passed)
  1    one or more critical checks failed (see "ACTION REQUIRED" block)
  130  interrupted by the user (Ctrl+C)
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parent
MODELS_DIR = WORKSPACE / "models"
LOGS_DIR = WORKSPACE / "logs"
SMOKE_DIR = LOGS_DIR / "smoke_assets"
GIB = 1024 ** 3

PYTHON_MIN = (3, 10)
PYTHON_MAX = (3, 12)

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_MODEL_GIB_ESTIMATE = 2.0
OLLAMA_SERVE_LOG = LOGS_DIR / "ollama_serve.log"

WHISPER_MODEL_SIZE = "base.en"
WHISPER_CPU_THREADS = 4
WHISPER_WEIGHTS_MB_ESTIMATE = 145

TEST_UTTERANCE = "System initialized."
TTS_SAMPLE_RATE_HZ = 24000

KOKORO_ONNX_DIR = MODELS_DIR / "kokoro-onnx"
KOKORO_ONNX_MODEL_URL = (
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx"
)
KOKORO_ONNX_VOICES_URL = (
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin"
)

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# (import_name, pip_name, pip_requirement)
REQUIRED_LIBS = [
    ("numpy", "numpy", "numpy>=1.24"),
    ("sounddevice", "sounddevice", "sounddevice>=0.4.6"),
    ("soundfile", "soundfile", "soundfile>=0.12.1"),
    ("requests", "requests", "requests>=2.31"),
    ("faster_whisper", "faster-whisper", "faster-whisper>=1.0.3"),
    ("kokoro", "kokoro", "kokoro>=0.9.2"),
]
PYAUDIO_PIP = "pyaudio"

# ---------------------------------------------------------------------------
# Windows console: UTF-8 output + ANSI escape sequences
# ---------------------------------------------------------------------------
USE_COLOR = True


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GREY = "\033[90m"


def _force_utf8_stdio() -> None:
    """Keep the check-mark glyphs intact on every Windows console code page."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _enable_windows_vt() -> bool:
    """Enable VT100 processing on classic conhost windows (Win11 terminals
    already support it). Returns False when colored output must be disabled."""
    global USE_COLOR
    if os.name != "nt":
        return True
    try:
        # Touching a fresh console buffer flips most conhost hosts into VT mode.
        os.system("")
    except Exception:
        pass
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            USE_COLOR = False
            return False
        if not (mode.value & 0x0004):  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        USE_COLOR = False
        return False
    return True


def _c(text: str, *codes: str) -> str:
    if not USE_COLOR or not codes:
        return text
    return "".join(codes) + text + C.RESET


def info(msg: str = "") -> None:
    print(f"  {_c('[i]', C.CYAN)} {msg}")


def ok(msg: str = "") -> None:
    print(f"  {_c('[OK]', C.GREEN + C.BOLD)} {msg}")


def warn(msg: str = "") -> None:
    print(f"  {_c('[!]', C.YELLOW + C.BOLD)} {msg}")


def err(msg: str = "") -> None:
    print(f"  {_c('[X]', C.RED + C.BOLD)} {msg}")


def rule(char: str = "-", width: int = 78) -> None:
    print("  " + _c(char * width, C.GREY))


def section(title: str, subtitle: str = "") -> None:
    print()
    print("  " + _c("=" * 78, C.MAGENTA))
    heading = f"{title}  -  {subtitle}" if subtitle else title
    print("   " + _c(heading, C.MAGENTA + C.BOLD))
    print("  " + _c("=" * 78, C.MAGENTA))


def banner(host_facts: dict) -> None:
    width = 78

    def boxed(text: str) -> str:
        return "| " + text.ljust(width - 4) + " |"

    print()
    print("  " + _c("+" + "-" * (width - 2) + "+", C.CYAN))
    print("  " + _c(boxed("TA PHASE 1  -  OFFLINE LOCAL TEACHING ASSISTANT"), C.CYAN + C.BOLD))
    print("  " + _c(boxed("Environment Setup, Provisioning & Diagnostic Verifier"), C.CYAN))
    print("  " + _c(boxed("CUDA budget: Ollama ONLY | STT + TTS pinned to CPU"), C.CYAN))
    print("  " + _c("+" + "-" * (width - 2) + "+", C.CYAN))
    print()
    print(f"   CPU ......: {host_facts['cpu']}")
    print(f"   RAM ......: {host_facts['ram']}")
    print(f"   GPU ......: {host_facts['gpu']}")
    print(f"   Python ...: {host_facts['py']}  ({host_facts['exe']})")
    print(f"   Workspace : {WORKSPACE}")
    print(f"   Started ..: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rule("=")

# ---------------------------------------------------------------------------
# Check registry / results
# ---------------------------------------------------------------------------
@dataclass
class Check:
    key: str
    label: str
    critical: bool = True
    passed: bool = False
    skipped: bool = False
    detail: str = ""
    fixes: list = field(default_factory=list)


CHECKS: dict = {}


def record(
    key: str,
    label: str,
    passed: bool,
    detail: str = "",
    fixes: list | None = None,
    critical: bool = True,
    skipped: bool = False,
) -> Check:
    check = Check(key, label, critical, passed, skipped, detail, fixes or [])
    CHECKS[key] = check
    return check


# ---------------------------------------------------------------------------
# Host / system information helpers
# ---------------------------------------------------------------------------
def cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(name).strip()
    except Exception:
        return platform.processor() or "unknown CPU"


def total_ram_gib() -> float:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / GIB
    except Exception:
        pass
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB, 1)
    except Exception:
        return 0.0


def gpu_summary() -> str:
    """'Name (x.x GiB VRAM, driver X)' via nvidia-smi, or a plain fallback."""
    exe = shutil.which("nvidia-smi")
    if exe is None and os.name == "nt":
        candidate = os.path.expandvars(
            r"%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        )
        if os.path.isfile(candidate):
            exe = candidate
    if exe:
        try:
            proc = subprocess.run(
                [
                    exe,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                name, mem, driver = [
                    p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")
                ]
                return f"{name} ({float(mem) / 1024:.1f} GiB VRAM, driver {driver})"
        except Exception:
            pass
    return "no NVIDIA GPU via nvidia-smi (CUDA offload for Ollama unavailable)"


def ensure_disk_space(min_free_gib: float = 6.0) -> None:
    try:
        free = shutil.disk_usage(str(WORKSPACE)).free / GIB
        if free < min_free_gib:
            warn(
                f"Only {free:.1f} GiB free on {WORKSPACE.drive} - model weights need "
                "~6 GiB (whisper ~0.15 + kokoro ~0.4 + llama3.2:3b ~2.0 + wheels)."
            )
        else:
            info(f"Disk space OK: {free:.1f} GiB free on {WORKSPACE.drive}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# pip / import helpers
# ---------------------------------------------------------------------------
def module_ok(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def module_imports(import_name: str) -> tuple:
    """Actually import the module; returns (ok, version_or_error)."""
    try:
        mod = __import__(import_name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def pip_present() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.returncode == 0
    except Exception:
        return False


def pip_install(spec: str, *extra: str) -> bool:
    """Run `python -m pip install <extra...> <spec>`; inherits the console so
    large wheels show their download progress. Returns True on success."""
    args = [sys.executable, "-m", "pip", "--disable-pip-version-check", "install"]
    args += list(extra)
    args.append(spec)
    info("pip install " + " ".join(list(extra) + [spec]))
    try:
        proc = subprocess.run(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        err(f"pip failed to run: {exc}")
        return False
    return proc.returncode == 0


def pip_uninstall(*packages: str) -> None:
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "uninstall",
                "-y",
                *packages,
            ]
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# STEP 1 - Python runtime & wheel provisioning
# ---------------------------------------------------------------------------
def ensure_pip() -> bool:
    if pip_present():
        return True
    warn("pip is missing - bootstrapping it with ensurepip ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=True,
            timeout=300,
        )
    except Exception as exc:
        err(f"ensurepip failed: {exc}")
        return False
    return pip_present()


def ensure_cpu_torch(keep_cuda: bool = False) -> None:
    """Install CPU-only PyTorch, or swap a CUDA build out.

    The RTX 3050 Ti has a hard 4 GB VRAM budget reserved for Ollama; the
    STT/TTS stack must never see CUDA. Installing torch from the CPU wheel
    index FIRST also stops `pip install kokoro` from pulling the default
    (CUDA) torch build later.
    """
    if not module_ok("torch"):
        info("PyTorch not found - installing the CPU-only wheel (~200 MB) ...")
        if not pip_install("torch", "--index-url", TORCH_CPU_INDEX):
            err("CPU-only PyTorch install failed (reported in the summary).")
        return

    try:
        import torch

        cuda_build = bool(getattr(torch.version, "cuda", None))
        cuda_active = bool(getattr(torch.cuda, "is_available", lambda: False)())
    except Exception as exc:
        warn(f"torch present but not importable ({exc}) - reinstalling CPU wheel ...")
        pip_uninstall("torch", "torchvision", "torchaudio")
        pip_install("torch", "--index-url", TORCH_CPU_INDEX)
        return

    if cuda_build or cuda_active:
        if keep_cuda:
            warn(
                f"PyTorch {torch.__version__} is a CUDA build - keeping it per "
                "--keep-cuda-torch (STT/TTS stay pinned to device='cpu')."
            )
            return
        warn(
            f"PyTorch {torch.__version__} is a CUDA build - swapping to the "
            "CPU-only wheel so STT/TTS can never leak VRAM away from Ollama ..."
        )
        pip_uninstall("torch")
        if not pip_install("torch", "--index-url", TORCH_CPU_INDEX):
            err("CPU-only PyTorch reinstall failed (reported in the summary).")
    else:
        ok(f"PyTorch {torch.__version__} is already CPU-only - guardrail intact.")


def install_missing_core_libs() -> None:
    for import_name, pip_name, requirement in REQUIRED_LIBS:
        if module_ok(import_name):
            continue
        if pip_name == "kokoro":
            info(
                f"'{pip_name}' missing - installing {requirement} "
                "(torch is already pinned to CPU) ..."
            )
        else:
            info(f"'{pip_name}' missing - installing {requirement} ...")
        pip_install(requirement)


def _pipwin_install(package: str) -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pipwin", "install", package], timeout=900
        )
        return proc.returncode == 0
    except Exception:
        return False


def ensure_pyaudio() -> None:
    """PyAudio/PortAudio is the classic Windows pain point: when pip falls
    back to source compilation it dies on 'portaudio.h'. Try wheels first,
    then precompiled-only wheels, then pipwin; else leave crisp instructions."""
    if module_ok("pyaudio"):
        return
    info(
        "PyAudio missing - trying the official PyPI wheel "
        "(0.2.14 ships prebuilt wheels for 3.10-3.12) ..."
    )
    if pip_install(PYAUDIO_PIP):
        return
    warn("Default install failed - forcing precompiled wheels only (--only-binary) ...")
    if pip_install(PYAUDIO_PIP, "--only-binary=:all:"):
        return
    warn("No usable wheel - trying pipwin's unofficial prebuilt binaries ...")
    if pip_install("pipwin") and _pipwin_install(PYAUDIO_PIP):
        return
    warn(
        "PyAudio could not be installed automatically - the stack falls back to "
        "sounddevice (bundles PortAudio via CFFI). Manual fix steps are printed "
        "in the final report."
    )

def verify_core_libs() -> None:
    missing, present, fixes = [], [], []
    for import_name, pip_name, _requirement in REQUIRED_LIBS:
        imported, version = module_imports(import_name)
        if imported:
            present.append(f"{pip_name}=={version}")
            continue
        missing.append(pip_name)
        if import_name == "kokoro" and not module_ok("torch"):
            fixes.append(
                "kokoro imports torch - install the CPU wheel first: "
                f"python -m pip install torch --index-url {TORCH_CPU_INDEX}"
            )
        else:
            fixes.append(f"python -m pip install {pip_name}")
    if missing:
        record(
            "core",
            "Python & Core Libraries",
            False,
            detail="missing: " + ", ".join(missing),
            fixes=fixes
            + ["Re-run setup.bat WITHOUT --skip-install to provision automatically."],
        )
        err(f"Core libraries incomplete - missing: {', '.join(missing)}")
    else:
        record("core", "Python & Core Libraries", True, detail=", ".join(present))
        ok(f"Core libraries verified: {', '.join(present)}")


def verify_torch_cpu_guardrail(keep_cuda: bool) -> None:
    label = "CPU-only PyTorch (VRAM guardrail)"
    if not module_ok("torch"):
        record(
            "torch_cpu",
            label,
            False,
            detail="torch is not installed",
            fixes=[
                f"python -m pip install torch --index-url {TORCH_CPU_INDEX}",
                "Re-run setup.bat without --skip-install to provision automatically.",
            ],
        )
        err("PyTorch is not installed - the CPU guardrail cannot be verified.")
        return
    try:
        import torch

        cuda_build = bool(getattr(torch.version, "cuda", None))
        cuda_active = bool(getattr(torch.cuda, "is_available", lambda: False)())
    except Exception as exc:
        record(
            "torch_cpu",
            label,
            False,
            detail=f"import failed: {exc}",
            fixes=[
                "python -m pip uninstall -y torch",
                f"python -m pip install torch --index-url {TORCH_CPU_INDEX}",
            ],
        )
        err(f"PyTorch cannot be imported: {exc}")
        return
    detail = (
        f"torch {torch.__version__}, cuda_build={cuda_build}, "
        f"torch.cuda.is_available()={cuda_active}"
    )
    if cuda_build or cuda_active:
        if keep_cuda:
            record(
                "torch_cpu",
                label,
                False,
                detail=detail + " (--keep-cuda-torch)",
                fixes=[
                    "Remove the CUDA build manually:",
                    "  python -m pip uninstall -y torch",
                    f"  python -m pip install torch --index-url {TORCH_CPU_INDEX}",
                ],
            )
            warn("CUDA PyTorch kept by request - STT/TTS will still be pinned to CPU.")
            return
        record(
            "torch_cpu",
            label,
            False,
            detail=detail,
            fixes=[
                "python -m pip uninstall -y torch",
                f"python -m pip install torch --index-url {TORCH_CPU_INDEX}",
                "Never install CUDA torch wheels: CUDA is reserved for Ollama only.",
            ],
        )
        err(f"CUDA PyTorch detected - VRAM guardrail violated! ({detail})")
    else:
        record("torch_cpu", label, True, detail=detail)
        ok(f"CPU-only PyTorch confirmed ({detail}).")

def verify_pyaudio() -> None:
    label = "PyAudio / PortAudio Bindings (optional)"
    if module_ok("pyaudio"):
        imported, version = module_imports("pyaudio")
        if imported:
            record("pyaudio", label, True, detail=f"pyaudio {version}", critical=False)
            ok(f"PyAudio {version} importable - PortAudio bindings OK.")
            return
        record(
            "pyaudio",
            label,
            False,
            detail=f"installed but import failed: {version}",
            critical=False,
            fixes=["python -m pip install --force-reinstall pyaudio"],
        )
        err(f"PyAudio present but import failed: {version}")
        return
    sounddevice_ok = module_ok("sounddevice")
    record(
        "pyaudio",
        label,
        False,
        detail=(
            "not installed - sounddevice (bundled PortAudio) is the active backend"
            if sounddevice_ok
            else "not installed and sounddevice missing: no audio backend at all"
        ),
        critical=False,
        fixes=[
            "On 3.10-3.12 the official wheel usually works: python -m pip install pyaudio",
            "Force a prebuilt wheel:  python -m pip install --only-binary=:all: pyaudio",
            "If pip compiles from source and fails on 'portaudio.h':",
            "  1) install 'Desktop development with C++' from Microsoft C++ Build Tools",
            "     https://visualstudio.microsoft.com/visual-cpp-build-tools/",
            "  2) provide PortAudio headers/libs (vcpkg install portaudio:x64-windows),",
            "     point INCLUDE/LIB at them, then re-run: python -m pip install pyaudio",
            "  3) OR skip PyAudio entirely - sounddevice ships its own PortAudio DLL",
            "     and is the recommended mic/speaker backend for this TA stack.",
        ],
    )
    if sounddevice_ok:
        warn("PyAudio not importable - sounddevice fallback covers the audio backend.")
    else:
        err("Neither PyAudio nor sounddevice is available - audio I/O is impossible.")


def step1_runtime(args) -> bool:
    section("STEP 1/5", "Python runtime & wheel provisioning")

    # -- 1a. Python version gate ---------------------------------------------
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if PYTHON_MIN <= (v.major, v.minor) <= PYTHON_MAX:
        record("python", "Python 3.10-3.12 Runtime", True, detail=detail)
        ok(f"{detail} - within the supported 3.10-3.12 range.")
    else:
        record(
            "python",
            "Python 3.10-3.12 Runtime",
            False,
            detail=detail + " - OUTSIDE supported range",
            fixes=[
                "Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/",
                "Tick 'Add python.exe to PATH' during installation.",
                "Recreate the venv:  py -3.11 -m venv .venv   (or 3.12), then re-run setup.bat",
            ],
        )
        err(
            f"Python {v.major}.{v.minor}.{v.micro} is NOT supported "
            "(need 3.10-3.12) - aborting."
        )
        return False

    # -- 1b. provisioning ------------------------------------------------------
    if not args.skip_install:
        if not ensure_pip():
            warn("pip is unavailable - automatic provisioning cannot continue.")
        ensure_cpu_torch(keep_cuda=args.keep_cuda_torch)
        install_missing_core_libs()
        ensure_pyaudio()
    else:
        warn("--skip-install given - verification only (no provisioning).")

    # -- 1c. verification (always runs) ------------------------------------------
    verify_core_libs()
    verify_torch_cpu_guardrail(args.keep_cuda_torch)
    verify_pyaudio()
    return True

# ---------------------------------------------------------------------------
# STEP 2 - Ollama daemon & GPU LLM verification
# ---------------------------------------------------------------------------
def find_ollama_binary() -> Path | None:
    exe = shutil.which("ollama")
    if exe:
        return Path(exe)
    candidates = []
    if os.name == "nt":
        candidates += [
            Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")),
            Path(os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe")),
            Path(os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")),
        ]
    else:
        candidates += [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ollama_reachable(host: str, timeout: float = 2.5) -> bool:
    try:
        import requests

        return requests.get(f"{host}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def start_ollama_detached(ollama_exe: Path) -> bool:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    try:
        with open(OLLAMA_SERVE_LOG, "ab") as log_file:
            proc = subprocess.Popen(
                [str(ollama_exe), "serve"],
                stdout=log_file,
                stderr=log_file,
                stdin=subprocess.DEVNULL,
                **kwargs,
            )
        info(f"Spawned detached `ollama serve` (pid {proc.pid}); log: {OLLAMA_SERVE_LOG}")
        return True
    except Exception as exc:
        err(f"Could not start `ollama serve`: {exc}")
        return False


def wait_for_ollama(host: str, seconds: float = 30.0) -> bool:
    info(f"Waiting up to {seconds:.0f}s for the daemon to accept connections ...")
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if ollama_reachable(host):
            print()
            return True
        print(_c(".", C.GREY), end="", flush=True)
        time.sleep(1.25)
    print()
    return ollama_reachable(host)


def pull_model(ollama_exe: Path, model: str) -> bool:
    info(
        f"Pulling '{model}' via subprocess (~{OLLAMA_MODEL_GIB_ESTIMATE:.1f} GB "
        "download; progress is streamed below) ..."
    )
    try:
        proc = subprocess.run([str(ollama_exe), "pull", model])
        return proc.returncode == 0
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        err(f"`ollama pull {model}` failed to run: {exc}")
        return False

def _ollama_generation_test(args) -> None:
    label = "GPU LLM Engine (Ollama llama3.2:3b via RTX 3050 Ti)"
    if args.skip_models:
        record("llm_gpu", label, False, detail="skipped by --skip-models", skipped=True)
        warn("Generation smoke test skipped (--skip-models).")
        return
    try:
        import requests
    except Exception as exc:
        record(
            "llm_gpu",
            label,
            False,
            detail=f"requests unavailable: {exc}",
            fixes=["python -m pip install requests"],
        )
        return

    host = args.host.rstrip("/")
    info(f"Sending a 1-token test generation to {host}/api/generate ...")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Reply with exactly one word: ready",
        "stream": False,
        "keep_alive": "15m",
        "options": {"num_predict": 1, "temperature": 0.0},
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=900)
        wall_s = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        record(
            "llm_gpu",
            label,
            False,
            detail=f"generation request failed: {exc}",
            fixes=[
                f"Verify manually: curl {host}/api/generate -d "
                f'\'{{"model":"{OLLAMA_MODEL}","prompt":"hi","stream":false}}\'',
                "Run `ollama ps` - if nothing is loaded, re-run the pull step.",
                "Watch the daemon live: Get-Content logs/ollama_serve.log -Wait",
            ],
        )
        err(f"1-token generation failed: {exc}")
        return

    text = (data.get("response") or "").strip()
    load_s = float(data.get("load_duration") or 0) / 1e9
    eval_count = int(data.get("eval_count") or 0)
    eval_dur = float(data.get("eval_duration") or 0) / 1e9
    tps = (eval_count / eval_dur) if eval_dur > 0 else 0.0
    ok(
        f"Generation OK in {wall_s:.2f}s wall (model load {load_s:.2f}s, "
        f"{tps:.1f} tok/s, reply: '{text or '<empty>'}')."
    )

    # -- VRAM offload proof via /api/ps ----------------------------------------
    vram_bytes = None
    try:
        ps = requests.get(f"{host}/api/ps", timeout=10).json()
        running = ps.get("models") or []
        if running:
            entry = running[0]
            vram_bytes = int(entry.get("size_vram") or 0)
            info(
                f"ollama ps: '{entry.get('name')}' - "
                f"{float(entry.get('size', 0)) / GIB:.2f} GiB total, "
                f"{vram_bytes / GIB:.2f} GiB resident in VRAM"
            )
    except Exception as exc:
        warn(f"Could not query /api/ps for VRAM residency: {exc}")

    if vram_bytes is None:
        record(
            "llm_gpu",
            label,
            True,
            detail=(
                f"generation OK ({wall_s:.2f}s wall, {tps:.1f} tok/s); VRAM "
                "residency unknown (/api/ps reported no loaded model)"
            ),
        )
        warn(
            "VRAM offload could not be confirmed - update the NVIDIA driver so "
            "Ollama keeps llama3.2:3b layers on the RTX 3050 Ti."
        )
        return

    if vram_bytes > 0:
        record(
            "llm_gpu",
            label,
            True,
            detail=(
                f"generation OK ({wall_s:.2f}s wall, {tps:.1f} tok/s); "
                f"{vram_bytes / GIB:.2f} GiB resident in VRAM (CUDA offload confirmed)"
            ),
        )
        ok(
            f"RTX 3050 Ti offload confirmed: {vram_bytes / GIB:.2f} GiB resident "
            "in the 4 GB VRAM."
        )
    else:
        record(
            "llm_gpu",
            label,
            False,
            detail="generation OK but 0 bytes resident in VRAM - model ran on CPU",
            fixes=[
                "Update the NVIDIA driver (RTX 3050 Ti needs a CUDA-capable driver).",
                "Update Ollama:  winget upgrade Ollama.Ollama",
                "Generate once, then `ollama ps` - the PROCESSOR column must show a GPU %.",
                "Check other apps are not occupying the 4 GB VRAM (browsers, games).",
            ],
        )
        err("Model is NOT offloaded to the GPU (0 bytes VRAM) - it is running on CPU.")

def step2_ollama(args) -> None:
    section("STEP 2/5", "Ollama daemon & GPU LLM verification (RTX 3050 Ti)")
    host = args.host.rstrip("/")

    # -- 2a. binary -------------------------------------------------------------
    ollama_exe = find_ollama_binary()
    if ollama_exe is None:
        record(
            "ollama_daemon",
            "Ollama Daemon (localhost:11434)",
            False,
            detail="ollama executable not found on PATH or in default install locations",
            fixes=[
                "Download and run the Windows installer: https://ollama.com/download/windows",
                "Or install silently via winget:  winget install --id Ollama.Ollama -e",
                "Launch Ollama once (tray icon) and re-run setup.bat.",
            ],
        )
        err("Ollama is not installed - the GPU LLM engine cannot start.")
        return
    ok(f"Ollama binary found: {ollama_exe}")
    try:
        ver = subprocess.run(
            [str(ollama_exe), "--version"], capture_output=True, text=True, timeout=30
        )
        version_line = (ver.stdout or ver.stderr or "").strip().splitlines()
        if version_line:
            info(f"Version: {version_line[0]}")
    except Exception:
        pass

    # -- 2b. daemon reachability ---------------------------------------------------
    if ollama_reachable(host):
        ok(f"Daemon reachable at {host}/api/tags")
    else:
        warn(f"Daemon not reachable at {host} - attempting a detached `ollama serve` ...")
        info("If this keeps failing: start 'Ollama' from the Start Menu (tray app) or run")
        info(f"`ollama serve` manually; the log lands in {OLLAMA_SERVE_LOG}.")
        started = start_ollama_detached(ollama_exe)
        if not started or not wait_for_ollama(host):
            record(
                "ollama_daemon",
                "Ollama Daemon (localhost:11434)",
                False,
                detail=f"no response from {host}/api/tags after the auto-start attempt",
                fixes=[
                    f"Inspect the daemon log: {OLLAMA_SERVE_LOG}",
                    "Start Ollama from the Start Menu, or run `ollama serve` in a terminal.",
                    f"Probe the endpoint manually: curl {host}/api/tags",
                    "Make sure no proxy env vars (HTTP_PROXY/HTTPS_PROXY) intercept localhost.",
                ],
            )
            err("Ollama daemon unreachable - remaining LLM checks are skipped.")
            return
        ok("Daemon is now accepting connections.")

    record(
        "ollama_daemon",
        "Ollama Daemon (localhost:11434)",
        True,
        detail=f"{host}/api/tags responding",
    )

    # -- 2c. model presence + 2d. generation ---------------------------------------
    _ollama_model_check(args, ollama_exe, host)

def _ollama_model_check(args, ollama_exe: Path, host: str) -> None:
    # -- 2c. model presence ---------------------------------------------------
    if not module_ok("requests"):
        record(
            "llm_model",
            f"Model Present ({OLLAMA_MODEL})",
            False,
            detail="requests library unavailable",
            fixes=["python -m pip install requests"],
        )
        err("The 'requests' library is missing - cannot query the Ollama API.")
        return
    import requests

    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json()
    except Exception as exc:
        record(
            "llm_model",
            f"Model Present ({OLLAMA_MODEL})",
            False,
            detail=f"/api/tags request failed: {exc}",
            fixes=["Restart the daemon (ollama serve) and re-run setup.bat."],
        )
        err(f"/api/tags request failed: {exc}")
        return

    models = tags.get("models", [])
    names = [m.get("name", "") for m in models]
    if names:
        info(f"Installed models ({len(names)}): " + ", ".join(names))
    else:
        warn("No models pulled yet.")

    model_entry = next((m for m in models if m.get("name") == OLLAMA_MODEL), None)
    if model_entry is None and args.skip_models:
        record(
            "llm_model",
            f"Model Present ({OLLAMA_MODEL})",
            False,
            detail=f"'{OLLAMA_MODEL}' not present; pull skipped by --skip-models",
            skipped=True,
        )
        warn(f"'{OLLAMA_MODEL}' is missing - pull skipped (--skip-models).")
        _ollama_generation_test(args)
        return
    if model_entry is None:
        warn(f"'{OLLAMA_MODEL}' is not present - pulling it now via subprocess ...")
        if not pull_model(ollama_exe, OLLAMA_MODEL):
            record(
                "llm_model",
                f"Model Present ({OLLAMA_MODEL})",
                False,
                detail="pull failed or was interrupted",
                fixes=[
                    f"Retry manually:  ollama pull {OLLAMA_MODEL}",
                    "Check free disk space (~2 GB) and your internet connection.",
                    "Model page: https://ollama.com/library/llama3.2",
                ],
            )
            err(f"'{OLLAMA_MODEL}' could not be pulled - LLM engine unavailable.")
            return
        try:
            tags = requests.get(f"{host}/api/tags", timeout=10).json()
            models = tags.get("models", [])
            model_entry = next((m for m in models if m.get("name") == OLLAMA_MODEL), None)
        except Exception:
            model_entry = None
        if model_entry is None:
            record(
                "llm_model",
                f"Model Present ({OLLAMA_MODEL})",
                False,
                detail="pull reported success but the model is not listed by /api/tags",
                fixes=[f"Run:  ollama pull {OLLAMA_MODEL}", "Then:  ollama list"],
            )
            err("Model missing from /api/tags after pull - see fixes below.")
            return

    size_gib = float(model_entry.get("size", 0)) / GIB
    record(
        "llm_model",
        f"Model Present ({OLLAMA_MODEL})",
        True,
        detail=f"on disk: {size_gib:.2f} GiB",
    )
    ok(f"'{OLLAMA_MODEL}' is ready ({size_gib:.2f} GiB on disk).")

    # -- 2d. generation + VRAM proof -------------------------------------------
    _ollama_generation_test(args)

# ---------------------------------------------------------------------------
# STEP 3 - Hardware & audio pipeline enumeration
# ---------------------------------------------------------------------------
def _pyaudio_enumeration() -> None:
    if not module_ok("pyaudio"):
        info(
            "PyAudio not importable - skipping its cross-check "
            "(sounddevice is authoritative)."
        )
        return
    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            count = pa.get_device_count()
            info(f"PyAudio cross-check: {count} device(s) visible:")
            usable = 0
            for idx in range(count):
                try:
                    dev = pa.get_device_info_by_index(idx)
                except Exception:
                    continue
                if int(dev.get("maxInputChannels", 0)) <= 0 and int(
                    dev.get("maxOutputChannels", 0)
                ) <= 0:
                    continue
                try:
                    api_name = pa.get_host_api_info_by_index(dev["hostApi"])["name"]
                except Exception:
                    api_name = "?"
                name = str(dev.get("name", "?"))[:44]
                print(
                    f"      [{idx:>2}] {name:<44} "
                    f"in={dev.get('maxInputChannels', 0):>2} "
                    f"out={dev.get('maxOutputChannels', 0):>2}  [{api_name}]"
                )
                usable += 1
            if usable == 0:
                warn("PyAudio loaded but reported no usable devices.")
        finally:
            pa.terminate()
    except Exception as exc:
        warn(f"PyAudio enumeration failed: {exc}")


def _mic_capture_probe(sd) -> None:
    try:
        import numpy as np
    except Exception:
        info("Mic capture probe skipped (numpy unavailable).")
        return
    try:
        info("Probing the default microphone with a 0.5 s capture (16 kHz, mono) ...")
        recording = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        peak = float(np.abs(np.asarray(recording)).max()) if recording.size else 0.0
        if peak < 1e-4:
            warn(f"Capture produced silence (peak {peak:.5f}) - mic muted or privacy-blocked?")
            record(
                "mic_capture",
                "Microphone Capture Probe (0.5 s)",
                False,
                detail=f"silence captured (peak {peak:.5f})",
                critical=False,
                fixes=[
                    "Check the mic is not muted and is the default input device.",
                    "Settings > Privacy & security > Microphone: allow desktop apps.",
                ],
            )
        else:
            record(
                "mic_capture",
                "Microphone Capture Probe (0.5 s)",
                True,
                detail=f"captured audio, peak level {peak:.3f}",
                critical=False,
            )
            ok(f"Mic capture OK (peak level {peak:.3f}).")
    except Exception as exc:
        record(
            "mic_capture",
            "Microphone Capture Probe (0.5 s)",
            False,
            detail=str(exc),
            critical=False,
            fixes=[
                "Close apps holding the microphone exclusively (Teams / Zoom / Discord).",
                "Settings > Privacy & security > Microphone: allow desktop apps.",
            ],
        )
        warn(f"Mic capture probe failed: {exc}")

def step3_audio(args) -> None:
    section("STEP 3/5", "Audio hardware enumeration (microphones / speakers / host APIs)")

    try:
        import sounddevice as sd
    except Exception as exc:
        detail = f"sounddevice unavailable: {exc}"
        record(
            "mic",
            "Microphone Input Device Detected",
            False,
            detail=detail,
            fixes=["python -m pip install sounddevice"],
        )
        record(
            "speaker",
            "Speaker Output Device Detected",
            False,
            detail=detail,
            fixes=["python -m pip install sounddevice"],
        )
        record(
            "mic_capture",
            "Microphone Capture Probe (0.5 s)",
            False,
            detail=detail,
            critical=False,
            fixes=["python -m pip install sounddevice"],
        )
        err("sounddevice could not be imported - audio enumeration impossible.")
        return

    # -- 3a. host APIs -----------------------------------------------------------
    try:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        info(
            f"sounddevice enumerates {len(hostapis)} host API(s), "
            f"{len(devices)} device(s):"
        )
        for idx, api in enumerate(hostapis):
            in_idx = int(api.get("default_input_device", -1))
            out_idx = int(api.get("default_output_device", -1))
            in_name = str(devices[in_idx]["name"]) if in_idx >= 0 else "-"
            out_name = str(devices[out_idx]["name"]) if out_idx >= 0 else "-"
            print(
                f"      [{idx:>2}] {str(api['name'])[:30]:<30} "
                f"in: {str(in_name)[:38]:<38} | out: {str(out_name)[:38]}"
            )
    except Exception as exc:
        warn(f"Host API enumeration failed: {exc}")

    # -- 3b. default input ---------------------------------------------------------
    try:
        dev = sd.query_devices(kind="input")
        api_name = sd.query_hostapis()[int(dev["hostapi"])]["name"]
        rate = float(dev.get("default_samplerate", 0.0))
        record(
            "mic",
            "Microphone Input Device Detected",
            True,
            detail=(
                f"{dev['name']} [{api_name}, {rate:g} Hz, "
                f"{int(dev.get('max_input_channels', 0))} in-ch]"
            ),
        )
        ok(f"Default INPUT : {dev['name']}  [{api_name}]  default {rate:g} Hz")
    except Exception as exc:
        record(
            "mic",
            "Microphone Input Device Detected",
            False,
            detail=f"no default input device ({exc})",
            fixes=[
                "Connect/enable a microphone: Settings > System > Sound > Input.",
                "Settings > Privacy & security > Microphone: allow desktop apps.",
            ],
        )
        err(f"No default input device: {exc}")

    # -- 3c. default output ----------------------------------------------------------
    try:
        dev = sd.query_devices(kind="output")
        api_name = sd.query_hostapis()[int(dev["hostapi"])]["name"]
        rate = float(dev.get("default_samplerate", 0.0))
        record(
            "speaker",
            "Speaker Output Device Detected",
            True,
            detail=(
                f"{dev['name']} [{api_name}, {rate:g} Hz, "
                f"{int(dev.get('max_output_channels', 0))} out-ch]"
            ),
        )
        ok(f"Default OUTPUT: {dev['name']}  [{api_name}]  default {rate:g} Hz")
    except Exception as exc:
        record(
            "speaker",
            "Speaker Output Device Detected",
            False,
            detail=f"no default output device ({exc})",
            fixes=["Connect speakers/headphones: Settings > System > Sound > Output."],
        )
        err(f"No default output device: {exc}")

    # -- 3d. pyaudio cross-check --------------------------------------------------------
    _pyaudio_enumeration()

    # -- 3e. 0.5 s mic capture probe (non-critical) ----------------------------------------
    _mic_capture_probe(sd)

# ---------------------------------------------------------------------------
# STEP 4 - CPU STT / TTS smoke tests (first-run weight downloads happen here)
# ---------------------------------------------------------------------------
def _write_test_tone_wav(
    path: Path, seconds: float = 1.0, freq: float = 440.0, rate: int = 16000
) -> Path:
    import numpy as np

    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False, dtype=np.float64)
    tone = (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    pcm = (tone * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return path


def smoke_faster_whisper(args) -> None:
    label = "CPU STT Engine (faster-whisper int8)"
    if args.skip_models:
        record("stt", label, False, detail="skipped by --skip-models", skipped=True)
        warn("STT smoke test skipped (--skip-models).")
        return
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        record(
            "stt",
            label,
            False,
            detail=f"import failed: {exc}",
            fixes=["python -m pip install faster-whisper"],
        )
        err(f"faster-whisper unavailable: {exc}")
        return

    download_root = MODELS_DIR / "faster-whisper"
    download_root.mkdir(parents=True, exist_ok=True)
    info(
        f'Loading WhisperModel("{WHISPER_MODEL_SIZE}", device="cpu", '
        f'compute_type="int8", cpu_threads={WHISPER_CPU_THREADS}) - first run '
        f"downloads ~{WHISPER_WEIGHTS_MB_ESTIMATE} MB of weights ..."
    )
    t0 = time.perf_counter()
    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            cpu_threads=WHISPER_CPU_THREADS,
            num_workers=1,
            download_root=str(download_root),
        )
    except Exception as exc:
        record(
            "stt",
            label,
            False,
            detail=f"model load failed: {exc}",
            fixes=[
                f"Delete partial weights and retry: {download_root}",
                "First download needs internet access to huggingface.co.",
                "python -m pip install -U faster-whisper",
            ],
        )
        err(f"faster-whisper model load failed: {exc}")
        return
    load_s = time.perf_counter() - t0
    ok(
        f"faster-whisper '{WHISPER_MODEL_SIZE}' (int8) loaded in {load_s:.1f}s on CPU "
        f"({WHISPER_CPU_THREADS} threads)."
    )

    try:
        SMOKE_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = _write_test_tone_wav(SMOKE_DIR / "smoke_tone_440hz.wav")
        t0 = time.perf_counter()
        segments, seg_info = model.transcribe(str(wav_path), language="en", beam_size=1)
        n_segments = 0
        heard_parts = []
        for seg in segments:
            n_segments += 1
            if seg.text.strip():
                heard_parts.append(seg.text.strip())
        inf_s = time.perf_counter() - t0
        heard = " ".join(heard_parts)
        record(
            "stt",
            label,
            True,
            detail=(
                f"loaded in {load_s:.1f}s; inference {inf_s:.2f}s, "
                f"lang={seg_info.language}, segments={n_segments}"
            ),
        )
        heard_note = f", heard: '{heard}'" if heard else ""
        ok(
            f"CPU transcription pass complete in {inf_s:.2f}s "
            f"(lang={seg_info.language}, {n_segments} segment(s){heard_note})."
        )
    except Exception as exc:
        record(
            "stt",
            label,
            False,
            detail=f"weights loaded ({load_s:.1f}s) but inference failed: {exc}",
            fixes=[
                "python -m pip install -U ctranslate2 faster-whisper",
                "CTranslate2 needs a CPU with AVX support (all modern x64 CPUs have it).",
            ],
        )
        err(f"faster-whisper inference pass failed: {exc}")
    finally:
        try:
            del model
        except Exception:
            pass

def _as_float_numpy(chunk):
    import numpy as np

    if hasattr(chunk, "detach"):  # torch tensor from the KPipeline
        return chunk.detach().cpu().float().numpy()
    return np.asarray(chunk, dtype=np.float32)


def _download_file(url: str, dest: Path, human_name: str) -> None:
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        info(f"{human_name} already cached: {dest}")
        return
    import requests

    info(f"Downloading {human_name} ...")
    tmp = Path(str(dest) + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print(
                            f"\r      {done / GIB:6.2f} GiB / {total / GIB:6.2f} GiB "
                            f"({pct:5.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r      {done / GIB:6.2f} GiB downloaded", end="", flush=True)
        print()
    except Exception:
        print()
        try:
            tmp.unlink()
        except Exception:
            pass
        raise
    tmp.replace(dest)
    ok(f"Saved {human_name} -> {dest}")


def _kokoro_onnx_fallback(args) -> tuple:
    """Lightweight offline fallback: ONNX Runtime inference, no torch needed."""
    info("Falling back to the lightweight offline engine: kokoro-onnx (ONNX Runtime, CPU).")
    if not module_ok("kokoro_onnx"):
        if args.skip_install:
            raise RuntimeError("kokoro-onnx is not installed and --skip-install was given")
        if not pip_install("kokoro-onnx"):
            raise RuntimeError("pip could not install kokoro-onnx")
    KOKORO_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = KOKORO_ONNX_DIR / "kokoro-v1.0.onnx"
    voices_path = KOKORO_ONNX_DIR / "voices-v1.0.bin"
    _download_file(KOKORO_ONNX_MODEL_URL, onnx_path, "kokoro-v1.0.onnx (~310 MB)")
    _download_file(KOKORO_ONNX_VOICES_URL, voices_path, "voices-v1.0.bin (~27 MB)")
    from kokoro_onnx import Kokoro

    tts = Kokoro(str(onnx_path), str(voices_path))
    t0 = time.perf_counter()
    samples, sample_rate = tts.create(
        TEST_UTTERANCE, voice="af_heart", speed=1.0, lang="en-us"
    )
    audio = _as_float_numpy(samples)
    engine = f"kokoro-onnx {module_imports('kokoro_onnx')[1]} (ONNX Runtime, CPU)"
    return audio, float(sample_rate), engine, time.perf_counter() - t0

def _playback_check(args, sd, audio, sample_rate: float) -> None:
    label = "Speaker Playback Test (kokoro sample)"
    if args.skip_playback:
        info("Playback skipped (--skip-playback); synthesis itself already passed.")
        record(
            "playback",
            label,
            False,
            detail="skipped by --skip-playback",
            critical=False,
            skipped=True,
        )
        return
    if sd is None:
        record(
            "playback",
            label,
            False,
            detail="sounddevice unavailable",
            critical=False,
            fixes=["python -m pip install sounddevice"],
        )
        return
    try:
        info("Playing the synthesized sample through the default output device ...")
        sd.play(audio, int(sample_rate))
        sd.wait()
        time.sleep(0.2)
        record(
            "playback",
            label,
            True,
            detail=f"played {len(audio) / float(sample_rate):.2f}s @ {int(sample_rate)} Hz",
            critical=False,
        )
        ok("Playback complete - default output device is wired up.")
    except Exception as exc:
        record(
            "playback",
            label,
            False,
            detail=str(exc),
            critical=False,
            fixes=[
                "Select the correct default output: Settings > System > Sound > Output.",
                "Close apps holding exclusive-mode audio (ASIO drivers, VoIP clients).",
            ],
        )
        warn(f"Playback failed: {exc}")


def smoke_kokoro(args, sd) -> None:
    label = "CPU TTS Engine (kokoro)"
    if args.skip_models:
        record("tts", label, False, detail="skipped by --skip-models", skipped=True)
        record(
            "playback",
            "Speaker Playback Test (kokoro sample)",
            False,
            detail="skipped by --skip-models",
            critical=False,
            skipped=True,
        )
        warn("TTS smoke test skipped (--skip-models).")
        return

    audio = None
    sample_rate = float(TTS_SAMPLE_RATE_HZ)
    engine = ""
    synth_s = 0.0

    # -- primary: kokoro (torch) KPipeline on CPU -------------------------------
    if module_ok("kokoro"):
        try:
            from kokoro import KPipeline

            info(
                "Initializing kokoro KPipeline on CPU (first run downloads "
                "~330 MB of weights from HuggingFace) ..."
            )
            t0 = time.perf_counter()
            pipeline = KPipeline(lang_code="a")  # American English
            chunks = pipeline(TEST_UTTERANCE, voice="af_heart", speed=1.0)
            for item in chunks:
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    chunk = item[2]
                else:
                    chunk = getattr(item, "audio", item)
                audio = _as_float_numpy(chunk)
                break
            synth_s = time.perf_counter() - t0
            engine = f"kokoro {module_imports('kokoro')[1]} (torch CPU)"
        except Exception as exc:
            warn(f"kokoro (torch) pipeline failed: {exc}")
            audio = None

    # -- fallback: kokoro-onnx (ONNX Runtime, no torch at all) --------------------
    if audio is None:
        try:
            audio, sample_rate, engine, synth_s = _kokoro_onnx_fallback(args)
        except Exception as exc:
            record(
                "tts",
                label,
                False,
                detail=f"both kokoro backends failed; last error: {exc}",
                fixes=[
                    "python -m pip install -U kokoro          (torch backend)",
                    "python -m pip install -U kokoro-onnx     (ONNX fallback)",
                    "First synthesis needs internet access to huggingface.co.",
                    f"Clear partial weights and retry: {KOKORO_ONNX_DIR}",
                ],
            )
            err(f"TTS smoke test failed: {exc}")
            return

    duration_s = len(audio) / float(sample_rate)
    record(
        "tts",
        label,
        True,
        detail=(
            f"{engine}; synthesized {duration_s:.2f}s @ {int(sample_rate)} Hz "
            f"(synthesis took {synth_s:.2f}s)"
        ),
    )
    ok(
        f"Kokoro synthesized '{TEST_UTTERANCE}' -> {duration_s:.2f}s of audio on CPU "
        f"via {engine}."
    )
    _playback_check(args, sd, audio, sample_rate)


def step4_smoke(args) -> None:
    section("STEP 4/5", "CPU STT / TTS smoke tests (weight downloads + playback)")
    sd = None
    try:
        import sounddevice as sd_module

        sd = sd_module
    except Exception:
        sd = None
    smoke_faster_whisper(args)
    smoke_kokoro(args, sd)

# ---------------------------------------------------------------------------
# STEP 5 - Readiness report
# ---------------------------------------------------------------------------
SUMMARY_ROWS = [
    ("Python & Core Libraries", ("python", "core")),
    ("Microphone Input Device Detected", ("mic",)),
    ("Speaker Output Device Detected", ("speaker",)),
    ("CPU STT Engine (faster-whisper int8)", ("stt",)),
    ("CPU TTS Engine (kokoro)", ("tts",)),
    (
        "GPU LLM Engine (Ollama llama3.2:3b via RTX 3050 Ti)",
        ("ollama_daemon", "llm_model", "llm_gpu"),
    ),
]

EXTRA_ROWS = [
    ("CPU-only PyTorch (VRAM guardrail)", ("torch_cpu",)),
    ("PyAudio / PortAudio Bindings (optional)", ("pyaudio",)),
    ("Microphone Capture Probe (0.5 s)", ("mic_capture",)),
    ("Speaker Playback Test (kokoro sample)", ("playback",)),
]


def _row_verdict(keys) -> tuple:
    parts = [CHECKS[k] for k in keys if k in CHECKS]
    if not parts:
        return "skip", "not evaluated"
    if all(p.skipped for p in parts):
        return "skip", " / ".join(p.detail or "skipped" for p in parts)
    if all(p.passed or p.skipped for p in parts):
        return "pass", " / ".join(p.detail for p in parts if p.detail)
    failed = [p for p in parts if not p.passed and not p.skipped]
    return "fail", " / ".join(p.detail or "failed" for p in failed)


def _print_row(label: str, verdict: str, detail: str) -> None:
    if verdict == "pass":
        sym, color = "[✓]", C.GREEN
    elif verdict == "fail":
        sym, color = "[✗]", C.RED
    else:
        sym, color = "[~]", C.YELLOW
    shown = detail if len(detail) <= 44 else detail[:41] + "..."
    print(f"   {_c(sym, color + C.BOLD)} {label:<51} {_c(shown, C.GREY)}")


def step5_report() -> int:
    section("STEP 5/5", "Readiness summary & exit code")

    print()
    print("  " + _c("=" * 78, C.CYAN))
    print(
        "   "
        + _c("PHASE 1 READINESS REPORT - offline local Teaching Assistant", C.BOLD + C.CYAN)
    )
    print("  " + _c("=" * 78, C.CYAN))
    print()

    for label, keys in SUMMARY_ROWS:
        verdict, detail = _row_verdict(keys)
        _print_row(label, verdict, detail)
    rule()
    for label, keys in EXTRA_ROWS:
        if all(k not in CHECKS for k in keys):
            continue
        verdict, detail = _row_verdict(keys)
        _print_row(label, verdict, detail)

    critical_failures = []
    noncritical_failures = []
    for check in CHECKS.values():
        if check.skipped or check.passed:
            continue
        if check.critical:
            critical_failures.append(check)
        else:
            noncritical_failures.append(check)

    if critical_failures or noncritical_failures:
        print()
        print("   " + _c("ACTION REQUIRED - apply these fixes in order:", C.BOLD + C.RED))
        n = 0
        for check in critical_failures + noncritical_failures:
            n += 1
            print(_c(f"   {n}. {check.label}", C.BOLD))
            for fix in check.fixes:
                print(f"        - {fix}")
    else:
        print()
        ok("All critical subsystems are 100% operational.")
        print()
        info("Next phase: real-time orchestration (VAD + turn-taking) can now be written against:")
        info("  - faster-whisper 'base.en' int8 on CPU (4 threads)")
        info("  - kokoro TTS on CPU (24 kHz output)")
        info("  - Ollama llama3.2:3b resident on the RTX 3050 Ti (keep_alive 15m)")

    print()
    rule("=")
    passed_n = sum(1 for c in CHECKS.values() if c.passed and not c.skipped)
    skipped_n = sum(1 for c in CHECKS.values() if c.skipped)
    print(
        f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   "
        f"Checks: {passed_n} passed / {len(critical_failures)} critical failed / "
        f"{len(noncritical_failures)} advisory failed / {skipped_n} skipped"
    )
    return 0 if not critical_failures else 1

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="verify_env.py",
        description=(
            "Phase 1 setup & diagnostics for the offline local Teaching Assistant "
            "(Windows 11, Ryzen 7 5800U + RTX 3050 Ti 4GB)."
        ),
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="skip all pip provisioning (assume packages are already present)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="skip heavy downloads & STT/TTS/LLM smoke tests (daemon ping and "
        "audio enumeration still run)",
    )
    parser.add_argument(
        "--skip-playback",
        action="store_true",
        help="synthesize the TTS sample but do not play it through speakers",
    )
    parser.add_argument(
        "--keep-cuda-torch",
        action="store_true",
        help="do NOT swap a CUDA PyTorch build for the CPU-only wheel "
        "(STT/TTS still pinned to device='cpu')",
    )
    parser.add_argument(
        "--host",
        default=OLLAMA_DEFAULT_HOST,
        help=f"Ollama base URL (default: {OLLAMA_DEFAULT_HOST})",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _force_utf8_stdio()
    _enable_windows_vt()

    host_facts = {
        "cpu": cpu_name(),
        "ram": f"{total_ram_gib():.1f} GiB",
        "gpu": gpu_summary(),
        "py": platform.python_version(),
        "exe": sys.executable,
    }
    banner(host_facts)
    ensure_disk_space()
    info("Guardrail: CUDA is reserved for the Ollama LLM; STT/TTS run on CPU only.")

    try:
        if not step1_runtime(args):
            return step5_report()
        step2_ollama(args)
        step3_audio(args)
        step4_smoke(args)
        return step5_report()
    except KeyboardInterrupt:
        print()
        err("Interrupted by user (Ctrl+C).")
        return 130


if __name__ == "__main__":
    sys.exit(main())