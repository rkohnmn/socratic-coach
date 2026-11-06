#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 TA Phase 4 - `vision_tracker.py`
 Optional low-overhead blackboard/whiteboard vision companion (CPU-only)
==============================================================================

Hardware context (verified in Phases 1-3):
    NVIDIA RTX 3050 Ti (4 GB VRAM) -> Ollama `llama3.2:3b` ONLY (~2.2 GB).
                                      No vision model is ever given CUDA here.
    AMD Ryzen 7 5800U (8C/16T)     -> camera IO, frame differencing, SSIM,
                                      diagram heuristics and OCR all run here.

VRAM guardrail
--------------
No torch/CUDA vision model is used at all. OpenCV runs on the CPU, and the one
optional multimodal engine (`--ocr ollama`) is forced to CPU offload with
`options.num_gpu = 0`, so `llama3.2:3b` keeps the GPU to itself. If Moondream is
ever pulled it munches *system* RAM, never the 3050 Ti's 4 GB.

Architecture
------------
  [OpenCV capture thread]  camera 720p, throttled to 1.5 FPS
        |  crop ROI -> grayscale -> 480px -> GaussianBlur          (~0.25 ms)
        v
  [scene-change FSM]  diff vs keyframe (MAD + changed-pixel ratio + SSIM)
        |  significant change -> MOTION ... no motion for 3.0 s -> SETTLE
        v
  [OCR worker thread]  runs only on a settled board, never per frame
        v
  [BoardSnapshot deque, 120 s]  --thread-safe--> get_visual_context()

Scene-change FSM
----------------
  CALIBRATING  measure the sensor noise floor (like Phase 2's room calibration)
  SCANNING     board is stable; watch for a significant structural change
  MOTION       board is being written on; track the last real motion instant
  SETTLE       motion stopped >= STABILIZE_S (3.0 s) -> OCR the settled frame

Heuristics that keep the signal clean:
  * sub-threshold frame deltas are treated as sensor noise and dropped
  * a settled frame that matches the last *emitted* keyframe (SSIM >= 0.97) is
    discarded as a transient occlusion - this is what kills "teacher walked past
    the board" events, because the picture returns to the same state afterwards
  * a non-stop writer is force-flushed after MAX_DWELL_S so context never goes
    stale during a long derivation

Measured on this host (Ryzen 7 5800U, 1280x720 synthetic board, 8 text lines):
  * Windows OCR (Windows.Media.Ocr, built into Windows 11) .. 850 ms/call
    zero install, 128/128 accurate on the probe board; the only quirk is that
    `P(A|B)` is read as `P(AIB)`, which `normalize_math()` repairs.
  * PaddleOCR 3.7.0 .. 22.2 s/call, and it CRASHES on this machine unless
    `enable_mkldnn=False` is passed (PaddlePaddle 3.3.1 oneDNN bug:
    "ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>]"). Kept as an opt-in accuracy
    engine via `--ocr paddle`, never as the default.
  * pytesseract .. requires a tesseract.exe binary that is NOT installed here,
    so the engine self-disables with an actionable message.
  * ONNX/moondream on the GPU .. deliberately not used; see VRAM guardrail.

Steady-state cost per sampled frame is ~0.3 ms of diffing plus a 1.5 FPS read,
and OCR is event-driven, which keeps average CPU far below the 10% budget.

Standalone CLI:
    python vision_tracker.py                     # live dashboard, Ctrl+C stops
    python vision_tracker.py --gui               # + annotated preview window
    python vision_tracker.py --calibrate         # drag a ROI, save it to JSON
    python vision_tracker.py --camera 1 --fps 2 --duration 60
    python vision_tracker.py --check             # engines/ROI/deps diagnostics
    python vision_tracker.py --from-image board.png   # offline single-image test

Import interface for Phase 3 (`main.py`):
    from vision_tracker import VisionTracker
    vt = VisionTracker().start()
    ctx = vt.get_visual_context()          # "[BOARD CONTEXT]: ..." or None
    vt.stop()

Exit codes: 0 clean stop | 1 fatal error | 2 no usable OCR engine | 130 Ctrl+C
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import numpy as np
# OpenCV is the one hard third-party requirement. It is imported defensively so
# `--check` and the import interface still work on a machine where it is absent.
try:
    import cv2
    CV2_VERSION = getattr(cv2, "__version__", "unknown")
    CV2_IMPORT_ERROR: Exception | None = None
except Exception as _exc:          # pragma: no cover - depends on the host
    cv2 = None
    CV2_VERSION = "missing"
    CV2_IMPORT_ERROR = _exc


def ensure_cv2(auto_install: bool = True) -> bool:
    """Make `cv2` importable, optionally pip-installing opencv-python.

    Mirrors the auto-install behaviour Phases 2/3 use for pyttsx3.
    """
    global cv2, CV2_VERSION, CV2_IMPORT_ERROR
    if cv2 is not None:
        return True
    if not auto_install:
        return False
    log_warn("OpenCV is not installed - attempting `pip install opencv-python` ...")
    cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check",
           "install", "opencv-python"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              creationflags=SUBPROC_FLAGS)
    except Exception as exc:
        CV2_IMPORT_ERROR = exc
        log_err(f"pip install opencv-python failed to run: {exc}")
        return False
    if proc.returncode != 0:
        CV2_IMPORT_ERROR = RuntimeError(proc.stderr[-400:])
        log_err("pip could not install opencv-python: "
                + (proc.stderr or "").strip()[-300:])
        return False
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        CV2_VERSION = getattr(cv2, "__version__", "unknown")
        CV2_IMPORT_ERROR = None
        log_ok(f"OpenCV {CV2_VERSION} installed.")
        return True
    except Exception as exc:
        CV2_IMPORT_ERROR = exc
        log_err(f"OpenCV still not importable after install: {exc}")
        return False


def require_cv2(auto_install: bool = True) -> None:
    """Abort with an actionable message if cv2 cannot be obtained."""
    if ensure_cv2(auto_install=auto_install):
        return
    raise RuntimeError(
        "OpenCV (cv2) is required for the vision companion. Fix: "
        "`python -m pip install opencv-python`  (already validated on this "
        f"host: opencv-python 5.0.0.93). Underlying error: {CV2_IMPORT_ERROR}"
    )


def silence_cv2_logs() -> None:
    """Suppress OpenCV's stderr chatter during camera index probing.

    Probing indexes 0-4 makes the DSHOW/MSMF/obsensor backends emit a wall of
    "[ WARN:0@...] VIDEOIO(DSHOW): backend is generally available but can't be
    used to capture by index" and "[ERROR:0@...] obsensor_uvc_stream_channel
    Camera index out of range" lines that bury our own diagnostics. Those
    backends log at ERROR, so the level is set to SILENT - we report camera
    problems ourselves in _list_cameras()/_open_camera().
    """
    if cv2 is None:
        return
    try:
        logging_utils = getattr(cv2, "utils", None)
        logging_utils = getattr(logging_utils, "logging", None)
        if logging_utils is not None:
            logging_utils.setLogLevel(logging_utils.LOG_LEVEL_SILENT)
            return
    except Exception:
        pass
    try:      # OpenCV < 4.5 exposes no python API; the env var must be pre-set
        os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    except Exception:
        pass
# ---------------------------------------------------------------------------
# Crash-isolated camera probing
# ---------------------------------------------------------------------------
# Enumerating several camera indexes through OpenCV in a single Windows process
# can raise a hard access violation (0xC0000005) that no Python `except` can
# catch. Verified on this host: `--check` died inside _list_cameras() while
# scanning indexes 0-4 (the DSHOW/obsensor backends fault on out-of-range
# indexes). Running every probe in a child process means a crashing probe costs
# only that index and the parent survives to report it.
CAMERA_PROBE_SOURCE = r'''
import json, os, sys, time
try:
    import cv2
except Exception:
    sys.exit(4)
try:
    _lg = getattr(getattr(cv2, "utils", None), "logging", None)
    if _lg is not None:
        _lg.setLogLevel(_lg.LOG_LEVEL_SILENT)
except Exception:
    pass
index = int(sys.argv[1]); width = int(sys.argv[2]); height = int(sys.argv[3])
wait = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
if os.name == "nt":
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
else:
    backends = [(cv2.CAP_ANY, "ANY")]
for backend, name in backends:
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0):
                sys.stdout.write(json.dumps({
                    "index": index, "w": int(frame.shape[1]),
                    "h": int(frame.shape[0]), "backend": name}) + "\n")
                sys.stdout.flush()
                sys.exit(0)
            time.sleep(0.05)
    except Exception:
        pass
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
sys.exit(3)
'''


def probe_camera_index(index: int, width: int = 0, height: int = 0,
                       wait: float = 1.2):
    """Probe one camera index in an isolated child process.

    Returns (index, width, height, backend) or None. A child that hangs is
    killed by the timeout; a child that segfaults returns a non-zero code. Both
    are reported as "no camera", which is exactly the truth we want.

    `width`/`height` default to CAPTURE_WIDTH/CAPTURE_HEIGHT at call time (they
    are resolved here, not in the signature, because this helper is defined
    above the constants block).
    """
    width = int(width or CAPTURE_WIDTH)
    height = int(height or CAPTURE_HEIGHT)
    payload = base64.b64encode(CAMERA_PROBE_SOURCE.encode("utf-8")).decode("ascii")
    code = f"import base64;exec(base64.b64decode('{payload}'))"
    cmd = [sys.executable, "-c", code, str(int(index)), str(int(width)),
           str(int(height)), str(float(wait))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=wait + 25.0, creationflags=SUBPROC_FLAGS)
    except subprocess.TimeoutExpired:
        LOG.warning("camera probe %s timed out - ignoring that index", index)
        return None
    except Exception as exc:
        LOG.warning("camera probe %s could not run: %s", index, exc)
        return None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        return (int(data.get("index", index)), int(data["w"]), int(data["h"]),
                str(data.get("backend", "?")))
    return None



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parent
LOGS_DIR = WORKSPACE / "logs"
BOARD_DIR = LOGS_DIR / "board"
ROI_CONFIG_PATH = WORKSPACE / "vision_roi.json"
VISION_LOG_PATH = LOGS_DIR / "vision.log"
OLLAMA_HOST_DEFAULT = "http://localhost:11434"

CAMERA_INDEX_DEFAULT = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 1.5               # 1-2 FPS: enough for handwriting, near-zero CPU

ANALYZE_WIDTH = 480             # ROI is downscaled to this width before diffing
BLUR_KSIZE = 11
BLUR_SIGMA = 1.5

# Scene-change thresholds - EVERY value below is measured, not guessed
# (see _probe_temporal.py). On this camera pairwise frame differencing is
# unusable: sparse impulse noise puts up to 9.3% of pixels above 15 grey levels
# (while MAD is only ~1.25), because that noise is heavy-tailed and temporally
# independent. Findings that drive the design:
#   * a 9-frame temporal median reference cuts the noise tail p99 from 5.4%
#     to 0.005% (~1000x), because independent impulses cancel in a median
#   * MAD cannot separate signal from noise (noise 3.6 vs a written line 0.54)
#     so MAD is used ONLY as a stillness indicator, never as significance
#   * SSIM change cannot separate them either (noise 0.032 vs line 0.012) so
#     SSIM is used ONLY for the occlusion/revert test, with a wider margin
#   * the changed-pixel RATIO does separate them: a written line 0.90% vs
#     noise 0.047% (19x) and vs the median-reference tail 0.005% (~180x)
# => significance is ratio-driven; ratio and MAD floors are calibrated at
#    startup against the real board (see NOISE_CALIBRATION_S).
PIXEL_DELTA = 15                # grey levels counted as "this pixel moved"
SIG_RATIO_FLOOR = 0.0015        # 0.15% of ROI - absolute floor, clears the
                                # 0.005% median-reference noise tail by ~30x
SIG_RATIO_MULT = 6.0            # ...or 6x the ratio noise measured at startup
SIG_MAD_MIN = 0.15              # keeps a 1-line change (MAD 0.54) valid while
                                # still rejecting a frozen/duplicate frame
STILL_MAD_DEFAULT = 2.0         # measured p99 of the median-reference MAD
NOISE_CALIBRATION_S = 3.0       # sensor noise-floor sampling window
NOISE_MULTIPLIER = 3.0          # still_eps = clamp(noise * 3, 0.4 .. 3.0)
TEMPORAL_MEDIAN_WINDOW = 9      # frames in the rolling median reference
                                # (9 -> ratio p99 0.005%, MAD p99 1.78)

STABILIZE_S = 3.0               # "no motion for >3 s" -> board updated
MAX_DWELL_S = 90.0              # force-flush a relentlessly busy writer
REVERT_SSIM = 0.95              # settled frame matching the last one we read
                                # means a transient occlusion (SSIM 0.95 = 5%
                                # change). Sits safely above the measured
                                # frame-vs-frame SSIM noise (max 0.0276) yet
                                # still catches a person crossing the board.
MIN_OCR_INTERVAL_S = 2.0        # hard floor between two OCR calls

OCR_ENGINES = ("winocr", "paddle", "tesseract", "ollama", "none")
OCR_MIN_SCORE = 0.45            # paddle/tesseract confidence filter
OCR_MIN_CHARS = 2               # drop 1-char OCR debris

CONTEXT_WINDOW_S = 120.0        # board snapshots retained for the LLM
MAX_CONTEXT_CHARS = 420         # keep the injected context compact
MAX_SUMMARY_ITEMS = 6

CV_THREADS_DEFAULT = 2          # whisper keeps 4, kokoro takes 8 during playback
JOB_QUEUE_MAXSIZE = 8

WINOCR_TIMEOUT_S = 45.0
PADDLE_TIMEOUT_S = 180.0
OLLAMA_VISION_TIMEOUT_S = 120.0
DEFAULT_OLLAMA_VISION_MODEL = "moondream:1.8b"

# Phase 5 HUD push target: matches webui.py's loopback defaults.
WEB_HOST_DEFAULT = "127.0.0.1"
WEB_PORT_DEFAULT = 8765

SUBPROC_FLAGS = 0
if os.name == "nt":
    SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PREVIEW_TITLE = "TA Phase 4 - board monitor  (q = quit)"
DASH_INTERVAL_S = 1.0           # console status-line refresh period

# ---------------------------------------------------------------------------
# Console / logging helpers (same visual language as Phases 1-3)
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
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _enable_windows_vt() -> None:
    global USE_COLOR
    if os.name != "nt":
        return
    try:
        os.system("")
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            USE_COLOR = False
            return
        if not (mode.value & 0x0004):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        USE_COLOR = False


def _c(text: str, *codes: str) -> str:
    if not USE_COLOR or not codes:
        return text
    return "".join(codes) + text + C.RESET


def _ts() -> str:
    now = datetime.now()
    return f"{now:%H:%M:%S}.{now.microsecond // 1000:03d}"


def log_info(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[i]', C.CYAN)} {msg}", flush=True)


def log_ok(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[OK]', C.GREEN + C.BOLD)} {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[!]', C.YELLOW + C.BOLD)} {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[X]', C.RED + C.BOLD)} {msg}", flush=True)


def log_board(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[BOARD UPDATED]', C.BLUE + C.BOLD)} {msg}",
          flush=True)


LOG = logging.getLogger("ta.vision")


def _setup_file_log(verbose: bool = False) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(VISION_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s", "%H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.propagate = False
# ---------------------------------------------------------------------------
# Image metrics: mean-abs-diff, changed-pixel ratio and SSIM (no skimage)
# ---------------------------------------------------------------------------
_SSIM_K = (11, 11)
_SSIM_SIGMA = 1.5
_SSIM_C1 = (0.01 * 255.0) ** 2       # 6.5025
_SSIM_C2 = (0.03 * 255.0) ** 2       # 58.5225


def ssim_score(a: "np.ndarray", b: "np.ndarray") -> float:
    """Mean structural similarity of two grey images, in [-1, 1].

    Implemented with cv2.GaussianBlur so the module needs no skimage. Validated
    against skimage.metrics.structural_similarity on this host: 0.9717 here vs
    0.9704 there for a pasted-block change; identical images give exactly 1.0.
    """
    if a.shape != b.shape:
        raise ValueError("ssim_score: shape mismatch")
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    k, s = _SSIM_K, _SSIM_SIGMA
    mu_a = cv2.GaussianBlur(a, k, s)
    mu_b = cv2.GaussianBlur(b, k, s)
    mu_aa = mu_a * mu_a
    mu_bb = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_aa = cv2.GaussianBlur(a * a, k, s) - mu_aa
    sig_bb = cv2.GaussianBlur(b * b, k, s) - mu_bb
    sig_ab = cv2.GaussianBlur(a * b, k, s) - mu_ab
    num = (2.0 * mu_ab + _SSIM_C1) * (2.0 * sig_ab + _SSIM_C2)
    den = (mu_aa + mu_bb + _SSIM_C1) * (sig_aa + sig_bb + _SSIM_C2)
    return float((num / den).mean())


@dataclass
class SceneThresholds:
    """Calibrated significance/stillness limits for this camera + board.

    Calibrated at startup against the real board (see NOISE_CALIBRATION_S)
    instead of being hard-coded, because the correct numbers depend on sensor
    noise, lighting and ROI size.
    """
    ratio_sig: float = SIG_RATIO_FLOOR   # changed-pixel ratio -> "board changed"
    mad_min: float = SIG_MAD_MIN         # mean-abs-diff floor (frozen-frame guard)
    still_eps: float = STILL_MAD_DEFAULT  # MAD below this -> the board is still
    ratio_noise: float = 0.0             # measured noise ratio (p95), for logs

    def describe(self) -> str:
        return (f"ratio>={self.ratio_sig * 100:.3f}%  "
                f"mad>={self.mad_min:.2f}  still_eps={self.still_eps:.2f}")


@dataclass
class FrameDelta:
    """Result of comparing a prepared ROI against the temporal median reference."""
    mad: float                # mean absolute difference, 0..255
    ratio: float              # fraction of pixels changed by >= PIXEL_DELTA
    ssim: float               # 1.0 == identical
    change: float             # 1.0 - ssim

    def significant(self, th: "SceneThresholds | None" = None) -> bool:
        """Layer 1: did real structure move, or is this just sensor noise?

        Ratio-driven on purpose. Measured on this host against a live camera:
        a single written line moves 0.90% of ROI pixels while frame-to-frame
        sensor noise moves 0.047% and the median-reference noise tail only
        0.005% - a 19x..180x margin. MAD and SSIM could NOT separate them
        (noise MAD 3.6 vs line 0.54; noise SSIM change 0.032 vs line 0.012),
        so those two are used only as stillness / occlusion indicators.

        `th` defaults to the pre-calibration floors so the helpers stay usable
        standalone; the tracker always passes its calibrated thresholds.
        """
        th = th or SceneThresholds()
        return self.ratio >= th.ratio_sig and self.mad >= th.mad_min


def frame_delta(a: "np.ndarray", b: "np.ndarray") -> FrameDelta:
    """Cheap structural comparison of two prepared grey ROI images."""
    diff = cv2.absdiff(a, b)
    mad = float(diff.mean())
    ratio = float((diff >= PIXEL_DELTA).mean())
    ssim = ssim_score(a, b)
    return FrameDelta(mad=mad, ratio=ratio, ssim=ssim, change=1.0 - ssim)


def prepare_roi(bgr: "np.ndarray", width: int = ANALYZE_WIDTH) -> "np.ndarray":
    """Crop -> grey -> resize to `width` -> blur. Basis for all diffing."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        raise ValueError("prepare_roi: empty frame")
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = grey.shape[:2]
    if width and w > int(width):
        new_h = max(1, int(round(h * (float(width) / w))))
        grey = cv2.resize(grey, (int(width), new_h), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(grey, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)


# ---------------------------------------------------------------------------
# Region of interest
# ---------------------------------------------------------------------------
@dataclass
class ROIRect:
    """Board area inside the camera frame. w/h <= 0 means "full frame"."""
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @property
    def is_full(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def clamp(self, frame_w: int, frame_h: int) -> "ROIRect":
        if self.is_full:
            return ROIRect(0, 0, frame_w, frame_h)
        x = max(0, min(int(self.x), max(0, frame_w - 2)))
        y = max(0, min(int(self.y), max(0, frame_h - 2)))
        w = max(2, min(int(self.w), frame_w - x))
        h = max(2, min(int(self.h), frame_h - y))
        return ROIRect(x, y, w, h)

    def crop(self, frame: "np.ndarray") -> "np.ndarray":
        if self.is_full:
            return frame
        fh, fw = frame.shape[:2]
        r = self.clamp(fw, fh)
        return frame[r.y:r.y + r.h, r.x:r.x + r.w]

    def line(self) -> str:
        return "full frame" if self.is_full else (
            f"x={self.x} y={self.y} w={self.w} h={self.h}")

    def to_dict(self) -> dict:
        return {"x": int(self.x), "y": int(self.y),
                "w": int(self.w), "h": int(self.h)}


def parse_roi(spec: str | None) -> ROIRect:
    """Accept 'full', 'auto' or 'x,y,w,h'."""
    if not spec or spec.strip().lower() in ("full", "auto", ""):
        return ROIRect(0, 0, 0, 0)
    parts = [p for p in re.split(r"[,\s]+", spec.strip()) if p]
    if len(parts) != 4:
        raise ValueError(f"--roi expects 'x,y,w,h' or 'full' (got {spec!r})")
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--roi values must be numeric (got {spec!r})") from exc
    return ROIRect(x, y, w, h)


def load_roi_config(path: Path = ROI_CONFIG_PATH) -> ROIRect | None:
    """Read the saved ROI; returns None when absent or unreadable."""
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ROIRect(int(data.get("x", 0)), int(data.get("y", 0)),
                       int(data.get("w", 0)), int(data.get("h", 0)))
    except Exception as exc:
        log_warn(f"Could not read {path.name}: {exc}")
        return None


def save_roi_config(roi: ROIRect, path: Path = ROI_CONFIG_PATH) -> None:
    payload = roi.to_dict()
    payload["note"] = ("Blackboard ROI for vision_tracker.py; "
                       "w/h <= 0 means full frame.")
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_ok(f"Saved ROI to {path} ({roi.line()})")


# ---------------------------------------------------------------------------
# OCR text sanitisation
# ---------------------------------------------------------------------------
_MATH_CHAR_MAP = {
    "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u00d7": "*",
    "\u00f7": "/", "\u2264": "<=", "\u2265": ">=", "\u2260": "!=",
    "\u2192": "->", "\u21d2": "=>", "\u03a3": "sum", "\u221a": "sqrt",
    "\u03c0": "pi", "\u03b1": "alpha", "\u03b2": "beta", "\u03bb": "lambda",
}

# `P(A|B)` is routinely OCR'd as `P(AIB)`, `P(AlB)` or `P(A1B)`. Both sides must
# be a single letter and the pair must sit inside parentheses, which keeps the
# rule away from ordinary prose such as "AI" or "the 1st".
_COND_PROB_RE = re.compile(r"\(([A-Za-z])\s*[lIi1|]\s*([A-Za-z])\)")

_SUPERSCRIPT_MAP = {
    "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
    "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
    "8": "\u2078", "9": "\u2079", "n": "\u207f", "-": "\u207b",
}

# Two narrow superscript rules (both applied only inside an equation, i.e. when
# the line contains "="). Deliberately not one loose rule, so that "top2",
# "H2O", "the 2nd" and "n1" survive untouched. Only 2-9 convert, because 0 and
# 1 are overwhelmingly indices/ordinals in lecture notes.
_SUPERSCRIPT_MC_RE = re.compile(r"\bmc([2-9])\b")
_SUPERSCRIPT_LONE_RE = re.compile(r"\b([A-Za-z])([2-9])\b")


# OCR puts a stray space before an argument list in maths: "H (X)", "P (A)".
# Deliberately narrow - a single letter plus a short parenthesised token - so
# ordinary prose like "a (short note)" is never rewritten.
_FUNC_SPACE_RE = re.compile(r"\b([A-Za-z]) \(([A-Za-z0-9|]{1,4}"
                            r"(?:\s*,\s*[A-Za-z0-9|]{1,4})*)\)")


def _fix_func_space(match: "re.Match") -> str:
    name = match.group(1)
    if name in ("a", "I"):          # English words, not function names
        return match.group(0)
    return f"{name}({match.group(2)})"


def normalize_math(text: str) -> str:
    """Conservative OCR repair for whiteboard maths notation."""
    if not text:
        return ""
    out = text
    for bad, good in _MATH_CHAR_MAP.items():
        out = out.replace(bad, good)
    out = _COND_PROB_RE.sub(r"(\1|\2)", out)
    # A digit glued to a letter inside an equation is nearly always a power.
    # Two narrow rules instead of one loose one, so "top2", "H2O" and
    # "the 2nd" are left alone:
    #   1. a lone letter followed by 2-9 at a word end ("a2 + b2 = c2")
    #   2. the canonical mass-energy token "mc2"
    # Only 2-9 is converted; 0 and 1 are far more often indices or ordinals.
    if "=" in out:
        out = _SUPERSCRIPT_MC_RE.sub(
            lambda m: "mc" + _SUPERSCRIPT_MAP.get(m.group(1), m.group(1)), out)
        out = _SUPERSCRIPT_LONE_RE.sub(
            lambda m: m.group(1) + _SUPERSCRIPT_MAP.get(m.group(2), m.group(2)),
            out)
    out = _FUNC_SPACE_RE.sub(_fix_func_space, out)
    return re.sub(r"\s+", " ", out).strip()


def clean_ocr_lines(lines: list) -> list:
    """Drop OCR debris: 1-char tokens, punctuation-only and symbol-only rows."""
    out = []
    for raw in lines:
        text = normalize_math(str(raw))
        if len(text) < OCR_MIN_CHARS:
            continue
        if not re.search(r"[A-Za-z0-9]", text):
            continue
        if re.fullmatch(r"[^\w]+", text):
            continue
        out.append(text)
    return out


# A bullet or numbered list row always starts its own item, so it must never be
# glued onto the line above - even though "- x" and "* x" also look like an
# operator-led equation fragment.
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022\u00b7o]|\(?\d{1,2}[.)])\s+")

# A line that STARTS with an operator, or whose predecessor ENDS with one, is
# an OCR line-fragment rather than a new written line. Measured on this host:
# "entropy H(X)" and "= -sum p log p" arrive as two rows, so the tail got
# bucketed as its own equation and the head was misfiled as a written term.
_CONTINUATION_RE = re.compile(
    r"^\s*(?:[=+\-*/^<>]|\u00d7|\u00f7|\u00b1|\u2264|\u2265|\u2248"
    r"|\u2211|\u221a)")
_OPERATOR_TAIL_RE = re.compile(
    r"(?:[=+\-*/^<>]|\u00d7|\u00f7|\u00b1|\u2264|\u2265|\u2248"
    r"|\u2211|\u221a)\s*$")


def merge_line_fragments(lines: list) -> list:
    """Rejoin OCR line-fragments that belong to one written equation.

    Only joins when the operator is unambiguous - the next line opens with one,
    or the current line ends with one - and never touches a bullet/list row, so
    separate sentences are not run together. The joined text is re-normalised as
    a unit, which lets the maths repairs see the whole expression.
    """
    out: list = []
    for raw in lines:
        text = str(raw).strip()
        if not text:
            continue
        if not _BULLET_RE.match(text) and out and (
                _CONTINUATION_RE.match(text)
                or _OPERATOR_TAIL_RE.search(out[-1])):
            out[-1] = f"{out[-1]} {text}".strip()
            continue
        out.append(text)
    return [normalize_math(line) for line in out]


_EQUATION_RE = re.compile(
    r"[=<>]|\bmc\b|[\u00b2\u00b3\u207f]|\d\s*[+\-*/^]\s*\d|\bsqrt\b|\bsum\b")

# Recognised page furniture / classroom noise that is not board content.
_NOISE_LINE_RE = re.compile(
    r"^(?:page|slide|chapter|lecture|date|time|name|copyright|\u00a9)\b"
    r"|^\d{1,3}$", re.IGNORECASE)


def classify_lines(lines: list) -> dict:
    """Bucket cleaned OCR lines into equations / bullets / written terms."""
    equations, bullets, keywords = [], [], []
    for text in lines:
        if _NOISE_LINE_RE.search(text):
            continue
        body = _BULLET_RE.sub("", text).strip()
        if not body:
            continue
        if _EQUATION_RE.search(body):
            equations.append(body)
        elif _BULLET_RE.match(text):
            bullets.append(body)
        else:
            keywords.append(body)
    return {"equations": equations, "bullets": bullets, "keywords": keywords}


def _dedupe(items: list, limit: int = MAX_SUMMARY_ITEMS) -> list:
    """Case/punctuation-insensitive dedupe that preserves order."""
    seen, out = set(), []
    for item in items:
        key = re.sub(r"[^a-z0-9]", "", str(item).lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def diagram_hints(bgr_roi: "np.ndarray") -> dict:
    """Cheap structural hints that something was *drawn*, not just written.

    Looks for long straight segments (HoughLinesP) and polygonal contours in the
    binarised ROI. Costs ~1 ms, so it can run on every settled frame.
    """
    hints = {"edges": 0, "ink_ratio": 0.0, "shapes": 0, "label": ""}
    try:
        grey = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY) \
            if bgr_roi.ndim == 3 else bgr_roi
        h, w = grey.shape[:2]
        scale = 480.0 / float(w) if w > 480 else 1.0
        small = cv2.resize(grey, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale != 1.0 else grey
        small = cv2.GaussianBlur(small, (5, 5), 1.0)
        _, dark = cv2.threshold(small, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hints["ink_ratio"] = round(float((dark > 0).mean()), 4)
        min_len = max(30, int(small.shape[1] * 0.18))
        segments = cv2.HoughLinesP(dark, 1, np.pi / 180.0, threshold=60,
                                   minLineLength=min_len, maxLineGap=8)
        if segments is not None:
            hints["edges"] = int(len(segments))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        shapes = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 0.0008 * small.shape[0] * small.shape[1]:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if 3 <= len(approx) <= 8:
                shapes += 1
        hints["shapes"] = shapes
        if hints["edges"] >= 3 and shapes >= 1:
            hints["label"] = (f"line diagram ({hints['edges']} edges, "
                              f"{shapes} shapes)")
        elif hints["edges"] >= 3:
            hints["label"] = f"line drawing ({hints['edges']} edges)"
        elif shapes >= 2 and hints["ink_ratio"] < 0.25:
            hints["label"] = f"sketched shapes ({shapes})"
    except Exception as exc:      # heuristics must never break the pipeline
        LOG.debug("diagram_hints failed: %s", exc)
    return hints


def build_summary(buckets: dict, hints: dict, added: list) -> str:
    """Render `[BOARD CONTEXT]: Equations: ...; Diagram: ...` for the LLM."""
    parts = []
    eqs = _dedupe(buckets.get("equations", []))
    bullets = _dedupe(buckets.get("bullets", []))
    keywords = _dedupe(buckets.get("keywords", []))
    if eqs:
        parts.append("Equations: " + "; ".join(eqs))
    if bullets:
        parts.append(f"Bullet points ({len(buckets.get('bullets', []))}): "
                     + "; ".join(bullets))
    if keywords:
        parts.append("Written terms: " + "; ".join(keywords))
    new_items = _dedupe([normalize_math(a) for a in added if a], limit=4)
    if new_items:
        parts.append("Just added: " + "; ".join(new_items))
    if hints.get("label"):
        parts.append("Diagram: " + str(hints["label"]))
    if not parts:
        return ""
    body = " | ".join(parts)
    if len(body) > MAX_CONTEXT_CHARS:
        body = body[:MAX_CONTEXT_CHARS - 3].rstrip(" ;|,") + "..."
    return "[BOARD CONTEXT]: " + body


def token_set(text: str) -> set:
    """Lowercase alphanumeric tokens, used to detect 'what is new on the board'."""
    return set(re.findall(r"[a-z0-9]{2,}", str(text).lower()))


# ---------------------------------------------------------------------------
# OCR engines
#
# The ladder is ordered by "what actually works on this machine right now":
#   winocr    -> Windows.Media.Ocr, built into Windows 11, zero install,
#                850 ms/call on the probe board, 128/128 chars accurate
#   paddle    -> PaddleOCR 3.7.0, opt-in for hard handwriting; 22 s/call and it
#                needs enable_mkldnn=False or PaddlePaddle 3.3.1 crashes with
#                "ConvertPirAttribute2RuntimeAttribute not support"
#   tesseract -> requires tesseract.exe (NOT installed on this host) -> disables
#   ollama    -> multimodal endpoint forced to CPU (`num_gpu: 0`), last resort
# ---------------------------------------------------------------------------
@dataclass
class OcrLine:
    text: str
    score: float = 1.0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    kind: str = "text"          # "text" | "diagram" | "bullet" | "equation"


@dataclass
class OcrResult:
    lines: list = field(default_factory=list)     # list[OcrLine]
    engine: str = "none"
    elapsed_ms: int = 0
    error: str = ""
    raw_text: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.lines) and not self.error

    @property
    def texts(self) -> list:
        return [ln.text for ln in self.lines]

    @property
    def mean_score(self) -> float:
        if not self.lines:
            return 0.0
        return sum(ln.score for ln in self.lines) / len(self.lines)


class BaseOCREngine:
    """Common contract: `available()` is cheap, `recognize()` may be slow."""

    name = "base"
    label = "base"

    def __init__(self, **kwargs) -> None:
        self.opts = kwargs
        self.detail = ""
        self._available: bool | None = None

    def available(self) -> bool:            # pragma: no cover - overridden
        raise NotImplementedError

    def recognize(self, bgr: "np.ndarray") -> OcrResult:   # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        state = self.detail or ("ready" if self._available else "unavailable")
        return f"{self.name}: {state}"

    # -- shared helper: persist a frame for engines that want a file ----------
    def _frame_path(self, tag: str = "frame") -> Path:
        BOARD_DIR.mkdir(parents=True, exist_ok=True)
        return BOARD_DIR / f"{tag}.png"

    def _write_png(self, bgr: "np.ndarray", tag: str = "frame") -> Path:
        path = self._frame_path(tag)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed for OCR frame")
        path.write_bytes(buf.tobytes())
        return path


class WinOCREngine(BaseOCREngine):
    """Windows.Media.Ocr via one-shot PowerShell - always available on Win11.

    PowerShell is used instead of a Python WinRT binding so the vision companion
    needs no extra wheels (`winsdk`/`winrt` are not installed here, and pulling
    them in for a 4 GB-VRAM laptop adds ABI risk for no benefit). The image path
    travels through a UTF-8 file, so paths with spaces cannot break the round
    trip, and the script is handed over with -EncodedCommand, so no quoting rule
    can corrupt it either.
    """

    name = "winocr"
    label = "Windows.Media.Ocr (built-in)"

    PS_TMPL = r"""
$ErrorActionPreference = "Stop"
try {
    [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    [Windows.Storage.StorageFile, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    })[0]
    function Await($op, $t) {
        $m = $asTaskGeneric.MakeGenericMethod($t)
        $task = $m.Invoke($null, @($op))
        $task.Wait(-1) | Out-Null
        $task.Result
    }
    $cfg = [System.IO.File]::ReadAllLines('__CFG__', [System.Text.Encoding]::UTF8)
    $imgPath = $cfg[0]
    $lang = ''
    if ($cfg.Count -gt 1) { $lang = $cfg[1] }

    $eng = $null
    if ($lang -ne '') {
        try {
            $l = New-Object Windows.Globalization.Language($lang)
            $eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($l)
        } catch { $eng = $null }
    }
    if ($null -eq $eng) {
        $eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    }
    if ($null -eq $eng) {
        [System.IO.File]::WriteAllText('__OUT__', '{"lines":[],"error":"no_ocr_engine"}', [System.Text.Encoding]::UTF8)
        exit 0
    }

    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imgPath)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

    $res = Await ($eng.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])

    $lines = New-Object System.Collections.ArrayList
    foreach ($line in $res.Lines) {
        $ws = @($line.Words)
        if ($ws.Count -eq 0) { continue }
        $x0 = [double]::MaxValue; $y0 = [double]::MaxValue; $x1 = 0.0; $y1 = 0.0
        foreach ($wd in $ws) {
            $r = $wd.BoundingRect
            if ($r.X -lt $x0) { $x0 = $r.X }
            if ($r.Y -lt $y0) { $y0 = $r.Y }
            if (($r.X + $r.Width) -gt $x1) { $x1 = $r.X + $r.Width }
            if (($r.Y + $r.Height) -gt $y1) { $y1 = $r.Y + $r.Height }
        }
        [void]$lines.Add([pscustomobject]@{
            t = $line.Text; n = $ws.Count
            x = [int]$x0; y = [int]$y0
            w = [int]($x1 - $x0); h = [int]($y1 - $y0)
        })
    }
    $json = ConvertTo-Json -InputObject @($lines) -Compress -Depth 4
    if ([string]::IsNullOrWhiteSpace($json)) { $json = '[]' }
    [System.IO.File]::WriteAllText('__OUT__', $json, [System.Text.Encoding]::UTF8)
} catch {
    $msg = ($_.Exception.Message -replace '\\', '/' -replace '"', "'") -replace '\s+', ' '
    [System.IO.File]::WriteAllText('__OUT__', '{"lines":[],"error":"' + $msg + '"}', [System.Text.Encoding]::UTF8)
}
"""

    def __init__(self, language: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.language = (language or "").strip()
        self._scratch: Path | None = None
        self._cfg_path: Path | None = None
        self._out_path: Path | None = None
        self._script_path: Path | None = None
        self._ps_exe: str | None = None

    # -- scratch files live in %APPDATA% so a read-only workdir is fine -------
    def _scratch_dir(self) -> Path:
        if self._scratch is None:
            base = (os.environ.get("APPDATA") or os.environ.get("TEMP")
                    or str(WORKSPACE))
            d = Path(base) / "TA_local_vision"
            d.mkdir(parents=True, exist_ok=True)
            self._scratch = d
            self._cfg_path = d / "winocr_cfg.txt"
            self._out_path = d / "winocr_out.json"
            self._script_path = d / "winocr.ps1"
            script = (self.PS_TMPL
                      .replace("__CFG__", str(self._cfg_path).replace("'", "''"))
                      .replace("__OUT__", str(self._out_path).replace("'", "''")))
            self._script_path.write_text(script, encoding="utf-8")
        return self._scratch

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if os.name != "nt":
            self._available = False
            self.detail = "Windows-only engine"
            return False
        exe = shutil.which("powershell") or shutil.which("pwsh")
        if not exe:
            self._available = False
            self.detail = "powershell.exe not found on PATH"
            return False
        self._ps_exe = exe
        try:
            self._scratch_dir()
        except Exception as exc:
            self._available = False
            self.detail = f"scratch dir failed: {exc}"
            return False
        probe = (
            "try { [Windows.Media.Ocr.OcrEngine, Windows.Foundation, "
            "ContentType = WindowsRuntime] | Out-Null; "
            "$e = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages(); "
            "if ($e) { 'yes' } else { 'no' } } catch { 'err' }")
        try:
            proc = self._run_ps(probe, timeout=30.0)
        except Exception as exc:
            self._available = False
            self.detail = f"probe failed: {exc}"
            return False
        out = (proc.stdout or "").strip().lower()
        if out.endswith("yes"):
            self._available = True
            self.detail = "ready (Windows.Media.Ocr)"
        elif out.endswith("no"):
            self._available = False
            self.detail = ("no OCR language pack - Settings > Time & Language > "
                           "Language & region > English (US) > Language options "
                           "> add 'Optical character recognition'")
        else:
            self._available = False
            self.detail = f"WinRT OCR could not be constructed ({out!r})"
        return self._available

    def _run_ps(self, script: str, timeout: float = WINOCR_TIMEOUT_S):
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return subprocess.run(
            [self._ps_exe, "-NoProfile", "-NonInteractive", "-EncodedCommand",
             encoded],
            capture_output=True, text=True, timeout=timeout,
            creationflags=SUBPROC_FLAGS,
        )

    def recognize(self, bgr: "np.ndarray") -> OcrResult:
        t0 = time.perf_counter()
        if not self.available():
            return OcrResult(engine=self.name,
                             error=f"unavailable ({self.detail})")

        def _ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        try:
            img_path = self._write_png(bgr, "winocr_frame")
            self._cfg_path.write_text(f"{img_path}\n{self.language}\n",
                                      encoding="utf-8")
            try:
                self._out_path.unlink()
            except OSError:
                pass
            proc = self._run_ps(self._script_path.read_text(encoding="utf-8"))
            raw = ""
            if self._out_path.exists():
                # PowerShell's [System.Text.Encoding]::UTF8 writes a BOM, so the
                # Python side must decode tolerantly (utf-8-sig accepts both).
                raw = self._out_path.read_text(encoding="utf-8-sig").strip()
            if not raw:
                raw = (proc.stdout or "").strip()
            if not raw:
                err = (proc.stderr or "").strip().splitlines()
                hint = err[-1] if err else "no output from OCR host"
                return OcrResult(engine=self.name, elapsed_ms=_ms(),
                                 error=f"empty OCR response ({hint[:160]})")
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("error"):
                return OcrResult(engine=self.name, elapsed_ms=_ms(),
                                 error=str(data["error"])[:200])
            if isinstance(data, dict):
                data = [data]
            lines = []
            for item in data or []:
                if isinstance(item, str):
                    text, score, x, y, w, h = item.strip(), 1.0, 0, 0, 0, 0
                else:
                    text = str(item.get("t", "")).strip()
                    words = int(item.get("n", 0) or 0)
                    # Windows OCR exposes no per-word confidence. A multi-word
                    # line is far more trustworthy than a stray glyph, so this
                    # proxy score lets a single OCR_MIN_SCORE rule apply to every
                    # engine without pretending to know more than we do.
                    score = 1.0 if words >= 2 else 0.6
                    x = int(item.get("x", 0) or 0)
                    y = int(item.get("y", 0) or 0)
                    w = int(item.get("w", 0) or 0)
                    h = int(item.get("h", 0) or 0)
                if not text:
                    continue
                lines.append(OcrLine(text=text, score=score, x=x, y=y, w=w, h=h))
            lines.sort(key=lambda ln: (ln.y, ln.x))
            return OcrResult(lines=lines, engine=self.name, elapsed_ms=_ms())
        except subprocess.TimeoutExpired:
            return OcrResult(engine=self.name, elapsed_ms=_ms(),
                             error=f"timeout after {WINOCR_TIMEOUT_S:.0f}s")
        except Exception as exc:
            LOG.debug("winocr recognize failed", exc_info=True)
            return OcrResult(engine=self.name, elapsed_ms=_ms(),
                             error=f"{type(exc).__name__}: {exc}"[:200])
class PaddleOCREngine(BaseOCREngine):
    """Optional high-accuracy engine (PaddleOCR PP-OCRv6).

    Measured on this host: excellent text (0.96-1.00 confidences, correct
    `P(A|B)`) but 22.2 s per 1280x720 frame, so it is never the default.

    The oneDNN crash first seen here -
        NotImplementedError (Unimplemented)
        ConvertPirAttribute2RuntimeAttribute not support
        [pir::ArrayAttribute<pir::DoubleAttribute>]
    - is a PaddlePaddle 3.3.1 bug that `enable_mkldnn=False` works around, so
    that flag is always passed and the constructor also tolerates older
    PaddleOCR releases that do not accept the 3.x document-preprocessing flags.
    """

    name = "paddle"
    label = "PaddleOCR PP-OCRv6 (accurate, slow)"

    def __init__(self, cpu_threads: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cpu_threads = cpu_threads
        self._ocr = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import paddleocr
            version = getattr(paddleocr, "__version__", "?")
            self._available = True
            self.detail = f"paddleocr {version} (mkldnn off, {self.cpu_threads} threads)"
        except Exception as exc:
            self._available = False
            self.detail = (f"paddleocr missing ({type(exc).__name__}) - fix: "
                           f"`python -m pip install paddleocr paddlepaddle`")
        return self._available

    def _ensure_model(self):
        """Lazily build the pipeline, degrading to the oldest constructor."""
        if self._ocr is not None:
            return self._ocr
        from paddleocr import PaddleOCR
        base = {
            "lang": "en",
            "enable_mkldnn": False,        # works around the oneDNN crash
            "cpu_threads": self.cpu_threads,
        }
        disabled = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        try:
            self._ocr = PaddleOCR(**base, **disabled)
        except TypeError:
            try:
                self._ocr = PaddleOCR(**base)
            except TypeError:
                self._ocr = PaddleOCR(lang="en")
        return self._ocr

    def recognize(self, bgr: "np.ndarray") -> OcrResult:
        t0 = time.perf_counter()

        def _ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        if not self.available():
            return OcrResult(engine=self.name, error=f"unavailable ({self.detail})")
        try:
            ocr = self._ensure_model()
            result = ocr.predict(bgr)
            lines = []
            for page in result or []:
                texts = list(page.get("rec_texts", []) or [])
                scores = list(page.get("rec_scores", []) or [])
                polys = list(page.get("rec_polys", []) or [])
                for idx, text in enumerate(texts):
                    try:
                        score = float(scores[idx]) if idx < len(scores) else 0.0
                    except (TypeError, ValueError):
                        score = 0.0
                    if score < OCR_MIN_SCORE:
                        continue
                    x = y = w = h = 0
                    if idx < len(polys) and polys[idx] is not None:
                        arr = np.asarray(polys[idx]).reshape(-1, 2)
                        if arr.size:
                            x = int(arr[:, 0].min())
                            y = int(arr[:, 1].min())
                            w = int(arr[:, 0].max()) - x
                            h = int(arr[:, 1].max()) - y
                    lines.append(OcrLine(
                        text=normalize_math(str(text)), score=round(score, 3),
                        x=x, y=y, w=w, h=h))
            lines.sort(key=lambda ln: (ln.y, ln.x))
            return OcrResult(lines=lines, engine=self.name, elapsed_ms=_ms())
        except Exception as exc:
            LOG.debug("paddleocr recognize failed", exc_info=True)
            text = f"{type(exc).__name__}: {exc}"
            if "ConvertPirAttribute" in text or "onednn" in text.lower():
                text += (" | known PaddlePaddle oneDNN bug; `--ocr winocr` avoids it")
            return OcrResult(engine=self.name, elapsed_ms=_ms(), error=text[:240])


class TesseractOCREngine(BaseOCREngine):
    """pytesseract wrapper. Self-disables when the tesseract binary is absent.

    Checked on this host: tesseract.exe is NOT on PATH and not in
    "C:\\Program Files\\Tesseract-OCR", so this engine reports the exact fix
    instead of raising. When present, `image_to_data` supplies both word
    confidences and word boxes, which are grouped into lines here.
    """

    name = "tesseract"
    label = "pytesseract / Tesseract OCR"

    CANDIDATE_PATHS = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\ProgramData\chocolatey\bin\tesseract.exe",
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pt = None
        self._exe: str | None = None

    def _find_exe(self) -> str | None:
        found = shutil.which("tesseract")
        if found:
            return found
        for candidate in self.CANDIDATE_PATHS:
            if Path(candidate).exists():
                return candidate
        return None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if self._find_exe() is None:
            self._available = False
            self.detail = ("tesseract.exe not found - install from "
                           "https://github.com/UB-Mannheim/tesseract/wiki "
                           "(default install path C:\\Program Files\\Tesseract-OCR)")
            return False
        try:
            import pytesseract
            self._pt = pytesseract
            self._exe = self._find_exe()
            pytesseract.pytesseract.tesseract_cmd = self._exe
            getter = getattr(pytesseract, "get_tesseract_version", None)
            version = str(getter()) if callable(getter) else "?"
            self._available = True
            self.detail = f"tesseract {version.splitlines()[0]} at {self._exe}"
        except Exception as exc:
            self._available = False
            self.detail = (f"pytesseract unusable ({type(exc).__name__}: {exc}) "
                           f"- fix: `python -m pip install pytesseract`")
        return self._available

    def recognize(self, bgr: "np.ndarray") -> OcrResult:
        t0 = time.perf_counter()

        def _ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        if not self.available():
            return OcrResult(engine=self.name, error=f"unavailable ({self.detail})")
        try:
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            # --psm 6 = "single uniform block of text", which suits a board of
            # short lines much better than the default page-layout mode.
            data = self._pt.image_to_data(grey, config="--oem 3 --psm 6",
                                          output_type=self._pt.Output.DICT)
            grouped: dict = {}
            for i, word in enumerate(data.get("text", [])):
                word = str(word).strip()
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError, KeyError, IndexError):
                    conf = -1.0
                if not word or conf < OCR_MIN_SCORE * 100.0:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                bucket = grouped.setdefault(key, {
                    "words": [], "confs": [],
                    "x0": 10 ** 9, "y0": 10 ** 9, "x1": 0, "y1": 0})
                bucket["words"].append(word)
                bucket["confs"].append(conf / 100.0)
                x, y = int(data["left"][i]), int(data["top"][i])
                w, h = int(data["width"][i]), int(data["height"][i])
                bucket["x0"] = min(bucket["x0"], x)
                bucket["y0"] = min(bucket["y0"], y)
                bucket["x1"] = max(bucket["x1"], x + w)
                bucket["y1"] = max(bucket["y1"], y + h)
            lines = []
            for key in sorted(grouped):
                bucket = grouped[key]
                if not bucket["words"]:
                    continue
                score = sum(bucket["confs"]) / len(bucket["confs"])
                lines.append(OcrLine(
                    text=" ".join(bucket["words"]), score=round(score, 3),
                    x=bucket["x0"], y=bucket["y0"],
                    w=bucket["x1"] - bucket["x0"], h=bucket["y1"] - bucket["y0"]))
            lines.sort(key=lambda ln: (ln.y, ln.x))
            return OcrResult(lines=lines, engine=self.name, elapsed_ms=_ms())
        except Exception as exc:
            LOG.debug("tesseract recognize failed", exc_info=True)
            return OcrResult(engine=self.name, elapsed_ms=_ms(),
                             error=f"{type(exc).__name__}: {exc}"[:200])


def build_image_payload(bgr: "np.ndarray", max_width: int = 896,
                        quality: int = 85) -> str:
    """Base64 JPEG for the Ollama multimodal endpoint, capped for CPU speed.

    896 px is the sweet spot: Moondream spends far fewer vision tokens at that
    width, so a CPU round trip stays seconds instead of tens of seconds.
    """
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = float(max_width) / w
        bgr = cv2.resize(bgr, (int(max_width), max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for the Ollama image payload")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class OllamaVisionEngine(BaseOCREngine):
    """Optional multimodal board reader via Ollama.

    VRAM guardrail: every request carries `options.num_gpu = 0`, so the vision
    model is pinned to system RAM and `llama3.2:3b` keeps the whole 4 GB of the
    RTX 3050 Ti. Because this only runs on a settled board update, a multi-second
    CPU answer is perfectly acceptable here.

    Three prompt strategies exist because boards are layout-heavy and a short
    caption sometimes beats a full transcription:
        transcribe (default) | describe | diagram
    """

    name = "ollama"
    label = "Ollama multimodal (CPU-offloaded)"

    PROMPTS = {
        "transcribe": ("Transcribe every piece of writing visible on this "
                       "blackboard or whiteboard. Preserve the reading order. "
                       "Render mathematical notation in plain text. End with the "
                       "line 'DIAGRAM:' followed by a short description of any "
                       "drawing, or 'DIAGRAM: none'. Output nothing else."),
        "describe": ("Describe what is written and drawn on this board in at "
                     "most three short sentences, naming equations and diagram "
                     "structure. No preamble."),
        "diagram": ("Ignore prose. Describe only the diagrams: node/edge counts, "
                    "arrows, axes and shapes, in at most two sentences."),
    }

    def __init__(self, host: str = OLLAMA_HOST_DEFAULT,
                 model: str = DEFAULT_OLLAMA_VISION_MODEL,
                 mode: str = "transcribe", **kwargs) -> None:
        super().__init__(**kwargs)
        self.host = host.rstrip("/")
        self.model = model
        self.mode = mode if mode in self.PROMPTS else "transcribe"
        self._session = None

    def _get_session(self):
        """One pooled session, so TCP setup is not paid per board update."""
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=2, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session = session
        return self._session

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests                            # noqa: F401
            resp = self._get_session().get(f"{self.host}/api/tags", timeout=(2, 4))
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            if self.model not in names:
                self._available = False
                self.detail = (f"'{self.model}' is not pulled - fix: "
                               f"`ollama pull {self.model}` (runs on CPU RAM; "
                               f"the 3050 Ti stays reserved for llama3.2:3b)")
                return False
            self._available = True
            self.detail = f"{self.model} via {self.host} (num_gpu=0)"
        except Exception as exc:
            self._available = False
            self.detail = (f"Ollama unreachable at {self.host} "
                           f"({type(exc).__name__}) - start the Ollama app or "
                           f"`ollama serve`, then `ollama pull {self.model}`")
        return self._available
    def recognize(self, bgr: "np.ndarray") -> OcrResult:
        t0 = time.perf_counter()

        def _ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        if not self.available():
            return OcrResult(engine=self.name, error=f"unavailable ({self.detail})")
        try:
            payload = {
                "model": self.model,
                "prompt": self.PROMPTS[self.mode],
                "images": [build_image_payload(bgr)],
                "stream": False,
                "options": {
                    # THE guardrail: never let this borrow the 3050 Ti.
                    "num_gpu": 0,
                    "temperature": 0.1,
                    "num_predict": 220,
                },
            }
            resp = self._get_session().post(
                f"{self.host}/api/generate", json=payload,
                timeout=OLLAMA_VISION_TIMEOUT_S)
            resp.raise_for_status()
            body = resp.json().get("response", "") or ""
            lines = []
            for raw in body.splitlines():
                text = normalize_math(
                    re.sub(r"^[\s\-*•·]+", "", raw.strip()))
                if not text:
                    continue
                low = text.lower()
                if low.startswith("diagram:"):
                    detail = text.split(":", 1)[1].strip()
                    if detail and detail.lower() != "none":
                        lines.append(OcrLine(text="Diagram: " + detail,
                                             score=0.6, kind="diagram"))
                    continue
                if low.startswith(("here is", "sure,", "in the image",
                                   "the image", "as an ai", "i can")):
                    continue
                lines.append(OcrLine(text=text, score=0.7))
            return OcrResult(lines=lines, engine=self.name, elapsed_ms=_ms(),
                             raw_text=body)
        except Exception as exc:
            LOG.debug("ollama vision failed", exc_info=True)
            return OcrResult(engine=self.name, elapsed_ms=_ms(),
                             error=f"{type(exc).__name__}: {exc}"[:200])

    def close(self) -> None:
        """Release the pooled HTTP session (called by VisionTracker.stop)."""
        session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


class NullOCREngine(BaseOCREngine):
    """`--ocr none`: still detects board updates, never reads text.

    Useful for a pure motion/preview sanity check, or on a machine with no OCR
    stack at all, so the tracker can be exercised without failing out.
    """

    name = "none"
    label = "disabled (scene-change detection only)"

    def available(self) -> bool:
        self._available = True
        self.detail = "OCR disabled by configuration"
        return True

    def recognize(self, bgr: "np.ndarray") -> OcrResult:
        return OcrResult(engine=self.name, lines=[], raw_text="", error="")


# ---------------------------------------------------------------------------
# Engine ladder
# ---------------------------------------------------------------------------
ENGINE_CLASSES = {
    "winocr": WinOCREngine,
    "paddle": PaddleOCREngine,
    "tesseract": TesseractOCREngine,
    "ollama": OllamaVisionEngine,
    "none": NullOCREngine,
}


def build_engine(name: str, **kwargs) -> BaseOCREngine:
    """Instantiate one concrete engine by name."""
    cls = ENGINE_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"unknown OCR engine {name!r} "
                         f"(choose from {', '.join(OCR_ENGINES)})")
    return cls(**kwargs)


def probe_engines(host: str = OLLAMA_HOST_DEFAULT,
                  vision_model: str = DEFAULT_OLLAMA_VISION_MODEL,
                  vision_mode: str = "transcribe",
                  verbose: bool = False) -> list:
    """Inspect every engine and return them in ladder order, best first.

    Order is by measured value on this host, not by reputation:
        winocr    - 0.9 s/call, 128/128 characters correct on the probe board
        ollama    - accurate but CPU-bound, and only when its model is pulled
        paddle    - excellent accuracy at 22.2 s/call
        tesseract - would rank high, but tesseract.exe is not installed here
    """
    engines = []
    for name in ("winocr", "ollama", "paddle", "tesseract"):
        if name == "ollama":
            engine = OllamaVisionEngine(host=host, model=vision_model,
                                        mode=vision_mode)
        else:
            engine = ENGINE_CLASSES[name]()
        ready = engine.available()
        marker = _c("ready", C.GREEN) if ready else _c("unavailable", C.YELLOW)
        print(f"  {name:10s} {marker:22s} {engine.describe()}", flush=True)
        if verbose and not ready:
            LOG.debug("engine %s unavailable: %s", name, engine.detail)
        engines.append(engine)
    return engines


def select_engine(preferred: str, engines: list):
    """Pick `preferred` if usable, else the first usable engine in the ladder.

    Returns (engine_or_None, reason_when_None).
    """
    if preferred not in ("auto", "none", ""):
        chosen = next((e for e in engines if e.name == preferred), None)
        if chosen is None or not chosen.available():
            reason = chosen.detail if chosen is not None else "unknown engine"
            return None, f"--ocr {preferred} is not usable ({reason})"
        return chosen, ""
    if preferred == "none":
        return NullOCREngine(), ""
    for engine in engines:
        if engine.available():
            return engine, ""
    return None, ("no OCR engine available. Fix any ONE of: run on Windows 11 "
                  "(its built-in OCR needs no install at all), "
                  "`ollama pull moondream:1.8b`, "
                  "`python -m pip install paddleocr paddlepaddle`, or install "
                  "Tesseract from https://github.com/UB-Mannheim/tesseract/wiki")

# ---------------------------------------------------------------------------
# Board state
# ---------------------------------------------------------------------------
class SceneState(IntEnum):
    """Scene-change FSM (mirrors the state-machine style of listener.py)."""
    CALIBRATING = 0     # measuring the sensor noise floor
    SCANNING = 1        # board stable, waiting for a significant change
    MOTION = 2          # board being written on / something moved in front
    SETTLE = 3          # motion stopped; waiting out the stabilisation window
    STOPPED = 4         # capture torn down


@dataclass
class BoardSnapshot:
    """One settled, OCR'd state of the board."""
    summary: str = ""                       # "[BOARD CONTEXT]: ..."
    texts: list = field(default_factory=list)        # cleaned OCR lines
    buckets: dict = field(default_factory=dict)      # equations/bullets/keywords
    new_items: list = field(default_factory=list)    # additions vs last snapshot
    hints: dict = field(default_factory=dict)        # diagram heuristics
    engine: str = "none"
    ocr_ms: int = 0
    ssim: float = 1.0
    mad: float = 0.0
    revision: int = 0
    capture_epoch: float = 0.0
    roi: str = "full frame"
    stored_at: float = field(default_factory=time.monotonic)
    wall: float = field(default_factory=time.time)

    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.stored_at)

    def timestamp(self) -> str:
        dt = datetime.fromtimestamp(self.wall)
        return f"{dt:%H:%M:%S}.{dt.microsecond // 1000:03d}"

    @property
    def is_empty_board(self) -> bool:
        """True when OCR found nothing worth telling the LLM about."""
        return not self.texts and not self.hints.get("label")

    @property
    def age_str(self) -> str:
        age = self.age_s()
        return f"{age:.1f}s" if age < 60 else f"{age / 60:.1f}min"

    def to_dict(self) -> dict:
        return {
            "revision": self.revision,
            "timestamp": self.timestamp(),
            "engine": self.engine,
            "ocr_ms": self.ocr_ms,
            "ssim": round(self.ssim, 4),
            "mad": round(self.mad, 3),
            "roi": self.roi,
            "texts": list(self.texts),
            "new_items": list(self.new_items),
            "hints": dict(self.hints),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Vision tracker
# ---------------------------------------------------------------------------
class VisionTracker:
    """Camera -> scene-change FSM -> CPU OCR -> thread-safe board context.

    Threads (all daemon, all joined on stop):
        cam-<index>   owns cv2.VideoCapture, throttles to `fps`
        vision-ocr    runs OCR only for settled boards, never per frame

    The camera thread never waits for OCR: when a board settles it pushes one
    job onto a bounded queue and immediately resumes watching. A full queue
    drops the job and logs it, so a slow OCR engine can never back up into the
    capture path (same philosophy as listener.py's PCM queue).
    """

    def __init__(self,
                 camera_index: int = CAMERA_INDEX_DEFAULT,
                 fps: float = CAPTURE_FPS,
                 roi: ROIRect | None = None,
                 ocr_pref: str = "auto",
                 vision_model: str = DEFAULT_OLLAMA_VISION_MODEL,
                 ollama_host: str = OLLAMA_HOST_DEFAULT,
                 stabilize_s: float = STABILIZE_S,
                 max_dwell_s: float = MAX_DWELL_S,
                 width: int = CAPTURE_WIDTH,
                 height: int = CAPTURE_HEIGHT,
                 gui: bool = False,
                 save_frames: bool = False,
                 cv_threads: int = CV_THREADS_DEFAULT,
                 verbose: bool = False,
                 web_publish: bool = False,
                 web_host: str = WEB_HOST_DEFAULT,
                 web_port: int = WEB_PORT_DEFAULT) -> None:
        require_cv2()
        self.camera_index = int(camera_index)
        self.fps = max(0.1, float(fps))
        self.roi = roi or ROIRect(0, 0, 0, 0)
        self.ocr_pref = ocr_pref
        self.vision_model = vision_model
        self.ollama_host = ollama_host
        self.stabilize_s = max(0.0, float(stabilize_s))
        self.max_dwell_s = max(self.stabilize_s + 1.0, float(max_dwell_s))
        self.width = int(width)
        self.height = int(height)
        self.gui = bool(gui)
        self.save_frames = bool(save_frames)
        self.cv_threads = max(1, int(cv_threads))
        self.verbose = bool(verbose)

        # Phase 5: optional push of settled boards into the HTML cockpit HUD.
        # Vision stays fully standalone - when the HUD is absent the push is
        # skipped with one warning and capture/OCR carry on untouched.
        self.web_publish = bool(web_publish)
        self.web_host = str(web_host)
        self.web_port = int(web_port)
        self._web_failures = 0
        self._web_warned = False
        self._webui = None

        self.engine: BaseOCREngine | None = None
        self._engine_name = "none"   # survives stop() nulling .engine
        self.engine_candidates: list = []

        # capture plumbing
        self._cap = None
        self._cam_thread: threading.Thread | None = None
        self._ocr_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._job_queue: "queue.Queue" = queue.Queue(maxsize=JOB_QUEUE_MAXSIZE)

        # board state
        self._lock = threading.RLock()
        self._board_cond = threading.Condition(self._lock)
        self._snapshots: deque = deque(maxlen=64)
        self._latest: BoardSnapshot | None = None
        self._last_emitted_frame = None   # occlusion/revert test only
        self._revision = 0

        # FSM bookkeeping
        self.state = SceneState.CALIBRATING
        self.noise_floor = 0.0
        self.noise_ratio_floor = 0.0
        self.still_eps = STILL_MAD_DEFAULT
        self.thresholds = SceneThresholds()
        self.last_motion_at = 0.0
        self.motion_started_at = 0.0
        self.last_significant_at = 0.0
        self.last_ocr_at = 0.0
        self.last_delta = FrameDelta(mad=0.0, ratio=0.0, ssim=1.0, change=0.0)
        self._noise_ratios: deque = deque(maxlen=240)
        self._noise_mads: deque = deque(maxlen=240)
        # Rolling reference window. NEVER refilled after a read: a window
        # holding copies of a single frame would lose the independent-noise
        # averaging that makes the median reference worth having.
        self._ref_window: deque = deque(maxlen=TEMPORAL_MEDIAN_WINDOW)
        self._pending_update = False
        self._calib_started = 0.0

        # stats
        self.frames_seen = 0
        self.frames_skipped = 0
        self.changes_detected = 0
        self.boards_updated = 0
        self.ocr_calls = 0
        self.ocr_failures = 0
        self.occlusions_rejected = 0
        self.blank_boards = 0
        self.jobs_dropped = 0
        self.ocr_ms_total = 0
        self._started_at = 0.0

        # gui state
        self._preview_frame = None
        self._preview_lock = threading.Lock()
        self._gui_stop = threading.Event()
        self._dash_thread: threading.Thread | None = None
        self.dashboard = False
        self._backend = "?"
        self._stopping = False
        self._last_ocr_lines: list = []

    # ------------------------------------------------------------- properties
    @property
    def running(self) -> bool:
        return self._cam_thread is not None and self._cam_thread.is_alive()

    @property
    def camera_label(self) -> str:
        return f"camera {self.camera_index} @ {self.fps:g} FPS"

    def _set_state(self, state: SceneState) -> None:
        if state != self.state:
            LOG.debug("state %s -> %s", self.state.name, state.name)
            self.state = state

    # ------------------------------------------------------------------ start
    def start(self) -> "VisionTracker":
        """Open the camera, choose an OCR engine, launch both worker threads."""
        if self.running:
            return self
        try:
            # Budget CPU: faster-whisper keeps 4 threads and Kokoro wants 8
            # during playback, so the vision side stays deliberately small.
            cv2.setNumThreads(self.cv_threads)
            LOG.info("cv2.setNumThreads(%d)", self.cv_threads)
        except Exception as exc:
            LOG.debug("cv2.setNumThreads failed: %s", exc)
        silence_cv2_logs()

        self._stop_event.clear()
        self._gui_stop.clear()
        if self.gui:
            self._open_preview_window()

        log_info("Probing OCR engines ...")
        self.engine_candidates = probe_engines(
            host=self.ollama_host, vision_model=self.vision_model,
            verbose=self.verbose)
        engine, reason = select_engine(self.ocr_pref, self.engine_candidates)
        if engine is None:
            raise RuntimeError(reason)
        self.engine = engine
        self._engine_name = engine.name
        log_ok(f"OCR engine selected: {engine.describe()}")

        cap, first_frame = self._open_camera()
        self._cap = cap
        self._started_at = time.monotonic()

        self._ocr_thread = threading.Thread(
            target=self._ocr_loop, name="vision-ocr", daemon=True)
        self._cam_thread = threading.Thread(
            target=self._capture_loop, name=f"cam-{self.camera_index}",
            args=(first_frame,), daemon=True)
        self._ocr_thread.start()
        self._cam_thread.start()
        if self.dashboard:
            self.start_dashboard()
        return self

    @staticmethod
    def _backend_choices() -> list:
        """Capture backends to try, best first.

        On Windows DSHOW is the reliable 720p path on this host and MSMF is the
        fallback. CAP_ANY is deliberately excluded: it resolves to the FFMPEG
        backend, which has no camera support and emits
        "OpenCV should be configured with libavdevice to open a camera device"
        without ever producing a frame.
        """
        if os.name == "nt":
            return [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
        return [(cv2.CAP_ANY, "ANY")]

    def _open_camera(self):
        """Open the capture device, trying the Windows backends in order.

        DSHOW is tried first because it is the one that reliably returns 720p on
        this host; MSMF sometimes reports 0x0 or needs seconds to warm up. The
        first few reads from a Windows camera are frequently black, so a short
        warm-up loop runs before success is declared.
        """
        backends = self._backend_choices()
        problems = []
        for backend, name in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(self.camera_index, backend)
                if not cap.isOpened():
                    problems.append(f"{name}: device did not open")
                    cap.release()
                    continue
                for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, self.width),
                                    (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                                    (cv2.CAP_PROP_BUFFERSIZE, 1)):
                    try:
                        cap.set(prop, value)
                    except Exception:
                        pass
                frame = None
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    ok, candidate = cap.read()
                    if ok and candidate is not None and getattr(candidate, "size", 0):
                        frame = candidate
                        break
                    time.sleep(0.15)
                if frame is None:
                    problems.append(f"{name}: no frame within 3 s (privacy switch?)")
                    cap.release()
                    continue
                self._backend = name
                real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
                real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
                log_ok(f"Camera {self.camera_index} open via {name} "
                       f"({real_w}x{real_h}, ROI: {self.roi.line()})")
                return cap, frame
            except Exception as exc:
                problems.append(f"{name}: {type(exc).__name__}: {exc}")
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        raise RuntimeError(
            f"Could not open camera index {self.camera_index}. Tried: "
            + "; ".join(problems)
            + ". Fix: close Teams/Zoom/Camera that may hold the device, enable "
              "Settings > Privacy & security > Camera > 'Let desktop apps "
              "access your camera', or pick another index with --camera N "
              "(0, 1, 2 ...). `--list-cameras` probes indexes 0-4 for you."
        )

    def _list_cameras(self, limit: int = 5, status=None,
                      configured_first: bool = True) -> list:
        """Probe camera indexes 0..limit-1 in isolated child processes.

        Each index is opened in its own child (see probe_camera_index) because
        probing several indexes in one process can hard-crash OpenCV on Windows.
        The configured index is tried first so the common case reports back
        immediately, then the remaining indexes follow.

        On a real host the capture resolution is requested before reading, so
        the report shows the size the daemon will actually use - a Windows
        camera otherwise defaults to 640x480, which is misleading to print.
        """
        found = []
        order = list(range(limit))
        if configured_first and self.camera_index in order:
            order.remove(self.camera_index)
            order.insert(0, self.camera_index)
        for index in order:
            if self._stop_event.is_set():
                break
            if status is not None:
                try:
                    status(index)
                except Exception:
                    pass
            hit = probe_camera_index(index, self.width, self.height)
            if hit is not None:
                found.append(hit)
                LOG.info("camera probe: index %s -> %sx%s (%s)", *hit)
        return found

    # ---------------------------------------------------------------- capture
    def _capture_loop(self, first_frame) -> None:
        """Throttled camera reader. Never blocks on OCR (see _emit_update)."""
        interval = 1.0 / self.fps
        cap = self._cap
        next_t = time.monotonic()
        read_failures = 0
        log_ok(f"Capture thread running at {self.fps:g} FPS "
               f"(analyze width {ANALYZE_WIDTH}px).")
        try:
            self._process_frame(first_frame)
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_t:
                    time.sleep(min(next_t - now, 0.2))
                    continue
                # Never burst-catch-up after a hitch: this is what keeps the
                # average CPU cost of a 1.5 FPS sampler negligible.
                next_t = max(now + interval * 0.5, next_t + interval)
                ok, frame = cap.read()
                if not ok or frame is None or not getattr(frame, "size", 0):
                    read_failures += 1
                    if read_failures in (3, 10) or read_failures % 100 == 0:
                        log_warn(f"Camera read failed {read_failures}x in a row "
                                 f"- is another app holding the device?")
                    time.sleep(0.5)
                    continue
                read_failures = 0
                self._process_frame(frame)
        except Exception as exc:
            LOG.exception("capture loop crashed")
            log_err(f"Camera loop stopped unexpectedly: {exc}")
        finally:
            self._set_state(SceneState.STOPPED)
            log_info("Capture loop ended.")

    def _process_frame(self, frame) -> None:
        """Crop -> prepare -> temporal-median reference -> diff -> scene FSM."""
        self.frames_seen += 1
        roi_frame = self.roi.crop(frame)
        try:
            prepared = prepare_roi(roi_frame, ANALYZE_WIDTH)
        except Exception as exc:
            LOG.debug("prepare_roi failed: %s", exc)
            return
        now = time.monotonic()

        # Rolling temporal-median reference. Diffing a frame against the MEDIAN
        # of the last N frames, rather than against one previous frame, is the
        # biggest noise win available here: temporally independent impulses
        # cancel inside a median. Measured on this camera it cuts the
        # changed-pixel ratio tail (p99) from 5.4% to 0.005% - a ~1000x margin
        # - and it also fixes the old FSM defect, where every frame was
        # differenced against a frozen keyframe, so motion never appeared to
        # stop, the FSM never left MOTION and no board update was ever emitted.
        self._ref_window.append(prepared)
        if len(self._ref_window) < TEMPORAL_MEDIAN_WINDOW:
            if len(self._ref_window) == 2:
                log_info(f"Noise calibration: warming up the "
                         f"{TEMPORAL_MEDIAN_WINDOW}-frame median reference, then "
                         f"sampling {NOISE_CALIBRATION_S:.1f}s of sensor noise "
                         f"(keep the board still) ...")
            self._update_preview(roi_frame)
            return

        reference = self._median_reference()
        if reference is None:
            self._update_preview(roi_frame)
            return

        delta = frame_delta(prepared, reference)
        self.last_delta = delta

        if self.state == SceneState.CALIBRATING:
            if self._calib_started == 0.0:
                self._calib_started = now
            self._noise_ratios.append(delta.ratio)
            self._noise_mads.append(delta.mad)
            self._calibrate(now)
            self._update_preview(roi_frame)
            return

        self._advance_fsm(now, delta, prepared, roi_frame)
        self._update_preview(roi_frame)

    def _median_reference(self):
        """Per-pixel median of the reference window (exact; ~15 ms at 480px).

        This is the reference every frame is diffed against. Using the median
        rather than a single stored frame is what makes the noise tail vanish;
        the window is never "refilled" after a board is read, because a window
        holding copies of one frame would lose the independent-noise averaging
        that the median provides.
        """
        window = list(self._ref_window)
        if len(window) < 3:
            return None
        try:
            stack = np.stack(window).astype(np.float32, copy=False)
            return np.median(stack, axis=0).astype(np.uint8)
        except Exception as exc:
            LOG.debug("median reference failed: %s", exc)
            return None

    def _percentile(self, values, q: float) -> float:
        """Nearest-rank percentile over a small sample."""
        ordered = sorted(values)
        if not ordered:
            return 0.0
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return float(ordered[idx])

    def _calibrate(self, now: float) -> None:
        """Derive the significance + stillness thresholds from real board noise.

        Mirrors listener.py's room calibration, but calibrates BOTH floors,
        because the correct numbers depend on the sensor, the lighting and the
        ROI size - hard-coded constants cannot be trusted here. The old
        hard-coded MAD floor (1.6) was in fact a latent bug: it sat ABOVE a
        genuine one-line change (measured MAD 0.54) and, being ANDed with the
        ratio test, would have blocked exactly the updates the module exists to
        catch.

            ratio_sig = max(SIG_RATIO_FLOOR, SIG_RATIO_MULT * ratio_p95)
            still_eps = clamp(NOISE_MULTIPLIER * mad_p99, 0.4 .. 3.0)

        A perfectly still camera reports ~0 noise and the floors take over, so
        the detector never becomes hair-triggered just because the room is quiet.
        """
        if len(self._noise_ratios) < 4:
            return
        if now - self._calib_started < NOISE_CALIBRATION_S:
            return

        ratio_p95 = self._percentile(self._noise_ratios, 0.95)
        mad_p99 = self._percentile(self._noise_mads, 0.99)
        self.noise_floor = round(mad_p99, 3)
        self.noise_ratio_floor = round(ratio_p95, 5)
        self.thresholds = SceneThresholds(
            ratio_sig=float(max(SIG_RATIO_FLOOR, SIG_RATIO_MULT * ratio_p95)),
            mad_min=SIG_MAD_MIN,
            still_eps=float(min(3.0, max(0.4, NOISE_MULTIPLIER * mad_p99))),
            ratio_noise=float(ratio_p95),
        )
        self.still_eps = self.thresholds.still_eps
        self.last_motion_at = now
        self._set_state(SceneState.SCANNING)
        log_ok(f"Camera noise floor: ratio p95 {ratio_p95 * 100:.3f}%, "
               f"MAD p99 {mad_p99:.3f} (n={len(self._noise_ratios)}) -> "
               f"{self.thresholds.describe()}")

    def _advance_fsm(self, now: float, delta: FrameDelta, prepared,
                     roi_frame) -> None:
        """SCANNING -> MOTION -> SETTLE -> (emit) -> SCANNING."""
        th = self.thresholds
        moving = (delta.mad >= th.still_eps) or (delta.ratio >= th.ratio_sig)

        if delta.significant(th):
            self.changes_detected += 1
            self._pending_update = True
            self.last_significant_at = now
            if self.state != SceneState.MOTION:
                self.motion_started_at = now
            self.last_motion_at = now
            self._set_state(SceneState.MOTION)
            return

        if self.state == SceneState.MOTION:
            if moving:
                self.last_motion_at = now
            else:
                # Structure stopped changing: start the stabilisation timer.
                self._set_state(SceneState.SETTLE)
            return

        if self.state == SceneState.SETTLE:
            if moving:
                self.last_motion_at = now
                self._set_state(SceneState.MOTION)
                return
            quiet = now - self.last_motion_at
            dwell = now - self.motion_started_at
            if quiet >= self.stabilize_s:
                self._emit_update(roi_frame, prepared, delta, "stabilised")
            elif dwell >= self.max_dwell_s:
                self._emit_update(roi_frame, prepared, delta, "max-dwell")
            return

        # SCANNING: sub-threshold motion is a teacher crossing the frame, which
        # deliberately does NOT count as a board update.

    def _emit_update(self, roi_frame, prepared, delta: FrameDelta,
                     reason: str) -> None:
        """Queue a settled board for OCR, applying the two rejection filters."""
        self._pending_update = False
        self._set_state(SceneState.SCANNING)
        now = time.monotonic()

        # Layer 2: if the settled picture matches the board we already read,
        # the change was transient (someone walked past, a shadow, a hand).
        if self._last_emitted_frame is not None:
            try:
                reverted = ssim_score(prepared, self._last_emitted_frame)
            except Exception:
                reverted = 0.0
            if reverted >= REVERT_SSIM:
                self.occlusions_rejected += 1
                log_info(f"Board change reverted (SSIM {reverted:.3f} vs last "
                         f"read) - transient occlusion ignored "
                         f"({self.occlusions_rejected} total)")
                return

        if now - self.last_ocr_at < MIN_OCR_INTERVAL_S:
            self.frames_skipped += 1
            return

        self.last_ocr_at = now
        try:
            self._job_queue.put_nowait((roi_frame.copy(), prepared.copy(), delta))
        except queue.Full:
            self.jobs_dropped += 1
            log_warn("OCR queue full - dropped one board update "
                     f"({self.jobs_dropped} total); the OCR engine is slower "
                     "than the lecture. Try --ocr winocr.")
            return
        log_info(f"Board settled ({reason}: SSIM {delta.ssim:.3f}, "
                 f"MAD {delta.mad:.2f}, pixels {delta.ratio * 100:.2f}%) "
                 f"-> queued for OCR")

    # ------------------------------------------------------------------ OCR
    def _ocr_loop(self) -> None:
        """Single OCR worker: reads settled boards only, never every frame."""
        log_ok(f"OCR worker running on '{self.engine.name}' "
               f"(thread '{threading.current_thread().name}').")
        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if job is None:
                break
            roi_frame, prepared, delta = job
            self.ocr_calls += 1
            t0 = time.perf_counter()
            try:
                result = self.engine.recognize(roi_frame)
            except Exception as exc:
                LOG.exception("OCR engine crashed")
                result = OcrResult(engine=self.engine.name,
                                   error=f"{type(exc).__name__}: {exc}"[:200])
            wall_ms = int((time.perf_counter() - t0) * 1000)
            try:
                self._store_snapshot(roi_frame, prepared, delta, result, wall_ms)
            except Exception as exc:
                LOG.exception("snapshot store failed")
                log_err(f"Could not store board snapshot: {exc}")
        log_info("OCR worker ended.")


# ------------------------------------------------------------ publishing
    def _store_snapshot(self, roi_frame, prepared, delta: FrameDelta,
                        result: OcrResult, wall_ms: int) -> None:
        """Turn one OCR result into a published board snapshot.

        Runs only on the OCR worker. `_last_emitted_frame` is updated here so
        the capture side can recognise a *reverted* (transient) scene change -
        the trick that makes "teacher walked past the board" a non-event.
        """
        self._last_emitted_frame = prepared
        self._last_ocr_lines = list(result.lines)

        if result.error:
            self.ocr_failures += 1
            log_warn(f"OCR engine '{result.engine}' returned an error: "
                     f"{result.error}")

        if result.engine.startswith("ollama"):
            # A multimodal model answers in prose, so keep its lines verbatim.
            lines = [s.strip() for s in result.texts if s and s.strip()]
        else:
            lines = merge_line_fragments(
                clean_ocr_lines([ln.text for ln in result.lines]))

        hints = diagram_hints(roi_frame)
        buckets = classify_lines(lines)

        with self._lock:
            previous = self._latest
        known = token_set(" ".join(previous.texts)) if previous else set()
        new_items = []
        for line in lines:
            tokens = token_set(line)
            if tokens and not tokens.issubset(known):
                new_items.append(line)

        summary = build_summary(buckets, hints, new_items)
        ocr_ms = int(result.elapsed_ms or wall_ms or 0)

        with self._board_cond:
            self._revision += 1
            snapshot = BoardSnapshot(
                summary=summary,
                texts=list(lines),
                buckets=buckets,
                new_items=list(new_items),
                hints=hints,
                engine=result.engine,
                ocr_ms=ocr_ms,
                ssim=delta.ssim,
                mad=delta.mad,
                revision=self._revision,
                capture_epoch=time.time(),
                roi=self.roi.line(),
            )
            self._snapshots.append(snapshot)
            self._latest = snapshot
            # Bound the context window in time, exactly like listener.py's
            # rolling transcript, so a 3-hour lecture cannot grow memory.
            cutoff = time.monotonic() - CONTEXT_WINDOW_S
            while self._snapshots and self._snapshots[0].stored_at < cutoff:
                self._snapshots.popleft()
            self._board_cond.notify_all()

        self.publish_to_hud(snapshot)
        if self.save_frames:
            self._save_debug_frame(roi_frame, result, snapshot)

        if snapshot.is_empty_board:
            self.blank_boards += 1
            log_info(f"Board settled but OCR found nothing useful "
                     f"({ocr_ms} ms, '{result.engine}') - nothing injected.")
        else:
            self.boards_updated += 1
            self.ocr_ms_total += ocr_ms
            body = summary
            prefix = "[BOARD CONTEXT]: "
            if body.startswith(prefix):
                body = body[len(prefix):]
            log_board(f"[rev {snapshot.revision}] ({ocr_ms} ms, "
                      f"{len(lines)} lines) {body}")
            if new_items:
                log_info("Just added: " + "; ".join(new_items[:4]))

    def publish_to_hud(self, snapshot: BoardSnapshot) -> bool:
        """Push one settled board into the Phase 5 HTML cockpit (optional).

        Returns True when the HUD accepted it. Failures are rate-limited to a
        single warning so a HUD that was never started cannot flood the log
        once per board revision.
        """
        if not self.web_publish:
            return False
        if self._webui is None:
            try:
                import webui
            except Exception as exc:
                if not self._web_warned:
                    log_warn(f"--web-publish disabled: cannot import webui.py "
                             f"({exc})")
                    self._web_warned = True
                return False
            self._webui = webui
        try:
            payload = self._webui.board_payload(snapshot)
        except Exception as exc:
            log_warn(f"board payload mapping failed: {exc}")
            return False
        ok = self._webui.publish_board(payload, host=self.web_host,
                                       port=self.web_port)
        if ok:
            self._web_failures = 0
            if self.verbose:
                log_info(f"Board rev {snapshot.revision} pushed to HUD "
                         f"{self.web_host}:{self.web_port}")
            return True
        self._web_failures += 1
        if self._web_failures == 3 and not self._web_warned:
            self._web_warned = True
            log_warn(f"--web-publish is not reaching a HUD on "
                     f"{self.web_host}:{self.web_port}; start "
                     f"'python main.py --web-ui' or drop the flag. "
                     f"Vision keeps running regardless.")
        return False

    def _save_debug_frame(self, roi_frame, result: OcrResult,
                          snapshot: BoardSnapshot) -> None:
        """Optional annotated frame dump (--save-frames) for offline tuning."""
        try:
            BOARD_DIR.mkdir(parents=True, exist_ok=True)
            annotated = roi_frame.copy()
            for line in result.lines:
                if line.w <= 0 or line.h <= 0:
                    continue
                cv2.rectangle(annotated, (line.x, line.y),
                              (line.x + line.w, line.y + line.h),
                              (60, 200, 60), 2)
            stamp = datetime.now().strftime("%H%M%S")
            path = BOARD_DIR / f"board_{stamp}_rev{snapshot.revision:04d}.png"
            cv2.imwrite(str(path), annotated)
            LOG.info("saved debug frame %s", path)
        except Exception as exc:
            LOG.debug("debug frame save failed: %s", exc)


# ------------------------------------------------------- preview & status
    def _update_preview(self, roi_frame) -> None:
        """Stash the newest ROI frame for the UI thread (never blocks capture)."""
        if not self.gui:
            return
        with self._preview_lock:
            self._preview_frame = roi_frame

    def _open_preview_window(self) -> None:
        """Create the --gui window now so failures are reported at startup."""
        try:
            cv2.namedWindow(PREVIEW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_TITLE, 720, 405)
            log_info("Preview window opened; press 'q' inside it or Ctrl+C in "
                     "the console to stop.")
        except Exception as exc:
            self.gui = False
            log_warn(f"Could not open the preview window ({exc}) - continuing "
                     "headless. Windows in a service/SSH session cannot be shown.")

    def _annotate(self, frame):
        """Draw OCR boxes + the live change score over the preview frame."""
        try:
            annotated = frame.copy()
            fh, fw = annotated.shape[:2]
            for line in list(self._last_ocr_lines):
                if line.w <= 0 or line.h <= 0:
                    continue
                cv2.rectangle(annotated, (line.x, line.y),
                              (line.x + line.w, line.y + line.h),
                              (60, 220, 60), 2)
                if line.text:
                    cv2.putText(annotated, line.text[:30],
                                (line.x, max(12, line.y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 60),
                                1, cv2.LINE_AA)
            delta = self.last_delta
            band = 26
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (fw, band), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
            text = (f"{self.state.name}  change={delta.change:.4f}  "
                    f"mad={delta.mad:.2f}  px={delta.ratio * 100:.2f}%  "
                    f"eps={self.still_eps:.2f}  rev={self._revision}  "
                    f"boards={self.boards_updated}  ocr={self.ocr_calls}")
            cv2.putText(annotated, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            return annotated
        except Exception as exc:
            LOG.debug("annotate failed: %s", exc)
            return frame

    def _print_status(self) -> None:
        """One-line live dashboard on the console (mirrors Phase 2's statuses)."""
        delta = self.last_delta
        line = (f"[STATUS: {self.state.name}] change {delta.change:.4f} | "
                f"mad {delta.mad:.2f} | eps {self.still_eps:.2f} | "
                f"rev {self._revision} | boards {self.boards_updated} | "
                f"ocr {self.ocr_calls} | rej {self.occlusions_rejected} | "
                f"frames {self.frames_seen} | engine {self.engine.name if self.engine else '-'}")
        if self.gui:
            # The window already shows these numbers; keep the console for events.
            return
        print("\r" + _c(line[:130].ljust(130), C.CYAN), end="", flush=True)

    def start_dashboard(self) -> "VisionTracker":
        """Start the single UI thread (console status line and/or --gui window).

        One thread only: OpenCV window calls must all originate from the same
        thread on Windows, and keeping it separate from capture means a slow
        `waitKey`/`imshow` can never stall the camera sampler.
        """
        if self._dash_thread is not None and self._dash_thread.is_alive():
            return self
        self._gui_stop.clear()
        self._dash_thread = threading.Thread(
            target=self._gui_loop, name="vision-ui", daemon=True)
        self._dash_thread.start()
        return self

    def _gui_loop(self) -> None:
        """UI thread body: paint the window, handle 'q', refresh the status."""
        draw = bool(self.gui)
        last_status = 0.0
        try:
            while not self._gui_stop.is_set() and not self._stop_event.is_set():
                with self._preview_lock:
                    frame = None if self._preview_frame is None \
                        else self._preview_frame.copy()
                if draw and frame is not None:
                    try:
                        cv2.imshow(PREVIEW_TITLE, self._annotate(frame))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            log_info("'q' pressed - stopping the vision tracker.")
                            self._gui_stop.set()
                            self._stop_event.set()
                            break
                    except Exception as exc:
                        log_warn(f"Preview window closed ({exc}) - continuing "
                                 "headless.")
                        draw = False
                now = time.monotonic()
                if now - last_status >= DASH_INTERVAL_S:
                    last_status = now
                    self._print_status()
                time.sleep(0.05)
        except Exception as exc:
            LOG.exception("UI loop crashed")
            log_err(f"UI loop stopped unexpectedly: {exc}")
        finally:
            if draw:
                self._close_preview()

    def _close_preview(self) -> None:
        """Release every OpenCV window handle (Windows needs the waitKey pump)."""
        try:
            cv2.destroyWindow(PREVIEW_TITLE)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            cv2.waitKey(1)
        except Exception:
            pass
# ------------------------------------------------------------- public API
    def get_visual_context(self, max_age_s: float | None = CONTEXT_WINDOW_S) -> str:
        """Latest `[BOARD CONTEXT]: ...` string; "" when unavailable or stale.

        This is the method Phase 3 interpolates into its Socratic prompt:

            ctx = vision.get_visual_context()
            prompt = f"[BOARD SHOWS]: {ctx} | [TEACHER SAID]: {transcript}"

        Returning "" rather than None means the caller can format it
        unconditionally. `max_age_s` guards freshness: a board that has not
        changed for longer than the window is treated as already-known rather
        than re-injected as if it were new.
        """
        with self._lock:
            snapshot = self._latest
            if snapshot is None:
                return ""
            if max_age_s is not None and snapshot.age_s() > max_age_s:
                return ""
            return snapshot.summary or ""

    def latest_snapshot(self) -> BoardSnapshot | None:
        """Newest snapshot object (thread-safe) or None before the first read."""
        with self._lock:
            return self._latest

    def snapshot_window(self) -> list:
        """Snapshots from the last CONTEXT_WINDOW_S seconds, oldest first."""
        with self._lock:
            cutoff = time.monotonic() - CONTEXT_WINDOW_S
            while self._snapshots and self._snapshots[0].stored_at < cutoff:
                self._snapshots.popleft()
            return list(self._snapshots)

    def board_revision(self) -> int:
        """Monotonic counter; lets Phase 3 cheaply detect 'nothing new'."""
        with self._lock:
            return self._revision

    def wait_for_update(self, revision: int = 0,
                        timeout: float | None = None) -> BoardSnapshot | None:
        """Block until a snapshot newer than `revision` exists (thread-safe).

        Returns immediately when the current snapshot is already newer, so a
        caller can poll cheaply without losing events. None means timeout or
        shutdown.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._board_cond:
            while True:
                snapshot = self._latest
                if snapshot is not None and snapshot.revision > revision:
                    return snapshot
                if self._stop_event.is_set():
                    return None
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._board_cond.wait(timeout=remaining)

    def iter_board_updates(self, timeout: float | None = None):
        """Generator of BoardSnapshots as they are published.

        Usage in Phase 3 (its own thread, or driven inline):
            for snap in vt.iter_board_updates():
                log(snap.summary)
        `timeout` applies per wait so the loop can re-check its stop condition.
        """
        last = 0
        while not self._stop_event.is_set():
            snapshot = self.wait_for_update(revision=last, timeout=timeout)
            if snapshot is None:
                if self._stop_event.is_set():
                    return
                continue
            last = snapshot.revision
            yield snapshot

    def stats(self) -> dict:
        """Runtime counters, mirroring listener.ListenerEngine.stats()."""
        with self._lock:
            revision = self._revision
        avg_ocr = int(self.ocr_ms_total / self.boards_updated) \
            if self.boards_updated else 0
        return {
            "state": self.state.name,
            "running": self.running,
            "camera": f"{self.camera_index} via {self._backend}",
            "roi": self.roi.line(),
            "fps_target": self.fps,
            "frames_seen": self.frames_seen,
            "frames_skipped": self.frames_skipped,
            "changes_detected": self.changes_detected,
            "occlusions_rejected": self.occlusions_rejected,
            "jobs_dropped": self.jobs_dropped,
            "ocr_calls": self.ocr_calls,
            "ocr_failures": self.ocr_failures,
            "boards_updated": self.boards_updated,
            "blank_boards": self.blank_boards,
            "board_revision": revision,
            "avg_ocr_ms": avg_ocr,
            "engine": (self.engine.name if self.engine
                       else self._engine_name),
            "noise_floor": self.noise_floor,
            "still_eps": round(self.still_eps, 3),
            "uptime_s": round(time.monotonic() - self._started_at, 1)
            if self._started_at else 0.0,
        }

    def summary_lines(self) -> list:
        """Human-readable session summary printed by stop()/the CLI."""
        st = self.stats()
        return [
            f"runtime .......: {st['uptime_s']:.0f}s "
            f"({st['frames_seen']} frames inspected at {st['fps_target']:g} "
            f"FPS target)",
            f"camera ........: {st['camera']} | ROI {st['roi']}",
            f"OCR engine ....: {st['engine']} ({st['ocr_calls']} calls, "
            f"avg {st['avg_ocr_ms']} ms, {st['ocr_failures']} failures)",
            f"board updates .: {st['boards_updated']} published, "
            f"{st['blank_boards']} blank",
            f"rejected ......: {st['occlusions_rejected']} transient changes, "
            f"{st['frames_skipped']} rate-limited",
            f"queue .........: {st['jobs_dropped']} dropped jobs",
        ]

    # ------------------------------------------------------------------ stop
    def stop(self, join_timeout: float = 5.0) -> None:
        """Graceful shutdown: camera -> workers -> UI, then every OpenCV handle.

        Order matters on Windows. The capture device is released first so no new
        frames can arrive while the worker drains; then the OCR thread is
        unblocked; then the UI thread (which owns the window) is stopped last,
        because `destroyAllWindows` must be the final OpenCV call. This mirrors
        listener.py's stream-first teardown.
        """
        if self._stopping:
            return
        self._stopping = True
        log_info("Shutting down vision tracker ...")
        self._stop_event.set()

        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                LOG.warning("cap.release(): %s", exc)
            log_ok("Camera released (VideoCapture.release).")

        try:
            self._job_queue.put_nowait(None)   # sentinel -> OCR worker exits
        except queue.Full:
            try:
                self._job_queue.get_nowait()
                self._job_queue.put_nowait(None)
            except Exception:
                pass

        for thread in (self._cam_thread, self._ocr_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    log_warn(f"{thread.name} did not exit in time (daemon).")

        self._gui_stop.set()
        if self._dash_thread is not None and self._dash_thread.is_alive():
            self._dash_thread.join(timeout=join_timeout)
        self._clear_status_line()
        self._close_preview()

        engine, self.engine = self.engine, None
        if engine is not None and hasattr(engine, "close"):
            try:
                engine.close()
            except Exception as exc:
                LOG.warning("engine.close(): %s", exc)

        with self._board_cond:
            self._board_cond.notify_all()
        self._set_state(SceneState.STOPPED)
        log_ok("Vision tracker stopped; all OpenCV handles released.")

    def _clear_status_line(self) -> None:
        """Blank the live status line so the console is not left mid-dashboard."""
        if self.dashboard and not self.gui:
            print("\r" + " " * 132 + "\r", end="", flush=True)

    def run(self, duration: float | None = None) -> "VisionTracker":
        """Blocking convenience loop used by the CLI (Ctrl+C stops it)."""
        self.start()
        deadline = None if not duration else time.monotonic() + duration
        log_info("Vision tracker active. Press Ctrl+C to stop.")
        try:
            while not self._stop_event.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    log_info(f"Duration {duration:g}s reached - stopping.")
                    break
                if self.gui and self._dash_thread is not None and \
                        not self._dash_thread.is_alive():
                    log_info("Preview window closed - stopping.")
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            log_warn("Ctrl+C received - stopping ...")
        finally:
            self.stop()
        return self



# ===========================================================================
# CLI
# ===========================================================================
# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vision_tracker.py",
        description=("Phase 4 - optional CPU-only blackboard vision companion. "
                     "Watches a camera, detects settled board updates and "
                     "publishes OCR'd context for the Socratic LLM."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python vision_tracker.py                     live dashboard\n"
            "  python vision_tracker.py --gui               + annotated preview\n"
            "  python vision_tracker.py --calibrate         drag the board ROI\n"
            "  python vision_tracker.py --check             diagnose engines/deps\n"
            "  python vision_tracker.py --list-cameras      find camera indexes\n"
            "  python vision_tracker.py --camera 1 --fps 2 --duration 120\n"
            "  python vision_tracker.py --from-image b.png  offline OCR test\n"
            "\n"
            "exit codes: 0 clean stop | 1 fatal error | 2 no OCR engine | "
            "130 Ctrl+C\n"),
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX_DEFAULT,
                        metavar="N", help="camera index (default: 0); "
                        "external USB capture cards are usually 1, 2, ...")
    parser.add_argument("--fps", type=float, default=CAPTURE_FPS, metavar="F",
                        help=f"inspection rate, 1-2 is plenty (default: "
                             f"{CAPTURE_FPS:g})")
    parser.add_argument("--roi", default=None, metavar="x,y,w,h",
                        help="blackboard ROI crop; 'full' for the whole frame. "
                             "Defaults to the saved vision_roi.json if present")
    parser.add_argument("--ocr", default="auto", choices=OCR_ENGINES,
                        help="OCR engine (default: auto = best available)")
    parser.add_argument("--vision-model", default=DEFAULT_OLLAMA_VISION_MODEL,
                        metavar="NAME",
                        help="Ollama multimodal model for --ocr ollama "
                             f"(default: {DEFAULT_OLLAMA_VISION_MODEL})")
    parser.add_argument("--ollama-host", default=OLLAMA_HOST_DEFAULT,
                        metavar="URL", help="Ollama base URL")
    parser.add_argument("--stabilize", type=float, default=STABILIZE_S,
                        metavar="S", help="seconds of stillness before a board "
                        f"counts as updated (default: {STABILIZE_S:g})")
    parser.add_argument("--max-dwell", type=float, default=MAX_DWELL_S,
                        metavar="S", help="force a read during non-stop writing "
                        f"(default: {MAX_DWELL_S:g})")
    parser.add_argument("--width", type=int, default=CAPTURE_WIDTH,
                        help=f"requested capture width (default: {CAPTURE_WIDTH})")
    parser.add_argument("--height", type=int, default=CAPTURE_HEIGHT,
                        help=f"requested capture height (default: {CAPTURE_HEIGHT})")
    parser.add_argument("--cv-threads", type=int, default=CV_THREADS_DEFAULT,
                        metavar="N", help="OpenCV worker threads; kept low so "
                        "faster-whisper (4) and Kokoro keep the CPU "
                        f"(default: {CV_THREADS_DEFAULT})")
    parser.add_argument("--duration", type=float, default=None, metavar="S",
                        help="stop automatically after S seconds")
    parser.add_argument("--gui", action="store_true",
                        help="show the annotated preview window (bounding boxes "
                             "+ change scores); press q in the window to quit")
    parser.add_argument("--save-frames", action="store_true",
                        help=f"keep each OCR'd board frame as PNG in "
                             f"{BOARD_DIR.name}/")
    parser.add_argument("--verbose", action="store_true",
                        help="debug logging to logs/vision.log")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print board snapshots as JSON lines")

    mode = parser.add_argument_group("diagnostic modes (no live capture)")
    mode.add_argument("--check", action="store_true",
                      help="report OpenCV/OCR engine/ROI readiness and exit")
    mode.add_argument("--list-cameras", action="store_true",
                      help="probe camera indexes 0-4 and exit")
    mode.add_argument("--probe-limit", type=int, default=5, metavar="N",
                      help="how many camera indexes to probe in --check / "
                           "--list-cameras (default: 5; each probe is an "
                           "isolated child process, ~2 s)")
    mode.add_argument("--calibrate", action="store_true",
                      help="open a frame and drag the board ROI with the mouse, "
                           "then save vision_roi.json")
    mode.add_argument("--from-image", default=None, metavar="PATH",
                      help="run the OCR pipeline on a still image and exit "
                           "(no camera needed)")
    return parser.parse_args(argv)


def _print_banner() -> None:
    print()
    print(_c("=" * 74, C.BLUE + C.BOLD))
    print(_c("  TA Phase 4 - vision_tracker.py  |  CPU-only board monitor",
             C.BLUE + C.BOLD))
    print(_c("  GPU stays reserved for Ollama llama3.2:3b (4 GB VRAM)",
             C.GREY))
    print(_c("=" * 74, C.BLUE + C.BOLD))


# ---------------------------------------------------------------------------
# Diagnostic modes
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    """`--check`: report readiness without opening the camera stream for long."""
    _print_banner()
    print(f"\n{_c('OpenCV', C.BOLD)}")
    if not ensure_cv2(auto_install=False):
        log_err(f"OpenCV (cv2) is not importable: {CV2_IMPORT_ERROR}")
        log_warn("Fix: python -m pip install opencv-python")
        print("\nRESULT: NOT READY (no OpenCV)")
        return 1
    silence_cv2_logs()
    threads_before = getattr(cv2, "getNumThreads", lambda: 0)()
    cv2.setNumThreads(args.cv_threads)
    log_ok(f"cv2 {CV2_VERSION} (GUI build: "
           f"{'yes' if hasattr(cv2, 'imshow') else 'no'}, "
           f"workers {threads_before} -> {cv2.getNumThreads()} "
           f"so faster-whisper keeps its 4)")
    if not hasattr(cv2, "imshow"):
        log_warn("This cv2 build has no GUI support - `--gui` will not work "
                 "(install opencv-python rather than opencv-python-headless).")

    print(f"\n{_c('OCR engines', C.BOLD)}")
    candidates = probe_engines(host=args.ollama_host,
                               vision_model=args.vision_model,
                               verbose=True)

    print(f"\n{_c('Region of interest', C.BOLD)}")
    saved = load_roi_config()
    if args.roi:
        roi = parse_roi(args.roi)
        log_ok(f"--roi on the command line: {roi.line()}")
    elif saved is not None:
        roi = saved
        log_ok(f"vision_roi.json found: {roi.line()}")
    else:
        roi = ROIRect(0, 0, 0, 0)
        log_info("No vision_roi.json yet - analysing the full frame. Run "
                 "`--calibrate` to restrict analysis to the board "
                 "(faster OCR, fewer false triggers).")

    print(f"\n{_c('Camera', C.BOLD)}")
    tracker = VisionTracker(roi=roi, camera_index=args.camera,
                            width=args.width, height=args.height,
                            verbose=args.verbose)

    def _progress(index: int) -> None:
        print(f"{_c(_ts(), C.GREY)} {_c('  ..', C.CYAN)} probing index "
              f"{index} (isolated child, ~2 s)", flush=True)

    cameras = tracker._list_cameras(limit=args.probe_limit, status=_progress)
    if not cameras:
        log_warn(f"No camera responded on indexes 0-{args.probe_limit - 1}. "
                 "Fix: allow desktop apps in Settings > Privacy & security > "
                 "Camera, close any app holding the device, then re-run "
                 "`--list-cameras`.")
    else:
        for index, w, h, backend in cameras:
            log_ok(f"camera {index}: {w}x{h} via {backend}"
                   + ("   <- configured index" if index == args.camera else ""))

    engine, reason = select_engine(args.ocr, candidates)
    print()
    if engine is None:
        log_err(reason)
    else:
        log_ok(f"Selected OCR engine: {engine.describe()}")

    print(f"\n{_c('-' * 74, C.GREY)}")
    ready = engine is not None and bool(cameras)
    if ready:
        print(_c("RESULT: READY  (run `python vision_tracker.py` to start)",
                 C.GREEN + C.BOLD))
        return 0
    if engine is None:
        print(_c("RESULT: NOT READY  (no usable OCR engine)", C.RED + C.BOLD))
        return 2
    print(_c("RESULT: PARTIAL  (no camera detected)", C.YELLOW + C.BOLD))
    return 1


def cmd_list_cameras(args) -> int:
    """`--list-cameras`: probe indexes in isolated child processes."""
    _print_banner()
    require_cv2()
    silence_cv2_logs()
    limit = max(1, args.probe_limit)
    tracker = VisionTracker(roi=ROIRect(0, 0, 0, 0), camera_index=args.camera,
                            width=args.width, height=args.height)
    print(f"\n{_c(f'Probing camera indexes 0-{limit - 1} '
                   f'(each in its own child process, ~2 s) ...', C.BOLD)}")

    def _progress(index: int) -> None:
        print(f"{_c(_ts(), C.GREY)} {_c('  ..', C.CYAN)} index {index}",
              flush=True)

    found = tracker._list_cameras(limit=limit, status=_progress)
    if not found:
        log_warn("No camera answered.")
        log_info("Fixes: (1) Settings > Privacy & security > Camera > enable "
                 "'Let desktop apps access your camera'; (2) close Teams/Zoom/"
                 "Camera/Chrome tabs that may hold the device; (3) plug the "
                 "external capture card in and retry (indexes usually become 1 "
                 "or 2); (4) raise the search with --probe-limit 8.")
        return 1
    print()
    for index, w, h, backend in found:
        log_ok(f"index {index}: {w}x{h} via {backend}  "
               f"-> python vision_tracker.py --camera {index}")
    return 0


def cmd_calibrate(args) -> int:
    """`--calibrate`: grab a frame and let the user drag the board ROI."""
    _print_banner()
    require_cv2()
    if not hasattr(cv2, "selectROI"):
        log_err("This OpenCV build lacks selectROI (headless build). Fix: "
                "python -m pip install opencv-python")
        return 1
    tracker = VisionTracker(camera_index=args.camera, width=args.width,
                            height=args.height, verbose=args.verbose)
    cap, frame = tracker._open_camera()
    try:
        log_info("Drag a box around the blackboard / whiteboard / slide area, "
                 "then press ENTER. Press c to cancel.")
        window = "TA Phase 4 - drag the board ROI, ENTER to accept"
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, min(1280, frame.shape[1]),
                             min(720, frame.shape[0]))
            box = cv2.selectROI(window, frame, showCrosshair=False,
                                fromCenter=False)
        finally:
            try:
                cv2.destroyWindow(window)
            except Exception:
                pass
            cv2.destroyAllWindows()
        x, y, w, h = (int(v) for v in box)
        if w < 8 or h < 8:
            log_warn("No usable rectangle selected - keeping the previous ROI.")
            return 1
        roi = ROIRect(x, y, w, h)
        frame_h, frame_w = frame.shape[:2]
        log_ok(f"Selected {roi.line()} inside {frame_w}x{frame_h} "
               f"({100.0 * w * h / float(frame_w * frame_h):.0f}% of the frame)")
        crops_dir = BOARD_DIR / "roi_preview"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crops_dir / "roi_crop.png"
        cv2.imwrite(str(crop_path), roi.crop(frame))
        save_roi_config(roi)
        log_info(f"ROI crop written to {crop_path} - open it and confirm the "
                 "board text is inside the box.")
        return 0
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


def cmd_from_image(args) -> int:
    """`--from-image`: exercise OCR + summary on a still image, no camera."""
    _print_banner()
    require_cv2()
    path = Path(args.from_image)
    if not path.exists():
        log_err(f"Image not found: {path}")
        return 1
    frame = cv2.imread(str(path))
    if frame is None:
        log_err(f"cv2.imread could not decode {path} "
                "(unsupported format or truncated file).")
        return 1
    roi = parse_roi(args.roi) if args.roi else (load_roi_config()
                                                or ROIRect(0, 0, 0, 0))
    log_info(f"Loaded {path.name} {frame.shape[1]}x{frame.shape[0]} "
             f"| ROI {roi.line()}")

    candidates = probe_engines(host=args.ollama_host,
                               vision_model=args.vision_model, verbose=True)
    engine, reason = select_engine(args.ocr, candidates)
    if engine is None:
        log_err(reason)
        return 2
    log_ok(f"Using {engine.describe()}")

    roi_frame = roi.crop(frame)
    hints = diagram_hints(roi_frame)
    result = engine.recognize(roi_frame)
    if result.error:
        log_err(f"OCR failed: {result.error}")
        return 1
    cleaned = merge_line_fragments(
        clean_ocr_lines([ln.text for ln in result.lines]))
    buckets = classify_lines(cleaned)
    summary = build_summary(buckets, hints, cleaned)

    print(f"\n{_c('Raw OCR lines', C.BOLD)} "
          f"({result.elapsed_ms} ms, mean score {result.mean_score:.2f})")
    if not result.lines:
        log_warn("OCR returned no text. Try `--roi` to crop tighter, raise the "
                 "camera resolution, or improve lighting/contrast.")
    for line in result.lines:
        print(f"  {_c(f'{line.score:.2f}', C.GREY)} "
              f"{_c(f'({line.x},{line.y},{line.w},{line.h})', C.GREY)} "
              f"{line.text}")

    print(f"\n{_c('Empty board?', C.BOLD)} "
          f"{'yes' if not cleaned else 'no'}")
    print(f"\n{_c('Diagram hints', C.BOLD)}: {hints}")
    print(f"\n{_c('LLM context that would be injected', C.BOLD)}")
    print("  " + (summary if summary
                  else _c("(nothing - board looks blank)", C.GREY)))

    if args.as_json:
        payload = {
            "image": str(path),
            "engine": result.engine,
            "elapsed_ms": result.elapsed_ms,
            "lines": [vars(ln) for ln in result.lines],
            "cleaned": cleaned,
            "buckets": buckets,
            "hints": hints,
            "context": summary,
        }
        print("\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _live_printer(tracker: "VisionTracker", as_json: bool,
                  stop_event: threading.Event) -> None:
    """Consume published board snapshots and print them (CLI mode only)."""
    try:
        for snapshot in tracker.iter_board_updates(timeout=0.5):
            if stop_event.is_set():
                return
            print("\r" + " " * 132 + "\r", end="", flush=True)
            if as_json:
                print(json.dumps(snapshot.to_dict(), ensure_ascii=False),
                      flush=True)
                continue
            meta = "{} {} ms, SSIM {:.3f}".format(
                snapshot.engine or "?", snapshot.ocr_ms, snapshot.ssim)
            header = "{} {} {}".format(
                _c(snapshot.timestamp(), C.GREY),
                _c("rev " + str(snapshot.revision), C.CYAN),
                _c("(" + meta + ")", C.GREY))
            log_board(header)
            if snapshot.is_empty_board:
                print("    " + _c("(board erased or unreadable)", C.YELLOW))
            else:
                print("    " + _c(snapshot.summary, C.BOLD))
                if snapshot.new_items:
                    new_text = "new: " + "; ".join(snapshot.new_items)
                    print("    " + _c(new_text, C.GREEN))
            tail_info = "{} text lines | age {}".format(
                len(snapshot.texts), snapshot.age_str)
            print("    " + _c(tail_info, C.GREY))
    except Exception as exc:      # a printer must never kill the daemon
        LOG.debug("live printer stopped: %s", exc)


def _print_session_summary(tracker: "VisionTracker") -> None:
    """Final report: counters + the exact Phase 3 wiring to copy."""
    print(f"\n{_c('-' * 74, C.GREY)}")
    print(_c("SESSION SUMMARY", C.BOLD))
    for line in tracker.summary_lines():
        print("  " + line)
    print(f"  last context ..: {tracker.get_visual_context() or '(none)'}")
    print(f"\n  {_c('Phase 3 integration', C.BOLD)}")
    print(f"    {_c('from vision_tracker import VisionTracker', C.CYAN)}")
    print(f"    {_c('vt = VisionTracker().start()', C.CYAN)}")
    print(f"    {_c('ctx = vt.get_visual_context()', C.CYAN)}")
    print(f"  {_c('Inject as:', C.BOLD)} "
          "\"[BOARD SHOWS]: {ctx} | [TEACHER SAID]: {transcript}\"")


def _resolve_roi(args) -> "ROIRect":
    """CLI --roi > saved vision_roi.json > full frame."""
    if args.roi:
        return parse_roi(args.roi)
    saved = load_roi_config()
    if saved is not None:
        log_info(f"Using saved ROI from vision_roi.json: {saved.line()}")
        return saved
    log_info("No ROI configured - analysing the full frame. `--calibrate` "
             "restricts analysis to the board area.")
    return ROIRect(0, 0, 0, 0)


def main(argv=None) -> int:
    args = parse_args(argv)
    _force_utf8_stdio()
    _enable_windows_vt()
    _setup_file_log(args.verbose)

    if args.check:
        return cmd_check(args)
    if args.list_cameras:
        return cmd_list_cameras(args)
    if args.calibrate:
        return cmd_calibrate(args)
    if args.from_image:
        return cmd_from_image(args)

    # ---------------------------------------------------------- live capture
    _print_banner()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    try:
        roi = _resolve_roi(args)
    except ValueError as exc:
        log_err(str(exc))
        return 1

    tracker = VisionTracker(
        camera_index=args.camera, fps=args.fps, roi=roi, ocr_pref=args.ocr,
        vision_model=args.vision_model, ollama_host=args.ollama_host,
        stabilize_s=args.stabilize, max_dwell_s=args.max_dwell,
        width=args.width, height=args.height, gui=args.gui,
        save_frames=args.save_frames, cv_threads=args.cv_threads,
        verbose=args.verbose)
    tracker.dashboard = not args.gui and not args.as_json

    # Ctrl+C must tear the camera down. Turning the signal into a flag (rather
    # than letting KeyboardInterrupt unwind anywhere) means stop() always runs
    # in a known place and the webcam handle is always released - which is what
    # otherwise leaves a camera "busy" on Windows until the machine reboots.
    interrupted = threading.Event()

    def _on_sigint(signum, frame):     # pragma: no cover - signal path
        interrupted.set()
        log_warn("Ctrl+C received - stopping ...")

    previous = None
    try:
        previous = signal.signal(signal.SIGINT, _on_sigint)
    except Exception as exc:
        LOG.debug("could not install SIGINT handler: %s", exc)

    printer = None
    try:
        tracker.start()
        if not args.gui:
            printer = threading.Thread(
                target=_live_printer, name="vision-printer",
                args=(tracker, args.as_json, interrupted), daemon=True)
            printer.start()
        log_info("Watching for board updates. Press Ctrl+C to stop."
                 + ("  (preview window: press q)" if args.gui else ""))
        deadline = (None if not args.duration
                    else time.monotonic() + args.duration)
        while not tracker._stop_event.is_set():
            if interrupted.is_set():
                break
            if deadline is not None and time.monotonic() >= deadline:
                log_info(f"Duration {args.duration:g}s reached - stopping.")
                break
            if args.gui and tracker._dash_thread is not None and \
                    not tracker._dash_thread.is_alive():
                log_info("Preview window closed - stopping.")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:          # pragma: no cover - interactive path
        log_warn("Ctrl+C received - stopping ...")
    except Exception as exc:
        LOG.exception("vision tracker crashed")
        log_err(f"Vision tracker failed: {exc}")
        try:
            tracker.stop()
        except Exception:
            pass
        return 1
    finally:
        try:
            tracker.stop()
        except Exception as exc:
            LOG.warning("stop() raised: %s", exc)
        if printer is not None and printer.is_alive():
            printer.join(timeout=1.0)
        if previous is not None:
            try:
                signal.signal(signal.SIGINT, previous)
            except Exception:
                pass

    _print_session_summary(tracker)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:          # pragma: no cover - interactive path
        print()
        log_warn("Interrupted before startup completed.")
        sys.exit(130)
    except RuntimeError as exc:
        log_err(str(exc))
        sys.exit(1)
