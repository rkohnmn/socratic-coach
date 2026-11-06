"""Phase 5 - HTML cockpit HUD for the Socratic TA (standard library only).

Implements the "SocraticCoach Studio System" from DESIGN.md as a local web HUD.
A threaded HTTP server bound to loopback streams live orchestrator state to
hud.html over Server-Sent Events and accepts single-keystroke control actions
back over a small JSON API.

Design contract enforced here and in hud.html:
  * OKLCH dark-slate tokens, 1px border elevation, no glassmorphism.
  * Semantic status hues are exclusive (green=listen, amber=think,
    violet=speak, red=mute) - DESIGN.md 4.2.
  * Compositor-only animation; the browser never animates layout.
  * Real microphone energy for the waveform, or an explicit unavailable
    state - never a fabricated idle render (DESIGN.md 9).

Transport (must match hud.html exactly):
  GET  /events       SSE, default 'message' events, {"kind":...,"data":...}
  POST /api/action   {"action": "ask"|"toggle_mute"|"dismiss"|"stop"}

Usage
-----
    python main.py --web-ui                  # orchestrator + HUD on :8765
    python webui.py --demo                   # UI smoke test, no mic/LLM
    python webui.py --selftest               # assertions only

The bridge is intentionally passive: it never imports main.py, listener.py or
vision_tracker.py, so those stay runnable on a machine with no HUD at all.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUD_HTML = HERE / "hud.html"
LOGS_DIR = HERE / "logs"
HUD_LOG = LOGS_DIR / "hud.log"

HUD_HOST = "127.0.0.1"          # loopback only: no firewall prompt, no LAN exposure
HUD_PORT = 8765
LEVEL_BARS = 64                 # waveform ring depth (DESIGN.md 7.4); hud.html mirrors this
TICK_HZ = 20.0                  # SSE telemetry tick; canvas interpolates to 60 fps
TRANSCRIPT_LIMIT = 500
ERROR_LIMIT = 40
HISTORICAL_AFTER_S = 60.0       # DESIGN.md 7.3 - older lines recede
PEAK_REF_RMS = 4000.0           # int16 RMS treated as full-scale energy
CLIENT_QUEUE = 256              # per-client SSE backlog before the client is dropped
TELEMETRY_PERIOD_S = 2.0
SMI_TIMEOUT_S = 6.0             # nvidia-smi stalls seconds when the GPU is saturated
SSE_HEARTBEAT_S = 15.0
ACTIONS = ("ask", "toggle_mute", "dismiss", "stop")


# ---------------------------------------------------------------------------
# Logging (mirrors the per-module helper style used by listener.py/main.py)
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()


def _stamp() -> str:
    now = time.time()
    return (time.strftime("%H:%M:%S", time.localtime(now))
            + f".{int(now * 1000) % 1000:03d}")


def _log(level: str, msg: str) -> None:
    line = f"{_stamp()} {level} ta-hud {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with _log_lock:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(HUD_LOG, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass


def log_info(msg: str) -> None:
    _log("INFO ", msg)


def log_warn(msg: str) -> None:
    _log("WARN ", msg)


def log_err(msg: str) -> None:
    _log("ERROR", msg)


def log_ok(msg: str) -> None:
    _log("OK   ", msg)

# ---------------------------------------------------------------------------
# Hardware telemetry (VRAM via nvidia-smi, CPU via GetSystemTimes)
# ---------------------------------------------------------------------------
class _CpuSampler:
    """Windows CPU load via kernel32!GetSystemTimes deltas.

    psutil is not a dependency of this project, so the sampler degrades to an
    explicit None (the HUD prints an em dash) rather than a fabricated number.
    """

    def __init__(self) -> None:
        self._ctypes = None
        self._filetime = None
        self._kernel32 = None
        self._prev: tuple | None = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLowDateTime", wintypes.DWORD),
                            ("dwHighDateTime", wintypes.DWORD)]

            self._ctypes = ctypes
            self._filetime = FILETIME
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._kernel32.GetSystemTimes.argtypes = [
                ctypes.POINTER(FILETIME)] * 3
            self._kernel32.GetSystemTimes.restype = wintypes.BOOL
        except Exception as exc:
            log_warn(f"CPU sampler unavailable: {exc}")
            self._kernel32 = None

    @staticmethod
    def _to_int(ft) -> int:
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    def sample(self) -> float | None:
        """Busy CPU percentage across all cores, or None when unmeasurable."""
        if self._kernel32 is None:
            return None
        idle, kernel, user = (self._filetime(), self._filetime(),
                              self._filetime())
        if not self._kernel32.GetSystemTimes(self._ctypes.byref(idle),
                                             self._ctypes.byref(kernel),
                                             self._ctypes.byref(user)):
            return None
        cur = (self._to_int(idle), self._to_int(kernel), self._to_int(user))
        prev, self._prev = self._prev, cur
        if prev is None:
            return None
        d_idle = cur[0] - prev[0]
        d_total = (cur[1] - prev[1]) + (cur[2] - prev[2])   # kernel includes idle
        if d_total <= 0:
            return None
        return round(max(0.0, min(100.0, 100.0 * (1.0 - (d_idle / d_total)))), 1)


class Telemetry:
    """Sampled GPU/CPU facts for the header pills. Never raises.

    `nvidia-smi` normally returns in ~0.08 s, but it can stall for seconds
    while the GPU is saturated with inference or the display is reconfiguring.
    A failed query therefore KEEPS the previous reading and flags it as stale,
    rather than blanking the pill with None - a real measurement from a few
    seconds ago is honest; None would be a lie of omission that flickers the
    header on every GPU hiccup.
    """

    def __init__(self, period_s: float = TELEMETRY_PERIOD_S,
                 timeout: float = SMI_TIMEOUT_S) -> None:
        self.period_s = period_s
        self.timeout = timeout
        self._smi = shutil.which("nvidia-smi")
        self._cpu = _CpuSampler()
        self._lock = threading.Lock()
        self._data = {"vram_used_mb": None, "vram_total_mb": None,
                      "gpu": None, "cpu_pct": None, "cores": os.cpu_count()}
        self._vram_ok = False           # has a GPU reading ever succeeded?
        self._vram_ts = 0.0             # monotonic stamp of the last good one
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        if self._smi is None:
            log_warn("nvidia-smi not found - VRAM pill stays unavailable")

    @property
    def data(self) -> dict:
        with self._lock:
            out = dict(self._data)
            if not self._vram_ok:
                # never measured: the client shows an explicit dash
                out["vram_stale"] = True
            else:
                # staleness is judged against the sampler's own cadence, so the
                # client never has to guess what "fresh" means for this machine
                age = time.monotonic() - self._vram_ts
                out["vram_stale"] = age > max(3.0 * self.period_s, self.timeout)
            return out

    def _query_vram(self) -> tuple:
        """One nvidia-smi query -> (gpu, used_mb, total_mb), or None on failure."""
        out = subprocess.run(
            [self._smi, "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=self.timeout, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 3:
            return None
        return parts[0], int(float(parts[1])), int(float(parts[2]))

    def sample_once(self, timeout: float | None = None) -> dict:
        if timeout is not None:
            self.timeout = timeout
        gpu = vram_used = vram_total = None
        if self._smi:
            try:
                got = self._query_vram()
                if got is not None:
                    gpu, vram_used, vram_total = got
            except subprocess.TimeoutExpired:
                log_warn(f"nvidia-smi exceeded {self.timeout:g}s "
                         "(GPU busy) - keeping last VRAM reading")
            except Exception as exc:
                log_warn(f"nvidia-smi query failed: {exc}")
        with self._lock:
            if vram_used is not None:
                self._data.update({"vram_used_mb": vram_used,
                                   "vram_total_mb": vram_total, "gpu": gpu})
                self._vram_ok = True
                self._vram_ts = time.monotonic()
            self._data["cpu_pct"] = self._cpu.sample()
            return dict(self._data)

    def _loop(self) -> None:
        """Sample immediately, then every `period_s`. Never blocks a caller."""
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:
                log_warn(f"telemetry sample failed: {exc}")
            self._stop.wait(self.period_s)

    def start(self) -> "Telemetry":
        """Start sampling in the background.

        Deliberately does NOT prime synchronously: the first nvidia-smi call
        can stall for seconds on a busy GPU, and this is called from the
        orchestrator's startup path where that delay would postpone the
        microphone coming up and drop the first teacher utterance.
        """
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop,
                                        name="ta-hud-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

# ---------------------------------------------------------------------------
# Level mapping (int16 RMS -> 0..1 bar heights)
# ---------------------------------------------------------------------------
def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _as_str_list(value) -> list:
    """Coerce an arbitrary JSON value into a clean list of short strings.

    Used for the board payload so a malformed push from another process can
    never raise inside a request handler.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text[:400])
    return out


def _to_norm(rms: float) -> float:
    """Map an int16-domain RMS to 0..1 with a perceptual square root."""
    try:
        return _clamp01((float(rms) / PEAK_REF_RMS) ** 0.5)
    except Exception:
        return 0.0


def normalise_levels(raw, floor_rms: float = 0.0, bars: int = LEVEL_BARS) -> list:
    """Coerce a raw level sequence into exactly `bars` normalised floats.

    Values at or below the measured noise floor collapse to 0 so the meter
    reflects speech rather than room tone; a malformed input yields all zeros
    (the client then shows its unavailable notice, never a fake waveform).
    """
    out = [0.0] * bars
    if not raw:
        return out
    try:
        floor_norm = _to_norm(floor_rms) if floor_rms else 0.0
        seq = list(raw)
        n = min(len(seq), bars)
        for i in range(n):
            try:
                v = _to_norm(float(seq[i]))
            except Exception:
                v = 0.0
            if v <= floor_norm:
                v = 0.0
            out[i] = round(_clamp01(v), 4)
    except Exception:
        return [0.0] * bars
    return out


# ---------------------------------------------------------------------------
# Live state + SSE fan-out
# ---------------------------------------------------------------------------
class WebBridge:
    """Thread-safe live HUD state with fan-out to SSE clients.

    The orchestrator thread is the only logical writer; HTTP handler threads
    read snapshots and drain per-client queues. Every mutation takes the lock,
    and `_emit` never blocks - a slow client is dropped rather than stalling
    the lecture loop (DESIGN.md 8 responsiveness).
    """

    def __init__(self, telemetry: "Telemetry | None" = None,
                 port: int = HUD_PORT) -> None:
        self._lock = threading.RLock()
        self._clients: dict = {}
        self._next_client = 0
        self._port = port
        self._started_ts = time.time()
        self._state = "idle"
        self._note = ""
        self._model = ""
        self._transcript: deque = deque(maxlen=TRANSCRIPT_LIMIT)
        self._line_id = 0
        self._question: dict | None = None
        self._errors: deque = deque(maxlen=ERROR_LIMIT)
        self._counters = {"turns": 0, "qualifying": 0, "questions": 0,
                          "thin": 0, "gap": 0, "cadence": 0}
        self._metrics = {"stt_ms": 0, "llm_ms": 0, "tts_ms": 0, "total_ms": 0,
                         "target_ms": 1500, "cycles": 0, "avg_ms": 0.0,
                         "best_ms": 0.0, "within": 0}
        self._board: dict | None = None
        self._levels = [0.0] * LEVEL_BARS
        self._levels_available = False
        self._level_provider = None
        self._floor_rms = 0.0
        self._cycle_ms: list = []
        # control actions: HUD (producer) -> orchestrator (consumer)
        self._actions: dict = {}
        self._action_cv = threading.Condition()
        self.telemetry = telemetry

    # -- SSE fan-out ---------------------------------------------------------
    def subscribe(self) -> tuple:
        with self._lock:
            cid = self._next_client
            self._next_client += 1
            q: queue.Queue = queue.Queue(maxsize=CLIENT_QUEUE)
            self._clients[cid] = q
            return cid, q

    def unsubscribe(self, cid: int) -> None:
        with self._lock:
            self._clients.pop(cid, None)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _emit(self, kind: str, data: dict) -> None:
        """Non-blocking publish to every SSE client."""
        with self._lock:
            targets = list(self._clients.items())
        payload = {"kind": kind, "data": data}
        for cid, q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                with self._lock:
                    self._clients.pop(cid, None)
                log_warn(f"SSE client {cid} dropped (backlog full)")

# -- state mutations -----------------------------------------------------
    def set_state(self, state: str, note: str = "") -> None:
        """idle -> listening -> thinking -> speaking -> muted (DESIGN.md 7.2)."""
        with self._lock:
            self._state = state
            self._note = note
        self._emit("state", {"state": state, "note": note})

    def set_model(self, model: str) -> None:
        with self._lock:
            self._model = model
        self._emit("session", {"model": model, "started_ts": self._started_ts})

    def add_transcript(self, text: str, audio_s: float = 0.0,
                       stt_ms: int = 0, seq: int = 0) -> dict:
        with self._lock:
            self._line_id += 1
            entry = {"id": self._line_id, "ts": time.time(), "text": str(text),
                     "audio_s": round(float(audio_s or 0.0), 2),
                     "stt_ms": int(stt_ms or 0), "seq": int(seq or 0)}
            self._transcript.append(entry)
            self._counters["turns"] += 1
        self._emit("transcript", entry)
        self._emit("counters", self.counters)
        return entry

    def set_question(self, text: str, latency_ms: int = 0,
                     tier: str = "") -> None:
        with self._lock:
            self._question = {"text": str(text), "latency_ms": int(latency_ms or 0),
                              "tier": str(tier or ""), "ts": time.time()}
            snap = dict(self._question)
        self._emit("question", snap)

    def resolve_question(self, reason: str = "acknowledged") -> bool:
        with self._lock:
            if self._question is None:
                return False
            self._question = None
        self._emit("question_resolved", {"reason": reason})
        return True

    def add_error(self, msg: str) -> None:
        entry = {"ts": time.time(), "text": str(msg)}
        with self._lock:
            self._errors.append(entry)
        self._emit("error", entry)

    def set_board(self, board: dict) -> None:
        """Whiteboard payload: equations[] terms[] bullets[] diagram (str)."""
        board = dict(board or {})
        board["ts"] = time.time()
        with self._lock:
            self._board = board
        self._emit("board", board)

    def clear_feed(self) -> None:
        with self._lock:
            self._transcript.clear()
            self._line_id = 0
        self._emit("feed_reset", {})

    # -- metrics / counters --------------------------------------------------
    @property
    def counters(self) -> dict:
        with self._lock:
            return dict(self._counters)

    def set_counts(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                self._counters[k] = int(v)
            snap = dict(self._counters)
        self._emit("counters", snap)

    def bump_count(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + by
            snap = dict(self._counters)
        self._emit("counters", snap)

    def set_metrics(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                if v is not None:
                    self._metrics[k] = v
            snap = dict(self._metrics)
        self._emit("metrics", snap)

    def record_cycle(self, stt_ms: int = 0, llm_ms: int = 0, tts_ms: int = 0,
                     total_ms: int = 0, target_ms: int = 1500) -> dict:
        """One completed question cycle; keeps the rolling avg/best/within."""
        stt_ms, llm_ms, tts_ms = int(stt_ms or 0), int(llm_ms or 0), int(tts_ms or 0)
        total_ms = int(total_ms or 0) or (stt_ms + llm_ms + tts_ms)
        with self._lock:
            self._cycle_ms.append(total_ms)
            cycles = len(self._cycle_ms)
            avg = sum(self._cycle_ms) / cycles
            best = min(self._cycle_ms)
            within = sum(1 for c in self._cycle_ms if c <= target_ms)
            self._metrics.update({"stt_ms": stt_ms, "llm_ms": llm_ms,
                                  "tts_ms": tts_ms, "total_ms": total_ms,
                                  "target_ms": int(target_ms), "cycles": cycles,
                                  "avg_ms": round(avg, 1), "best_ms": best,
                                  "within": within})
            self._counters["questions"] += 1
            snap, cnt = dict(self._metrics), dict(self._counters)
        self._emit("metrics", snap)
        self._emit("counters", cnt)
        return snap

    # -- telemetry -----------------------------------------------------------
    def publish_telemetry(self) -> None:
        if self.telemetry is None:
            return
        self._emit("telemetry", self.telemetry.data)

    # -- microphone levels ---------------------------------------------------
    def set_level_provider(self, fn) -> None:
        """Register a callable returning a raw level sequence (or None).

        listener.py exposes this so the waveform reflects real captured audio;
        without it the HUD shows its explicit unavailable notice.
        """
        with self._lock:
            self._level_provider = fn
        if fn is None:
            self._levels_available = False

    @property
    def levels_available(self) -> bool:
        with self._lock:
            return self._levels_available

    def set_floor_rms(self, floor_rms: float) -> None:
        with self._lock:
            self._floor_rms = float(floor_rms or 0.0)

    def sample_levels(self) -> dict:
        """Pull the provider once, normalise, and cache for late joiners."""
        with self._lock:
            provider = self._level_provider
            floor_rms = self._floor_rms
        raw = None
        if provider is not None:
            try:
                raw = provider()
            except Exception as exc:
                log_warn(f"level provider failed: {exc}")
                raw = None
        levels = normalise_levels(raw, floor_rms)
        available = bool(raw) and provider is not None
        floor_norm = round(_to_norm(floor_rms), 4) if floor_rms else 0.0
        with self._lock:
            self._levels = levels
            self._levels_available = available
        return {"levels": levels, "levels_available": available,
                "floor": floor_norm}

    def tick_levels(self) -> dict:
        payload = self.sample_levels()
        self._emit("levels", payload)
        return payload

    # -- control actions (HUD -> orchestrator) -------------------------------
    def push_action(self, action: str) -> dict:
        action = str(action or "").strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        with self._action_cv:
            self._actions[action] = self._actions.get(action, 0) + 1
            self._action_cv.notify_all()
        log_info(f"action queued from HUD: {action}")
        return {"ok": True, "action": action}

    def drain_actions(self) -> dict:
        with self._action_cv:
            pending, self._actions = self._actions, {}
        return pending

    def wait_action(self, timeout: float = 0.25,
                    stop_flag: threading.Event | None = None) -> str | None:
        """Block up to `timeout` (or until stop_flag) for one HUD command.

        Returns the action name or None. Lets main.py honour a button press
        the way it honours Ctrl+C, from inside its existing consumption loop.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._action_cv:
            while not self._actions:
                if stop_flag is not None and stop_flag.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._action_cv.wait(min(0.1, remaining))
            for name in ("stop", "ask", "dismiss", "toggle_mute"):
                if self._actions.get(name):
                    self._actions[name] -= 1
                    if not self._actions[name]:
                        del self._actions[name]
                    return name
        return None

    # -- snapshot ------------------------------------------------------------
    def snapshot(self) -> dict:
        """Full state for a newly connected client (DESIGN.md 7.3 - late join)."""
        with self._lock:
            return {
                "started_ts": self._started_ts,
                "state": self._state,
                "note": self._note,
                "model": self._model,
                "metrics": dict(self._metrics),
                "counters": dict(self._counters),
                "telemetry": (self.telemetry.data if self.telemetry
                              else {"vram_used_mb": None, "vram_total_mb": None,
                                    "cpu_pct": None}),
                "transcript": [dict(x) for x in self._transcript],
                "errors": [dict(x) for x in self._errors],
                "board": dict(self._board) if self._board else None,
                "question": dict(self._question) if self._question else None,
                "levels": list(self._levels),
                "levels_available": self._levels_available,
                "floor": (round(_to_norm(self._floor_rms), 4)
                          if self._floor_rms else 0.0),
            }

# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------
class HudHandler(BaseHTTPRequestHandler):
    """Loopback-only request handler: document, SSE stream, JSON action API."""

    protocol_version = "HTTP/1.1"
    server_version = "SocraticHUD/1.0"

    def log_message(self, fmt, *args) -> None:      # silence stdlib stderr spam
        pass

    def handle_one_request(self) -> None:
        """Swallow connection-abort noise when a browser/tab vanishes.

        socketserver prints a traceback via handle_error() for any exception
        escaping a request; on Windows a closed SSE tab raises
        ConnectionAbortedError deep inside rfile.readline(), which is pure
        operator noise here.
        """
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError,
                BrokenPipeError, TimeoutError):
            self.close_connection = True

    # -- helpers -------------------------------------------------------------
    @property
    def bridge(self) -> WebBridge:
        return self.server.bridge          # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except Exception:
            n = 0
        if n <= 0 or n > 65536:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) or {}
        except Exception:
            return {}

    @staticmethod
    def _route(path: str) -> str:
        return (path or "/").split("?", 1)[0].rstrip("/") or "/"

    # -- GET -----------------------------------------------------------------
    def do_GET(self) -> None:                       # noqa: N802 (stdlib naming)
        route = self._route(self.path)
        if route in ("/", "/hud.html", "/index.html"):
            try:
                body = HUD_HTML.read_bytes()
            except Exception as exc:
                log_err(f"cannot read {HUD_HTML.name}: {exc}")
                self._send(500, b"hud.html missing", "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if route == "/events":
            self._serve_events()
            return
        if route == "/api/state":
            self._send_json(self.bridge.snapshot())
            return
        if route == "/api/health":
            self._send_json({"ok": True, "clients": self.bridge.client_count,
                             "state": self.bridge.snapshot()["state"],
                             "levels_available": self.bridge.levels_available})
            return
        if route == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    # -- POST ----------------------------------------------------------------
    def do_POST(self) -> None:                      # noqa: N802
        route = self._route(self.path)
        if route == "/api/board":
            self._accept_board()
            return
        if route != "/api/action":
            self._send_json({"ok": False, "error": "unknown endpoint"}, 404)
            return
        action = self._read_json().get("action", "")
        try:
            self._send_json(self.bridge.push_action(action))
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _accept_board(self) -> None:
        """Board push from vision_tracker.py --web-publish (separate process).

        Loopback-only like every other route, so an arbitrary web page cannot
        reach it. Only the four documented keys are read, so an oversized or
        malformed payload cannot poison the HUD state.
        """
        payload = self._read_json()
        if not isinstance(payload, dict):
            self._send_json({"ok": False, "error": "board payload must be an object"}, 400)
            return
        board = {
            "equations": _as_str_list(payload.get("equations")),
            "terms": _as_str_list(payload.get("terms")),
            "bullets": _as_str_list(payload.get("bullets")),
            "diagram": str(payload.get("diagram") or ""),
        }
        self.bridge.set_board(board)
        self._send_json({"ok": True, "keys": sorted(board)})

    # -- SSE -----------------------------------------------------------------
    @staticmethod
    def _frame(kind: str, data: dict) -> bytes:
        """Default 'message' event: hud.html only listens on es.onmessage."""
        obj = {"kind": kind, "data": data}
        return ("data: " + json.dumps(obj, ensure_ascii=False, default=str)
                + "\n\n").encode("utf-8")

    def _serve_events(self) -> None:
        bridge = self.bridge
        cid, q = bridge.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-transform")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            bridge.unsubscribe(cid)
            return

        log_info(f"SSE client {cid} connected ({bridge.client_count} live)")
        try:
            self.wfile.write(self._frame("snapshot", bridge.snapshot()))
            self.wfile.flush()
            last_ping = time.monotonic()
            while True:
                try:
                    payload = q.get(timeout=0.5)
                except queue.Empty:
                    if time.monotonic() - last_ping >= SSE_HEARTBEAT_S:
                        self.wfile.write(self._frame("ping", {}))
                        self.wfile.flush()
                        last_ping = time.monotonic()
                    continue
                self.wfile.write(self._frame(payload["kind"], payload["data"]))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                OSError):
            pass
        finally:
            bridge.unsubscribe(cid)
            self.close_connection = True       # SSE ends this connection
            log_info(f"SSE client {cid} disconnected")


class HudServer:
    """Owns the HTTP server thread and the shared sampling ticker."""

    def __init__(self, bridge: WebBridge | None = None, host: str = HUD_HOST,
                 port: int = HUD_PORT,
                 telemetry: Telemetry | None = None) -> None:
        self.bridge = bridge or WebBridge(telemetry=telemetry, port=port)
        if telemetry is not None:
            self.bridge.telemetry = telemetry
        self.host = host
        self.port = int(port)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._ticker: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._owns_telemetry = self.bridge.telemetry is None
        if self._owns_telemetry:
            self.bridge.telemetry = Telemetry()

    @property
    def url(self) -> str:
        shown = "127.0.0.1" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return f"http://{shown}:{self.port}/"

    @staticmethod
    def _port_help(exc: OSError) -> str:
        return (f"port {exc.args[1] if len(exc.args) > 1 else ''} busy - "
                "close the other HUD, or pass --web-port <other>")

    def _tick_loop(self) -> None:
        period = 1.0 / max(1.0, TICK_HZ)
        last_tel = 0.0
        while not self._stop_flag.is_set():
            t0 = time.monotonic()
            try:
                if self.bridge.client_count:
                    self.bridge.tick_levels()
                    if t0 - last_tel >= TELEMETRY_PERIOD_S:
                        last_tel = t0
                        self.bridge.publish_telemetry()
            except Exception as exc:
                log_warn(f"tick loop error: {exc}")
            self._stop_flag.wait(max(0.0, period - (time.monotonic() - t0)))

    def start(self, open_browser: bool = False) -> "HudServer":
        tel = self.bridge.telemetry
        if self._owns_telemetry and tel is not None:
            tel.start()
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), HudHandler)
        except OSError as exc:
            raise SystemExit(f"HUD server error: {self._port_help(exc)}") from exc
        self._httpd.daemon_threads = True
        self._httpd.bridge = self.bridge        # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="ta-hud-http", daemon=True)
        self._thread.start()
        self._ticker = threading.Thread(target=self._tick_loop,
                                        name="ta-hud-tick", daemon=True)
        self._ticker.start()
        log_ok(f"HUD live at {self.url}  (loopback only, Ctrl+C to stop)")
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception as exc:
                log_warn(f"could not open browser: {exc}")
        return self

    def stop(self) -> None:
        self._stop_flag.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception as exc:
                log_warn(f"HTTP shutdown: {exc}")
            self._httpd = None
        for th in (self._thread, self._ticker):
            if th is not None and th.is_alive():
                th.join(timeout=2.0)
        if self._owns_telemetry and self.bridge.telemetry is not None:
            self.bridge.telemetry.stop()
        log_info("HUD server stopped")

# ---------------------------------------------------------------------------
# Demo + selftest + CLI
# ---------------------------------------------------------------------------
DEMO_LINES = (
    "Right, so Bayes theorem tells us how to update a prior belief once we see "
    "some evidence.",
    "The posterior is proportional to the likelihood times the prior, divided "
    "by the marginal probability of the data.",
    "For the entropy of a discrete source it is minus the sum of p log p over "
    "all symbols.",
    "If the channel is memoryless then each use of the channel is independent.",
    "Notice that the cross-entropy is never smaller than the entropy itself.",
)

DEMO_QUESTIONS = (
    "Could you explain why the marginal probability sits in the denominator?",
    "When would the entropy be maximised for a fixed alphabet size?",
    "Does the memoryless assumption change the capacity calculation at all?",
)

DEMO_BOARD = {
    "equations": ["p(A|B) = p(B|A) * P(A) / P(B)", "H(X) = -sum p log p"],
    "terms": ["Bayes Theorem", "prior", "likelihood", "posterior", "entropy"],
    "bullets": ["- prior: P(A)", "- likelihood: p(B|A)"],
    "diagram": "line diagram (5 edges, 3 shapes)",
}

ok_count = 0


def check(label: str, cond: bool, extra="") -> None:
    """Same contract as _test_vision.py: print, count, abort on failure."""
    global ok_count
    if cond:
        ok_count += 1
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {extra}")
        raise SystemExit(1)


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind((HUD_HOST, 0))
        return s.getsockname()[1]

def selftest() -> int:
    """Assertions over the pure parts of the HUD contract. No browser, no mic."""
    global ok_count
    ok_count = 0
    print("--- level mapping ---")
    z = normalise_levels(None)
    check("None levels -> LEVEL_BARS zeros",
          isinstance(z, list) and len(z) == LEVEL_BARS and not any(z))
    check("full-scale RMS -> 1.0",
          normalise_levels([PEAK_REF_RMS])[0] == 1.0,
          normalise_levels([PEAK_REF_RMS])[0])
    check("over-range clamps to 1.0",
          normalise_levels([PEAK_REF_RMS * 50])[0] == 1.0)
    check("negative clamps to 0.0", normalise_levels([-5.0])[0] == 0.0)
    check("silence -> all zeros", not any(normalise_levels([0.0] * 8)))
    check("output length is exactly LEVEL_BARS",
          len(normalise_levels(list(range(500)))) == LEVEL_BARS)

    gated = normalise_levels([400.0, 4000.0], floor_rms=1000.0)
    check("sub-floor energy collapses to 0", gated[0] == 0.0, gated[0])
    check("above-floor energy survives", gated[1] > 0.0, gated[1])

    junk = normalise_levels(["not-a-number", None, {}])
    check("malformed levels -> zeros, never a crash",
          len(junk) == LEVEL_BARS and not any(junk))

    print("--- sse frame shape ---")
    frame = HudHandler._frame("state", {"state": "listening"})
    check("frame uses the default message event",
          frame.startswith(b"data: ") and b"event:" not in frame, frame[:40])
    check("frame terminates with a blank line", frame.endswith(b"\n\n"))
    payload = json.loads(frame.decode("utf-8")[len("data: "):].strip())
    check("frame carries kind + data",
          payload.get("kind") == "state"
          and payload["data"]["state"] == "listening", payload)

    print("--- WebBridge state + fan-out ---")
    bridge = WebBridge(telemetry=None)
    cid, q = bridge.subscribe()
    check("subscribe registers a client", bridge.client_count == 1)

    bridge.set_state("listening")
    m = q.get_nowait()
    check("state event delivered",
          m["kind"] == "state" and m["data"]["state"] == "listening", m)

    bridge.set_model("llama3.2:3b")
    m = q.get_nowait()
    check("session event carries the model", m["data"]["model"] == "llama3.2:3b")

    entry = bridge.add_transcript("Bayes theorem updates a prior.", 3.5, 220, 7)
    check("transcript entry is numbered",
          entry["id"] == 1 and entry["text"].startswith("Bayes"), entry)
    check("transcript increments turns", bridge.counters["turns"] == 1)
    check("transcript event emitted", q.get_nowait()["kind"] == "transcript")
    check("counters event emitted", q.get_nowait()["kind"] == "counters")

    bridge.set_question("Why is the marginal in the denominator?", 640, "kokoro")
    mq = q.get_nowait()
    check("question event carries text + latency",
          mq["kind"] == "question" and mq["data"]["latency_ms"] == 640, mq)

    check("resolve_question returns True once", bridge.resolve_question() is True)
    q.get_nowait()
    check("second resolve_question is a no-op",
          bridge.resolve_question() is False)

    bridge.add_error("ollama unreachable")
    check("error event emitted", q.get_nowait()["kind"] == "error")

    bridge.set_board({"equations": ["H(X) = -sum p log p"],
                      "terms": ["entropy"], "bullets": [], "diagram": ""})
    mb = q.get_nowait()
    check("board event carries equations",
          mb["data"]["equations"][0].startswith("H(X)"), mb)

    print("--- snapshot (late joiner contract) ---")
    snap = bridge.snapshot()
    for key in ("started_ts", "state", "note", "model", "metrics", "counters",
                "telemetry", "transcript", "errors", "board", "question",
                "levels", "levels_available", "floor"):
        check(f"snapshot has '{key}'", key in snap)
    check("snapshot state is the last one set", snap["state"] == "listening",
          snap["state"])
    check("snapshot transcript replays history",
          len(snap["transcript"]) == 1, len(snap["transcript"]))
    check("snapshot levels are LEVEL_BARS long",
          len(snap["levels"]) == LEVEL_BARS)

    print("--- level availability is honest ---")
    avail = bridge.levels_available
    check("no provider -> levels_available is False", avail is False, avail)
    bridge.set_level_provider(lambda: [1000.0] * LEVEL_BARS)
    bridge.sample_levels()
    check("provider registered -> levels_available is True",
          bridge.levels_available is True)
    bridge.set_level_provider(None)
    check("provider removed -> back to False",
          bridge.levels_available is False)

    print("--- control actions ---")
    for name in ACTIONS:
        bridge.push_action(name)
    seen = {bridge.wait_action(timeout=0.01) for _ in range(len(ACTIONS))}
    check("all four actions survive the round trip", seen == set(ACTIONS), seen)
    check("unknown action is rejected",
          _raises(lambda: bridge.push_action("sudo")))
    check("queue is empty afterwards", bridge.wait_action(timeout=0.0) is None)

    print("--- clear_feed ---")
    bridge.clear_feed()
    check("clear_feed empties the transcript",
          bridge.snapshot()["transcript"] == [])

    print(f"\nHUD SELFTEST: {ok_count} assertions passed")
    return 0


def _raises(fn) -> bool:
    """True when fn raises - used for the negative-path assertions."""
    try:
        fn()
    except Exception:
        return True
    return False

# ---------------------------------------------------------------------------
# Demo driver + CLI
# ---------------------------------------------------------------------------
DEMO_TICK_S = 2.4


def _demo_level_provider():
    """Synthetic energy for --demo only; logged loudly, never used in a session."""
    state = {"t": 0.0}

    def provider():
        state["t"] += 0.16
        t = state["t"]
        base = 1100.0 + 620.0 * math.sin(t) + 430.0 * math.sin(t * 2.7)
        sparkle = 300.0 * math.sin(t * 9.1)
        wobble = 260.0 * math.sin(t * 17.0)
        return [max(0.0, base + sparkle + wobble) for _ in range(LEVEL_BARS)]

    return provider


def run_demo(host: str = HUD_HOST, port: int = HUD_PORT,
             open_browser: bool = True, duration: float = 0.0) -> int:
    """Drive the HUD with a scripted lecture. No mic, no LLM, no camera."""
    log_warn("DEMO MODE - synthetic transcript, questions and microphone levels")
    bridge = WebBridge(telemetry=Telemetry(), port=port)
    bridge.set_model("llama3.2:3b (demo)")
    bridge.set_level_provider(_demo_level_provider())
    bridge.set_floor_rms(600.0)
    bridge.set_board(dict(DEMO_BOARD))
    server = HudServer(bridge=bridge, host=host, port=port,
                       telemetry=bridge.telemetry)
    server.start(open_browser=open_browser)
    log_ok(f"open {server.url} in any browser (loopback only)")

    t0 = time.monotonic()
    idx = 0
    try:
        bridge.set_state("listening")
        while True:
            if duration and time.monotonic() - t0 >= duration:
                log_info(f"--duration {duration:g}s reached - stopping demo")
                break
            action = bridge.wait_action(timeout=DEMO_TICK_S)
            if action == "stop":
                log_info("demo stop requested from the HUD")
                break
            if action:
                log_info(f"demo action from the HUD: {action}")

            bridge.set_state("thinking")
            bridge.add_transcript(DEMO_LINES[idx % len(DEMO_LINES)],
                                  audio_s=3.2, stt_ms=240, seq=idx + 1)
            bridge.bump_count("qualifying")
            bridge.bump_count("gap")
            bridge.bump_count("thin")

            bridge.set_metrics(stt_ms=240, llm_ms=610, tts_ms=180)
            bridge.set_question(DEMO_QUESTIONS[idx % len(DEMO_QUESTIONS)],
                                latency_ms=610, tier="kokoro")
            bridge.set_state("speaking")
            time.sleep(2.0)
            bridge.resolve_question("demo playback finished")
            bridge.record_cycle(stt_ms=240, llm_ms=610, tts_ms=180,
                                total_ms=1340)
            bridge.set_state("listening")
            idx += 1
        return 0
    except KeyboardInterrupt:
        print()
        log_info("Ctrl+C received - demo shutting down")
        return 130
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Board push client (used by vision_tracker.py --web-publish)
# ---------------------------------------------------------------------------
def board_payload(snapshot) -> dict:
    """Map a vision_tracker BoardSnapshot onto the HUD's board contract.

    vision_tracker buckets cleaned OCR lines as
    {equations, bullets, keywords} and describes drawings via
    hints["label"]; the HUD renders {equations, terms, bullets, diagram}.
    Keeping the translation here (rather than in vision_tracker.py) means the
    vision module never has to know the HUD's wire format.
    """
    buckets = dict(getattr(snapshot, "buckets", None) or {})
    hints = dict(getattr(snapshot, "hints", None) or {})
    return {
        "equations": _as_str_list(buckets.get("equations")),
        "terms": _as_str_list(buckets.get("keywords")),
        "bullets": _as_str_list(buckets.get("bullets")),
        "diagram": str(hints.get("label") or ""),
    }


def publish_board(board: dict, host: str = HUD_HOST, port: int = HUD_PORT,
                  timeout: float = 2.0) -> bool:
    """POST one board payload to a running HUD. Never raises.

    Vision is an *optional* companion: if the HUD is not up (or is bound to a
    different port) the caller keeps running and simply reports False.
    """
    body = json.dumps(board or {}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/api/board", data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except urllib.error.URLError as exc:
        log_warn(f"board push skipped: HUD not reachable on "
                 f"{host}:{port} ({exc.reason})")
        return False
    except Exception as exc:
        log_warn(f"board push failed: {exc}")
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="webui.py",
        description=("Phase 5 HUD server for the Socratic TA (usually started "
                     "with 'python main.py --web-ui')."),
    )
    parser.add_argument("--host", default=HUD_HOST,
                        help=f"bind address (default: {HUD_HOST}; loopback only)")
    parser.add_argument("--port", type=int, default=HUD_PORT,
                        help=f"HTTP port (default: {HUD_PORT})")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="open the HUD in the default browser")
    parser.add_argument("--demo", action="store_true",
                        help="drive the HUD with a scripted lecture (no mic/LLM)")
    parser.add_argument("--selftest", action="store_true",
                        help="run the HUD contract assertions and exit")
    parser.add_argument("--duration", type=float, default=0,
                        help="auto-stop after N seconds (0 = until Ctrl+C)")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    if args.demo:
        return run_demo(host=args.host, port=args.port,
                        open_browser=args.open_browser,
                        duration=args.duration)

    # Bare serve: HUD + telemetry with no orchestrator attached.
    server = HudServer(host=args.host, port=args.port)
    t0 = time.monotonic()
    try:
        server.start(open_browser=args.open_browser)
        log_info("HUD serving with no orchestrator attached - start "
                 "'python main.py --web-ui' for live lecture data")
        while True:
            time.sleep(0.25)
            if args.duration and time.monotonic() - t0 >= args.duration:
                log_info(f"--duration {args.duration:g}s reached - stopping")
                break
        return 0
    except KeyboardInterrupt:
        print()
        log_info("Ctrl+C received - HUD shutting down")
        return 130
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
