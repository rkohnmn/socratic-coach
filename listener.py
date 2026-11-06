#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 TA Phase 2 - `listener.py`
 Ultra-reliable audio capture & transcription daemon (CPU-only, Windows 11)
==============================================================================

Hardware context (verified in Phase 1):
    AMD Ryzen 7 5800U  (8C/16T, 16 GB RAM)  -> ALL DSP + ASR happens here
    NVIDIA RTX 3050 Ti (4 GB VRAM)          -> NOT touched by this module;
                                               CUDA stays reserved for the
                                               Ollama LLM (Phase 1 guard).

Architecture
------------
  [PortAudio callback] --bytes--> queue.Queue(128) --> [FSM thread]
  mic, 16 kHz / 16-bit / mono, 1024-sample chunks; the callback never blocks.

  FSM states:
    IDLE_LISTENING   - monitor RMS vs adaptive threshold, maintain a 0.5 s
                       pre-roll circular buffer (no clipped first syllables).
    RECORDING_SPEECH - energy above threshold: append pre-roll + live frames.
    EVALUATING_PAUSE - energy below threshold: count consecutive silence.

  Turn rules:
    * bursts shorter than MIN_SPEECH_S (1.2 s) are discarded (coughs, bumps)
    * silence >= PAUSE_TRIGGER_S (2.2 s, inside the 2.2-2.6 s teacher-pause
      band) after valid speech closes the utterance
    * hard cap MAX_UTTERANCE_S (30 s) forces transcription mid-speech

  Transcriber worker (single background thread, lazy model load):
    faster-whisper WhisperModel("base.en", device="cpu", compute_type="int8",
    cpu_threads=4); beam_size=1, best_of=1, temperature=0.0, vad_filter=True,
    condition_on_previous_text=False; hallucination filter applied.

  Rolling transcript: deque pruned to the last 90 seconds.

External interface for Phase 3 (`main.py`):
    from listener import ListenerEngine, pause_recording
    engine = ListenerEngine().start()
    text = engine.get_latest_utterance(timeout=30)   # blocking, thread-safe
    for utt in engine.iter_utterances(): ...         # generator pattern
    pause_recording.set()   # mute ingestion during TTS playback

Standalone CLI:
    python listener.py [--duration N] [--model base.en] [--chunk {512,1024}]
                       [--device NAME] [--no-stt] [--list-devices] [--verbose]

Exit codes: 0 clean stop | 1 fatal error | 130 Ctrl+C
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import queue
import re
import signal
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR = 16_000                     # capture sample rate (Hz)
SAMPLE_WIDTH = 2                # int16 -> 2 bytes
CHANNELS = 1                    # mono
DEFAULT_CHUNK = 1024            # samples per callback (64 ms @ 16 kHz)
LEVEL_RING = 96                 # waveform history depth (64 bars @ 64 ms = ~6 s)
ALLOWED_CHUNKS = (512, 1024)

CALIBRATION_S = 1.5             # ambient noise measurement window
PRE_ROLL_S = 0.5                # circular pre-speech buffer

BASELINE_MULTIPLIER = 2.2       # threshold = baseline_rms * 2.2 + margin
SAFETY_MARGIN = 120.0           # int16-domain RMS units
SILENT_ROOM_FALLBACK_RMS = 60.0  # fallback baseline if room is dead silent
MIN_BASELINE_RMS = 30.0

MIN_SPEECH_S = 1.2              # discard shorter bursts (coughs, bumps)
PAUSE_TRIGGER_S = 2.2           # teacher-pause band lower bound -> boundary
PAUSE_BAND_MAX_S = 2.6          # teacher-pause band upper bound (headroom)
MAX_UTTERANCE_S = 30.0          # hard cutoff for non-stop speech

TRANSCRIPT_WINDOW_S = 90.0      # rolling transcript retention

QUEUE_MAXSIZE = 128             # bounded PCM queue (~8 s of 64 ms chunks)
INT16_SCALE = 1.0 / 32768.0

LOGS_DIR = Path(__file__).resolve().parent / "logs"
UTTERANCE_WAV_DIR = LOGS_DIR / "utterances"
DIAG_LOG_PATH = LOGS_DIR / "listener.log"

HALLUCINATION_PATTERNS = (
    "thank you.", "thank you for watching", "thanks for watching",
    "thank you very much", "bye.", "goodbye.", "see you next",
    "subscribe", "like and subscribe", "okay.", "ok.",
    "[music]", "[applause]", "[laughter]", "[inaudible]",
    "[blank_audio]", "[silence]", "you",
)
MIN_TRANSCRIPT_CHARS = 2
MIN_UNIQUE_WORD_RATIO = 0.34    # collapse "the the the the"-style loops
MAX_CONSECUTIVE_REPEATS = 3

# Global programmatic mute flag (Phase 3 sets this during TTS playback).
pause_recording = threading.Event()

# ---------------------------------------------------------------------------
# Console / logging helpers (same visual language as Phase 1)
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
    print(f"{_c(_ts(), C.GREY)} {_c('[i]', C.CYAN)} {msg}")


def log_ok(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[OK]', C.GREEN + C.BOLD)} {msg}")


def log_warn(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[!]', C.YELLOW + C.BOLD)} {msg}")


def log_err(msg: str) -> None:
    print(f"{_c(_ts(), C.GREY)} {_c('[X]', C.RED + C.BOLD)} {msg}")


LOG = logging.getLogger("ta.listener")

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------
@dataclass
class Utterance:
    """A finished, transcribed teacher utterance."""
    text: str
    started_at: float          # time.monotonic() of speech onset
    finished_at: float         # time.monotonic() of boundary trigger
    audio_s: float             # seconds of PCM audio captured
    transcribe_ms: int         # wall time spent in faster-whisper (ms)
    waveform: bytes = b""      # raw int16 PCM (kept for optional debugging)
    seq: int = 0               # monotonically increasing utterance index

    @property
    def wall_s(self) -> float:
        return self.finished_at - self.started_at


class State(IntEnum):
    IDLE_LISTENING = 0
    RECORDING_SPEECH = 1
    EVALUATING_PAUSE = 2


STATE_LABELS = {
    State.IDLE_LISTENING: _c("IDLE_LISTENING ", C.BLUE + C.BOLD),
    State.RECORDING_SPEECH: _c("RECORDING_SPEECH", C.GREEN + C.BOLD),
    State.EVALUATING_PAUSE: _c("EVALUATING_PAUSE", C.YELLOW + C.BOLD),
}


def rms_int16(pcm_bytes: bytes) -> float:
    """Root-mean-square energy of raw 16-bit little-endian PCM bytes."""
    import numpy as np

    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def pcm_to_float32(pcm_bytes: bytes) -> "np.ndarray":
    """Raw int16 PCM bytes -> normalized float32 in [-1.0, 1.0]."""
    import numpy as np

    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    return samples.astype(np.float32) * np.float32(INT16_SCALE)


# ---------------------------------------------------------------------------
# Phase A - dynamic noise calibration (startup)
# ---------------------------------------------------------------------------
class NoiseCalibrator:
    """Measures 1.5 s of room tone and derives the adaptive speech threshold:
    threshold = baseline_rms * 2.2 + margin. Falls back to a sane default when
    the room is completely silent.

    Implementation note: the stream runs in callback mode, so PortAudio
    forbids stream.read() here (PaErrorCode -9977). Calibration therefore
    consumes the PCM chunks the callback is already pushing into the queue.
    """

    def __init__(self, pcm_queue: "queue.Queue") -> None:
        self._queue = pcm_queue
        self.baseline_rms = 0.0
        self.threshold = 0.0
        self.samples_seen = 0
        self.peak_rms = 0.0

    def calibrate(self) -> float:
        import numpy as np

        rms_values: list = []
        print()
        log_info(
            f"Calibrating ambient noise for {CALIBRATION_S:.1f}s "
            "- please stay quiet ..."
        )
        # drain anything queued before calibration began (startup noise)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        deadline = time.monotonic() + CALIBRATION_S
        spinner = "|/-\\"
        spin_idx = 0
        while time.monotonic() < deadline:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
            pcm, _dropped = item
            if not pcm:
                continue
            value = rms_int16(bytes(pcm))
            rms_values.append(value)
            self.peak_rms = max(self.peak_rms, value)
            self.samples_seen += len(pcm) // (SAMPLE_WIDTH * CHANNELS)
            print(
                f"\r      {spinner[spin_idx % 4]} measuring room tone: "
                f"{len(rms_values)} chunks, live RMS {value:7.1f}   ",
                end="", flush=True,
            )
            spin_idx += 1
        print()

        if not rms_values:
            log_warn("No audio captured during calibration - using fallback.")
            self.baseline_rms = SILENT_ROOM_FALLBACK_RMS
        else:
            self.baseline_rms = float(np.median(rms_values))

        if self.baseline_rms < MIN_BASELINE_RMS:
            log_warn(
                f"Room is virtually silent (baseline {self.baseline_rms:.1f} "
                f"< {MIN_BASELINE_RMS:.0f}) - applying fallback baseline "
                f"{SILENT_ROOM_FALLBACK_RMS:.1f}."
            )
            self.baseline_rms = SILENT_ROOM_FALLBACK_RMS

        self.threshold = self.baseline_rms * BASELINE_MULTIPLIER + SAFETY_MARGIN

        # -- clean calibration report ---------------------------------------
        print()
        print(_c("  +---------------------------------------------------+", C.CYAN))
        print(_c("  |          DYNAMIC NOISE CALIBRATION REPORT          |", C.CYAN + C.BOLD))
        print(_c("  +---------------------------------------------------+", C.CYAN))
        print(f"   Window ............: {CALIBRATION_S:.1f} s "
              f"({self.samples_seen} samples @ {SR} Hz)")
        print(f"   Baseline RMS ......: {self.baseline_rms:8.1f} (median of "
              f"{len(rms_values)} chunks)")
        print(f"   Peak RMS ..........: {self.peak_rms:8.1f}")
        print(f"   Speech threshold ..: {self.threshold:8.1f}   "
              f"= baseline * {BASELINE_MULTIPLIER} + {SAFETY_MARGIN:.0f}")
        print(f"   Dynamic range .....: x{max(self.threshold / max(self.baseline_rms, 1e-9), 1.0):.1f}")
        print(_c("  +---------------------------------------------------+", C.CYAN))
        print()
        return self.threshold

# ---------------------------------------------------------------------------
# Phase C - hallucination filtering
# ---------------------------------------------------------------------------
_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_WS_RE = re.compile(r"\s+")


def is_hallucination(text: str) -> tuple:
    """Return (is_hallucination, reason) for a raw Whisper output string."""
    cleaned = _WS_RE.sub(" ", _BRACKET_RE.sub("", text)).strip()
    if len(cleaned) < MIN_TRANSCRIPT_CHARS:
        return True, "blank/too short"
    lowered = cleaned.lower()
    for pattern in HALLUCINATION_PATTERNS:
        if lowered == pattern or lowered.rstrip(".!? ") == pattern.rstrip(".!? "):
            if len(cleaned) <= 14:
                return True, f"generic subtitle token: '{pattern}'"
    words = lowered.split()
    if not words:
        return True, "empty after cleaning"
    unique_ratio = len(set(words)) / len(words)
    if len(words) >= 4 and unique_ratio < MIN_UNIQUE_WORD_RATIO:
        return True, f"repetitive loop (unique ratio {unique_ratio:.2f})"
    # exact immediate repetition: "hello hello hello hello"
    if len(words) >= MAX_CONSECUTIVE_REPEATS * 2:
        run, worst = 1, 1
        for i in range(1, len(words)):
            run = run + 1 if words[i] == words[i - 1] else 1
            worst = max(worst, run)
        if worst >= MAX_CONSECUTIVE_REPEATS:
            return True, f"token repeated {worst}x in a row"
    return False, ""


# ---------------------------------------------------------------------------
# Phase C - fast int8 transcription worker (single background thread)
# ---------------------------------------------------------------------------
class Transcriber:
    """Lazy-loading faster-whisper wrapper pinned to the CPU.

    VRAM guardrail: device="cpu" + compute_type="int8" + cpu_threads=4 means
    the RTX 3050 Ti is never initialised by this process - zero VRAM cost.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        cpu_threads: int = 4,
        download_root: Path | None = None,
        save_wavs: bool = False,
    ) -> None:
        self.model_size = model_size
        self.cpu_threads = cpu_threads
        self.download_root = Path(download_root) if download_root else (
            Path(__file__).resolve().parent / "models" / "faster-whisper"
        )
        self.save_wavs = save_wavs
        self._model = None
        self._lock = threading.Lock()
        self.transcribe_times_ms: deque = deque(maxlen=200)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel

            self.download_root.mkdir(parents=True, exist_ok=True)
            log_info(
                f'Loading WhisperModel("{self.model_size}", device="cpu", '
                f'compute_type="int8", cpu_threads={self.cpu_threads}) '
                "- first run downloads the weights ..."
            )
            t0 = time.perf_counter()
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
                num_workers=1,
                download_root=str(self.download_root),
                local_files_only=False,
            )
            log_ok(f"Whisper '{self.model_size}' (int8, CPU) ready "
                   f"in {time.perf_counter() - t0:.1f}s.")
            return self._model

    def transcribe_pcm(self, pcm_bytes: bytes) -> tuple:
        """Raw int16 PCM -> (text, duration_s, transcribe_ms)."""
        audio = pcm_to_float32(pcm_bytes)
        duration_s = audio.size / float(SR)
        if audio.size < SR // 4:  # < 0.25 s of audio: nothing to do
            return "", duration_s, 0

        model = self._ensure_model()
        t0 = time.perf_counter()
        try:
            segments, _info = model.transcribe(
                audio,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            parts = [seg.text.strip() for seg in segments if seg.text]
            raw_text = " ".join(p for p in parts if p).strip()
        except Exception as exc:
            log_err(f"Whisper transcription failed: {exc}")
            LOG.exception("transcription failure")
            return "", duration_s, int((time.perf_counter() - t0) * 1000)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        self.transcribe_times_ms.append(elapsed_ms)

        hallucinated, reason = is_hallucination(raw_text)
        if hallucinated:
            log_warn(f"Dropped Whisper hallucination ({reason}): "
                     f"'{raw_text[:60]}'")
            return "", duration_s, elapsed_ms
        if self.save_wavs:
            self._save_wav(pcm_bytes)
        return raw_text, duration_s, elapsed_ms

    def _save_wav(self, pcm_bytes: bytes) -> None:
        try:
            UTTERANCE_WAV_DIR.mkdir(parents=True, exist_ok=True)
            path = UTTERANCE_WAV_DIR / (
                datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".wav"
            )
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SR)
                wf.writeframes(pcm_bytes)
            LOG.info("utterance wav saved: %s", path.name)
        except Exception as exc:
            LOG.warning("wav save failed: %s", exc)

# ---------------------------------------------------------------------------
# Phase B - finite state machine with energy-based VAD
# ---------------------------------------------------------------------------
class TurnDetector:
    """Consumes PCM chunks from a queue and emits finished utterances.

    Memory safety: every buffer is a bounded deque; the pre-roll buffer holds
    exactly PRE_ROLL_S of audio, the active speech buffer is capped at
    MAX_UTTERANCE_S + headroom, so hours-long sessions cannot grow unbounded.
    """

    _last_status_print = 0.0

    def __init__(self, threshold: float, chunk_samples: int = DEFAULT_CHUNK) -> None:
        self.threshold = float(threshold)
        self.chunk_samples = chunk_samples
        self.chunk_s = chunk_samples / float(SR)

        pre_roll_chunks = max(1, int(round(PRE_ROLL_S / self.chunk_s)))
        self._pre_roll: deque = deque(maxlen=pre_roll_chunks)

        max_speech_chunks = int(MAX_UTTERANCE_S / self.chunk_s) + 8
        self._speech_buf: deque = deque(maxlen=max_speech_chunks)

        self.state = State.IDLE_LISTENING
        self.speech_started_at = 0.0
        self.silence_started_at = 0.0
        self.silence_chunks = 0
        self.speech_chunks = 0      # total chunks buffered (incl. pause tail)
        self.voiced_chunks = 0      # chunks at/above threshold (true speech)
        self.last_rms = 0.0
        self.utterances_emitted = 0
        self.bursts_discarded = 0
        self.max_cuts = 0
        self._seq = 0

        self.status_lock = threading.Lock()
        self.chunks_total = 0
        self.chunks_dropped = 0

    # -- helpers -------------------------------------------------------------
    def _speech_seconds(self) -> float:
        return self.speech_chunks * self.chunk_s

    def _silence_seconds(self) -> float:
        return self.silence_chunks * self.chunk_s

    def _reset_speech(self) -> None:
        self._speech_buf.clear()
        self.speech_chunks = 0
        self.voiced_chunks = 0
        self.silence_chunks = 0

    def _voiced_seconds(self) -> float:
        return self.voiced_chunks * self.chunk_s

    # -- state handlers -------------------------------------------------------
    def _handle_idle(self, pcm: bytes, rms: float) -> None:
        self._pre_roll.append(pcm)
        if rms >= self.threshold:
            # speech onset: seed the active buffer with the pre-roll so the
            # first syllables are never clipped; count pre-roll chunks that
            # were genuinely voiced so the min-speech rule stays honest
            self._speech_buf.extend(self._pre_roll)
            self.speech_chunks += len(self._pre_roll)
            self.voiced_chunks = sum(
                1 for p in self._pre_roll if rms_int16(p) >= self.threshold
            )
            self.speech_started_at = time.monotonic()
            self.silence_chunks = 0
            self.state = State.RECORDING_SPEECH
            LOG.info("speech onset (rms=%.1f)", rms)
            print(f"\r{_c('>> voice detected' + ' ' * 40, C.GREEN)}", flush=True)

    def _handle_recording(self, pcm: bytes, rms: float) -> None:
        self._speech_buf.append(pcm)
        self.speech_chunks += 1
        if rms < self.threshold:
            self.silence_chunks += 1
            if self._silence_seconds() >= 0.24:  # debounce: real pause
                self.state = State.EVALUATING_PAUSE
                self.silence_started_at = time.monotonic()
                LOG.info("energy dropped below threshold; evaluating pause")
        else:
            self.silence_chunks = 0
            self.voiced_chunks += 1
        if self._speech_seconds() >= MAX_UTTERANCE_S:
            print(_c(">> 30 s hard cutoff - forcing boundary", C.MAGENTA), flush=True)
            self.max_cuts += 1
            self._finalize(reason="max-utterance-cutoff")

    def _handle_pause(self, pcm: bytes, rms: float) -> None:
        # keep buffering the tail (breath noise etc.) while tracking silence
        self._speech_buf.append(pcm)
        self.speech_chunks += 1
        if rms >= self.threshold:
            self.silence_chunks = 0
            self.voiced_chunks += 1
            self.state = State.RECORDING_SPEECH
            return
        self.silence_chunks += 1
        if self._speech_seconds() >= MAX_UTTERANCE_S:
            print(_c(">> 30 s hard cutoff - forcing boundary", C.MAGENTA), flush=True)
            self.max_cuts += 1
            self._finalize(reason="max-utterance-cutoff")
            return
        if self._silence_seconds() >= PAUSE_TRIGGER_S:
            # teacher-pause band hit (2.2 s; 2.6 s is band headroom)
            self._finalize(reason="teacher-pause")

    def _finalize(self, reason: str) -> None:
        total_s = self._speech_seconds()     # all buffered audio (incl. tail)
        voiced_s = self._voiced_seconds()    # genuinely above-threshold audio
        pcm = b"".join(self._speech_buf)
        started_at = self.speech_started_at
        finished_at = time.monotonic()
        self._reset_speech()
        self.state = State.IDLE_LISTENING
        if voiced_s < MIN_SPEECH_S:
            self.bursts_discarded += 1
            LOG.info("burst discarded (voiced %.2fs < %.2fs, buffered %.2fs, "
                     "reason=%s)", voiced_s, MIN_SPEECH_S, total_s, reason)
            print(
                f"\r{_c('-- burst discarded (voiced '
                       f'{voiced_s:.2f}s < {MIN_SPEECH_S:.1f}s)' + ' ' * 20,
                       C.GREY)}",
                end="", flush=True,
            )
            return
        self._seq += 1
        self.utterances_emitted += 1
        log_info(
            f"Utterance boundary #{self._seq} ({reason}): voiced "
            f"{voiced_s:.2f}s, buffered {total_s:.2f}s"
        )
        self._on_utterance(pcm, started_at, finished_at, total_s)

    # -- override point --------------------------------------------------------
    def _on_utterance(self, pcm: bytes, started_at: float,
                      finished_at: float, speech_s: float) -> None:
        raise NotImplementedError

    # -- main loop --------------------------------------------------------------
    def feed_loop(self, pcm_queue: "queue.Queue",
                  stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                item = pcm_queue.get(timeout=0.1)
            except queue.Empty:
                self._refresh_status(True)
                continue
            if item is None:  # sentinel -> drain and exit
                break
            pcm, dropped = item
            with self.status_lock:
                self.chunks_total += 1
                self.chunks_dropped += dropped
            pcm = bytes(pcm)
            rms = rms_int16(pcm)
            self.last_rms = rms
            if self.state == State.IDLE_LISTENING:
                self._handle_idle(pcm, rms)
            elif self.state == State.RECORDING_SPEECH:
                self._handle_recording(pcm, rms)
            else:
                self._handle_pause(pcm, rms)
            self._refresh_status()

    def _refresh_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_print < 0.25:
            return
        self._last_status_print = now
        label = STATE_LABELS[self.state]
        if self.state == State.IDLE_LISTENING:
            detail = f"rms {self.last_rms:6.0f} / thr {self.threshold:6.0f}"
        elif self.state == State.RECORDING_SPEECH:
            detail = f"speech {self._speech_seconds():5.2f}s / cap {MAX_UTTERANCE_S:.0f}s"
        else:
            detail = (f"silence {self._silence_seconds():5.2f}s / "
                      f"trigger {PAUSE_TRIGGER_S:.1f}s")
        line = f"   STATE: {label}  {detail}   "
        print(f"\r{_c(line, C.GREY)}", end="", flush=True)

# ---------------------------------------------------------------------------
# The ListenerEngine - capture + FSM + transcription orchestration
# ---------------------------------------------------------------------------
class ListenerEngine(TurnDetector):
    """Top-level engine. Import-safe for Phase 3; standalone-runnable too.

    Threads:
      * PortAudio callback (PortAudio's internal thread; only enqueues bytes)
      * FSM thread            - self.feed_loop()
      * transcription worker  - self._transcription_worker()
    """

    def __init__(
        self,
        model_size: str = "base.en",
        chunk_samples: int = DEFAULT_CHUNK,
        device: "int | str | None" = None,
        cpu_threads: int = 4,
        save_wavs: bool = False,
        enable_stt: bool = True,
        verbose: bool = False,
        threshold_override: float | None = None,
    ) -> None:
        if chunk_samples not in ALLOWED_CHUNKS:
            raise ValueError(f"chunk_samples must be one of {ALLOWED_CHUNKS}")
        super().__init__(threshold=0.0, chunk_samples=chunk_samples)
        self.model_size = model_size
        self.device = device
        self.cpu_threads = cpu_threads
        self.enable_stt = enable_stt
        self.verbose = verbose
        self.threshold_override = threshold_override

        self.pcm_queue: "queue.Queue" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.stop_event = threading.Event()
        self.calibrated = threading.Event()
        self.threshold_final = 0.0

        self.transcriber = Transcriber(
            model_size=model_size,
            cpu_threads=cpu_threads,
            save_wavs=save_wavs,
        )

        # rolling transcript: bounded deque, pruned to the last 90 seconds
        self._transcript: deque = deque()
        self._transcript_lock = threading.Lock()

        # Phase 5 waveform feed: raw int16 RMS per callback, oldest first.
        # Bounded and lock-guarded so the HUD can read it without ever
        # touching PortAudio's real-time thread.
        self._level_ring: deque = deque(maxlen=LEVEL_RING)
        self._level_lock = threading.Lock()

        self._latest: Utterance | None = None
        self._latest_cond = threading.Condition()
        self._seq_counter = 0

        self._stream = None
        self._fsm_thread: threading.Thread | None = None
        self._stt_thread: threading.Thread | None = None
        self._stt_queue: "queue.Queue" = queue.Queue(maxsize=8)
        self._dropped_chunks = 0
        self._started_at = 0.0

    # ------------------------------------------------------------------ API
    def start(self) -> "ListenerEngine":
        """Open the mic, calibrate, and launch worker threads."""
        import sounddevice as sd

        LOG.info("listener engine starting (chunk=%d, model=%s)",
                 self.chunk_samples, self.model_size)
        in_dev, out_dev = self._resolve_device()
        log_ok(f"Input device: {_c(in_dev['name'], C.BOLD)} "
               f"[{sd.query_hostapis()[int(in_dev['hostapi'])]['name']}] "
               f"@ {int(in_dev['default_samplerate'])} Hz")
        if out_dev is not None:
            log_info(f"Output device (for Phase 3 TTS): {out_dev['name']}")

        self._stream = sd.InputStream(
            samplerate=SR,
            blocksize=self.chunk_samples,
            channels=CHANNELS,
            dtype="int16",
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        log_ok(f"InputStream live: {SR} Hz / 16-bit / mono / "
               f"{self.chunk_samples}-sample chunks ({self.chunk_s * 1000:.0f} ms)")

        # -- dynamic noise calibration (consumes callback-fed PCM queue) ----
        calib = NoiseCalibrator(self.pcm_queue)
        threshold = calib.calibrate()
        if self.threshold_override is not None:
            threshold = float(self.threshold_override)
            log_warn(f"Threshold overridden via config: {threshold:.1f}")
        self.threshold = threshold
        self.threshold_final = threshold
        self.calibrated.set()

        self._started_at = time.monotonic()
        self._fsm_thread = threading.Thread(
            target=self.feed_loop,
            args=(self.pcm_queue, self.stop_event),
            name="ta-fsm",
            daemon=True,
        )
        self._fsm_thread.start()
        if self.enable_stt:
            self._stt_thread = threading.Thread(
                target=self._transcription_worker,
                name="ta-stt",
                daemon=True,
            )
            self._stt_thread.start()
        log_ok("Listener engine is LIVE - start talking. "
               "Pause 2.2 s to close an utterance; Ctrl+C to stop.")
        return self

    # ------------------------------------------------------- capture plumbing
    def _resolve_device(self) -> tuple:
        import sounddevice as sd

        try:
            in_dev = sd.query_devices(kind="input")
        except Exception as exc:
            raise RuntimeError(f"No default input device available: {exc}") from exc
        out_dev = None
        try:
            out_dev = sd.query_devices(kind="output")
        except Exception:
            pass
        return in_dev, out_dev

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """PortAudio callback - runs on PortAudio's own thread.

        HARD RULE: never block here. bytes(indata) is O(n) memcpy; the only
        lock-free risk is queue.Full, which we resolve by dropping the OLDEST
        chunk (put_nowait + get_nowait) so the newest audio always wins and
        the callback returns within microseconds.
        """
        if status:
            LOG.warning("portaudio status: %s", status)
        if pause_recording.is_set():
            self._push_level(0.0)   # Phase 3 mute: meter flatlines, mic swallowed
            return  # Phase 3 mute: swallow mic input during TTS playback
        pcm = bytes(indata)
        self._push_level(rms_int16(pcm))
        dropped = 0
        try:
            self.pcm_queue.put_nowait((pcm, 0))
        except queue.Full:
            try:
                self.pcm_queue.get_nowait()
                dropped = 1
            except queue.Empty:
                pass
            try:
                self.pcm_queue.put_nowait((pcm, dropped))
            except queue.Full:
                pass

    def _push_level(self, value: float) -> None:
        """Append one waveform sample. Runs on PortAudio's real-time thread.

        Only a float append under a short lock - no allocation of buffers,
        no blocking calls, so the callback keeps returning in microseconds.
        """
        try:
            with self._level_lock:
                self._level_ring.append(float(value))
        except Exception:
            pass

    def level_snapshot(self) -> list:
        """Recent int16 RMS values, oldest first (Phase 5 HUD waveform).

        Returns [] before the stream is live so the HUD can show its explicit
        'level unavailable' notice rather than a fabricated idle waveform.
        """
        try:
            with self._level_lock:
                if not self._level_ring:
                    return []
                return list(self._level_ring)
        except Exception:
            return []

    # ------------------------------------------------------- utterance handling
    def _on_utterance(self, pcm: bytes, started_at: float,
                      finished_at: float, speech_s: float) -> None:
        """Called on the FSM thread the instant a boundary is detected.

        The raw PCM is handed to the background worker immediately so the
        microphone stream never waits for Whisper.
        """
        if self.enable_stt:
            try:
                self._stt_queue.put_nowait((pcm, started_at, finished_at, speech_s))
            except queue.Full:
                log_warn("STT queue saturated - dropping the oldest utterance")
                try:
                    self._stt_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._stt_queue.put_nowait(
                        (pcm, started_at, finished_at, speech_s)
                    )
                except queue.Full:
                    log_err("STT queue drop failed - utterance lost")
        else:
            log_info(f"[--no-stt] utterance captured ({speech_s:.2f}s, "
                     f"{len(pcm)} bytes PCM)")

    def _transcription_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                pcm, started_at, finished_at, speech_s = (
                    self._stt_queue.get(timeout=0.2)
                )
            except queue.Empty:
                continue
            try:
                text, audio_s, elapsed_ms = self.transcriber.transcribe_pcm(pcm)
            except Exception as exc:
                log_err(f"Transcription worker crashed: {exc}")
                LOG.exception("transcription worker")
                continue
            if not text:
                continue
            with self._transcript_lock:
                self._seq_counter += 1
                utt = Utterance(
                    text=text,
                    started_at=started_at,
                    finished_at=finished_at,
                    audio_s=audio_s if audio_s > 0 else speech_s,
                    transcribe_ms=elapsed_ms,
                    waveform=b"",  # release audio memory after use
                    seq=self._seq_counter,
                )
                self._transcript.append(utt)
                self._prune_transcript_locked()
                self._latest = utt
            with self._latest_cond:
                self._latest_cond.notify_all()
            self._print_transcription(utt)

    def _prune_transcript_locked(self) -> None:
        cutoff = time.monotonic() - TRANSCRIPT_WINDOW_S
        while self._transcript and self._transcript[0].finished_at < cutoff:
            self._transcript.popleft()

    def _print_transcription(self, utt: Utterance) -> None:
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        header = (f"[TRANSCRIPTION ({utt.transcribe_ms} ms)] "
                  f"audio {utt.audio_s:.2f}s")
        print(f"\r{' ' * 100}\r", end="")
        print(f"{_c(stamp, C.GREY)} {_c(header, C.MAGENTA + C.BOLD)}: "
              f"{_c('\"' + utt.text + '\"', C.BOLD)}", flush=True)

    # ------------------------------------------------------- public API (P3)
    def get_latest_utterance(self, timeout: float | None = None):
        """Block until a new utterance is available (thread-safe).

        Returns the Utterance object, or None on timeout / shutdown.
        Only returns utterances the caller has not seen yet - safe for a
        single consumer loop (Phase 3's conversation orchestrator).
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._latest_cond:
            while True:
                if self._latest is not None and self._latest.seq > self._consumed_seq:
                    self._consumed_seq = self._latest.seq
                    return self._latest
                if self.stop_event.is_set():
                    return None
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._latest_cond.wait(timeout=remaining)

    _consumed_seq = 0

    def iter_utterances(self):
        """Async-friendly generator: yields Utterance objects as they close.

        Usage in Phase 3:
            for utt in listener.iter_utterances():
                handle(utt.text)
        """
        while True:
            utt = self.get_latest_utterance(timeout=None)
            if utt is None:
                return
            yield utt

    def transcript_window(self) -> list:
        """Snapshot of the last 90 s of transcribed speech (oldest first)."""
        with self._transcript_lock:
            self._prune_transcript_locked()
            return list(self._transcript)

    def transcript_text(self, joiner: str = " ") -> str:
        return joiner.join(u.text for u in self.transcript_window())

    def stats(self) -> dict:
        with self.status_lock:
            total = self.chunks_total
            dropped = self.chunks_dropped
        stt_avg = 0
        if self.transcriber.transcribe_times_ms:
            stt_avg = sum(self.transcriber.transcribe_times_ms) // len(
                self.transcriber.transcribe_times_ms
            )
        return {
            "state": self.state.name,
            "uptime_s": round(time.monotonic() - self._started_at, 1)
            if self._started_at else 0.0,
            "chunks": total,
            "chunks_dropped": dropped,
            "utterances": self.utterances_emitted,
            "bursts_discarded": self.bursts_discarded,
            "max_cuts": self.max_cuts,
            "avg_transcribe_ms": stt_avg,
            "threshold": round(self.threshold_final, 1),
            "paused": pause_recording.is_set(),
        }

    # ------------------------------------------------------------------ stop
    def stop(self, join_timeout: float = 5.0) -> None:
        """Graceful shutdown - releases the audio hardware cleanly.

        Order matters on Windows: stop the stream first (silences the
        PortAudio callback), then close it, then signal worker threads, then
        unblock the FSM loop with a sentinel and join everything.
        """
        if getattr(self, "_stopping", False):
            return
        self._stopping = True
        log_info("Shutting down listener engine ...")
        self.stop_event.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                LOG.warning("stream.stop(): %s", exc)
            try:
                stream.close()
            except Exception as exc:
                LOG.warning("stream.close(): %s", exc)
            log_ok("PortAudio stream stopped and closed (device released).")
        try:
            self.pcm_queue.put_nowait(None)  # sentinel -> FSM thread exits
        except queue.Full:
            try:
                self.pcm_queue.get_nowait()
                self.pcm_queue.put_nowait(None)
            except Exception:
                pass
        for thread in (self._fsm_thread, self._stt_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    log_warn(f"{thread.name} did not exit in time "
                             "(daemon thread - will die with process)")
        try:
            self._stt_queue.put_nowait(None)  # unblock STT wait
        except Exception:
            pass
        with self._latest_cond:
            self._latest_cond.notify_all()
        stats = self.stats()
        print()
        print(_c("  +---------------------- SESSION SUMMARY "
                  "----------------------+", C.CYAN))
        print(f"   Uptime ............: {stats['uptime_s']:.1f} s")
        print(f"   Chunks processed ..: {stats['chunks']} "
              f"({stats['chunks_dropped']} dropped under load)")
        print(f"   Utterances ........: {stats['utterances']} "
              f"({stats['bursts_discarded']} sub-1.2s bursts discarded, "
              f"{stats['max_cuts']} 30s hard cuts)")
        print(f"   Avg transcription .: {stats['avg_transcribe_ms']} ms "
              f"per utterance (int8 CPU)")
        print(f"   Speech threshold ..: {stats['threshold']}")
        print(_c("  +--------------------------------------------------+",
                 C.CYAN))
        log_ok("Goodbye.")

    _stopping = False

# ---------------------------------------------------------------------------
# Standalone CLI mode
# ---------------------------------------------------------------------------
def list_devices() -> None:
    import sounddevice as sd

    print(_c("  AUDIO DEVICES (sounddevice / PortAudio)", C.BOLD + C.CYAN))
    try:
        in_dev = sd.query_devices(kind="input")
        print(f"   Default INPUT : {in_dev['name']} "
              f"({int(in_dev['max_input_channels'])} in-ch)")
    except Exception as exc:
        print(f"   Default INPUT : <none> ({exc})")
    try:
        out_dev = sd.query_devices(kind="output")
        print(f"   Default OUTPUT: {out_dev['name']} "
              f"({int(out_dev['max_output_channels'])} out-ch)")
    except Exception as exc:
        print(f"   Default OUTPUT: <none> ({exc})")
    print()
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev["max_input_channels"]) > 0:
            print(f"   [{idx:>2}] IN  {dev['name']}  "
                  f"[{sd.query_hostapis()[int(dev['hostapi'])]['name']}]")
    print()


def print_banner(model: str, chunk: int) -> None:
    width = 78

    def boxed(text: str) -> str:
        return "| " + text.ljust(width - 4) + " |"

    print()
    print("  " + _c("+" + "-" * (width - 2) + "+", C.CYAN))
    print("  " + _c(boxed("TA PHASE 2  -  LISTENER: CAPTURE + VAD + TRANSCRIBE"), C.CYAN + C.BOLD))
    print("  " + _c(boxed("CPU-only pipeline | 16 kHz int16 mono | zero VRAM"), C.CYAN))
    print("  " + _c("+" + "-" * (width - 2) + "+", C.CYAN))
    print()
    print(f"   Whisper model ...: {model} (int8, cpu_threads=4)")
    print(f"   Audio format ....: {SR} Hz / 16-bit / mono / {chunk}-sample chunks")
    print(f"   FSM .............: min-speech {MIN_SPEECH_S}s | pause "
          f"{PAUSE_TRIGGER_S}-{PAUSE_BAND_MAX_S}s | max {MAX_UTTERANCE_S:.0f}s")
    print(f"   Transcript mem ..: rolling {TRANSCRIPT_WINDOW_S:.0f}s window")


def run_cli(args) -> int:
    _force_utf8_stdio()
    _enable_windows_vt()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(DIAG_LOG_PATH),
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s",
        datefmt="%H:%M:%S",
    )
    print_banner(args.model, args.chunk)

    if args.list_devices:
        list_devices()

    engine = ListenerEngine(
        model_size=args.model,
        chunk_samples=args.chunk,
        device=args.device,
        enable_stt=not args.no_stt,
        save_wavs=args.save_wavs,
        verbose=args.verbose,
    )
    try:
        engine.start()
    except RuntimeError as exc:
        log_err(str(exc))
        return 1

    exit_code = 0
    try:
        if args.duration and args.duration > 0:
            log_info(f"Running for {args.duration}s, then stopping "
                     "(--duration mode) ...")
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                time.sleep(0.1)
        else:
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print()
        log_info("Ctrl+C received - shutting down gracefully ...")
        exit_code = 130
    finally:
        engine.stop()
    return 0 if exit_code == 0 else exit_code

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="listener.py",
        description=(
            "TA Phase 2 - CPU-only audio capture, VAD and transcription daemon. "
            "Run standalone, or import ListenerEngine from Phase 3."
        ),
    )
    parser.add_argument("--model", default="base.en",
                        help="faster-whisper model size (default: base.en)")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                        choices=ALLOWED_CHUNKS,
                        help="samples per callback block (default: 1024)")
    parser.add_argument("--device", default=None,
                        help="input device name/substring/index (default: OS default)")
    parser.add_argument("--duration", type=float, default=0,
                        help="auto-stop after N seconds (0 = run until Ctrl+C)")
    parser.add_argument("--no-stt", action="store_true",
                        help="run VAD/FSM only; skip loading Whisper entirely")
    parser.add_argument("--save-wavs", action="store_true",
                        help="save each utterance to logs/utterances/*.wav")
    parser.add_argument("--list-devices", action="store_true",
                        help="list input devices at startup, then continue")
    parser.add_argument("--verbose", action="store_true",
                        help="debug-level file logging")
    return parser


def _sigterm_handler(signum, frame) -> None:
    """Map SIGTERM onto the same graceful path as Ctrl+C."""
    raise KeyboardInterrupt


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    # SIGINT arrives on the main thread on Windows; KeyboardInterrupt handling
    # inside run_cli performs the graceful teardown.
    signal.signal(signal.SIGTERM, _sigterm_handler)
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())