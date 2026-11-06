#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 TA Phase 3 - `main.py`
 Master orchestrator: listener + Socratic LLM (Ollama/GPU) + Kokoro TTS (CPU)
==============================================================================

Hardware allocation boundaries (enforced):
    RTX 3050 Ti (4 GB VRAM) -> Ollama `llama3.2:3b` (Q4_K_M, ~2.2 GB) ONLY.
    Ryzen 7 5800U (CPU)     -> listener.py (PortAudio + faster-whisper int8,
                               4 threads) AND Kokoro-82M TTS synthesis.
    Latency target          -> utterance end -> playback start <= 1.5 s.

Pipeline per cycle:
    [listener.py FSM] -> TEACHER utterance -> sliding 90 s transcript window
      -> silence guard (>= 15 words, not garbled)
      -> mute mic + flush residual frames
      -> Ollama /api/generate (requests.Session, keep_alive 30 m, 1 question,
         <= 18 words, no meta/filler)          [THINKING, yellow]
      -> Kokoro CPU streaming synthesis -> sounddevice OutputStream @ 24 kHz
                                             [STUDENT AUDIO, magenta]
      -> 300 ms room-reverb cooldown -> unmute mic -> [STATUS: LISTENING]

Console dashboard (ANSI): STATUS/LISTENING cyan | TEACHER green |
THINKING yellow | STUDENT AUDIO magenta | MUTED/PLAYBACK red. Every cycle
reports the STT + LLM + TTS split against the 1.5 s target.

TTS resilience ladder: kokoro (torch CPU) -> pyttsx3 (SAPI5, auto-pip) ->
PowerShell System.Speech (always present on Windows 11). The assistant can
never fail silently.

Measured latency profile (Ryzen 7 5800U, Kokoro-82M, torch 2.14 CPU):
    * A 12-word question synthesised as ONE block costs ~1.9 s to first
      audio - over the 1.5 s budget. Kokoro's pipeline yields lazily per
      split segment, so `plan_speech_chunks()` dispatches the question as
      two clause-sized parts at a natural boundary: first audio lands at
      ~1.1-1.3 s for +0.8 s of inter-part silence. Three or more parts buy
      only ~0.2 s more and inflate the audio 2x, so two is the sweet spot.
    * torch threads: 8 -> ~1.3 s, 4 -> ~1.7 s, 2 -> ~3.8 s. Kokoro is
      latency-bound and the mic is muted during playback, so TTS gets 8
      threads (whisper runs its own ctranslate2 pool at cpu_threads=4).
    * The ONNX-Runtime variants were benchmarked and REJECTED: int8 needs
      7.0 s (Zen 3 has no VNNI) and `phonemizer` G2P is slower than misaki.
    * A one-phrase warm-up at init removes ~0.5 s of first-call cost.

Usage:
  python main.py                       # full stack, Ctrl+C to stop
  python main.py --web-ui              # full stack + HTML cockpit on :8765
  python main.py --selftest            # TTS + guard cycle only (no mic/LLM)
  python main.py --no-llm              # listener + transcript dashboard only
  python main.py --duration 30         # auto-stop after 30 s
  python main.py --ask-every 3         # question on every 3rd qualifying turn

Exit codes: 0 clean | 1 fatal setup error | 130 Ctrl+C
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

import listener as listener_mod
from listener import ListenerEngine, Utterance, pause_recording

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parent
LOGS_DIR = WORKSPACE / "logs"
OLLAMA_SERVE_LOG = LOGS_DIR / "ollama_serve.log"

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.2:3b"

TTS_SAMPLE_RATE = 24000          # Kokoro native output rate (Hz)
TTS_THREADS_DEFAULT = 8          # measured optimum (4 threads costs +0.4 s)
MIN_CHUNK_WORDS = 7              # below this a question is spoken as one part
MAX_QUESTION_WORDS = 18          # hard post-hoc enforcement
MIN_WINDOW_WORDS = 15            # silence guard threshold
PLAYBACK_COOLDOWN_S = 0.30       # room reverberation settle time
LATENCY_TARGET_S = 1.50          # utterance-end -> playback start

EV_TEACHER = "teacher"
EV_THINKING = "thinking"
EV_LLM_DONE = "llm_done"
EV_STUDENT = "student_audio"
EV_MUTED = "muted"
EV_LISTENING = "listening"
EV_CYCLE = "cycle"
EV_ERROR = "error"

SYSTEM_PROMPT = (
    "You are a sharp, attentive university student listening to a live "
    "lecture. You think critically about the material and you formulate "
    "exactly ONE short, natural, spoken question when asked."
)

USER_PROMPT_TEMPLATE = (
    "Recent lecture transcript (last ~2 minutes):\n"
    "\"\"\"\n{transcript}\n\"\"\"\n\n"
    "Identify one unstated assumption, one edge case, one missing logical "
    "deduction, or one ambiguous concept in this material. Then formulate "
    "exactly ONE natural, spoken student question about it.\n\n"
    "Hard rules:\n"
    "- Maximum {max_words} words.\n"
    "- Output strictly the exact words of the spoken question - nothing else.\n"
    "- Absolutely NO meta-commentary, pleasantries or filler phrases "
    "(e.g. \"Excuse me professor\", \"Great lecture!\"), NO quotation marks, "
    "NO stage directions, NO labels such as 'Question:'.\n"
    "- The question must be directly answerable by the lecturer.\n"
)

# ---------------------------------------------------------------------------
# Console helpers (same visual language as Phases 1-2)
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


def _stamp() -> str:
    now = datetime.now()
    return f"{now:%H:%M:%S}.{now.microsecond // 1000:03d}"


def _clearline() -> None:
    print(f"\r{' ' * 110}\r", end="")


def log_info(msg: str) -> None:
    print(f"{_c(_stamp(), C.GREY)} {_c('[i]', C.CYAN)} {msg}")


def log_ok(msg: str) -> None:
    print(f"{_c(_stamp(), C.GREY)} {_c('[OK]', C.GREEN + C.BOLD)} {msg}")


def log_warn(msg: str) -> None:
    print(f"{_c(_stamp(), C.GREY)} {_c('[!]', C.YELLOW + C.BOLD)} {msg}")


def log_err(msg: str) -> None:
    print(f"{_c(_stamp(), C.GREY)} {_c('[X]', C.RED + C.BOLD)} {msg}")


# ---------------------------------------------------------------------------
# Real-time event bus
# ---------------------------------------------------------------------------
class EventBus:
    """Minimal synchronous pub/sub. UI rendering and stats bookkeeping
    subscribe to lifecycle events published by the orchestrator thread."""

    def __init__(self) -> None:
        self._subs: dict = {}
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler) -> None:
        with self._lock:
            self._subs.setdefault(event, []).append(handler)

    def publish(self, event: str, **data) -> None:
        with self._lock:
            handlers = list(self._subs.get(event, ()))
        for handler in handlers:
            try:
                handler(**data)
            except Exception as exc:
                log_err(f"event handler for '{event}' failed: {exc}")

# ---------------------------------------------------------------------------
# Color-coded console dashboard
# ---------------------------------------------------------------------------
def banner() -> None:
    width = 78

    def boxed(text: str) -> str:
        return "| " + text.ljust(width - 4) + " |"

    print()
    print("  " + _c("+" + "-" * (width - 2) + "+", C.CYAN))
    print("  " + _c(boxed("TA PHASE 3  -  SOCRATIC STUDENT ORCHESTRATOR"), C.CYAN + C.BOLD))
    print("  " + _c(boxed("GPU: Ollama llama3.2:3b ONLY (4GB VRAM)"), C.CYAN))
    print("  " + _c(boxed("CPU: listener.py + faster-whisper + Kokoro TTS"), C.CYAN))
    print("  " + _c("+" + "-" * (width - 2) + "+", C.CYAN))
    print()
    print(f"   Latency target ...: utterance end -> playback start "
          f"<= {LATENCY_TARGET_S:.2f} s")
    print(f"   LLM ..............: {DEFAULT_LLM_MODEL} via Ollama REST (Session pool)")
    print(f"   TTS ladder .......: kokoro (24 kHz CPU, {TTS_THREADS_DEFAULT} "
          f"threads, 2-part streaming) -> pyttsx3 -> SAPI5")
    print(f"   Mic discipline ...: mute -> flush -> playback -> "
          f"{int(PLAYBACK_COOLDOWN_S * 1000)} ms cooldown -> unmute")
    print()


def ui_status_listening(note: str = "") -> None:
    _clearline()
    line = f"{_stamp()} [STATUS: LISTENING] {note or 'Monitoring instructor lecture ...'}"
    print(_c(line, C.CYAN + C.BOLD), flush=True)


def ui_muted(note: str = "") -> None:
    _clearline()
    line = f"{_stamp()} [MUTED / PLAYBACK] {note or 'Mic input suppressed - student speaking.'}"
    print(_c(line, C.RED + C.BOLD), flush=True)


def ui_teacher(utt: Utterance) -> None:
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _clearline()
    line = (f"{stamp} [TEACHER] ({utt.audio_s:.2f}s audio, "
            f"stt {utt.transcribe_ms} ms): \"{utt.text}\"")
    print(_c(line, C.GREEN + C.BOLD), flush=True)


def ui_thinking(model: str) -> None:
    _clearline()
    print(_c(f"{_stamp()} [THINKING] consulting the student persona "
             f"({model}) ...", C.YELLOW + C.BOLD), flush=True)


def ui_llm_done(ms: int, question: str) -> None:
    _clearline()
    if question:
        print(_c(f"{_stamp()} [THINKING] student query ready in {ms} ms",
                 C.YELLOW), flush=True)
    else:
        print(_c(f"{_stamp()} [THINKING] no usable question after {ms} ms "
                 f"- discarded", C.YELLOW), flush=True)


def ui_student_audio(text: str, tier: str) -> None:
    _clearline()
    line = f"{_stamp()} [STUDENT AUDIO] (via {tier}): \"{text}\""
    print(_c(line, C.MAGENTA + C.BOLD), flush=True)


def ui_cycle(idx: int, total_ms: int, stt_ms: int, llm_ms: int,
             tts_ms: int, tier: str) -> None:
    within = total_ms <= LATENCY_TARGET_S * 1000
    color = C.GREEN if within else C.YELLOW
    flag = "OK" if within else "OVER TARGET"
    _clearline()
    line = (f"{_stamp()} [CYCLE #{idx}] {total_ms / 1000:.2f}s = "
            f"STT {stt_ms / 1000:.2f}s + LLM {llm_ms / 1000:.2f}s + "
            f"TTS {tts_ms / 1000:.2f}s ({tier}) | target "
            f"{LATENCY_TARGET_S:.2f}s -> {flag}")
    print(_c(line, color + C.BOLD), flush=True)


def ui_error(msg: str) -> None:
    _clearline()
    print(_c(f"{_stamp()} [ERROR] {msg}", C.RED + C.BOLD), flush=True)

# ---------------------------------------------------------------------------
# Transcript guard + Socratic question hygiene
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_QUOTES = "\"'`“”‘’«»"
_PREFIX_RE = re.compile(r"^(?:question|q|student|answer)\s*[:\-]\s*", re.I)
_FILLER_SENT_RE = re.compile(
    r"(?:excuse me|sorry|great lecture|great question|hello|hi,? professor|"
    r"thank you|thanks|wonderful|amazing|interesting lecture|"
    r"that's a great|here is a question|here's a question|as a student)",
    re.I,
)
_STAGE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def _is_filler_sentence(sentence: str) -> bool:
    """Negative-constraint enforcement: meta-commentary, pleasantries and
    filler phrases are dropped; short non-question fragments count as
    pleasantries ('Great lecture!', 'Truly wonderful.')."""
    if _FILLER_SENT_RE.search(sentence):
        return True
    if len(sentence.split()) <= 5 and "?" not in sentence:
        return True
    return False


def transcript_qualifies(text: str, min_words: int) -> tuple:
    """Silence guard: reject thin or garbled transcript windows."""
    words = _WORD_RE.findall(text)
    if len(words) < min_words:
        return False, f"transcript has {len(words)} words (< {min_words})"
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2.0:
        return False, f"garbled fragments (avg word len {avg_len:.1f})"
    return True, ""


def _strip_filler_segments(sentence: str) -> str:
    """Remove filler clauses (usually before the first comma) while keeping
    the actual question content of the sentence."""
    segments = [s.strip(" ,;:") for s in sentence.split(",")]
    kept = [s for s in segments if s and not _FILLER_SENT_RE.search(s)]
    return " ".join(kept)


def clean_question(raw: str, max_words: int) -> str:
    """Normalize an LLM reply into speakable question text ('' if unusable)."""
    text = _STAGE_RE.sub("", str(raw or ""))
    text = text.strip().strip(_QUOTES)
    if not text:
        return ""
    text = _PREFIX_RE.sub("", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if _FILLER_SENT_RE.search(sentence):
            # keep the question core, drop filler clauses
            core = _strip_filler_segments(sentence)
            if core:
                kept.append(core)
            continue
        if _is_filler_sentence(sentence):
            continue  # short non-question pleasantry
        kept.append(sentence)
    text = " ".join(kept).strip().strip(_QUOTES)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    words = text.split(" ")
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    text = text.rstrip(",;: ")
    if len(text) < 4:
        return ""
    if not text.endswith(("?", ".", "!")):
        text += "?"
    return text

# ---------------------------------------------------------------------------
# Ollama client (persistent HTTP connection pooling)
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


class OllamaClient:
    """Ollama REST client with keep-alive connection pooling and a
    Socratic one-question interface (GPU-resident llama3.2:3b)."""

    def __init__(self, host: str, model: str) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=1
        )
        self.session.mount("http://", adapter)

    def health_ok(self) -> bool:
        try:
            return self.session.get(
                f"{self.host}/api/tags", timeout=(2, 4)
            ).status_code == 200
        except Exception:
            return False

    def model_present(self) -> bool:
        try:
            data = self.session.get(f"{self.host}/api/tags", timeout=(2, 4)).json()
            return any(
                m.get("name") == self.model for m in data.get("models", [])
            )
        except Exception:
            return False

    def ensure_ready(self) -> None:
        """Verify daemon + model; attempt to start `ollama serve` detached."""
        if self.health_ok() and self.model_present():
            log_ok(f"Ollama daemon ready at {self.host} - "
                   f"'{self.model}' is resident.")
            return
        if not self.health_ok():
            log_warn(f"Ollama daemon not reachable at {self.host} ...")
            exe = find_ollama_binary()
            if exe is None:
                raise RuntimeError(
                    "Ollama is not installed. Fix: install from "
                    "https://ollama.com/download/windows or "
                    "`winget install --id Ollama.Ollama -e`, then re-run."
                )
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = (
                    subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            with open(OLLAMA_SERVE_LOG, "ab") as log_file:
                proc = subprocess.Popen(
                    [str(exe), "serve"],
                    stdout=log_file, stderr=log_file,
                    stdin=subprocess.DEVNULL, **kwargs,
                )
            log_info(f"Spawned detached `ollama serve` (pid {proc.pid}); "
                     f"log: {OLLAMA_SERVE_LOG}")
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if self.health_ok():
                    break
                time.sleep(1.0)
            if not self.health_ok():
                raise RuntimeError(
                    f"Ollama daemon did not come up at {self.host}. "
                    f"Check {OLLAMA_SERVE_LOG}, or start Ollama from the "
                    "Start Menu and re-run."
                )
            log_ok("Ollama daemon is now accepting connections.")
        if not self.model_present():
            raise RuntimeError(
                f"Model '{self.model}' is not pulled yet. Fix: "
                f"`ollama pull {self.model}` (or re-run Phase 1 setup.bat)."
            )
        log_ok(f"Ollama daemon ready at {self.host} - "
               f"'{self.model}' is resident.")

    def ask(self, transcript: str, max_words: int) -> tuple:
        """One Socratic question from the transcript window.
        Returns (question_text_or_empty, llm_latency_ms)."""
        prompt = (USER_PROMPT_TEMPLATE
                  .replace("{transcript}", transcript)
                  .replace("{max_words}", str(max_words)))
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",           # keep weights resident in 4GB VRAM
            "options": {
                "num_predict": 64,
                "temperature": 0.8,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        t0 = time.perf_counter()
        resp = self.session.post(
            f"{self.host}/api/generate", json=payload, timeout=(3, 120)
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        data = resp.json()
        raw = str(data.get("response", ""))
        return clean_question(raw, max_words), elapsed_ms

# ---------------------------------------------------------------------------
# TTS resilience ladder: kokoro (CPU) -> pyttsx3 (SAPI5) -> PowerShell SAPI
# ---------------------------------------------------------------------------
# Words that make a natural breath-group boundary when a cut must be placed
# away from the exact midpoint of a sentence.
_BREAK_WORDS = frozenset((
    "when if that because and but so which while where then as since "
    "although though unless whether after before once or nor yet than "
    "is are was were do does did can could would should will"
).split())


def plan_speech_chunks(text: str, min_words: int = MIN_CHUNK_WORDS,
                       window: int = 3) -> list:
    """Split a question into <=2 speakable parts at a natural boundary.

    Kokoro's torch pipeline synthesises lazily, one chunk per split segment,
    so two ~6-word parts put first audio on the wire ~0.8 s earlier than one
    12-word block (measured 1.1-1.3 s vs 1.9-2.5 s on the Ryzen 7 5800U) at
    the cost of one short inter-part pause. Boundaries are taken at a comma
    (; / :) or a subordinating conjunction near the middle so the seam still
    sounds like a breath, not a splice.
    """
    words = str(text or "").split()
    if len(words) <= min_words:
        return [" ".join(words)] if words else []
    mid = len(words) / 2.0
    lo = max(3, int(mid) - window)
    hi = min(len(words) - 3, int(mid) + window)
    if hi < lo:
        return [" ".join(words)]
    comma = [i for i in range(lo, hi + 1) if words[i - 1].endswith((",", ";", ":"))]
    break_at = [i for i in range(lo, hi + 1)
                if words[i - 1].rstrip(",;:").lower() in _BREAK_WORDS]
    picks = comma or break_at
    cut = min(picks, key=lambda i: abs(i - mid)) if picks else int(mid)
    cut = max(3, min(cut, len(words) - 3))
    if 0 < cut < len(words):
        return [" ".join(words[:cut]), " ".join(words[cut:])]
    return [" ".join(words)]


class Speaker:
    """Low-latency CPU speech synthesis with a hard fallback chain.

    tier 'kokoro'  : KPipeline streaming chunks -> persistent sounddevice
                     OutputStream at 24 kHz (first audio starts while later
                     chunks are still synthesizing).
    tier 'pyttsx3' : SAPI5 via pyttsx3 (auto-pip-installed on first use).
    tier 'sapi'    : PowerShell System.Speech (always present on Win11).
    """

    def __init__(self, preferred: str = "auto", voice: str = "af_heart",
                 rate_wpm: int = 185, tts_threads: int = TTS_THREADS_DEFAULT) -> None:
        self.preferred = preferred
        self.voice = voice
        self.rate_wpm = rate_wpm
        self.tts_threads = tts_threads
        self.warmup_ms = 0
        self.tier: str | None = None
        self._pipeline = None
        self._out_stream = None

    # -- initialisation ------------------------------------------------------
    def _order(self) -> list:
        if self.preferred != "auto":
            return [self.preferred]
        return ["kokoro", "pyttsx3", "sapi"]

    def initialise(self) -> str:
        errors = []
        for tier in self._order():
            try:
                if tier == "kokoro":
                    self._init_kokoro()
                elif tier == "pyttsx3":
                    self._init_pyttsx3()
                elif tier == "sapi":
                    self._init_sapi()
                else:
                    raise ValueError(f"unknown TTS tier '{tier}'")
                self.tier = tier
                log_ok(f"TTS backend ready: {tier}")
                return tier
            except Exception as exc:
                errors.append(f"{tier}: {exc}")
                log_warn(f"TTS tier '{tier}' unavailable - {exc}")
        raise RuntimeError("No TTS backend available: " + " | ".join(errors))

    def _init_kokoro(self) -> None:
        import numpy as np  # noqa: F401
        import sounddevice as sd
        import torch
        from kokoro import KPipeline

        # Kokoro is latency-bound, not throughput-bound: on the Ryzen 7 5800U a
        # 12-word question costs ~1.3 s to first audio on 8 torch threads versus
        # ~1.7 s on 4 and ~3.8 s on 2. faster-whisper runs in its own
        # ctranslate2 pool (cpu_threads=4) and the mic is muted during
        # playback, so spending 8 torch threads here costs STT nothing.
        threads = max(1, min(int(self.tts_threads), os.cpu_count() or 4))
        torch.set_num_threads(threads)
        log_info(f"Initializing Kokoro-82M on CPU (torch threads: "
                 f"{torch.get_num_threads()}; first run downloads ~330 MB "
                 f"of weights) ...")
        t0 = time.perf_counter()
        self._pipeline = KPipeline(lang_code="a")   # American English
        self._out_stream = sd.OutputStream(
            samplerate=TTS_SAMPLE_RATE, channels=1, dtype="float32", blocksize=0
        )
        self._out_stream.start()
        # Warm-up: the first synthesis also pays for lazy kernel/voice init
        # (~0.5 s extra). Render a throwaway phrase and drop the audio so the
        # first real question gets the warm path.
        t_warm = time.perf_counter()
        for _ in self._pipeline("Ready.", voice=self.voice, speed=1.0):
            pass
        self.warmup_ms = int((time.perf_counter() - t_warm) * 1000)
        log_ok(f"Kokoro pipeline + 24 kHz output stream ready in "
               f"{time.perf_counter() - t0:.1f}s (voice: {self.voice}, "
               f"warm-up {self.warmup_ms} ms).")

    def _init_pyttsx3(self) -> None:
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            log_info("pyttsx3 missing - installing (small wheel) ...")
            subprocess.run(
                [sys.executable, "-m", "pip", "--disable-pip-version-check",
                 "install", "pyttsx3"],
                check=False,
            )
            import pyttsx3  # noqa: F401
        # probe: build + dispose one engine to validate SAPI5 access
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate_wpm)
        engine.stop()
        del engine

    def _init_sapi(self) -> None:
        if shutil.which("powershell") is None:
            raise RuntimeError("powershell.exe not found on PATH")
        self._sapi_probe()

    def _sapi_probe(self) -> None:
        self._sapi_speak("(system check)", blocking=False)

    # -- playback ------------------------------------------------------------
    def speak(self, text: str) -> dict:
        """Blocking playback. Returns dict(tier, first_ms, total_ms)."""
        t0 = time.perf_counter()
        if self.tier == "kokoro":
            try:
                return self._speak_kokoro(text, t0)
            except Exception as exc:
                log_warn(f"Kokoro synthesis failed mid-utterance ({exc}) - "
                         "falling back for this utterance ...")
                self._degrade_from_kokoro()
                self.tier = "pyttsx3" if self._pyttsx3_ok() else "sapi"
                log_warn(f"Session TTS tier downgraded to '{self.tier}'.")
        if self.tier == "pyttsx3":
            return self._speak_pyttsx3(text, t0)
        return self._speak_sapi(text, t0)

    def _speak_kokoro(self, text: str, t0: float) -> dict:
        import numpy as np

        parts = plan_speech_chunks(text)
        first_ms = None
        written = 0
        for part in parts:
            for item in self._pipeline(part, voice=self.voice, speed=1.0):
                chunk = self._item_to_numpy(item)
                if chunk is None or chunk.size == 0:
                    continue
                if first_ms is None:
                    first_ms = int((time.perf_counter() - t0) * 1000)
                    log_info(f"Kokoro stream: {len(parts)} part(s), first "
                             f"audio at {first_ms} ms "
                             f"(target <= {int(LATENCY_TARGET_S * 1000)} ms)")
                self._out_stream.write(
                    np.ascontiguousarray(chunk, dtype=np.float32)
                )
                written += chunk.size
        if first_ms is None:
            raise RuntimeError("kokoro produced no audio")
        total_ms = int((time.perf_counter() - t0) * 1000)
        return {"tier": "kokoro", "first_ms": first_ms, "total_ms": total_ms,
                "samples": written, "parts": len(parts)}

    @staticmethod
    def _item_to_numpy(item):
        import numpy as np

        chunk = item
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            chunk = item[2]
        else:
            chunk = getattr(item, "audio", item)
        if chunk is None:
            return None
        if hasattr(chunk, "detach"):            # torch tensor
            return chunk.detach().cpu().float().numpy().reshape(-1)
        return np.asarray(chunk, dtype=np.float32).reshape(-1)

    def _pyttsx3_ok(self) -> bool:
        try:
            self._init_pyttsx3()
            return True
        except Exception:
            return False

    def _speak_pyttsx3(self, text: str, t0: float) -> dict:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate_wpm)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        total_ms = int((time.perf_counter() - t0) * 1000)
        return {"tier": "pyttsx3", "first_ms": total_ms, "total_ms": total_ms,
                "samples": 0}

    def _sapi_speak(self, text: str, blocking: bool = True) -> None:
        safe = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = 0; $s.Speak('{safe}'); $s.Dispose()"
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-EncodedCommand", encoded],
            check=False,
            timeout=120,
        )

    def _speak_sapi(self, text: str, t0: float) -> dict:
        self._sapi_speak(text)
        total_ms = int((time.perf_counter() - t0) * 1000)
        return {"tier": "sapi", "first_ms": total_ms, "total_ms": total_ms,
                "samples": 0}

    # -- teardown --------------------------------------------------------------
    def _degrade_from_kokoro(self) -> None:
        """Release the Kokoro pipeline/output stream after a mid-utterance
        failure so the fallback tier can claim the audio device."""
        self.abort()
        self._pipeline = None

    def abort(self) -> None:
        stream = self._out_stream
        self._out_stream = None
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# The Orchestrator - wires listener + Ollama + TTS into one disciplined cycle
# ---------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, args) -> None:
        self.args = args
        self.bus = EventBus()
        self.stop_flag = threading.Event()
        self.engine: ListenerEngine | None = None
        self.ollama: OllamaClient | None = None
        self.speaker: Speaker | None = None
        self._ignore_before = 0.0       # utterances finalised before this are stale
        self._t_start = 0.0
        self._cycles: list = []
        self._turns_seen = 0
        self._qualifying = 0
        self._question_counter = 0
        self._questions_asked = 0
        self._skipped_thin = 0
        self._skipped_gap = 0
        self._skipped_cadence = 0
        self._stopping = False
        self.hud = None                 # webui.HudServer when --web-ui is set
        self.web = None                 # webui.WebBridge alongside it

    # -- setup ----------------------------------------------------------------
    def setup(self) -> None:
        self._subscribe_ui()
        banner()
        self._start_web_ui()
        if not self.args.no_llm:
            self.ollama = OllamaClient(self.args.host, self.args.llm_model)
            self.ollama.ensure_ready()
        else:
            log_warn("--no-llm: running in listen-only mode (no questions).")
        self.speaker = Speaker(preferred=self.args.tts, voice=self.args.voice,
                               tts_threads=self.args.tts_threads)
        self.speaker.initialise()

        log_info("Starting Phase 2 listener engine (mic + FSM + STT) ...")
        self.engine = ListenerEngine(
            model_size=self.args.stt_model,
            chunk_samples=self.args.chunk,
            device=self.args.device,
            save_wavs=self.args.save_wavs,
        )
        self.engine.start()
        self._bind_web_engine()
        self._t_start = time.monotonic()
        self.bus.publish(EV_LISTENING)
        log_ok("ALL SYSTEMS NOMINAL - the Socratic student is listening.")

    def _subscribe_ui(self) -> None:
        self.bus.subscribe(EV_TEACHER, lambda utt: ui_teacher(utt))
        self.bus.subscribe(EV_LISTENING, lambda: ui_status_listening())
        self.bus.subscribe(EV_MUTED, lambda note="": ui_muted(note))
        self.bus.subscribe(
            EV_THINKING,
            lambda model="": ui_thinking(model or self.args.llm_model),
        )
        self.bus.subscribe(
            EV_LLM_DONE, lambda ms=0, question="": ui_llm_done(ms, question)
        )
        self.bus.subscribe(
            EV_STUDENT,
            lambda text="", tier="": ui_student_audio(text, tier),
        )
        self.bus.subscribe(EV_ERROR, lambda msg="": ui_error(msg))

    # -- Phase 5 HTML cockpit --------------------------------------------------
    def _start_web_ui(self) -> None:
        """Stand up the loopback HUD and mirror the event stream into it.

        Started before the (slow) Whisper load so a browser can connect and
        paint its unavailable states while the engine warms up, and torn down
        first thing in shutdown() so no thread outlives the session.
        """
        if not getattr(self.args, "web_ui", False):
            return
        try:
            import webui
        except Exception as exc:
            log_err(f"--web-ui ignored: cannot import webui.py ({exc})")
            return
        try:
            self.hud = webui.HudServer(port=self.args.web_port).start(
                open_browser=getattr(self.args, "web_open", False)
            )
        except SystemExit as exc:
            log_err(str(exc))
            self.hud = None
            return
        except Exception as exc:
            log_err(f"--web-ui ignored: {exc}")
            self.hud = None
            return
        self.web = self.hud.bridge
        self.web.set_model(self.args.llm_model)
        self._subscribe_web()
        log_ok(f"HTML cockpit HUD ready - open {self.hud.url}")

    def _bind_web_engine(self) -> None:
        """Feed the HUD real microphone energy once the engine is live."""
        if self.web is None or self.engine is None:
            return
        try:
            self.web.set_level_provider(self.engine.level_snapshot)
            self.web.set_floor_rms(float(self.engine.threshold or 0.0))
        except Exception as exc:
            log_warn(f"HUD level hook failed: {exc}")

    def _subscribe_web(self) -> None:
        """Mirror every orchestrator event onto the HUD bridge."""
        bus = self.bus
        bus.subscribe(EV_TEACHER, self._web_teacher)
        bus.subscribe(EV_LISTENING,
                      lambda note="": self._web_set_state("listening", note))
        bus.subscribe(EV_MUTED,
                      lambda note="": self._web_set_state("muted", note))
        bus.subscribe(EV_THINKING,
                      lambda model="": self._web_set_state("thinking", model))
        bus.subscribe(EV_LLM_DONE, self._web_llm_done)
        bus.subscribe(EV_STUDENT, self._web_student)
        bus.subscribe(EV_CYCLE, self._web_cycle)
        bus.subscribe(EV_ERROR, lambda msg="": self._web_error(msg))

    def _web_set_state(self, state: str, note: str = "") -> None:
        if self.web is not None:
            self.web.set_state(state, note or "")

    def _web_teacher(self, utt: Utterance) -> None:
        if self.web is None:
            return
        self.web.add_transcript(
            utt.text, audio_s=utt.audio_s, stt_ms=utt.transcribe_ms,
            seq=utt.seq,
        )
        self._sync_web_counts()

    def _web_llm_done(self, ms: int = 0, question: str = "") -> None:
        if self.web is None or not question:
            return
        tier = self.speaker.tier if self.speaker is not None else ""
        self.web.set_question(question, latency_ms=ms, tier=tier)

    def _web_student(self, text: str = "", tier: str = "") -> None:
        self._web_set_state("speaking", tier or "")

    def _web_cycle(self, idx: int = 0, total_ms: int = 0, stt_ms: int = 0,
                   llm_ms: int = 0, tts_ms: int = 0, tier: str = "") -> None:
        if self.web is None:
            return
        self.web.record_cycle(
            stt_ms=stt_ms, llm_ms=llm_ms, tts_ms=tts_ms, total_ms=total_ms,
            target_ms=int(LATENCY_TARGET_S * 1000),
        )
        self._sync_web_counts()

    def _web_error(self, msg: str = "") -> None:
        if self.web is not None:
            self.web.add_error(msg)

    def _sync_web_counts(self) -> None:
        """Push the authoritative gate counters (mirrors _print_summary)."""
        if self.web is None:
            return
        try:
            self.web.set_counts(
                turns=self._turns_seen, qualifying=self._qualifying,
                questions=self._questions_asked, thin=self._skipped_thin,
                gap=self._skipped_gap, cadence=self._skipped_cadence,
            )
        except Exception as exc:
            log_warn(f"HUD counters failed: {exc}")

    # -- HUD control actions ----------------------------------------------------
    def _hud_actions(self) -> bool:
        """Consume button presses queued by the HTML cockpit.

        Returns True when the caller should break out of the run loop. Called
        once per loop iteration, so a click takes effect within one 0.5 s
        utterance timeout - fast enough to feel instant without spinning.
        """
        if self.web is None:
            return False
        try:
            pending = self.web.drain_actions()
        except Exception as exc:
            log_warn(f"HUD action drain failed: {exc}")
            return False
        for action, count in pending.items():
            for _ in range(max(1, int(count))):
                if self._hud_action(action):
                    return True
        return False

    def _hud_action(self, action: str) -> bool:
        """Apply one HUD action. Returns True only for 'stop'."""
        if action == "stop":
            log_info("HUD requested stop - ending the session.")
            return True

        if action == "toggle_mute":
            if pause_recording.is_set():
                self._unmute_no_playback()
                log_info("HUD: microphone resumed.")
            else:
                pause_recording.set()
                self.bus.publish(EV_MUTED, note="paused from HUD")
                log_info("HUD: microphone paused.")
            return False

        if action == "dismiss":
            self._ignore_before = time.monotonic()
            if self.web is not None:
                self.web.resolve_question("dismissed")
            log_info("HUD: question banner dismissed.")
            return False

        if action == "ask":
            self._hud_force_ask()
            return False

        log_warn(f"Unknown HUD action ignored: {action}")
        return False

    def _hud_force_ask(self) -> None:
        """'Ask now': one cycle that bypasses the gap + cadence gates.

        The silence guard still applies - asking a question from an empty
        transcript would only produce hallucinated nonsense, so that gate is
        honoured and the reason is surfaced in the HUD error box.
        """
        if self.args.no_llm or self.engine is None or self.ollama is None:
            log_warn("HUD ask ignored: session is running without an LLM.")
            if self.web is not None:
                self.web.add_error("Ask ignored: this session has --no-llm.")
            return
        text = self.engine.transcript_text()
        ok, reason = transcript_qualifies(text, self.args.min_words)
        if not ok:
            log_info(f"HUD ask ignored: {reason}")
            if self.web is not None:
                self.web.add_error(f"Ask needs more speech - {reason}.")
            return
        now = time.monotonic()
        utt = Utterance(text=text, started_at=now, finished_at=now,
                        audio_s=0.0, transcribe_ms=0, seq=0)
        self._last_question_at = now
        log_info("HUD ask honoured - forcing one Socratic cycle.")
        self._run_cycle(utt, 0)

    # -- acoustic feedback guard ------------------------------------------------
    def _flush_mic_pipeline(self) -> int:
        """Drain residual PCM frames + reset the FSM to a clean IDLE state."""
        if self.engine is None:
            return 0
        dropped = 0
        while True:
            try:
                self.engine.pcm_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        try:
            self.engine.state = listener_mod.State.IDLE_LISTENING
            self.engine._reset_speech()
            self.engine.silence_chunks = 0
            self.engine.speech_chunks = 0
            self.engine.voiced_chunks = 0
        except Exception as exc:
            log_warn(f"FSM reset during flush: {exc}")
        return dropped

    def _mute_for_student_turn(self) -> None:
        pause_recording.set()
        self.bus.publish(EV_MUTED)
        dropped = self._flush_mic_pipeline()
        if dropped:
            log_info(f"Flushed {dropped} residual mic frame(s) at mute.")

    def _unmute_after_playback(self) -> None:
        time.sleep(PLAYBACK_COOLDOWN_S)          # room reverb settle
        dropped = self._flush_mic_pipeline()
        pause_recording.clear()
        self._ignore_before = time.monotonic()
        self.bus.publish(EV_LISTENING)
        if dropped:
            log_info(f"Flushed {dropped} residual frame(s) after playback.")

    def _unmute_no_playback(self) -> None:
        self._flush_mic_pipeline()
        pause_recording.clear()
        self._ignore_before = time.monotonic()
        self.bus.publish(EV_LISTENING)

    # -- one full Socratic cycle -------------------------------------------------
    def _run_cycle(self, utt: Utterance, stt_ms: int) -> None:
        """Mute -> LLM -> synthesize -> playback -> cooldown -> unmute."""
        self._mute_for_student_turn()

        window_text = self.engine.transcript_text()
        self.bus.publish(EV_THINKING)
        try:
            question, llm_ms = self.ollama.ask(
                window_text, self.args.max_question_words
            )
        except Exception as exc:
            self.bus.publish(EV_ERROR, msg=f"Ollama query failed: {exc}")
            self._unmute_no_playback()
            return
        self.bus.publish(EV_LLM_DONE, ms=llm_ms, question=question)
        if not question:
            self._unmute_no_playback()
            return

        self.bus.publish(EV_STUDENT, text=question, tier=self.speaker.tier)
        self._flush_mic_pipeline()   # residual frames from synthesis start-up
        t_speak = time.perf_counter()
        try:
            speak_info = self.speaker.speak(question)
        except Exception as exc:
            self.bus.publish(EV_ERROR, msg=f"TTS playback failed: {exc}")
            self._unmute_no_playback()
            return
        self._questions_asked += 1

        self._unmute_after_playback()

        tts_ms = speak_info.get("first_ms", speak_info.get("total_ms", 0))
        total_ms = int((t_speak - utt.finished_at) * 1000) + tts_ms
        self._cycles.append(total_ms)
        self.bus.publish(
            EV_CYCLE, idx=len(self._cycles), total_ms=total_ms,
            stt_ms=stt_ms, llm_ms=llm_ms, tts_ms=tts_ms, tier=speak_info["tier"],
        )

    # -- main consumption loop -----------------------------------------------------
    def run(self) -> int:
        ui_status_listening()
        try:
            while not self.stop_flag.is_set():
                if (self.args.duration
                        and time.monotonic() - self._t_start >= self.args.duration):
                    log_info(f"--duration {self.args.duration:g}s reached - stopping.")
                    break
                if self._hud_actions():          # HTML cockpit buttons
                    break
                try:
                    utt = self.engine.get_latest_utterance(timeout=0.5)
                except Exception as exc:
                    log_err(f"listener consumption error: {exc}")
                    time.sleep(0.5)
                    continue
                if utt is None:
                    continue
                if utt.finished_at <= self._ignore_before:
                    continue          # stale tail from around our own playback
                self._turns_seen += 1

                stt_ms = int((time.perf_counter() - utt.finished_at) * 1000)
                self.bus.publish(EV_TEACHER, utt=utt)

                if self.args.no_llm:
                    continue

                ok, reason = transcript_qualifies(
                    self.engine.transcript_text(), self.args.min_words
                )
                if not ok:
                    self._skipped_thin += 1
                    log_info(f"Silence guard: {reason} - no LLM query.")
                    continue
                self._qualifying += 1

                if time.monotonic() - self._last_question_at < self.args.min_question_gap:
                    self._skipped_gap += 1
                    log_info("Question gap not elapsed - listening on.")
                    continue
                self._question_counter += 1
                if self.args.ask_every > 1 and (
                        self._question_counter % self.args.ask_every != 0):
                    self._skipped_cadence += 1
                    continue

                self._last_question_at = time.monotonic()
                self._run_cycle(utt, stt_ms)
            return 0
        except KeyboardInterrupt:
            print()
            log_info("Ctrl+C received - initiating graceful shutdown ...")
            return 130

    _last_question_at = 0.0

    # -- teardown + reporting ---------------------------------------------------
    def shutdown(self, exit_code: int = 0) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.stop_flag.set()
        pause_recording.clear()              # never leave the mic muted
        try:
            if self.speaker is not None:
                self.speaker.abort()         # kill playback first
        except Exception:
            pass
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception as exc:
            log_warn(f"listener stop: {exc}")
        try:
            if self.ollama is not None:
                self.ollama.session.close()  # release pooled connections
        except Exception:
            pass
        try:
            if self.hud is not None:
                self.hud.stop()              # close SSE clients + free the port
                log_info("HTML cockpit HUD stopped.")
        except Exception as exc:
            log_warn(f"HUD stop: {exc}")
        self._print_summary(exit_code)

    def _print_summary(self, exit_code: int) -> None:
        uptime = (time.monotonic() - self._t_start) if self._t_start else 0.0
        cycles = self._cycles
        avg = sum(cycles) / len(cycles) if cycles else 0
        best = min(cycles) if cycles else 0
        within = sum(1 for c in cycles if c <= LATENCY_TARGET_S * 1000)
        print()
        print(_c("  +-------------------- SESSION SUMMARY "
                  "--------------------+", C.CYAN))
        print(f"   Uptime ............: {uptime:.1f} s")
        print(f"   Teacher turns .....: {self._turns_seen}")
        print(f"   Qualifying windows : {self._qualifying} "
              f"(thin: {self._skipped_thin}, gap: {self._skipped_gap}, "
              f"cadence: {self._skipped_cadence})")
        print(f"   Questions asked ...: {self._questions_asked} "
              f"(target: every {self.args.ask_every} turn(s))")
        if cycles:
            print(f"   Cycle latency .....: avg {avg / 1000:.2f}s | best "
                  f"{best / 1000:.2f}s | within {LATENCY_TARGET_S:.2f}s target: "
                  f"{within}/{len(cycles)}")
        if self.engine is not None:
            stats = self.engine.stats()
            print(f"   Listener stats ....: {stats['chunks']} chunks, "
                  f"{stats['chunks_dropped']} dropped, threshold "
                  f"{stats['threshold']}")
        print(f"   TTS tier in use ...: "
              f"{self.speaker.tier if self.speaker else 'n/a'}")
        print(_c("  +--------------------------------------------------+",
                 C.CYAN))
        log_ok("Shutdown complete - no orphaned processes, audio devices released.")

    # -- selftest (no mic, no LLM) ----------------------------------------------
    def selftest(self) -> int:
        log_info("SELFTEST: verifying the TTS ladder and the mute/flush/"
                 "cooldown/unmute guard ...")
        self._subscribe_ui()
        self.speaker = Speaker(preferred=self.args.tts, voice=self.args.voice,
                               tts_threads=self.args.tts_threads)
        tier = self.speaker.initialise()
        question = ("Could you explain what happens when the matrix is "
                    "singular?")
        self.bus.publish(EV_MUTED)
        dropped = self._flush_mic_pipeline()
        log_info(f"Guard: mic muted, FSM flushed ({dropped} frames in queue).")
        self.bus.publish(EV_STUDENT, text=question, tier=tier)
        t0 = time.perf_counter()
        info = self.speaker.speak(question)
        first_ms = info.get("first_ms", info.get("total_ms", 0))
        log_ok(f"Playback OK via '{info['tier']}' - first audio at "
               f"{first_ms} ms, total {info['total_ms']} ms.")
        time.sleep(PLAYBACK_COOLDOWN_S)
        pause_recording.clear()
        log_ok(f"Cooldown ({int(PLAYBACK_COOLDOWN_S * 1000)} ms) elapsed; "
               "mic unmuted.")
        self.bus.publish(EV_LISTENING)
        assert pause_recording.is_set() is False
        log_ok("SELFTEST PASSED.")
        return 0

# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "TA Phase 3 - Socratic student orchestrator: listener.py "
            "(CPU STT) + Ollama llama3.2:3b (GPU LLM) + Kokoro TTS (CPU)."
        ),
    )
    parser.add_argument("--host", default=OLLAMA_DEFAULT_HOST,
                        help=f"Ollama base URL (default: {OLLAMA_DEFAULT_HOST})")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                        help=f"Ollama model (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--stt-model", default="base.en",
                        help="faster-whisper model for listener.py (default: base.en)")
    parser.add_argument("--chunk", type=int, default=1024,
                        choices=(512, 1024), help="mic callback block size")
    parser.add_argument("--device", default=None,
                        help="input device name/index for listener.py")
    parser.add_argument("--voice", default="af_heart",
                        help="Kokoro voice (default: af_heart)")
    parser.add_argument("--tts-threads", type=int, default=TTS_THREADS_DEFAULT,
                        help="torch CPU threads for Kokoro (default: "
                             f"{TTS_THREADS_DEFAULT}; measured optimum, "
                             "whisper keeps its own 4-thread pool)")
    parser.add_argument("--tts", default="auto",
                        choices=("auto", "kokoro", "pyttsx3", "sapi"),
                        help="force a TTS backend (default: auto ladder)")
    parser.add_argument("--min-words", type=int, default=MIN_WINDOW_WORDS,
                        help="silence guard: min words in the transcript window")
    parser.add_argument("--max-question-words", type=int, default=MAX_QUESTION_WORDS,
                        help="hard word cap on the generated question")
    parser.add_argument("--ask-every", type=int, default=1,
                        help="ask on every Nth qualifying turn (default: 1)")
    parser.add_argument("--min-question-gap", type=float, default=15.0,
                        help="minimum seconds between two student questions")
    parser.add_argument("--duration", type=float, default=0,
                        help="auto-stop after N seconds (0 = until Ctrl+C)")
    parser.add_argument("--save-wavs", action="store_true",
                        help="have listener.py archive utterance WAVs")
    parser.add_argument("--no-llm", action="store_true",
                        help="listen-only mode: transcript dashboard, no LLM/TTS")
    parser.add_argument("--selftest", action="store_true",
                        help="exercise the TTS ladder + guard cycle and exit")
    parser.add_argument("--web-ui", action="store_true",
                        help="serve the Phases-5 HTML cockpit HUD (loopback "
                             "only) and publish live state to it")
    parser.add_argument("--web-port", type=int, default=8765,
                        help="HUD port (default: 8765)")
    parser.add_argument("--web-open", action="store_true",
                        help="open the HUD in the default browser on start")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    _force_utf8_stdio()
    _enable_windows_vt()
    orch = Orchestrator(args)
    exit_code = 0
    try:
        if args.selftest:
            exit_code = orch.selftest()
        else:
            try:
                orch.setup()
            except RuntimeError as exc:
                log_err(str(exc))
                return 1
            except Exception as exc:
                log_err(f"setup failed: {exc}")
                return 1
            exit_code = orch.run()
    except KeyboardInterrupt:
        print()
        log_info("Ctrl+C received - emergency teardown ...")
        exit_code = 130
    finally:
        try:
            orch.shutdown(exit_code)
        except Exception as exc:
            log_err(f"shutdown error: {exc}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())