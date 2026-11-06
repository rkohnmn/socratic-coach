# SocraticCoach: Offline AI Teaching Assistant & Lesson Rehearsal Studio

> **Project Status: Active Work-in-Progress**  
> ⚠️ *Note:* This project is an early-stage development effort created for personal pedagogical training. **There are many bugs and rough edges that are still being actively worked out**, including microphone pause sensitivity under varied room acoustics, OCR math symbol normalization, occasional audio buffer delays, and HUD connection recovery.

---

## 1. Overview & Pedagogical Purpose

**SocraticCoach** is an offline, real-time AI teaching assistant and rehearsal cockpit designed specifically for educators. 

As an aspiring high school educator preparing to lead classrooms, mastering lesson plan execution requires far more than knowing the subject matter—it requires:
- **Pacing & Cadence:** Becoming conscious of pauses, lecture rhythm, and transitions.
- **Anticipating Confusion:** Identifying unstated assumptions, missing logical steps, and ambiguous concepts before students get lost.
- **In-the-Moment Adaptability:** Developing the reflexive skill to field unexpected, sharp questions and explain difficult concepts from alternate angles on the fly.
- **Board Coordination:** Synchronizing spoken explanations with chalkboard and whiteboard notes.

Practicing a lesson plan alone in an empty room usually feels artificial because there is no feedback loop. SocraticCoach acts as a simulated, inquisitive student sitting in your lecture. As you deliver your lesson plan aloud:
1. It listens to your lecture in real-time through your microphone.
2. It tracks natural pauses in your delivery.
3. It analyzes your recent speech transcript and board notes to spot logical leaps, unstated assumptions, or ambiguities.
4. It formulates and speaks aloud a single, concise Socratic question (in under 18 words) using local text-to-speech, while surfacing it on a glanceable studio HUD.
5. You pause, address the question as you would with a real student, and continue your lesson.

---

## 2. Key Capabilities & Design Pillars

- **100% Offline & Private:** Zero cloud dependencies, zero telemetry, and zero subscription APIs. All speech recognition, LLM inference, vision tracking, and speech synthesis run locally on device, preserving classroom and student privacy.
- **Predictable Local Hardware Allocation:**
  - **GPU (4 GB VRAM Boundary):** Reserved exclusively for local LLM inference via Ollama (`llama3.2:3b`, ~2.2 GB VRAM).
  - **CPU (AMD Ryzen / multi-core):** Dedicated to speech-to-text (`faster-whisper`), text-to-speech (`Kokoro-82M`), blackboard computer vision (`OpenCV`), and orchestrator logic.
- **Low-Latency Socratic Interjections:** Utterance end to audio playback starts in ~1.1–1.5 seconds. The question is streamed in natural clause chunks via Kokoro TTS.
- **Acoustic Discipline & Echo Prevention:** The microphone automatically mutes when the student persona speaks, flushes residual frames, and waits for a 300 ms room-reverberation cooldown before unmuting.
- **Blackboard & Whiteboard Vision Companion (`vision_tracker.py`):** Uses an optical camera to detect writing motion, filters out teacher occlusions (walking past the board), and transcribes written equations and notes via Windows Media OCR or local vision models.
- **Studio Cockpit HUD (`hud.html` + `webui.py`):** A distraction-free, high-contrast dark dashboard (OKLCH dark-slate, WCAG AAA 7:1 contrast ratio) built according to the `DESIGN.md` specification. Readable from a standing distance of 3–6 feet with real-time waveform, question cards, and keyboard navigation.

---

## 3. System Architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                      Instructor Lecture                      │
 └──────────────┬──────────────────────────────┬────────────────┘
                │ (Microphone Audio)           │ (Chalkboard / Whiteboard)
                ▼                              ▼
     ┌──────────────────────┐      ┌───────────────────────────┐
     │     listener.py      │      │     vision_tracker.py     │
     │  - PortAudio Callback│      │  - OpenCV Frame Diffing   │
     │  - Energy / RMS FSM  │      │  - SSIM Occlusion Filter  │
     │  - faster-whisper CPU│      │  - Windows Media OCR / CPU│
     └──────────┬───────────┘      └─────────────┬─────────────┘
                │ Spoken Transcript (90s window) │ Board Notes (120s window)
                └──────────────┬─────────────────┘
                               ▼
     ┌─────────────────────────────────────────────────────────┐
     │                         main.py                         │
     │                   Master Orchestrator                   │
     │                                                         │
     │  - Evaluates silence guard (>=15 words, valid speech)   │
     │  - Mutes microphone input & flushes audio pipeline      │
     │  - GPU Inference: Ollama llama3.2:3b (Socratic Persona) │
     │  - CPU Synthesis: Kokoro-82M (2-part chunk streaming)   │
     │  - 300 ms room reverb cooldown -> Unmutes microphone    │
     └──────────┬───────────────────────────────┬──────────────┘
                │ Audio Output                  │ SSE Event Stream
                ▼                               ▼
     ┌──────────────────────┐      ┌───────────────────────────┐
     │  Student Question    │      │         webui.py          │
     │  Spoken via Speakers │      │  Studio Cockpit Dashboard │
     │  (Kokoro-82M @ 24kHz)│      │  http://127.0.0.1:8765    │
     └──────────────────────┘      └───────────────────────────┘
```

---

## 4. Hardware & Software Requirements

- **Operating System:** Windows 11 (64-bit).
- **Python:** Python 3.10, 3.11, or 3.12.
- **Processor:** Modern multi-core CPU (tested on AMD Ryzen 7 5800U, 8C/16T, 16 GB RAM).
- **GPU:** Dedicated NVIDIA GPU with at least 4 GB VRAM (tested on RTX 3050 Ti Laptop GPU).
- **Local LLM Backend:** [Ollama for Windows](https://ollama.com/) with model `llama3.2:3b` pulled (`ollama pull llama3.2:3b`).
- **Audio Equipment:** Standard microphone and speakers/headphones.
- **Camera (Optional):** Webcam or USB capture card for the chalkboard vision tracker.

---

## 5. Quickstart & Installation

### Step 1: Install Ollama and Pull the Model
Download and install [Ollama](https://ollama.com/). In PowerShell or Command Prompt, run:
```powershell
ollama pull llama3.2:3b
```

### Step 2: Run Automated Environment Setup
The repository includes a Windows batch script (`setup.bat`) that provisions an isolated `.venv`, installs CPU-safe PyTorch (preventing CUDA conflicts with Ollama), installs dependencies, and verifies audio devices:
```cmd
setup.bat
```
To run a fast structural check without re-downloading model weights:
```cmd
setup.bat --skip-models
```

You can also run verification directly via Python:
```cmd
.venv\Scripts\python.exe verify_env.py
```

---

## 6. Usage Guide

### Launching the Studio with HUD Cockpit
To run the full stack (Speech-to-Text, Socratic LLM, Kokoro TTS, and the Web HUD):
```cmd
.venv\Scripts\python.exe main.py --web-ui
```
This opens the cockpit in your browser at `http://127.0.0.1:8765`.

### Common CLI Options for `main.py`
| Flag | Description |
| --- | --- |
| `--web-ui` | Launches the orchestrator and opens the browser HUD cockpit on `:8765`. |
| `--ask-every N` | Generates a question every $N$-th qualifying speech turn (default: 1). Useful for longer lectures. |
| `--no-llm` | Runs the listener and live transcript dashboard without generating questions. |
| `--selftest` | Tests TTS synthesis and silence guards without requiring a live microphone or Ollama. |
| `--duration S` | Automatically stops after $S$ seconds. |
| `--device NAME` | Selects a specific audio input device by substring name. |

### Running the Whiteboard / Blackboard Vision Companion
To track what you write on the board in real time:
```cmd
# Calibrate your camera view to outline the blackboard region:
.venv\Scripts\python.exe vision_tracker.py --calibrate

# Run vision companion with visual bounding box preview:
.venv\Scripts\python.exe vision_tracker.py --gui --camera 0
```

---

## 7. Studio Cockpit Controls & Shortcuts

The HUD dashboard is designed to be fully operable from anywhere in the classroom via single-keystroke shortcuts (no mouse hunting required while teaching):

| Key | Action | Description |
| :---: | :--- | :--- |
| **`Space`** | **Acknowledge Question** | Marks the current Socratic question as addressed and resolves the alert card. |
| **`M`** | **Toggle Mute** | Manually mutes/unmutes the microphone (visible across the room in red). |
| **`Esc`** | **Dismiss Question** | Immediately dismisses the active question and logs the dismissal. |
| **`F`** | **Fullscreen Mode** | Toggles the presentation cockpit into distraction-free fullscreen. |
| **`C`** | **Calibrate Noise Floor**| Re-samples ambient room sound to set the speech-detection floor. |

---

## 8. Repository Structure

```
.
├── main.py              # Master orchestrator (integrates listener, LLM, TTS, event bus)
├── listener.py          # Audio capture FSM & faster-whisper transcription daemon
├── vision_tracker.py    # Blackboard motion detection & OCR companion (OpenCV + Windows OCR)
├── webui.py             # HTTP & SSE server for the live cockpit dashboard
├── hud.html             # High-contrast studio HUD web interface
├── DESIGN.md            # OKLCH design system specification & WCAG AAA tokens
├── setup.bat            # Automated Windows 11 environment bootstrap script
├── verify_env.py        # Diagnostic suite checking Python, Ollama, GPU, and audio devices
├── .gitignore           # Git ignore rules for virtual environments, models, and caches
└── README.md            # Project documentation and educator use case guide
```

---

## 9. Active Development & Known Issues

This project is actively maintained and evolved as lesson planning needs grow. Current areas undergoing improvement:
- **Audio Driver Latencies:** Handling Windows CoreAudio / PortAudio default device switches gracefully during active sessions.
- **Pause Trigger Tuning:** Fine-tuning speech pause detection (`PAUSE_TRIGGER_S`) to distinguish between normal pedagogical dramatic pauses and end-of-concept pauses.
- **Math Symbol OCR:** Improving OCR normalization for mathematical notation (e.g. superscripts, fractions, and pipe symbols in probability statements).
- **HUD Event Reconnect:** Enhancing Server-Sent Events (SSE) automatic reconnection resilience when laptops enter power-saving states.

---

## 10. Contributing & Feedback

Suggestions from fellow teachers, educators, and developers are welcome! If you have recommendations for pedagogical prompts, Socratic questioning strategies, or performance optimizations, please feel free to open an issue or submit a pull request.
