# EdgeVox

**Sub-second local voice AI for robots and edge devices.**

No cloud APIs. No internet after setup. Fully private. Powered by Gemma 4.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![EdgeVox TUI Screenshot](docs/public/screenshot.png)

---

**0.8s** end-to-end latency &nbsp;|&nbsp; **15** languages &nbsp;|&nbsp; **56** voices &nbsp;|&nbsp; **4** wake words &nbsp;|&nbsp; **4** interfaces (TUI, Web, CLI, Text)

---

## Why EdgeVox?

Most voice assistants send your audio to the cloud. EdgeVox runs **everything locally** — VAD, STT, LLM, and TTS — on hardware as small as a Jetson Orin Nano. The streaming pipeline starts speaking the first sentence while the LLM is still generating the rest, keeping latency under one second on a GPU.

```
Microphone → Silero VAD → faster-whisper (STT) → Gemma 4 via llama.cpp → Kokoro 82M (TTS) → Speaker
```

## Features

- **Sub-second streaming** — speaks the first sentence while the LLM generates the rest (0.8s on RTX 3080)
- **15 languages** — English, Vietnamese, French, Spanish, Hindi, Italian, Portuguese, Japanese, Chinese, Korean, German, Thai, Russian, Arabic, Indonesian
- **56 voices** across 4 TTS backends — Kokoro (25), Piper (20), Supertonic (10), PyThaiTTS (1)
- **Voice interrupt** — speak while the bot is talking to cut it off naturally
- **4 wake words** — "Hey Jarvis", "Alexa", "Hey Mycroft", "Okay Nabu" via pymicro-wakeword
- **15 slash commands** — `/lang`, `/voice`, `/say`, `/mictest`, `/model`, `/devices`, and more
- **Web UI** — FastAPI + WebSocket server with a Vue.js frontend
- **Beautiful TUI** — ASCII logo, sparkline waveform, latency history, GPU/RAM monitor
- **ROS2 bridge** — full robotics integration with streaming tokens, text input, interrupt, language/voice switching, and query APIs
- **Chat export** — save conversations as markdown
- **Auto-detects hardware** — GPU layers, model size, and STT model selection

## Hardware Requirements

| Device | RAM | GPU | Expected Latency |
|--------|-----|-----|-------------------|
| PC (i9 + RTX 3080 16GB) | 64GB | CUDA | **~0.8s** |
| Jetson Orin Nano | 8GB | CUDA | ~1.5-2s |
| MacBook Air M1 | 8GB | Metal | ~2-3s |
| Any modern laptop | 16GB+ | CPU only | ~2-4s |

## Quick Start

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# 3. Install llama-cpp-python with CUDA (prebuilt wheels)
uv pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# For Apple Silicon (Metal):
# CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python

# For CPU only:
# uv pip install llama-cpp-python

# 4. Install EdgeVox
uv pip install -e .

# 5. Download all models (~3GB total)
edgevox-setup

# 6. Run!
edgevox
```

## Usage

```bash
# TUI mode (default, recommended)
edgevox

# Web UI mode
edgevox --web-ui

# With wake word
edgevox --wakeword "hey jarvis"

# With ROS2 bridge (for robotics)
edgevox --ros2

# ROS2 with custom namespace (multi-robot)
edgevox --ros2 --ros2-namespace /robot1/voice

# CLI mode (simpler, no TUI)
edgevox-cli

# Text mode (no microphone)
edgevox-cli --text-mode

# Custom options
edgevox \
    --whisper-model large-v3-turbo \
    --voice am_adam \
    --language en
```

## Languages & Backends

| Language | STT Backend | TTS Backend | Voices |
|----------|-------------|-------------|--------|
| English, French, Spanish, Hindi, Italian, Portuguese, Japanese, Chinese | faster-whisper | Kokoro | 25 |
| Vietnamese | sherpa-onnx (Zipformer) | Piper | 20 |
| German, Russian, Arabic, Indonesian | faster-whisper | Piper | varies |
| Korean | faster-whisper | Supertonic | 10 |
| Thai | faster-whisper | PyThaiTTS | 1 |

Models are hosted on [`nrl-ai/edgevox-models`](https://huggingface.co/nrl-ai/edgevox-models) (HuggingFace) with fallback to upstream repos.

## TUI Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset conversation |
| `M` | Mute/Unmute mic |
| `/` | Open command input |
| `Ctrl+S` | Export chat to markdown |

### Slash Commands

| Command | Action |
|---------|--------|
| `/reset` | Reset conversation |
| `/lang XX` | Switch language (en, vi, fr, ko, ...) |
| `/langs` | List all supported languages |
| `/say TEXT` | TTS preview — speak text directly |
| `/mictest` | Record 3s + playback to test audio |
| `/model SIZE` | Switch Whisper model (small/medium/large-v3-turbo) |
| `/voice XX` | Switch TTS voice |
| `/voices` | List available voices |
| `/mic` | Switch microphone device |
| `/spk` | Switch speaker device |
| `/devices` | List audio devices |
| `/export` | Export chat to markdown |
| `/mute` | Mute microphone |
| `/unmute` | Unmute microphone |
| `/help` | Show all commands |

## ROS2 Integration

EdgeVox publishes voice pipeline events to ROS2 topics with proper QoS profiles, making it easy to add voice interaction to any robot. Topics use relative names under a configurable namespace (default `/edgevox`).

```bash
# Source ROS2 workspace (rclpy must be importable)
source /opt/ros/jazzy/setup.bash

# Run with ROS2 bridge
edgevox --ros2

# Custom namespace for multi-robot setups
edgevox --ros2 --ros2-namespace /robot1/voice

# Or use the launch file
ros2 launch edgevox edgevox.launch.py namespace:=/robot1/voice language:=vi
```

### Published Topics

All topic names are relative — shown here with the default `/edgevox` namespace.

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `transcription` | `String` | Reliable | User's speech (STT output) |
| `response` | `String` | Reliable | Bot's full response text |
| `state` | `String` | Transient Local | Pipeline state (listening, transcribing, thinking, speaking, interrupted) |
| `audio_level` | `Float32` | Best Effort | Mic level (0.0-1.0) |
| `metrics` | `String` | Reliable | JSON latency metrics |
| `bot_token` | `String` | Best Effort | Streaming LLM tokens |
| `bot_sentence` | `String` | Reliable | Completed sentences (TTS chunks) |
| `wakeword` | `String` | Reliable | Wake word detection events |
| `info` | `String` | Reliable | JSON responses to query commands |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `tts_request` | `String` | Send text for the bot to speak |
| `text_input` | `String` | Send text to LLM (bypass STT) |
| `interrupt` | `String` | Interrupt current bot response |
| `set_language` | `String` | Switch language (e.g. `vi`, `fr`) |
| `set_voice` | `String` | Switch TTS voice (e.g. `af_bella`) |
| `command` | `String` | Commands: reset, mute, unmute, list_voices, list_languages, hardware_info, model_info |

### Node Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `en` | Language ISO 639-1 code |
| `voice` | string | `""` | TTS voice name |
| `muted` | bool | `false` | Mute/unmute the microphone |

```bash
ros2 param set /edgevox/edgevox language vi
ros2 param set /edgevox/edgevox muted true
```

### Example: Robot Integration

```python
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# Listen to what the user says
node.create_subscription(String, '/edgevox/transcription', on_user_speech, 10)

# Make the robot say something
pub = node.create_publisher(String, '/edgevox/tts_request', 10)
msg = String()
msg.data = "I detected an obstacle ahead."
pub.publish(msg)

# Stream bot tokens (match BEST_EFFORT QoS)
sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
node.create_subscription(String, '/edgevox/bot_token', on_token, sensor_qos)

# Switch language via parameter
# ros2 param set /edgevox/edgevox language vi
```

## Architecture

```
                        EdgeVox Pipeline
 +-----------+     +------------+     +----------------+
 | Microphone|---->| Silero VAD |---->| faster-whisper  |
 |           |     | (32ms)     |     | (STT)          |
 +-----------+     +------------+     +--------+-------+
                                               |
                                               v
                                      +----------------+
                                      | Gemma 4 E2B IT |
                                      | (streaming)    |
                                      +--------+-------+
                                               | sentence by sentence
                                               v
 +-----------+     +------------+     +----------------+
 |  Speaker  |<----| Kokoro 82M |<----| Sentence       |
 |           |     | (TTS)      |     | Splitter       |
 +-----------+     +------------+     +----------------+
                         |
                         v (optional)
                   +------------+
                   | ROS2 Bridge|----> <namespace>/* topics
                   +------------+
```

## Model Sizes

| Component | Model | Size | RAM |
|-----------|-------|------|-----|
| VAD | Silero VAD v6 | ~2MB | ~10MB |
| STT | whisper-small | 500MB | ~600MB |
| STT | whisper-large-v3-turbo | 1.5GB | ~2GB |
| LLM | Gemma 4 E2B IT Q4_K_M | 1.8GB | ~2.5GB |
| TTS | Kokoro 82M | 200MB | ~300MB |
| Wake | pymicro-wakeword | ~5MB | ~10MB |

**M1 Air (8GB):** whisper-small + Q4_K_M = **3.4GB**
**PC with GPU:** whisper-large-v3-turbo + Q4_K_M = **5.8GB**

## Documentation

Full docs: [EdgeVox Docs](https://edgevox-ai.github.io/edgevox/) (built with VitePress)

```bash
cd docs && npm run dev
```

## License

MIT
