# EdgeVox

**Sub-second local voice AI for robots and edge devices.**

No cloud APIs. No internet after setup. Fully private. Powered by Gemma 4.

```
    ______    __         _    __
   / ____/___/ /___ ____| |  / /___  _  __
  / __/ / __  / __ `/ _ \ | / / __ \| |/_/
 / /___/ /_/ / /_/ /  __/ |/ / /_/ />  <
/_____/\__,_/\__, /\___/|___/\____/_/|_|
            /____/
```

**Stack:** Silero VAD -> faster-whisper (STT) -> Gemma 4 E2B IT via llama.cpp (LLM) -> Kokoro 82M (TTS)

**Tested latency:** **0.80s** end-to-end on RTX 3080 (STT 0.40s + LLM 0.33s + TTS 0.08s)

## Features

- **Streaming pipeline** — speaks first sentence while LLM generates the rest
- **Interrupt support** — speak while bot is talking to cut it off
- **Wake word detection** — "Hey Jarvis" / "Lily" (optional, via OpenWakeWord)
- **Beautiful TUI** — ASCII logo, sparkline waveform, latency history, GPU/RAM monitor, model info panel
- **ROS2 bridge** — full robotics integration: streaming tokens, text input, interrupt, language/voice switching, wake word events, query APIs
- **Slash commands** — `/reset`, `/lang`, `/voice`, `/say`, `/mictest`, `/model` in the TUI
- **Chat export** — Ctrl+S to save conversation as markdown
- **15 languages** — English, Vietnamese, French, Spanish, Hindi, Italian, Portuguese, Japanese, Chinese, Korean, German, Thai, Russian, Arabic, Indonesian
- **Auto-detects hardware** — GPU layers, model size, STT model

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
| Wake | OpenWakeWord | ~2MB | ~10MB |

**M1 Air (8GB):** whisper-small + Q4_K_M = **3.4GB**
**PC with GPU:** whisper-large-v3-turbo + Q4_K_M = **5.8GB**

## Documentation

Full docs: [EdgeVox Docs](https://edgevox-ai.github.io/edgevox/) (built with VitePress)

```bash
cd website && npm run dev
```

## License

MIT
