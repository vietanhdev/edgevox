# ROS2 Integration

EdgeVox includes an optional ROS2 bridge for robotics applications. It exposes the full voice pipeline — streaming tokens, sentences, state changes, wake word events, and query APIs — as standard ROS2 topics with appropriate QoS profiles.

## Enable

```bash
# Source your ROS2 workspace (rclpy must be importable)
source /opt/ros/jazzy/setup.bash

# Launch with ROS2
edgevox --ros2

# Launch with a custom namespace (for multi-robot setups)
edgevox --ros2 --ros2-namespace /robot1/voice
```

If `rclpy` is not available, the bridge falls back to a `NullBridge` (no-op) and EdgeVox runs normally without ROS2.

### Using the Launch File

```bash
# Basic launch
ros2 launch edgevox edgevox.launch.py

# With parameters
ros2 launch edgevox edgevox.launch.py namespace:=/robot1/voice language:=vi
```

## Namespace and Topic Names

All topic names are **relative** to the node namespace. The default namespace is `/edgevox`, so topics appear as `/edgevox/transcription`, `/edgevox/state`, etc.

To run multiple instances (e.g., on different robots), set a custom namespace:

```bash
edgevox --ros2 --ros2-namespace /robot1/voice   # → /robot1/voice/transcription
edgevox --ros2 --ros2-namespace /robot2/voice   # → /robot2/voice/transcription
```

## QoS Profiles

Topics use QoS profiles matched to their data characteristics:

| Profile | Reliability | Durability | Depth | Used by |
|---------|------------|------------|-------|---------|
| **State** | Reliable | Transient Local | 1 | `state` — late joiners get the current state immediately |
| **Sensor** | Best Effort | Volatile | 5 | `audio_level`, `bot_token` — high-frequency, ephemeral data |
| **Reliable** | Reliable | Volatile | 10 | All other topics — commands and events must not be lost |

When subscribing from your own nodes, match the QoS profile or use a compatible one (e.g., `BEST_EFFORT` subscribers can receive from `RELIABLE` publishers).

## Node Parameters

The EdgeVox node exposes configuration as ROS2 parameters, allowing integration with launch files, YAML config, and `ros2 param`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `en` | Language ISO 639-1 code |
| `voice` | string | `""` | TTS voice name |
| `muted` | bool | `false` | Mute/unmute the microphone |

```bash
# Change language at runtime via parameter
ros2 param set /edgevox/edgevox language vi

# Mute the microphone
ros2 param set /edgevox/edgevox muted true
```

Parameters and topic-based commands (`set_language`, `command`) both work — use whichever fits your integration pattern.

## Published Topics

### Pipeline Output

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `transcription` | `String` | Reliable | User's speech text (STT result) |
| `response` | `String` | Reliable | Bot's full response text |
| `state` | `String` | State | Pipeline state: `listening`, `transcribing`, `thinking`, `speaking`, `interrupted` |
| `audio_level` | `Float32` | Sensor | Microphone level (0.0 - 1.0) |
| `metrics` | `String` | Reliable | JSON latency metrics (stt, llm, tts, ttfs, total) |

### Streaming Output

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `bot_token` | `String` | Sensor | Individual LLM tokens as they are generated |
| `bot_sentence` | `String` | Reliable | Complete sentences as they are sent to TTS |

### Events

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `wakeword` | `String` | Reliable | Wake word detection event (e.g. "hey jarvis") |
| `info` | `String` | Reliable | JSON responses to query commands |

## Subscribed Topics

### Actions

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `tts_request` | `String` | Reliable | Text to synthesize and play aloud |
| `text_input` | `String` | Reliable | Text sent to LLM, bypassing STT (like typing in the TUI) |
| `interrupt` | `String` | Reliable | Interrupt current bot response (any message triggers it) |
| `set_language` | `String` | Reliable | Switch language at runtime (ISO 639-1 code, e.g. `vi`, `fr`) |
| `set_voice` | `String` | Reliable | Switch TTS voice at runtime (e.g. `af_bella`, `de-thorsten`) |

### Commands

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `command` | `String` | Reliable | Commands: `reset`, `mute`, `unmute`, or query commands (see below) |

## Query Commands

Send one of these strings to `command` and the response is published on `info` as JSON:

| Command | Response fields |
|---------|-----------------|
| `list_voices` | `language`, `current_voice`, `voices[]` |
| `list_languages` | `current`, `languages{code: {name, stt_backend, tts_backend}}` |
| `hardware_info` | `cuda`, `metal`, `gpu_name`, `vram_total_gb`, `vram_used_mb`, `ram_gb` |
| `model_info` | `language`, `voice`, `stt{backend, model_size, device}`, `llm{model_path, device}`, `tts{backend, voice, sample_rate}` |

All responses include a `"query"` field with the command name.

## Docker

The `Dockerfile.ros2` provides a multi-stage build that keeps the runtime image small. The container runs as a non-root `edgevox` user for security.

```bash
# Build and run
docker compose -f docker-compose.ros2.yml up --build

# Interactive shell
docker compose -f docker-compose.ros2.yml run edgevox-ros2 bash
```

Set `ROS_DOMAIN_ID` in the compose file to isolate ROS2 traffic between robots.

## Examples

### Listen to transcriptions and responses

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class VoiceListener(Node):
    def __init__(self):
        super().__init__('voice_listener')
        self.create_subscription(String, '/edgevox/transcription', self.on_user, 10)
        self.create_subscription(String, '/edgevox/response', self.on_bot, 10)

    def on_user(self, msg):
        self.get_logger().info(f'User said: {msg.data}')

    def on_bot(self, msg):
        self.get_logger().info(f'Bot replied: {msg.data}')
```

### Make the robot speak

```python
pub = node.create_publisher(String, '/edgevox/tts_request', 10)
msg = String()
msg.data = "I detected an obstacle ahead."
pub.publish(msg)
```

### Send text to the LLM (bypass microphone)

```python
pub = node.create_publisher(String, '/edgevox/text_input', 10)
msg = String()
msg.data = "What is the weather like today?"
pub.publish(msg)
```

### Interrupt the bot mid-response

```python
pub = node.create_publisher(String, '/edgevox/interrupt', 10)
pub.publish(String(data=""))
```

### Switch language at runtime

```python
# Via topic
pub = node.create_publisher(String, '/edgevox/set_language', 10)
pub.publish(String(data="vi"))  # Switch to Vietnamese

# Or via parameter
# ros2 param set /edgevox/edgevox language vi
```

### Switch voice at runtime

```python
pub = node.create_publisher(String, '/edgevox/set_voice', 10)
pub.publish(String(data="af_bella"))
```

### Query available voices

```python
import json
from std_msgs.msg import String

# Subscribe to responses
def on_info(msg):
    data = json.loads(msg.data)
    if data.get("query") == "list_voices":
        print(f"Current voice: {data['current_voice']}")
        print(f"Available: {data['voices']}")

node.create_subscription(String, '/edgevox/info', on_info, 10)

# Send query
cmd_pub = node.create_publisher(String, '/edgevox/command', 10)
cmd_pub.publish(String(data="list_voices"))
```

### Stream bot tokens for a display

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Match the BEST_EFFORT QoS used by bot_token
sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

class TokenDisplay(Node):
    def __init__(self):
        super().__init__('token_display')
        self._buffer = ""
        self.create_subscription(String, '/edgevox/bot_token', self.on_token, sensor_qos)
        self.create_subscription(String, '/edgevox/state', self.on_state, 10)

    def on_token(self, msg):
        self._buffer += msg.data
        # Update robot screen with partial response

    def on_state(self, msg):
        if msg.data == "listening":
            self._buffer = ""  # Reset for next turn
```

### React to wake word

```python
def on_wakeword(msg):
    print(f"Wake word detected: {msg.data}")
    # Robot turns to face the user, activates LEDs, etc.

node.create_subscription(String, '/edgevox/wakeword', on_wakeword, 10)
```
