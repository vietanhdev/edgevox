# Server API Reference

EdgeVox exposes a FastAPI server with REST endpoints and a WebSocket for real-time voice interaction.

Start the server with `edgevox --web-ui` or `edgevox-serve`.

## REST Endpoints

### `GET /api/health`

Liveness check.

```json
{ "status": "ok", "active_sessions": 2 }
```

### `GET /api/info`

Returns current server configuration.

```json
{
  "language": "en",
  "languages": ["ar", "de", "en", "en-gb", "es", "fr", "hi", "id", "it", "ja", "ko", "pt", "ru", "th", "vi", "zh"],
  "voice": "af_heart",
  "voices": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael"],
  "stt": "WhisperSTT",
  "tts": "KokoroTTS",
  "tts_sample_rate": 24000,
  "active_sessions": 1
}
```

### `GET /`

Serves the React SPA (built from `webui/`). Returns a 503 with build instructions if the SPA hasn't been built.

---

## WebSocket (`/ws`)

The WebSocket endpoint handles real-time voice interaction. Each connection gets an isolated session with its own conversation history and VAD state.

### Connection Flow

```
Client connects → Server sends "ready" event → Server sends "listening" state
Client streams audio frames (binary) and/or sends control messages (JSON)
Server streams back: state updates, transcriptions, LLM tokens, TTS audio, metrics
```

### Client → Server

#### Binary (Audio Frames)

Send raw **int16 PCM at 16kHz mono** as binary WebSocket frames. The server runs VAD on each frame and automatically segments speech into utterances.

#### JSON Control Messages

| Type | Fields | Description |
|------|--------|-------------|
| `text_input` | `text` | Send text to the LLM (bypasses STT) |
| `say` | `text` | TTS preview — synthesize and play text (no LLM) |
| `interrupt` | — | Stop current speech and return to listening |
| `reset` | — | Clear conversation history |
| `set_language` | `language` | Switch language by code (e.g., `"vi"`, `"ko"`) |
| `set_voice` | `voice` | Switch TTS voice (e.g., `"bf_emma"`, `"ko-M2"`) |

**Examples:**

```json
{"type": "text_input", "text": "What is the weather today?"}
{"type": "say", "text": "Hello, this is a test."}
{"type": "interrupt"}
{"type": "reset"}
{"type": "set_language", "language": "fr"}
{"type": "set_voice", "voice": "ff_siwis"}
```

### Server → Client

#### JSON Events

| Type | Fields | Description |
|------|--------|-------------|
| `ready` | `session_id`, `language`, `languages[]`, `voice`, `voices[]`, `tts_sample_rate`, `sample_rate`, `frame_size` | Sent once on connection |
| `state` | `value`: `"listening"` \| `"transcribing"` \| `"thinking"` \| `"speaking"` | Pipeline state change |
| `level` | `value`: 0.0–1.0 | Microphone input level (RMS) |
| `user_text` | `text`, `latency` | Transcribed user speech (latency = STT time in seconds) |
| `bot_token` | `text` | Single LLM token (streaming) |
| `bot_sentence` | `text`, `audio_id`, `sample_rate`, `bytes` | Sentence metadata — binary audio follows immediately |
| `bot_text` | `text`, `latency` | Complete LLM response (latency = LLM time in seconds) |
| `metrics` | `stt`, `llm`, `ttft`, `tts`, `total`, `audio_duration` | Performance metrics in seconds |
| `language_changed` | `language`, `voice`, `voices[]` | After successful language switch |
| `voice_changed` | `voice` | After successful voice switch |
| `info` | `message` | Informational notification |
| `error` | `message` | Error notification |

#### Binary (TTS Audio)

After each `bot_sentence` JSON event, the server sends the corresponding audio as a binary WebSocket frame containing **WAV bytes (int16 PCM)**. Match audio to sentences using the `audio_id` field.

### Metrics Fields

The `metrics` event provides timing for each pipeline stage:

| Field | Description |
|-------|-------------|
| `stt` | Speech-to-text time (seconds) |
| `llm` | LLM generation time (seconds) |
| `ttft` | Time to first LLM token (seconds) |
| `tts` | Total TTS synthesis time (seconds) |
| `total` | End-to-end time (seconds) |
| `audio_duration` | Duration of the input audio segment (seconds) |

### Session Behavior

- Each WebSocket connection creates an isolated session with its own conversation history
- VAD is paused while the pipeline is busy (prevents echo during TTS)
- Only one turn runs at a time per session — new audio segments are ignored while busy
- The `interrupt` message cancels the current turn immediately
- Sessions are cleaned up on disconnect
