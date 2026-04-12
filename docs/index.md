---
layout: home

hero:
  name: EdgeVox
  text: Sub-second local voice AI
  tagline: VAD > STT > LLM > TTS — fully local, fully private
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: View on GitHub
      link: https://github.com/vietanhdev/edgevox

  image:
    src: /screenshot.png
    alt: EdgeVox TUI Screenshot

features:
  - title: Sub-second Latency
    icon: ⚡
    details: Streaming pipeline with sentence-level TTS delivers first audio in under 0.8 seconds.
  - title: 15 Languages
    icon: 🌍
    details: English, Vietnamese, French, Korean, Thai, German, Russian, and more with language-specific STT/TTS backends.
  - title: 100% Local
    icon: 🔒
    details: No cloud APIs. Whisper/Sherpa STT + Gemma LLM + Kokoro/Piper/Supertonic TTS all run on your hardware.
  - title: Voice Interrupt
    icon: 🎤
    details: Speak over the bot mid-response to cut it off for natural conversational flow.
  - title: TUI + Web UI
    icon: 📟
    details: Rich terminal UI with slash commands, plus a Web UI served via FastAPI + WebSocket.
  - title: ROS2 Ready
    icon: 🤖
    details: Full ROS2 bridge for robotics — streaming tokens, text input, interrupt, language/voice switching, wake word events, and query APIs.
---
