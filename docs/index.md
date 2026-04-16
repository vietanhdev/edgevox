---
layout: home

hero:
  name: EdgeVox
  text: Offline voice agent framework for robots
  tagline: Agents + Skills + Workflows — sub-second voice pipeline, 2D/3D simulation, fully local
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
  - title: Agent Framework
    icon: 🧠
    details: "@tool and @skill decorators, LLMAgent with handoffs, behavior-tree workflows (Sequence, Fallback, Loop, Router), cancellable skills with GoalHandle."
  - title: 2D + 3D Simulation
    icon: 🤖
    details: "IR-SIM for 2D navigation, MuJoCo for 3D pick-and-place. Same SimEnvironment protocol — swap backends without changing agent code."
  - title: Sub-second Voice
    icon: ⚡
    details: Streaming STT + LLM + TTS pipeline delivers first audio in ~0.8s. 15 languages, 56 voices, 4 TTS backends.
  - title: 100% Offline
    icon: 🔒
    details: No cloud APIs, no telemetry. Whisper STT + Gemma 4 LLM + Kokoro/Piper TTS all run on your hardware (CPU/CUDA/Metal).
  - title: Safety-First Robotics
    icon: 🛡️
    details: "SafetyMonitor preempts before the LLM is consulted. Stop-words halt in-flight skills in ~200ms. The LLM never enters the reactive layer."
  - title: ROS2 Native
    icon: 📡
    details: Full ROS2 bridge with pub/sub topics, proper QoS, and multi-robot namespace support. Planned RosActionSkill for Nav2/MoveIt2.
---

## Demos

<div style="display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 2rem 0;">
  <div style="flex: 1; min-width: 300px;">
    <img src="/screenshot.png" alt="EdgeVox TUI — voice pipeline" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
    <p style="text-align: center; margin-top: 0.5rem; color: var(--vp-c-text-2);"><strong>Voice Pipeline TUI</strong> — streaming STT + LLM + TTS</p>
  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="/robot_mujoco.png" alt="MuJoCo Franka Panda pick-and-place demo" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
    <p style="text-align: center; margin-top: 0.5rem; color: var(--vp-c-text-2);"><strong>MuJoCo 3D Demo</strong> — voice-controlled Franka Panda</p>
  </div>
</div>

```bash
edgevox                                    # voice pipeline TUI
edgevox-agent robot-panda --text-mode      # MuJoCo pick-and-place
edgevox-agent robot-irsim --text-mode      # IR-SIM 2D navigation
```
