---
layout: home

hero:
  name: EdgeVox
  text: Voice agents for robots.
  tagline: Offline agents, skills, and workflows — sub-second voice pipeline, fully on-device.
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
  - title: Full Sim Stack
    icon: 🤖
    details: "ToyWorld · IR-SIM (2D) · MuJoCo Franka (arm) · Unitree G1/H1 (humanoid) · External ROS2 (Gazebo, Isaac, real robots). One SimEnvironment protocol across all tiers."
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
    details: "Full ROS2 surface — voice + robot_state + agent_event topics, TF2 / Nav2 cmd_vel / LaserScan / Image, `execute_skill` ActionServer, std_srvs query services."
---

## Demos

<div style="display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 2rem 0;">
  <div style="flex: 1; min-width: 300px;">
    <img src="/screenshot.png" alt="EdgeVox TUI — voice pipeline" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
    <p style="text-align: center; margin-top: 0.5rem; color: var(--vp-c-text-2);"><strong>Voice Pipeline TUI</strong> — streaming STT + LLM + TTS</p>
  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="/robot_panda.png" alt="MuJoCo Franka Panda pick-and-place demo" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
    <p style="text-align: center; margin-top: 0.5rem; color: var(--vp-c-text-2);"><strong>MuJoCo 3D Demo</strong> — voice-controlled Franka Panda</p>
  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="/robot_unitree_g1.png" alt="Unitree G1 humanoid demo" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
    <p style="text-align: center; margin-top: 0.5rem; color: var(--vp-c-text-2);"><strong>Unitree G1 Humanoid</strong> — voice-controlled procedural gait</p>
  </div>
</div>

```bash
edgevox                                       # voice pipeline TUI
edgevox-agent robot-panda --text-mode         # MuJoCo Franka pick-and-place
edgevox-agent robot-irsim --text-mode         # IR-SIM 2D navigation
edgevox-agent robot-humanoid --simple-ui      # Unitree G1 humanoid (auto-fetched)
edgevox-agent robot-external --text-mode      # drive any external ROS2 sim / robot
```

Any of those composes with `--ros2` to publish `/edgevox/robot_state` + `/agent_event`, accept `cmd_vel` / `goal_pose` + `text_input`, and expose the `execute_skill` action.
