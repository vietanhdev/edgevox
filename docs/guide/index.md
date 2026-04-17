# Introduction

EdgeVox is an **offline voice agent framework for robots** — agents, skills, workflows, and a sub-second voice pipeline, all running locally on CPU/CUDA/Metal with no cloud dependencies.

![EdgeVox TUI Screenshot](/screenshot.png)

![MuJoCo Panda Demo](/robot_panda.png)

![Unitree G1 Humanoid Demo](/robot_unitree_g1.png)

## What is EdgeVox?

EdgeVox combines two things:

1. **An agent framework** — `@tool` and `@skill` decorators, `LLMAgent` with handoffs, behavior-tree workflows (`Sequence`, `Fallback`, `Loop`, `Router`), cancellable skills with `GoalHandle`, and a `SafetyMonitor` that preempts before the LLM is consulted.
2. **A streaming voice pipeline** — Mic → VAD → STT → LLM → TTS → Speaker, delivering first audio in ~0.8s. The pipeline is the substrate that agents run on top of.

Agent code is sim-agnostic: the same Python works on `ToyWorld` (stdlib), `IrSimEnvironment` (2D navigation), `MujocoArmEnvironment` (3D pick-and-place), `MujocoHumanoidEnvironment` (Unitree G1 / H1), and `ExternalROS2Environment` (any Gazebo / Isaac / real robot over ROS2).

## Key Design Principles

- **Voice is the interface** — sub-second streaming pipeline on an RTX 3080, runs on a Jetson Orin Nano, CPU fallback on a laptop
- **Agents are the program model** — write `@tool` and `@skill` functions; compose with workflows; delegate across agents with handoffs
- **Robots are the target** — cancellable skills, safety monitor, three simulation tiers, ROS2 bridge
- **Everything is offline** — no cloud APIs, no telemetry, no vendor lock

## Simulation Tiers

| Tier | Sim | Dependencies | Role | Status |
|------|-----|-------------|------|--------|
| 0 | `ToyWorld` | stdlib only | unit tests, trivial examples | shipped |
| 1 | `IrSimEnvironment` | `pip install ir-sim` | 2D visual demo (matplotlib, diff-drive, LiDAR) | shipped |
| 2a | `MujocoArmEnvironment` | `pip install mujoco` | 3D physics, Franka pick-and-place | shipped |
| 2b | `MujocoHumanoidEnvironment` | `pip install mujoco` | Unitree G1 / H1 from Menagerie, procedural gait + ONNX policy slot | shipped |
| 3 | `ExternalROS2Environment` | sourced ROS2 workspace | drive Gazebo / Isaac / real robots over standard topics | shipped |

## Voice Pipeline Components

| Component | Default Model | Purpose |
|-----------|--------------|---------|
| **VAD** | Silero VAD v6 | Voice activity detection (32ms chunks) |
| **STT** | Faster-Whisper | Speech-to-text (auto-sizes by VRAM) |
| **LLM** | Gemma 4 E2B IT Q4_K_M | Chat via llama-cpp-python |
| **TTS** | Kokoro-82M | Text-to-speech (15 languages, 56 voices) |

## Next Steps

- [Quick Start](/guide/quickstart) — install and run in 5 minutes
- [Agents & Tools](/guide/agents) — full agent framework reference
- [Architecture](/guide/architecture) — deep dive into the streaming pipeline
- [Languages](/guide/languages) — all supported languages and backends
