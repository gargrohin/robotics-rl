# Robotics RL

RL fine-tuning for Vision-Language-Action (VLA) models in robotics simulation.

## Overview

This project implements PPO-based reinforcement learning for fine-tuning pretrained VLA models (Octo, OpenVLA, Pi0) on robotic manipulation tasks.

## Setup

```bash
uv sync
```

## Project Structure

```
robotics-rl/
├── configs/              # Training configurations
├── envs/                 # Environment wrappers (robosuite, Isaac Lab)
├── models/               # VLA model loading and wrappers
├── training/             # PPO trainer, rewards, baselines
├── scripts/              # Training scripts
└── notebooks/            # Experimentation
```

## Dependencies

- **Simulation**: MuJoCo + robosuite (or Isaac Lab for GPU-parallel)
- **VLA Models**: Octo (93M), OpenVLA (7B), Pi0 (3B)
- **RL**: Custom PPO implementation with KL penalty
