"""
Evaluate a trained PPO policy on robosuite.

Usage:
    python scripts/eval_policy.py --checkpoint checkpoints/ppo_best.pt
    python scripts/eval_policy.py --checkpoint checkpoints/ppo_best.pt --num_episodes 20
    python scripts/eval_policy.py --checkpoint checkpoints/ppo_best.pt --save_video
"""

import os
import argparse

# Parse args early so we can set GPU env vars before importing mujoco
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for PyTorch (auto-switches to CPU when saving video)")
    parser.add_argument("--save_video", action="store_true",
                        help="Save video of evaluation (requires imageio)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (mean instead of sampling)")
    parser.add_argument("--custom_reward", action="store_true",
                        help="Use custom reward environment (must match training)")
    return parser.parse_args()

# Parse args and set env vars before importing mujoco/robosuite
_args = parse_args()
os.environ["MUJOCO_GL"] = "egl"

# Use CPU for inference when saving video to avoid GPU contention with MuJoCo EGL
if _args.save_video and _args.device.startswith("cuda"):
    print("Note: Using CPU for inference when saving video (avoids GPU contention with MuJoCo EGL)")
    _args.device = "cpu"

import logging
logging.disable(logging.INFO)

import numpy as np
import torch
from pathlib import Path

from envs.robosuite_wrapper import RobosuiteGymWrapper
from training.ppo import ActorCritic

import imageio


def evaluate(policy, env, num_episodes, deterministic=False, device="cuda", save_video=False, video_dir="."):
    """Run evaluation episodes and return statistics."""
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        frames = [] if save_video and ep == 0 else None

        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                mean, std, _ = policy(obs_tensor)
                if deterministic:
                    action = mean
                else:
                    action = torch.distributions.Normal(mean, std).sample()

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            if frames is not None:
                frames.append(env.render())

        if frames is not None and len(frames) > 0:
            video_path = Path(video_dir) / "eval_episode.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            print(f"Saved video to {video_path}")
            frames = []
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(env._check_success())

        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, Success = {successes[-1]}")

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean(successes),
        "episode_rewards": episode_rewards,
    }


def main():
    args = _args  # Use already-parsed args

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint['config']

    # Create environment
    render_mode = "rgb_array" if args.save_video else None
    env = RobosuiteGymWrapper(
        env_name="Lift",
        render_mode=render_mode,
        use_custom_reward=args.custom_reward,
    )

    # Create and load policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = ActorCritic(obs_dim, action_dim, config.hidden_dims).to(args.device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    print(f"\nEvaluating for {args.num_episodes} episodes...")
    print("-" * 50)

    # Run evaluation (save video in checkpoint directory)
    checkpoint_dir = Path(args.checkpoint).parent
    stats = evaluate(
        policy, env, args.num_episodes,
        deterministic=args.deterministic,
        device=args.device,
        save_video=args.save_video,
        video_dir=str(checkpoint_dir),
    )

    # Print summary
    print("-" * 50)
    print(f"\nResults over {args.num_episodes} episodes:")
    print(f"  Mean reward:  {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  Min reward:   {stats['min_reward']:.2f}")
    print(f"  Max reward:   {stats['max_reward']:.2f}")
    print(f"  Mean length:  {stats['mean_length']:.1f}")
    print(f"  Success rate: {stats['success_rate'] * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
