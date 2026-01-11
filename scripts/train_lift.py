"""
Training script for PPO on robosuite Lift task.

Usage:
    python scripts/train_lift.py --total_timesteps 100000
    python scripts/train_lift.py --total_timesteps 500000 --wandb_project my-project
    python scripts/train_lift.py --no_wandb  # disable wandb logging
"""

import argparse
import logging
import os
from pathlib import Path

# Suppress robosuite INFO messages
logging.disable(logging.INFO)

import torch
from envs.robosuite_wrapper import RobosuiteGymWrapper
from training.ppo import PPOTrainer
from configs.ppo_config import PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on robosuite Lift task")

    # Training args
    parser.add_argument("--total_timesteps", type=int, default=100_000,
                        help="Total timesteps to train for")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda/cpu)")

    # Environment args
    parser.add_argument("--env_name", type=str, default="Lift",
                        help="Robosuite environment name")
    parser.add_argument("--robot", type=str, default="Panda",
                        help="Robot type")
    parser.add_argument("--max_episode_steps", type=int, default=500,
                        help="Max steps per episode")

    # Custom reward args
    parser.add_argument("--custom_reward", action="store_true",
                        help="Use custom reward shaping (recommended)")
    parser.add_argument("--reaching_weight", type=float, default=0.1,
                        help="Weight for reaching reward (custom reward)")
    parser.add_argument("--grasp_reward", type=float, default=5.0,
                        help="Reward for grasping (custom reward)")
    parser.add_argument("--lift_reward", type=float, default=10.0,
                        help="Reward for lifting progress (custom reward)")
    parser.add_argument("--success_reward", type=float, default=50.0,
                        help="Reward for successful lift (custom reward)")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Max gradient norm for clipping")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of PPO epochs per update")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for PPO updates")
    parser.add_argument("--rollout_steps", type=int, default=2048,
                        help="Steps to collect per rollout")

    # Logging args
    parser.add_argument("--wandb_project", type=str, default="ppo-robosuite",
                        help="Wandb project name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")

    # Save args
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (used in checkpoint filename)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    # Create checkpoint directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    print(f"Creating {args.env_name} environment with {args.robot} robot...")
    if args.custom_reward:
        print("Using custom reward shaping")
        custom_reward_kwargs = {
            "reaching_weight": args.reaching_weight,
            "grasp_reward": args.grasp_reward,
            "lift_reward": args.lift_reward,
            "success_reward": args.success_reward,
        }
    else:
        custom_reward_kwargs = None

    env = RobosuiteGymWrapper(
        env_name=args.env_name,
        robots=args.robot,
        max_episode_steps=args.max_episode_steps,
        use_custom_reward=args.custom_reward,
        custom_reward_kwargs=custom_reward_kwargs,
    )
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Create config
    config = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
    )

    # Create trainer
    print(f"Creating PPO trainer on {args.device}...")
    trainer = PPOTrainer(config, env, device=args.device)

    # Train
    print(f"Starting training for {args.total_timesteps} timesteps...")
    trainer.train(
        total_timesteps=args.total_timesteps,
        project_name=args.wandb_project,
        save_dir=str(save_dir),
    )

    # Save final model
    run_name = args.run_name or f"{args.env_name}_{args.total_timesteps}"
    save_path = save_dir / f"ppo_{run_name}_final.pt"
    trainer.save(str(save_path))
    print(f"Best model saved at: {save_dir}/ppo_best.pt")

    # Cleanup
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
