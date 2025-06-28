#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any


def default_config() -> Dict[str, Any]:
    """
    Returns the default configuration dictionary for ROV RL training.
    """
    return {
        "task": "path_following",  # or 'pose_control'
        "environment": {
            "name": "ROVEnv",
            # Path following settings
            "waypoints": 5,
            "bounds": [
                [-1.0, -1.0, -1.0],
                [ 1.0,  1.0,  1.0]
            ],
            "tolerance": 0.1  # anomaly detection tolerance
        },
        "hyperparameters": {
            # Common RL settings
            "max_episodes": 500,
            "max_steps_per_episode": 1000,
            "replay_buffer_size": 1000000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "explore_noise": 0.1,
            "eval_interval": 20,
            "save_interval": 50
        },
        "algorithms": {
            "sac": {
                "lr": 1e-4,
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "automatic_entropy_tuning": True,
                "target_entropy": -4,
                "alpha": 0.1,
                "use_value_net": True,
                "log_std_min": -20,
                "log_std_max": 2,
                "hidden_dims": [256, 256]
            },
            "td3": {
                "actor_lr": 3e-4,
                "critic_lr": 3e-4,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_freq": 2
            },
            "ddpg": {
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "noise_std": 0.1,
                "noise_clip": 0.2
            }
        },
        "reward_scales": {
            "progress_scale": 200.0,   # delta position reward multiplier
            "angle_penalty": 0.1,      # quaternion error penalty
            "distance_penalty": 0.01,  # distance error penalty
            "time_penalty": 0.0001     # per-step time penalty
        }
    }


def generate_config(path: Path) -> None:
    """
    Create or overwrite a default config.yaml for ROV RL training.
    Writes the file to the specified path.
    """
    config = default_config()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Generated default config at: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate default config.yaml for ROV RL training"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path.cwd() / "training" / "config.yaml",
        help="Output path for config.yaml"
    )
    args = parser.parse_args()
    generate_config(args.output)


if __name__ == "__main__":
    main()
