import argparse
from pathlib import Path
import yaml
from typing import Dict, Any

# Default configuration template for ROV RL training
DEFAULT_CONFIG: Dict[str, Any] = {
    "task": "path_following",  # or 'pose_control'
    "algorithm": "sac",        # 'sac', 'td3', or 'ddpg'
    "environment": {
        # path_following settings
        "waypoints": 5,             # number of intermediate waypoints
        "bounds": [                # [min_xyz, max_xyz]
            [-1.0, -1.0, -1.0],
            [ 1.0,  1.0,  1.0]
        ],
        "tolerance": 0.1            # evaluation anomaly threshold
    },
    "hyperparameters": {
        # common RL settings
        "max_episodes": 500,
        "max_steps_per_episode": 1000,
        "replay_buffer_size": 1000000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "explore_noise": 0.1,
        "eval_interval": 20,
        "save_interval": 50,
        # algorithm-specific defaults (merged)
        # SAC
        "lr": 3e-4,
        "automatic_entropy_tuning": True,
        "target_entropy": -8,
        "use_value_net": True,
        "log_std_min": -20,
        "log_std_max": 2,
        # TD3 (will be ignored if algo!='td3')
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        # DDPG (ignored if algo!='ddpg')
        "noise_std": 0.1,
        "noise_clip": 0.2
    }
}


def generate_config(path: Path) -> None:
    """
    Create or overwrite a default config.yaml for ROV RL training.
    The file will be written to the specified path.
    """
    # Ensure parent exists
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dump YAML
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False, allow_unicode=True)
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
