# training/generate_config.py
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "task": "path_following",
    "algorithm": "sac",
    "environment": {
        "name": "ROVEnv",
        "params": {}
    },
    "hyperparameters": {
        # 通用设置
        "max_episodes": 500,
        "max_steps_per_episode": 1000,
        "replay_buffer_size": 1_000_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "explore_noise": 0.1,
        "eval_interval": 20,
        "save_interval": 50
    },
    "algorithms": {
        "sac": {
            "lr": 3e-4,
            "automatic_entropy_tuning": True,
            "target_entropy": -8,
            "use_value_net": True,
            "log_std_min": -20,
            "log_std_max": 2
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
    }
}


def generate_config(path: Path) -> None:
    """
    创建或覆盖默认训练配置文件。
    Args:
        path: 配置文件输出路径，支持 Path 对象。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False, allow_unicode=True)
    print(f"Generated default config at: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate default config.yaml for ROV RL training"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="输出 config.yaml 的路径"
    )
    args = parser.parse_args()
    generate_config(args.output)


if __name__ == "__main__":
    main()
