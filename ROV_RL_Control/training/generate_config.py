# 文件：rov_rl_control/training/generate_config.py

import os
import argparse
import yaml
from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    "hyperparameters": {
        # 通用设置
        "max_episodes": 500,
        "max_steps_per_episode": 1000,
        "replay_buffer_size": 1000000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "explore_noise": 0.1,
        "eval_interval": 20,
        "save_interval": 50,
        # 如果需要给每个算法单独传参，可以按下面模板添加 key
        "sac": {
            "lr": 3e-4,
            "automatic_entropy_tuning": True,
            "target_entropy": -8,    # 如果 action_dim=8
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


def generate_config(path: str) -> None:
    """
    在指定路径创建默认的 config.yaml 文件，如果文件已存在，则会覆盖。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)
    print(f"Default config.yaml has been generated at: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate default config.yaml for ROV RL training")
    parser.add_argument(
        "--output",
        type=str,
        default="rov_rl_control/training/config.yaml",
        help="输出 config.yaml 的路径（相对或绝对都可）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_config(args.output)
