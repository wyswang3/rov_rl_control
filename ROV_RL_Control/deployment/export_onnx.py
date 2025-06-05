# 文件：ROV_RL_Control/deployment/export_onnx.py

import os
import argparse
import torch
import numpy as np
import yaml

from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.ddpg_agent import DDPGAgent


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def export_model(algo: str, checkpoint: str, config: dict, output_path: str):
    """
    根据算法类型加载对应 Agent，载入 checkpoint，然后导出 ONNX 文件到 output_path。
    """
    # 1) 填写 state_dim / action_dim
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    replay_size = config["hyperparameters"]["replay_buffer_size"]
    agent = None

    # 2) 实例化对应的 Agent 架构（与训练时一致）
    if algo == "sac":
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["hyperparameters"]
        )
    elif algo == "td3":
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["hyperparameters"]
        )
    elif algo == "ddpg":
        agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["hyperparameters"]
        )
    else:
        raise ValueError(f"Unsupported algorithm for export: {algo}")

    # 3) 加载权重
    agent.load(checkpoint)

    # 4) 创建一个假的输入（batch=1），与 env 里 _get_observation() 输出维度一致
    dummy_input = torch.zeros((1, state_dim), dtype=torch.float32)

    # 5) 导出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if algo == "sac":
        # SAC 的 forward 可能需要从 actor 中拿 mu / log_std，直接选 mu 作为网络输出
        torch.onnx.export(
            agent.actor,                 # 导出 actor 网络
            dummy_input,                 # 输入
            output_path,                 # 输出 ONNX 路径
            export_params=True,
            opset_version=13,
            input_names=["obs"],
            output_names=["z"],          # 我们假设 actor.forward 返回 (mu, log_std)，这里只输出 mu
            dynamic_axes={
                "obs": {0: "batch"},
                "z": {0: "batch"}
            }
        )
    else:
        # 对于 TD3 / DDPG，actor.forward 输出 mu, log_std（一样只用 mu）
        torch.onnx.export(
            agent.actor,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            input_names=["obs"],
            output_names=["z"],
            dynamic_axes={
                "obs": {0: "batch"},
                "z": {0: "batch"}
            }
        )

    print(f"Exported {algo} model to ONNX at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained PyTorch model to ONNX for deployment")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["sac", "td3", "ddpg"],
        required=True,
        help="Which algorithm to export"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="rov_rl_control/training/config.yaml",
        help="Path to the training config.yaml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rov_rl_control/deployment/models/model.onnx",
        help="Where to save the exported ONNX file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 自动把 state_dim、action_dim 加入 cfg（因为存储在训练环境里）
    # 在训练脚本里请确保保存了 env.observation_space.shape[0] / action_space.shape[0]
    cfg["state_dim"] = args.__dict__.get("state_dim", cfg.get("state_dim"))
    cfg["action_dim"] = args.__dict__.get("action_dim", cfg.get("action_dim"))
    # 如果 cfg 里没有 state_dim、action_dim，需要手动填写。示例：
    if "state_dim" not in cfg or "action_dim" not in cfg:
        # 假设 obs_dim=25, action_dim=8
        cfg["state_dim"] = cfg.get("state_dim", 25)
        cfg["action_dim"] = cfg.get("action_dim", 8)

    export_model(args.algo, args.checkpoint, cfg, args.output)
