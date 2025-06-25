# ROV_RL_Control

本项目基于强化学习（Reinforcement Learning, RL）技术，实现水下机器人（ROV）的闭环运动控制。目标是通过训练智能体在模拟环境中学习，使 ROV 能够在复杂的水下环境中完成姿态保持、轨迹跟踪等任务，并能够部署到嵌入式设备上进行真实试验。

---

## 项目目录结构

```text
rov_rl_control/
├─ .gitignore                        # Git 忽略文件配置
├─ README.md                         # 项目说明文档
├─ requirements-train.txt           # Python训练依赖包列表

├─ env/                              # 训练环境封装
│   ├─ rov_env.py                    # 核心Gym环境接口
│   ├─ utils_env.py                  # 状态构建与reward工具
│   └─ README.md

├─ agents/                           # 强化学习智能体模块
│   ├─ sac_agent.py                  # Soft Actor-Critic
│   ├─ td3_agent.py                  # Twin Delayed DDPG
│   ├─ ddpg_agent.py                 # Deep Deterministic Policy Gradient
│   ├─ replay_buffer.py              # 经验回放缓存
│   ├─ network.py                    # 策略/价值网络结构定义
│   └─ README.md

├─ training/                         # 策略训练逻辑
│   ├─ train.py                      # 主训练脚本
│   ├─ config.yaml                   # 训练参数配置文件
│   ├─ logger.py                     # 日志记录工具
│   └─ README.md

├─ evaluation/                       # 策略评估与复现
│   ├─ evaluate.py                   # 模型评估脚本
│   ├─ replay.py                     # 策略轨迹回放
│   └─ README.md

├─ sim/                              # ROV仿真模型
│   ├─ hydro_dynamics.py             # 水动力六自由度动力学模型
│   ├─ thruster_allocation.py        # 推进器分配逻辑
│   ├─ parameters.py                 # 物理/惯性/流体参数
│   └─ README.md

└─ deployment/                       # 部署模块
    ├─ export_onnx.py                # 导出策略为ONNX模型
    ├─ pi_runner.cpp                 # C++部署于树莓派或嵌入式设备
    └─ README.md
                
