# ROV_RL_Control

本项目基于强化学习（Reinforcement Learning, RL）技术，实现水下机器人（ROV）的闭环运动控制。目标是通过训练智能体在模拟环境中学习，使 ROV 能够在复杂的水下环境中完成姿态保持、轨迹跟踪等任务，并能够部署到嵌入式设备上进行真实试验。

---

## 项目目录结构

```text
rov_rl_control/
├─ .gitignore                       
├─ README.md                        
├─ requirements-train.txt           

├─ env/                             
│   ├─ __init__.py                  
│   ├─ rov_env.py                   
│   ├─ utils_env.py                 
│   └─ README.md                    

├─ agents/                          
│   ├─ __init__.py                  
│   ├─ sac_agent.py                 
│   ├─ td3_agent.py                 
│   ├─ ddpg_agent.py                
│   ├─ replay_buffer.py             
│   ├─ network.py                   
│   └─ README.md                    

├─ training/                        
│   ├─ train.py                     
│   ├─ config.yaml                  
│   ├─ logger.py                    
│   └─ README.md                    

├─ evaluation/                      
│   ├─ evaluate.py                  
│   ├─ replay.py                    
│   └─ README.md                    

├─ sim/                             
│   ├─ hydro_dynamics.py            
│   ├─ thruster_allocation.py       
│   ├─ parameters.py                
│   └─ README.md                    

└─ deployment/                      
    ├─ export_onnx.py               
    ├─ pi_runner.cpp                
    └─ README.md                    
