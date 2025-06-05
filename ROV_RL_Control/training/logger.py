# 文件：rov_rl_control/training/logger.py

import os
import time
from typing import Optional, Dict

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    简易日志工具，基于 TensorBoard SummaryWriter 记录训练与评估指标。

    用法示例：
        logger = Logger(log_dir="path/to/logs")
        logger.log_scalar("Train/EpisodeReward", reward, episode)
        logger.log_hyperparams(config_dict, metrics_dict, run_name="experiment1")
        ...
        logger.close()
    """

    def __init__(self, log_dir: str, run_name: Optional[str] = None) -> None:
        """
        Args:
            log_dir (str): 日志根目录，实际会在此目录下创建子目录
            run_name (Optional[str]): 可选的本次训练名称（作为子目录）。如果为 None，则自动生成带时间戳的目录名。
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if run_name is None:
            run_name = f"run_{timestamp}"
        self.log_path = os.path.join(log_dir, run_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        记录标量指标。

        Args:
            tag (str): 指标名称（例如 "Train/EpisodeReward"、"Eval/AverageReward"）
            value (float): 指标值
            step (int): 全局步数或 Episode 数，用于 x 轴
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_prefix: str, scalar_dict: Dict[str, float], step: int) -> None:
        """
        一次性记录多个标量，自动在 tag 前加 prefix。

        Args:
            tag_prefix (str): 前缀（例如 "Train" 或 "Eval"）
            scalar_dict (Dict[str, float]): 键值对映射的多个指标
            step (int): 全局步数或 Episode 数
        """
        for key, val in scalar_dict.items():
            self.writer.add_scalar(f"{tag_prefix}/{key}", val, step)

    def log_hyperparams(
        self,
        param_dict: Dict[str, any],
        metric_dict: Dict[str, float],
        run_name: Optional[str] = None
    ) -> None:
        """
        记录超参数与对应最终指标，方便在 TensorBoard 中的 HParams 标签页查看。

        Args:
            param_dict (Dict[str, any]): 超参数字典，例如 {"lr": 3e-4, "gamma": 0.99}
            metric_dict (Dict[str, float]): 最终指标字典，例如 {"AvgReturn": 15.2}
            run_name (Optional[str]): 本次运行名称，会显示在 TensorBoard 表格中
        """
        # TensorBoard HParams 接口需要先设置 run name，这里使用同一个 writer
        if run_name is not None:
            self.writer.add_hparams(param_dict, metric_dict, run_name=run_name)
        else:
            self.writer.add_hparams(param_dict, metric_dict)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        记录文本信息，可用于打印超参数或训练配置。

        Args:
            tag (str): 文本标签，例如 "Config/Hyperparameters"
            text (str): 文本内容
            step (int): 全局步数或 Episode 数
        """
        self.writer.add_text(tag, text, step)

    def close(self) -> None:
        """关闭 SummaryWriter，确保所有数据写入磁盘。"""
        self.writer.close()
