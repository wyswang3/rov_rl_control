#!/usr/bin/env python3
# utils/log_helper.py

"""
日志系统配置助手

提供 setup_logging() 函数，用于初始化根日志记录器，
默认输出带时间戳和日志级别的控制台日志。
"""

import logging
from typing import Optional

__all__ = ["setup_logging"]

def setup_logging(level: int = logging.INFO,
                  fmt: Optional[str] = None) -> None:
    """
    配置根日志记录器。

    参数:
      - level: 日志级别，例如 logging.INFO
      - fmt:   日志格式字符串，默认 '[%(asctime)s] %(levelname)s: %(message)s'
    """
    if fmt is None:
        fmt = "[%(asctime)s] %(levelname)s: %(message)s"

    logger = logging.getLogger()
    logger.setLevel(level)

    # 如果还没有 handler，就添加一个 StreamHandler
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
