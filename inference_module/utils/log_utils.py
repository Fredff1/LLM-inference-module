# utils/logger.py
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

def init_logging(task_name: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    初始化日志系统并返回一个 Logger 对象。日志同时输出到控制台和文件。
    
    参数:
      - task_name: 日志文件名中使用的任务名，若为空则用当前时间生成文件名。
      - log_dir: 日志存放目录，若为空，则在当前模块目录下创建一个 "log" 文件夹。
      - level: 日志级别，默认为 logging.INFO。
    
    返回:
      - logging.Logger 对象，可用于后续的日志记录。
    """
    # 确定日志目录
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if log_dir is None:
        log_dir = os.path.join(module_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 根据任务名或时间生成日志文件名
    if task_name is None:
        log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    else:
        log_file_name = f"{task_name}.log"
    log_file_path = os.path.join(log_dir, log_file_name)
    
    # 配置日志格式、日期格式，以及输出通道
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode="a")
        ]
    )
    # 使用 task_name 作 logger 名称，若 task_name 为空则使用 __name__
    logger = logging.getLogger(task_name if task_name else __name__)
    logger.info("日志系统已初始化")
    logger.info(f"日志文件路径: {log_file_path}")
    return logger

def log_config(logger: logging.Logger, config: Dict[str, Any], parent_key: str = "") -> None:
    """
    递归记录配置字典内容到 logger 中，支持多层嵌套字典。
    
    参数:
      - logger: 用于记录日志的 logger 对象。
      - config: 要记录的配置字典。
      - parent_key: 用于标记当前记录键的前缀，通常由内部递归调用设置。
    """
    for key, value in config.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            logger.info(f"{current_key}:")
            log_config(logger, value, current_key)
        else:
            logger.info(f"{current_key}: {value}")
