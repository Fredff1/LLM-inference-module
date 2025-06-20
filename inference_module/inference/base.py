# inference/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List,Union
import os

from inference_module.gpu_manager.gpu_manager import GPUManager
from inference_module.utils.log_utils import init_logging

class BaseInference(ABC):
    def __init__(self, config: Dict[str, Any], gpu_manager:GPUManager,logger):
        """
        初始化推理器。
        :param config: 配置字典（通常来自 FullConfig.to_dict()）
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.gpu_manager = gpu_manager
        self.logger = logger
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_type = config.get("model_type","auto")
        self.chat_type = config.get("chat_type")
        self.full_path = os.path.join(self.model_path,self.model_name)
        
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """
        根据配置初始化模型、分词器及其它所需资源
        """
        pass

    @abstractmethod
    def run(self, input_content) -> Union[str,Any]:
        """
        单个输入的推理生成接口
        :param input_text: 单条输入文本
        :return: 生成文本结果
        """
        pass

    @abstractmethod
    def run_batch(self, input_contents: List) -> Union[List[str],List[Any]]:
        """
        批量输入的推理生成接口
        :param input_texts: 包含多条输入文本的列表
        :return: 每条输入对应的生成文本结果的列表
        """
        pass
    
    @abstractmethod
    def validate_input(
        self,
        input: Union[str, List[str], List[Dict[str, Any]]]
    ) -> Union[str, List[str], List[Dict[str, Any]]]:
        """
        校验并归一化输入，合法时返回子类 run/run_batch 能直接接受的格式，
        否则抛 ValueError。
        """
        pass
