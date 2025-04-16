from .base import BaseInference
from .api import APIInference
from .classical import ClassicalInference
from .vllm import VLLMInference

from gpu_manager.gpu_manager import GPUManager

def create_inference_engine(config:dict,gpu_manager:GPUManager,logger) -> BaseInference:
    type = config.get("chat_type")
    if type == "classical":
        return ClassicalInference(config,gpu_manager,logger)
    elif type == "api":
        return APIInference(config,gpu_manager,logger)
    elif type == "vllm":
        return VLLMInference(config,gpu_manager,logger)
    else:
        raise NotImplementedError("Chat type not supported")
        