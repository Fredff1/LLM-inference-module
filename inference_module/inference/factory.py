from inference_module.inference.base import BaseInference
from inference_module.gpu_manager.gpu_manager import GPUManager

def create_inference_engine(config:dict,gpu_manager:GPUManager,logger) -> BaseInference:
    type = config.get("chat_type")
    if type == "classical":
        from inference_module.inference.classical import ClassicalInference
        return ClassicalInference(config,gpu_manager,logger)
    elif type == "api":
        from inference_module.inference.api import APIInference
        return APIInference(config,gpu_manager,logger)
    elif type == "vllm":
        from inference_module.inference.vllm import VLLMInference
        return VLLMInference(config,gpu_manager,logger)
    elif type == "mock":
        from inference_module.inference.mock import MockInference
        return MockInference(config,gpu_manager,logger)
    else:
        raise NotImplementedError("Chat type not supported")
        