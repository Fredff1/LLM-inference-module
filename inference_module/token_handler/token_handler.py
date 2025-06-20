# inference/token_handler.py

from abc import ABC, abstractmethod
from inference_module.inference.base import BaseInference

def handle_missing_tokens(inference_obj:BaseInference):
    """
    根据 inference_obj.model_type 选择合适的处理策略，调用对应的 handle 方法。
    要求 inference_obj 中已存在以下属性：model_type, model_name, chat_type,
    model_path, tokenizer, model（可选，对于 classical 模式），sampling_params, logger
    """
    handler_map = {
        "llama3": LlamaTokenHandler(),
        "llama": LlamaTokenHandler(),
        "qwen2.5": QwenTokenHandler(),
        "qwen2": QwenTokenHandler(),
        "qwen3": QwenTokenHandler(),
        "qwen":QwenTokenHandler()
    }
    
    if inference_obj.config["model_type"] == "auto":
        if "llama" in inference_obj.model_name.lower():
            inference_obj.config["model_type"] = "llama3"
        elif "qwen" in inference_obj.model_name.lower():
            inference_obj.config["model_type"] = "qwen2.5"
        else:
            raise ValueError("Invalid model_type")
    
    handler = handler_map.get(inference_obj.config["model_type"])
    if handler:
        handler.handle(inference_obj)
    else:
        inference_obj.logger.warning("No token handler available for model type: " + inference_obj.model_type)

class BaseTokenHandler(ABC):
    @abstractmethod
    def handle(self, inference_obj):
        """
        根据推理器对象，对其 tokenizer、model.generation_config 和采样参数做相应的设置。
        :param inference_obj: 推理器实例，要求包含以下属性：
                              - chat_type
                              - model_name
                              - model_path
                              - tokenizer
                              - model (对于 classical 模式)
                              - sampling_params
                              - logger
        """
        pass

class LlamaTokenHandler(BaseTokenHandler):
    def handle(self, inference_obj):
        try:
            if inference_obj.chat_type == "vllm":
                inference_obj.tokenizer.pad_token = '<|end_of_text|>'
                inference_obj.tokenizer.add_bos_token = False
                inference_obj.tokenizer.add_eos_token = False
                stop_token_ids = [128001, 128008, 128009]
                inference_obj.sampling_params.stop_token_ids = stop_token_ids
            else:  
                from transformers import GenerationConfig
                inference_obj.model.generation_config = GenerationConfig.from_pretrained(
                    inference_obj.full_path, pad_token_id=inference_obj.tokenizer.pad_token_id)
                inference_obj.model.generation_config.eos_token_id = [128001, 128008, 128009]
                inference_obj.model.generation_config.pad_token_id = 128001
                inference_obj.tokenizer.pad_token = '<|end_of_text|>'
            inference_obj.logger.info("Added missing special tokens for " + inference_obj.model_name)
        except Exception as e:
            inference_obj.logger.warning("Failed to set special tokens for " + inference_obj.model_name)
            import traceback
            traceback.print_exc()

class QwenTokenHandler(BaseTokenHandler):
    def handle(self, inference_obj):
        try:
            if inference_obj.chat_type == "vllm":
                inference_obj.tokenizer.pad_token = '<|endoftext|>'
                inference_obj.tokenizer.pad_token_id = 151643
                stop_token_ids = [151645, 151643]
                inference_obj.sampling_params.stop_token_ids = stop_token_ids
            else:
                from transformers import GenerationConfig
                inference_obj.model.generation_config = GenerationConfig.from_pretrained(
                    inference_obj.full_path, pad_token_id=inference_obj.tokenizer.pad_token_id)
                inference_obj.model.generation_config.eos_token_id = [151645, 151643]
                inference_obj.model.generation_config.pad_token_id = 151643
                inference_obj.tokenizer.pad_token = '<|endoftext|>'
            inference_obj.logger.info("Added missing special tokens for " + inference_obj.model_name)
        except Exception as e:
            inference_obj.logger.warning("Failed to set special tokens for " + inference_obj.model_name)
            import traceback
            traceback.print_exc()
