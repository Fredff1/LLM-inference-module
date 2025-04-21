# model_inference/model_inference.py

from typing import List
from inference_module.config.inference_config import InferenceConfig
from inference_module.inference.factory import create_inference_engine
from inference_module.utils.log_utils import init_logging, log_config
from inference_module.utils.message_utils import merge_messages
from inference_module.gpu_manager.gpu_manager import GPUManager

from typing import Union,Dict,Any

class ModelInference:
    """
    顶层推理封装类：
      - 通过传入统一配置（FullConfig）构造内部推理器。
      - 对外提供 run_inference（单条推理）和 run_group_inference（批量推理）接口，
        使用户无需关心底层具体推理方式差异。
    """
    def __init__(self, full_config: InferenceConfig, task_name: str = None,gpu_manager:GPUManager = None):
        """
        构造推理实例，并基于 full_config 创建具体的推理器实例。
        
        参数：
          full_config (FullConfig): 全局配置对象，包含模型生成、分词器、设备、采样及推理模式相关参数。
          task_name (str): 任务名称，用于日志记录和日志文件命名（可选）。
        """
        # 初始化日志
        self.logger = init_logging(task_name=task_name)
        
        self.gpu_manager = gpu_manager
        
        # 保存配置并记录完整配置到日志中
        self.full_config = full_config
        config_dict = full_config.to_dict()
        self.logger.info("Initializing ModelInference with the following configuration:")
        log_config(self.logger, config_dict)
        
        # 根据配置生成具体推理器实例，通过工厂方法统一创建，如 ClassicalInference、APIInference、VLLMInference 等
        self.inference_engine = create_inference_engine(config_dict,self.gpu_manager,self.logger)
        
        self.logger.info("ModelInference instance successfully initialized.")
    
    
        
    def infer(self,input_content:Union[str,list[str],List])->str:
        apply_chat_template = self.full_config["apply_chat_template"]
        
        
        if isinstance(input_content,str):
            if apply_chat_template:
                input_content = self.apply_chat_template(input_content)
            return self._run_inference(input_content)
        elif isinstance(input_content,list):
            if len(input_content) <1:
                raise ValueError("Input list cannot be empty")
            content_example = input_content[0]
            if isinstance(content_example,str):
                pass
            else:
                if apply_chat_template:
                    for content in input_content:
                        content = self.apply_chat_template(content)
            return self._run_group_inference(input_content)
        else:
            raise ValueError("Invalid input format.Only support str or list[str]")


    def _run_inference(self, input_text: str) -> str:
        """
        单条推理接口，接受一个字符串作为输入。
        说明：
          - 对于基础模型推理，可根据配置自动决定是否应用 chat template；
          - 对于 API 模式，则内部会转换为消息格式（由对应的 APIInference 实现）。
        
        参数：
          input_text (str): 输入提示文本。
        
        返回：
          str: 模型生成的文本结果。
        """
        result = self.inference_engine.run(input_text)
        return result

    def _run_group_inference(self, input_texts: List[str]) -> List[str]:
        """
        批量推理接口，接受一个字符串列表作为输入，每个字符串对应一条提示文本。
        
        参数：
          input_texts (List[str]): 多条输入提示文本的列表。
        
        返回：
          List[str]: 每条输入对应的生成文本结果列表。
        """
        results = self.inference_engine.run_batch(input_texts)
        return results
    
    def format_message(user_prompt: Union[str, List[str]],
                   sys_prompt: str = "",
                   assist_prompt: str = "") -> List[Dict[str, Any]]:
        """
        构造用于推理的消息列表，每条消息包含 role 和 content 字段。
        
        参数:
        - user_prompt: 用户输入的提示文本，可为 str 或字符串列表（若需批量处理）。
        - sys_prompt: 系统提示文本（可选）。
        - assist_prompt: 助手提示文本（可选）。
        
        返回:
        - List[Dict[str, Any]]: 格式化后的消息列表，至少包含 user 消息。
        
        注意:
        1. 若 user_prompt 为空则抛出异常。
        2. 不支持同时传入多条 user_prompt 时与单条 sys_prompt/assist_prompt 混合使用，
            调用者需确保输入格式一致性。
        """
        messages = []
        
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        if not user_prompt:
            raise ValueError("不合法的用户输入，你必须指定一个有效的用户输入作为 prompt")
        
        # 如果 user_prompt 为列表，则对每个文本构造单独的消息
        if isinstance(user_prompt, list):
            for prompt in user_prompt:
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        if assist_prompt:
            messages.append({"role": "assistant", "content": assist_prompt})
        
        return messages
    
    
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        tokenizer = self.inference_engine.tokenizer
        template_available = self.inference_engine.config.get("apply_chat_template")
        if template_available:
            text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=tokenize,
                        add_generation_prompt=add_generation_prompt
                    )
            return text
        else:
            raise ValueError("Tokenizer does not support chat template")
            
