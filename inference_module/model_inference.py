# model_inference/model_inference.py

from typing import List
from inference_module.config.inference_config import *
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
       
        
        self.gpu_manager = gpu_manager
        
        self.full_config = full_config.to_dict()

        self.logger = init_logging(task_name=task_name,log_dir=self.full_config["log_dir"])
        
        self.logger.info("Initializing ModelInference with the following configuration:")
        log_config(self.logger, self.full_config)
        
        self.inference_engine = create_inference_engine(self.full_config,self.gpu_manager,self.logger)
        
        self.logger.info("ModelInference instance successfully initialized.")
    
    
        
    def infer(self,input_content:Union[str,list[str],List[Any]])->Union[str,List[str],Any,List[Any]]:
        """推理入口

        Args:
            input_content (Union[str,list[str],List]): 用于推理的内容，不同的推理类型支持不同的输入内容
            - 支持单个输入或批次输入，并会自动返回单个结果或结果列表
            - classical 支持字符串或字符串列表
            - vllm 支持字符串或字符串列表
            - api 支持单个包含角色信息的messages或其列表
            - mock 支持字符串或字符串列表
            
        Raises:
            ValueError: 当不支持推理输入的内容时抛出

        Returns:
            Union[str,List[str],Any,List[Any]]: 推理结果，不同的推理类型返回不同的推理结果
            - 根据是否是batch输入决定是否返回列表形式结果
            - classical 返回字符串或字符串列表
            - vllm 返回字符串或字符串列表
            - api 根据api调用类型返回不同类型的结果
            - mock 返回mock的字符串或字符串列表
        """
        proc,content_type = self.inference_engine.validate_input(input_content)

        if content_type=="list":
            return self.inference_engine.run_batch(proc)
        elif content_type=="single":
            return self.inference_engine.run(proc)
        else:
            raise ValueError("Invalid input content")


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

    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,**additional_params):
        tokenizer = self.inference_engine.tokenizer
        template_available = self.inference_engine.config.get("apply_chat_template")
        if template_available:
            text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=tokenize,
                        add_generation_prompt=add_generation_prompt,
                        **additional_params
                    )
            return text
        else:
            raise ValueError("Tokenizer does not support chat template")
    
    
    
    @staticmethod
    def format_message(user_prompt: Union[str, List[str]],
                   sys_prompt: str = None,
                   assist_prompt: str = None) -> List[Dict[str, Any]]:
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
        
        if isinstance(user_prompt, list):
            for prompt in user_prompt:
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        if assist_prompt:
            messages.append({"role": "assistant", "content": assist_prompt})
        
        return messages
    
    @staticmethod
    def from_mock(model_name="mock",task_name:str=None, log_dir:str=None) -> "ModelInference":
        """快速构造mock模式的推理实例

        Args:
            model_name (str, optional): 模型名称. Defaults to "mock".
            task_name (str, optional): 任务名称. Defaults to None.
            log_dir (str, optional): 日志路径. Defaults to None.

        Returns:
            ModelInference:推理实例
        """
        config = InferenceConfig(model_name = model_name,chat_type="mock",log_dir=log_dir)
        model_infer = ModelInference(config,task_name,None)
        return model_infer
    
    @staticmethod
    def from_api(
        model_name:str,
        api_key:str,
        url:str,
        max_retries :int = 3,
        backoff_factor :float = 1.0,
        max_concurrent_requests: int = 4,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        task_name:str = None,
        log_dir:str = None,
    ) -> "ModelInference":
        """快速构造api模式的推理实例

        Args:
            model_name (str): 模型名称
            api_key (str): api密钥
            url (str): api的url
            max_retries (int, optional): 最大重试次数. Defaults to 3.
            backoff_factor (float, optional): 抖动因子. Defaults to 1.0.
            max_concurrent_requests (int, optional): 最大并行请求数. Defaults to 4.
            max_new_tokens (int, optional): 最大新token. Defaults to 1024.
            temperature (float, optional): 温度. Defaults to 1.0.
            task_name (str, optional): 任务名称. Defaults to None.
            log_dir (str, optional): 日志路径. Defaults to None.

        Returns:
            ModelInference:推理实例
        """
        config = InferenceConfig(
            model_name=model_name,
            chat_type="api",
            generation_params=GenerationParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature
            ),
            api_config=ApiConfig(
                api_key=api_key,
                url=url,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                max_concurrent_requests=max_concurrent_requests
            ),
            log_dir=log_dir
        )
        model_infer = ModelInference(
            full_config=config,
            task_name=task_name,
        )
        return model_infer
    
    @staticmethod
    def from_classical(
        model_name: str,                    
        model_path: str,                       
        model_type:str = 'auto',
        max_new_tokens: int = 1024,         
        temperature: float = 1.0,     
        apply_chat_template = False,  
        log_dir:str = None,   
        task_name:str = None,
        top_p: float = 1.0,               
        top_k: int = 50,                   
        do_sample: bool = True,            
        num_beams: int = 1,                
        length_penalty: float = 1.0,       
        use_cache: bool = True,            
        repetition_penalty: float = 1.0,
    )-> "ModelInference":
        """快速构造经典模式的推理实例

        Args:
            model_name (str): 模型目录名称
            model_path (str): 到模型的路径，会与model_name拼接得到完整路径
            model_type (str, optional): 模型类型. Defaults to 'auto'.
            max_new_tokens (int, optional): 最大新token. Defaults to 1024.
            temperature (float, optional): 温度. Defaults to 1.0.
            apply_chat_template (bool, optional): 是否支持chat_template. Defaults to False.
            log_dir (str, optional): 日志路径. Defaults to None.
            task_name (str, optional): 任务名称. Defaults to None.
            top_p (float, optional): top_p参数. Defaults to 1.0.
            top_k (int, optional): top_k参数. Defaults to 50.
            do_sample (bool, optional): 采样启用参数. Defaults to True.
            num_beams (int, optional): num_beams参数. Defaults to 1.
            length_penalty (float, optional): 长度惩罚. Defaults to 1.0.
            use_cache (bool, optional): 启用缓存. Defaults to True.
            repetition_penalty (float, optional): 重复惩罚. Defaults to 1.0.

        Returns:
            ModelInference: 推理实例
        """
        config = InferenceConfig(
            model_name=model_name,
            model_path=model_path,
            chat_type="classical",
            model_type=model_type,
            apply_chat_template=apply_chat_template,
            generation_params=GenerationParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_beams=num_beams,
                length_penalty=length_penalty,
                use_cache=use_cache,
                repetition_penalty=repetition_penalty
            ),
            log_dir=log_dir
        )
        model_infer = ModelInference(
            full_config=config,
            task_name=task_name,
            gpu_manager=None
        )
        return model_infer
    
    @staticmethod
    def from_vllm(
        model_name: str,                    
        model_path: str,                       
        model_type:str = 'auto',
        max_tokens: int = 1024,                
        temperature: float = 1.0,
        apply_chat_template = False,    
        tensor_parallel_size: Union[int, str] = 1,  
        gpu_memory_utilization: float = 0.8,
        max_num_seqs: int = 16,     
        log_dir:str = None,   
        task_name:str = None,                   
        top_p: float = 1.0,                   
        top_k: int = 50,                       
        repetition_penalty: float = 1.0,       
    )-> "ModelInference":
        """快速构造vllm的推理实例

        Args:
            model_name (str): 模型目录名称
            model_path (str): 到模型的路径，会与model_name拼接得到完整路径
            model_type (str, optional): 模型类型. Defaults to 'auto'.
            max_tokens (int, optional): 最大token数. Defaults to 1024.
            temperature (float, optional): 温度. Defaults to 1.0.
            apply_chat_template (bool, optional): 是否支持推理模板. Defaults to False.
            tensor_parallel_size (Union[int, str], optional): 张量并行度. Defaults to 1.
            gpu_memory_utilization (float, optional): gpu显存使用率. Defaults to 0.8.
            max_num_seqs (int, optional): 最大seq数. Defaults to 16.
            log_dir (str, optional): 日志路径. Defaults to None.
            task_name (str, optional): 任务名称. Defaults to None.
            top_p (float, optional): top_p参数. Defaults to 1.0.
            top_k (int, optional): top_k参数. Defaults to 50.
            repetition_penalty (float, optional): 重复惩罚. Defaults to 1.0.

        Returns:
            ModelInference: 推理实例
        """
        config = InferenceConfig(
            model_name=model_name,
            model_path=model_path,
            chat_type="vllm",
            model_type=model_type,
            apply_chat_template=apply_chat_template,
            vllm_params=VLLMParams(
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=max_num_seqs
            ),
            sampling_params=VllmSamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            ),
            log_dir=log_dir
        )
        model_infer = ModelInference(
            full_config=config,
            task_name=task_name,
            gpu_manager=None
        )
        return model_infer
    
    
    
            
