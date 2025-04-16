# modules/model_inference.py
from .gpu_manager import GPUManager
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizer,PreTrainedModel,pipeline,GenerationConfig
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import logging

import torch
import re
import math
import traceback
from openai import OpenAI
from ctypes import ArgumentError

try:
    from vllm import LLM, SamplingParams # type: ignore
    vllm_available=True
except ImportError:
    vllm_available = False
from datetime import datetime

import torch.distributed as dist
import os

try:
    import deepspeed # type: ignore
    deepspeed_available = True
except ImportError:
    deepspeed_available = False




class ModelActivity:
    """
    
    ## 类作用
        在只进行模型输出的时候管理模型和分词器,调用模型生成回答
        不支持训练阶段的模型
    ## 初始化参数
        model_name
    ## 使用方法
        (1)调用format_message函数生成输入信息,你需要提供三种角色的prompt(除了user prompt都可以为空),可以通过validate_messages方法检测输入是否合法
        (2)调用run_inference方法生成回复,或者调用run_group_inference方法生成一组回复

    """
    default_model_path="/home/workspace/Models/"
    
    default_deep_speed_config=default_deep_speed_config = {

    }
    
    @staticmethod
    def format_message(user_prompt:str|list[str],sys_prompt="",assist_prompt="")->list[str]:
        """
        ## 函数功能
        这个函数可以为你格式化一个用于inference的messages
        ## 参数
        三个参数分别对应system user assistant的role种的content
        ## 返回值
        一个符合模型输入要求的messages，必定包含user角色和content，可能有system和assistant角色的content
        如果你需要进行group回答，请多次调用这个函数
        """
        messages=[]
        if sys_prompt=="" :
            pass
        else:
            system_message={"role": "system", "content": sys_prompt} 
            messages.append(system_message)
        
        if user_prompt=="" or not user_prompt:
            raise(ArgumentError("不合法的用户输入，你必须指定一个有效的用户输入作为prompt"))
        user_message={"role": "user", "content":user_prompt}
        messages.append(user_message) 
        
        if assist_prompt!="":
            assistant_message={"role":"assistant","content":assist_prompt}
            messages.append(assistant_message)
        return messages

    @staticmethod
    def validate_messages(messages:list):
        """
        ## 函数功能
        验证messages中的每个消息，确保其角色只能是'system'、'user' 或 'assistant'。
        如果有其他角色，抛出异常。
        ## 返回值
        一个bool值或抛出异常
        """
        if len(messages)<1:
            raise ValueError(f"输入消息为空")
        valid_roles = {"system", "user", "assistant"}  # 定义有效角色集合
        for message in messages:
            role = message.get("role", "")
            if role not in valid_roles:
                raise ValueError(f"非法的角色: {role}，仅允许'system', 'user' 或 'assistant'角色。")
        return True  # 如果通过验证，返回True
    
    
    
    @staticmethod
    def format_model_generation_params(
        max_new_tokens: int = 256,               # 与 Hugging Face 的默认 max_new_tokens 对齐
        temperature: float = 1.0,               # 默认值为 1.0，代表不改变生成的多样性
        top_p: float = 1.0,                     # 默认值为 1.0，意味着不裁剪 token 的概率
        top_k: int = 50,                        # 默认值为 50
        do_sample: bool = True,                 # 默认值为 True,防止警告
        num_beams: int = 1,                     # 默认值为 1，表示不使用 beam search
        length_penalty: float = 1.0,            # 默认值为 1.0，表示不惩罚生成的长度
        use_cache: bool = True,                 # 默认值为 True，启用缓存来加速生成
        repetition_penalty: float = 1.0,        # 默认值为 1.0，表示没有重复惩罚
        **additional_params                     # 其他可选参数
    ) -> dict:
        """
        格式化生成参数，以支持 Hugging Face 模型的所有常见参数。
        参数默认值与huggingface的模型相同，你可以自行修改
        
        参数:
            - max_new_tokens: 最大新生成的 token 数量。
            - temperature: 控制生成的多样性，默认 1.0。
            - top_p: Nucleus sampling 的累计概率阈值，默认 1.0。
            - top_k: 限制前 k 个 token 的选择范围，默认 50。
            - do_sample: 是否进行随机采样，默认 False。
            - num_beams: 用于 beam search 的 beam 数量，默认 1。
            - length_penalty: 控制生成长度的惩罚，默认 1.0。
            - use_cache: 启用缓存，默认 True。
            - repetition_penalty: 控制重复生成的惩罚，默认 1.0。
            - additional_params: 其他可选的 Hugging Face 参数。
        
        返回:
            返回包含所有设置的生成参数字典。
        """
        logger = logging.getLogger("ModelActivity.format_model_generation_params")
        logger.debug("Set model generation parameters")
        # 基础生成参数，与 Hugging Face 的默认值对齐
        generation_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "use_cache": use_cache,
            "repetition_penalty": repetition_penalty
        }
        
        # 添加用户提供的额外参数，以支持 Hugging Face 的其他参数
        generation_params.update(additional_params)
        
        return generation_params
    
    
    @staticmethod
    def format_tokenizer_params(
        padding: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        **additional_params
    ) -> dict:
        """
        格式化分词器参数，支持 Hugging Face 分词器的所有常见参数。
        
        参数:
            - padding: 是否填充序列，默认 False。
            - add_special_tokens: 是否添加特殊 token，默认 True。
            - return_tensors: 返回的张量格式，默认 "pt" (PyTorch)。
            - additional_params: 其他可选的分词器参数。
        
        返回:
            返回包含所有设置的分词器参数字典。
        """
        tokenizer_params = {
            "padding": padding,
            "add_special_tokens": add_special_tokens,
            "return_tensors": return_tensors
        }
        
        tokenizer_params.update(additional_params)
        
        return tokenizer_params
    
    
    @staticmethod
    def format_model_init_args(
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        cache_dir: str = None,
        **additional_params
    ) -> dict:
        """
        格式化模型初始化参数，支持 Hugging Face 模型的所有常见初始化参数。

        参数:
            - trust_remote_code: 是否信任远程代码，默认 True。
            - torch_dtype: 模型的浮点类型，默认 "auto"。
            - cache_dir: 模型的缓存目录，默认 None。
            - additional_params: 其他 Hugging Face 支持的模型初始化参数。

        返回:
            包含所有设置的模型初始化参数字典。
        """
        model_init_args = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "cache_dir": cache_dir
        }
        
        # 添加用户提供的其他参数
        model_init_args.update(additional_params)
        
        return model_init_args
    
    @staticmethod
    def format_tokenizer_init_args(
        padding_side: str = "left",
        trust_remote_code: bool = True,
        cache_dir: str = None,
        **additional_params
    ) -> dict:
        """
        格式化分词器初始化参数，支持 Hugging Face 分词器的所有常见初始化参数。

        参数:
            - padding_side: 填充方向 ("left" 或 "right")，默认 "left"。
            - trust_remote_code: 是否信任远程代码，默认 True。
            - cache_dir: 分词器的缓存目录，默认 None。
            - additional_params: 其他 Hugging Face 支持的分词器初始化参数。

        返回:
            包含所有设置的分词器初始化参数字典。
        """
        tokenizer_init_args = {
            "padding_side": padding_side,
            "trust_remote_code": trust_remote_code,
            "cache_dir": cache_dir
        }
        
        # 添加用户提供的其他参数
        tokenizer_init_args.update(additional_params)
        
        return tokenizer_init_args
    
    @staticmethod
    def format_sampling_params(
        max_tokens: int = 1024,
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        sampling_times=1,
        **kwargs
    ) -> dict:
        """
        格式化vllm采样参数。

        参数:
            - max_tokens: 最大生成 Token 数。
            - temperature: 采样温度。
            - top_p: Nucleus Sampling 截断概率。
            - top_k: Top-K 采样大小。
            - repetition_penalty: 重复惩罚。
            - presence_penalty: Presence 惩罚。
            - frequency_penalty: 频率惩罚。
            - sampling_times 采样次数
            - kwargs: 额外补充参数。
            

        返回:
            采样参数字典。
        """
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "n":sampling_times
        }
        params.update(kwargs)
        return params
    
    @staticmethod
    def format_vllm_args(
        tensor_parallel_size: int | str = "auto",
        gpu_memory_utilization: float = 0.5,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        enable_chunked_prefill:bool=False,
        max_num_seqs:int=8,
        **kwargs
    ) -> dict:
        """
        格式化 vLLM 加载参数。

        参数:
            - tensor_parallel_size: 张量并行大小（默认 "auto"）。
            - gpu_memory_utilization: GPU 内存利用率（默认 0.8）。
            - dtype: 数据类型（默认 "float16"）。
            - enforce_eager: 是否强制使用 Eager 模式（默认 True）。
            - kwargs: 其他参数。

        返回:
            vLLM 加载参数字典。
        """
        args = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "enable_chunked_prefill":enable_chunked_prefill,
            "max_num_seqs":max_num_seqs
        }
        args.update(kwargs)  # 合并额外参数
        return args


    
    

    @staticmethod
    def build_api_config(model_name: str,
                         api_key: str="default",
                         url: str="default",
                         model_generate_args: dict = "default", 
                         apply_chat_template: bool = True,
                         **kwargs) -> dict:
        """
        生成 OpenAI API 配置。

        参数:
            - api_key: OpenAI API 密钥。
            - model_name: 使用的模型名称。
            - url: API 请求的 URL。
            - apply_chat_template: 是否使用模型的chat_template
            - kwargs: 其他生成参数。
        
        返回:
            OpenAI API 配置字典。
        """
        
        if model_generate_args == "default":
            model_generate_args = ModelActivity.format_model_generation_params()
        config = {
            "model_name": model_name,
            "chat_type": "api",
            "model_generate_args": model_generate_args,
            "apply_chat_template":apply_chat_template
        }
        if api_key!="default":
            config["api_key"]=api_key
        if url!="default":
            config["url"]=url
        config.update(kwargs)
        return config

    @staticmethod
    def build_deepspeed_config(
        model_path: str,
        mp_size: int = 1,
        dtype: str = "float16",
        replace_method: str = "auto",
        model_init_args: dict = "default",
        tokenizer_init_args: dict = "default",
        model_generate_args: dict = "default",
        tokenizer_generate_args: dict = "default",
        model_type:str="auto",
        apply_chat_template: bool = True,
        **kwargs
    ) -> dict:
        """
        生成 DeepSpeed 模式配置。

        参数:
            - model_path: 模型路径。
            - mp_size: DeepSpeed 的模型并行大小（默认 1）。
            - dtype: 模型数据类型（默认 "float16"）。
            - replace_method: 模型替换方法（默认 "auto"）。
            - kwargs: 其他生成参数。

        返回:
            DeepSpeed 配置字典。
        """
        model_name = os.path.basename(model_path.rstrip("/\\"))
        model_parent_path = os.path.dirname(model_path.rstrip("/\\")) + "/"
        
        if model_init_args == "default":
            model_init_args = ModelActivity.format_model_init_args()
        if tokenizer_init_args == "default":
            tokenizer_init_args = ModelActivity.format_tokenizer_init_args()
        if model_generate_args == "default":
            model_generate_args = ModelActivity.format_model_generation_params()
        if tokenizer_generate_args == "default":
            tokenizer_generate_args = ModelActivity.format_tokenizer_params()

        
        config = {
            "model_path": model_parent_path,
            "model_name": model_name,
            "chat_type": "deepspeed",
            "deepspeed_args": {
                "mp_size": mp_size,
                "dtype": dtype,
                "replace_method": replace_method,
            },
            "model_init_args": model_init_args,
            "tokenizer_init_args": tokenizer_init_args,
            "model_generate_args": model_generate_args,
            "tokenizer_generate_args": tokenizer_generate_args,
            "model_type":model_type,
            "apply_chat_template":apply_chat_template
        }
        
        config.update(kwargs)
        return config

    @staticmethod
    def build_vllm_config(
        model_path: str,
        tokenizer_init_args: dict = "default",
        vllm_args: dict = "default",
        sampling_params: dict = "default",
        model_type:str="auto",
        apply_chat_template: bool = True,
        **kwargs
    ) -> dict:
        """
        生成 vLLM 模式配置。

        参数:
            - model_path: 模型路径。
            - model_init_args: 模型初始化参数（默认 "default"，使用 format_model_init_args 的值）。
            - tokenizer_init_args: 分词器初始化参数（默认 "default"，使用 format_tokenizer_init_args 的值）。
            - vllm_args: vLLM 加载参数（默认 "default"，使用 format_vllm_args 的值）。
            - sampling_params: 采样参数（默认 "default"，使用 format_sampling_params 的值）。
            - kwargs: 用于覆盖默认参数的额外补充参数。

        返回:
            vLLM 配置字典。
        """
        model_name = os.path.basename(model_path.rstrip("/\\"))
        model_parent_path = os.path.dirname(model_path.rstrip("/\\")) + "/"

        # 格式化各部分配置
      
        if tokenizer_init_args == "default":
            tokenizer_init_args = ModelActivity.format_tokenizer_init_args(**kwargs)
        if vllm_args == "default":
            vllm_args = ModelActivity.format_vllm_args(**kwargs)
        if sampling_params == "default":
            sampling_params = ModelActivity.format_sampling_params(**kwargs)

        # 整合最终配置
        config = {
            "model_path": model_parent_path,
            "model_name": model_name,
            "chat_type": "vllm",
            "vllm_args": vllm_args,
            "tokenizer_init_args": tokenizer_init_args,
            "sampling_params": sampling_params,
            "model_type":model_type,
            "apply_chat_template":apply_chat_template
        }
        
        return config
    


    
    
    @staticmethod
    def build_classical_config(
        model_path: str,
        chat_type="classical",
        model_init_args: dict = "default",
        tokenizer_init_args: dict = "default",
        model_generate_args: dict = "default",
        tokenizer_generate_args: dict = "default",
        model_type:str="auto",
        apply_chat_template: bool = True,
        **kwargs
    ) -> dict:
        """
        生成经典 hugging face 模型的配置模式

        参数:
            - model_path: 模型的完整路径。
            - chat_type: 方式，支持classical或pipeline
            - model_init_args: 模型初始化参数（默认使用 format_model_init_args 的默认值,你可以调用这个函数自行进行细化的调整）。
            - tokenizer_init_args: 分词器初始化参数（默认使用 format_tokenizer_init_args 的默认值，你可以调用这个函数自行进行细化的调整）。
            - model_generate_args: 模型生成参数（默认使用 format_model_generation_params 的默认值，你可以调用这个函数自行进行细化的调整）。
            - tokenizer_generate_args: 分词器生成参数（默认使用 format_tokenizer_params 的默认值，你可以调用这个函数自行进行细化的调整）。

        返回:
            包含所有配置的完整字典。
        """
        # 提取模型名称
        model_name = os.path.basename(model_path.rstrip("/\\"))
        model_parent_path = os.path.dirname(model_path.rstrip("/\\"))+"/"  # 去除末尾的斜杠并获取父路径4
        

        if model_init_args == "default":
            model_init_args = ModelActivity.format_model_init_args()
        if tokenizer_init_args == "default":
            tokenizer_init_args = ModelActivity.format_tokenizer_init_args()
        if model_generate_args == "default":
            model_generate_args = ModelActivity.format_model_generation_params()
        if tokenizer_generate_args == "default":
            tokenizer_generate_args = ModelActivity.format_tokenizer_params()

        if chat_type!="classical" and chat_type!="pipeline":
            raise ValueError("Invalid chat type for classical generation")
        
        # 整合所有配置
        full_config = {
            "model_path": model_parent_path,
            "model_name": model_name,
            "chat_type":chat_type,

            "model_init_args": model_init_args,
            "tokenizer_init_args": tokenizer_init_args,
            "model_generate_args": model_generate_args,
            "tokenizer_generate_args": tokenizer_generate_args,
            "model_type":model_type,
            "apply_chat_template":apply_chat_template,
        }
        full_config.update(kwargs)
        return full_config
    
    @staticmethod
    def build_full_config(
        model_path: str,
        model_name:str="default",
        chat_type="classical",
        model_init_args: dict = "default",
        tokenizer_init_args: dict = "default",
        model_generate_args: dict = "default",
        tokenizer_generate_args: dict = "default",
        vllm_args: dict = "default",
        sampling_params: dict = "default",
        model_type:str="auto",
        api_key: str="default",
        url: str="default",
        apply_chat_template: bool = True,
        **kwargs
    ):
        """
        配置完整的config,包括了所有可能的chat方式
        """
        # 提取模型名称
        if model_name=="default":
            model_name = os.path.basename(model_path.rstrip("/\\"))
        model_parent_path = os.path.dirname(model_path.rstrip("/\\"))+"/"  # 去除末尾的斜杠并获取父路径4
        

        if model_init_args == "default":
            model_init_args = ModelActivity.format_model_init_args()
        if tokenizer_init_args == "default":
            tokenizer_init_args = ModelActivity.format_tokenizer_init_args()
        if model_generate_args == "default":
            model_generate_args = ModelActivity.format_model_generation_params()
        if tokenizer_generate_args == "default":
            tokenizer_generate_args = ModelActivity.format_tokenizer_params()
        if vllm_args == "default":
            vllm_args = ModelActivity.format_vllm_args()
        if sampling_params == "default":
            sampling_params = ModelActivity.format_sampling_params()

        
        
        # 整合所有配置
        full_config = {
            "model_path": model_parent_path,
            "model_name": model_name,
            "chat_type":chat_type,
            "model_init_args": model_init_args,
            "tokenizer_init_args": tokenizer_init_args,
            "model_generate_args": model_generate_args,
            "tokenizer_generate_args": tokenizer_generate_args,
            "model_type":model_type,
            "vllm_args": vllm_args,
            "sampling_params": sampling_params,
            "apply_chat_template": apply_chat_template,
        }
        if api_key!="default":
            full_config["api_key"]=api_key
        if url!="default":
            full_config["url"]=url
        
        full_config.update(kwargs)
        return full_config
    
    
    def __init_deep_speed(self):
        """初始化deepspeed设置"""
        self.using_gpt=False
        self.using_deep_speed=True
        self.logger.info("正在从 "+self.model_path+"导入模型，请稍等")
        try:
            self.ds_config=self.config["deep_speed"]
        except :
            self.logger.info("未找到deep speed配置文件，使用默认配置")
        self.ds_config = self.default_deep_speed_config
        world_size = len(self.gpu_manager.get_all_avai_gpu_id())

        
       

        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=1)
        self.model:AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(self.model_path,**self.model_init_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **self.tokenizer_init_args)
        # 使用 DeepSpeed 初始化模型
        self.model = self.model = deepspeed.init_inference(
            self.model,                         # 传入模型
            mp_size=len(self.gpu_manager.get_all_avai_gpu_id()),  # GPU 的数量
            dtype=torch.float16,            # 使用半精度浮点
            replace_method='auto',          # 自动替换部分模块
            config=self.ds_config         # DeepSpeed 配置文件
        )
        
    def _init_sampling_params(self):
        self.logger.info("Initializing sampling parameters...")
        sp_config=self.config.get("sampling_params",None)
        if sp_config is not None:
            self.sampling_params = SamplingParams(**sp_config)
        else:
            self.sampling_params=self._get_sampling_params()
        
    def _get_sampling_params(self):
        """
        从 self.model_generate_args 读取生成参数并初始化 SamplingParams。
        """
        # 从 self.model_generate_args 中获取参数或设置默认值
        max_tokens = self.model_generate_args.get("max_tokens", 1024)
        temperature = self.model_generate_args.get("temperature", 0)
        top_p = self.model_generate_args.get("top_p", 0.9)
        top_k = self.model_generate_args.get("top_k", 50)
        if temperature is None:
            temperature=0
        if top_p is None:
            top_p=0.9
        if top_k is None:
            top_k=50
        repetition_penalty = self.model_generate_args.get("repetition_penalty", 1.0)
        presence_penalty = self.model_generate_args.get("presence_penalty", 0.0)
        frequency_penalty = self.model_generate_args.get("frequency_penalty", 0.0)

        # 初始化 SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        return sampling_params
        
    def _init_vllm(self):
        self.logger.info("Using vllm as inference method")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **self.tokenizer_init_args)
        
        vllm_args:dict = self.config.get("vllm_args",ModelActivity.format_vllm_args())
        default_tensor_parallel_size=1
        
        if isinstance(self.device_id,list):
            default_tensor_parallel_size=len(self.device_id)
            
        tensor_parallel_size=vllm_args.pop("tensor_parallel_size")
        if tensor_parallel_size=="auto":
            tensor_parallel_size=default_tensor_parallel_size
            
        
        
        
            
        
        self.model = LLM(self.model_path,
                         tokenizer_mode="auto", 
                         tensor_parallel_size=tensor_parallel_size,
                         **vllm_args) 
        
        self.model:LLM
        self.model.set_tokenizer(self.tokenizer)
        self._init_sampling_params()
        
        
    def __init_api(self):
        
        self.client=OpenAI(
            api_key=self.config.get("api_key","sk-xfu4tqc814iJQJGrD3A53540855a4cC095254366422d8dBe"),
            base_url=self.config.get("url","https://api3.apifans.com/v1")
        )
        self.logger.info(f"Initializing api key as "+self.client.api_key)
    
    def __init_multi_gpu(self):
        """加载多gpu上的模型和分词器"""
        self.logger.info("Loading model from "+self.model_path+"...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_path, 
                                device_map="auto",  # 自动分配模型到多张 GPU
                                max_memory=self.max_memory,
                                **self.model_init_args
                                )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,**self.tokenizer_init_args)
        self.logger.info("Successfully load model from "+self.model_path)
        
    def __init_single_gpu(self):
        """加载单gpu上都模型和分词器"""
        self.logger.info("Loading model from "+self.model_path+"...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.model_init_args).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,**self.tokenizer_init_args)
        self.logger.info("Successfully load model from "+self.model_path)
        
    # tokernizer整合message额外添加的字符长度
    
    def _handle_missing_tokens(self):
        """
        初始化时补全缺失的token,会根据模型自动选择和时代方法
        目前仅对llama和qwen有适配
        """
        self.model_type=self.config.get("model_type","auto")
        if  self.model_type=="auto":
            if "llama" in self.model_name.lower():
                self.model_type="llama3"
            elif "qwen" in self.model_name.lower():
                self.model_type="qwen2.5"
            else:
                raise ValueError("Invalid model_type")
        
        
        if self.model_type=="llama3" :
            try:
                if self.chat_type=="vllm":
                    self.tokenizer.pad_token = '<|end_of_text|>'
                    self.tokenizer.add_bos_token = False
                    self.tokenizer.add_eos_token = False
                    stop_token_ids=[128001,128008,128009] # 128001:<|end_of_text|> 128008:<|eom_id|> 128009:<|eot_id|>
                    self.sampling_params.stop_token_ids=stop_token_ids
                else:
                    # 处理缺失的标记并调整embedding
                    #ModelActivity.handle_missing_tokens_and_resize_embeddings(self.tokenizer,self.model)
                    self.model.generation_config = GenerationConfig.from_pretrained(self.model_path, pad_token_id=self.tokenizer.pad_token_id)
                    self.model.generation_config.eos_token_id =[128001, 12808, 128009]
                    self.model.generation_config.pad_token_id = 128001
                    self.tokenizer.pad_token = '<|end_of_text|>'
                self.logger.info("Adding missing special tokens for "+self.model_name)
            except Exception as e:
                self.logger.warning("Failed to set some special tokens for "+self.model_name+".This may cause problem when generating responses")
                traceback.print_exc()
        elif self.model_type=="qwen2.5" or self.model_type=="qwen2":
            if self.chat_type=="vllm":
                self.tokenizer.pad_token = '<|endoftext|>'
                self.tokenizer.pad_token_id = 151643
                stop_token_ids=[151645,151643]
                self.sampling_params.stop_token_ids=stop_token_ids
            else:
                self.model.generation_config = GenerationConfig.from_pretrained(self.model_path, pad_token_id=self.tokenizer.pad_token_id)
                self.model.generation_config.eos_token_id = [151645, 151643]
                self.model.generation_config.pad_token_id = 151643
            self.logger.info("Adding missing special tokens for "+self.model_name)
            
    def _init_pipeline(self,task="text-generation"):
        """初始化pipeline"""
        self.logger.info("Initializing pipeline for "+self.model_name)
       
        if isinstance(self.device_id,int):
            qa_pipeline = pipeline(task, model=self.model, tokenizer=self.tokenizer,device=self.device)
        elif isinstance(self.device_id,list):
            qa_pipeline = pipeline(task, model=self.model, tokenizer=self.tokenizer,device_map="auto")
        self.specific_pipeline=qa_pipeline
        return qa_pipeline
    
    def _init_config_args(self,config:dict):
        """
        从config读取默认的模型生成参数，如果没有设置则为默认只有max_new_tokens=256
        """
        self.logger.info("Initializing generation config...")
        self.model_init_args=config.get("model_init_args",None)
        self.tokenizer_init_args=config.get("tokenizer_init_args",None)
        self.model_generate_args=config.get("model_generate_args",{"max_new_tokens": 256})
        self.tokenizer_generate_args=config.get("tokenizer_generate_args",{"padding": False})
        self.remove_prompt=True
        
       
    def _init_logging(self):
        # 确定日志目录
        module_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(module_dir, "log")
        
        # 如果日志目录不存在，创建它
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 根据时间生成日志文件名
        if self.task_name is None:
            log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        else:
            log_file_name = self.task_name+".log"
        log_file_path = os.path.join(log_dir, log_file_name)

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别
            format="[%(asctime)s] [%(levelname)s] %(message)s",  # 定义日志格式
            datefmt="%Y-%m-%d %H:%M:%S",  # 日期时间格式
            handlers=[
                logging.StreamHandler(),  # 输出到控制台
                logging.FileHandler(log_file_path, mode="a")  # 输出到文件，追加模式
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("日志系统已初始化")
        self.logger.info(f"日志文件路径: {log_file_path}")
        try:
            self._log_config()
        except:
            pass

    def _log_config(self):
        """
        将当前对象的 config 内容写入日志文件，支持多层嵌套字典。
        """
        def log_dict_recursively(d, parent_key=""):
            """
            递归记录字典内容。

            :param d: 当前字典。
            :param parent_key: 父级键，用于多层结构记录时的前缀。
            """
            for key, value in d.items():
                current_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    self.logger.info(f"{current_key}:")
                    log_dict_recursively(value, current_key)
                else:
                    self.logger.info(f"{current_key}: {value}")

        if hasattr(self, 'config') and self.config:
            self.logger.info("当前配置 (config) 内容如下:")
            if isinstance(self.config, dict):
                log_dict_recursively(self.config)
            else:
                self.logger.warning("config 不是字典，记录其值:")
                self.logger.info(f"config: {self.config}")
        else:
            self.logger.warning("未找到 config 或 config 为空")
        

    
    def __init__(self,device_id:int|list[int]=0,config:dict=None,gpu_manager:GPUManager=None,task_name:str=None):
        if gpu_manager is None and config.get("chat_type")!="api":
            gpu_manager=GPUManager()
        if config is None:
            raise ValueError("Config cannot be None")
        self.task_name=task_name
        self.config=config
        self._init_logging()
        self.device_id=device_id
        
        
        self.model_name=config["model_name"]
        self.gpu_manager=gpu_manager
        if self.gpu_manager:
            self.device = gpu_manager.get_device_from_id(self.device_id)
            self.max_memory=gpu_manager.get_max_memory_map(device_id=self.device_id)
        
        self.using_deep_speed=config.get("using_deepspeed",False)
        self.using_vllm=config.get("using_vllm",False)
        
        #读取并设置模型生成回答的参数
        self._init_config_args(config=self.config)
        
        
        if "model_path" in self.config.keys():
            self.default_model_path=self.config["model_path"]
        self.model_path=self.default_model_path+self.model_name
        
        print(self.model_path)
        
        
        self.chat_type=config.get("chat_type","classical")
        
        if self.chat_type=="api":
            self.__init_api()
            return
        elif self.chat_type=="vllm":
            self._init_vllm()
            pass
        elif self.chat_type=="deepspeed":
            raise RuntimeError("deepspeed is not supported now")
            self.__init_deep_speed()
            pass
        elif self.chat_type=="pipeline" or "classical":
            if isinstance(self.device_id,list):
                self.__init_multi_gpu()
            # Init on single gpu
            else:
                self.__init_single_gpu()
            if self.chat_type=="pipeline":
                self._init_pipeline()
            
        else:
            raise ValueError("Invalid chat_type "+self.chat_type+".Only api,vllm,pipeline,classical are valid")
                
        self._handle_missing_tokens()
            

        
        
    
    

    



   

    
        
        
        
    def _test_model_when_initializing(self):
        """进行一个测试性质的回答"""
        print("--------------------------------------------------------------------------------")
        print("正在进行测试性问答")
        test_msg=ModelActivity.format_message("这是一个测试,请回答你是什么模型",sys_prompt="测试",assist_prompt="")
        output=self.run_inference(test_msg)
        print(output)
        print("测试性回答完成，模型可生成回复，请检查其生成是否符合你的格式要求")
        print("--------------------------------------------------------------------------------")

        
    
    

    def _remove_prompt(self,prompt:str,output:str):
        new_output = output[(len(prompt)):]
        return new_output

    

    
    def _merge_message(self,messages:list):
        ModelActivity.validate_messages(messages)
        output_text = ""
        apply_chat_template=self.config.get("apply_chat_template")
        if self.chat_type=="api" or apply_chat_template==False:
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    output_text += f"system\n{content}\n "
                elif role == "user":
                    output_text += f"user\n{content}\n "
                elif role == "assistant":
                    output_text += f"assistant\n{content}\n "
        elif apply_chat_template:
            output_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
        else:
            raise NotImplementedError("apply-chat-template must be set true or false")
        return output_text
    
    def generate_api_response(self,messages:list)->str:
        """
        ## 函数作用
        调用gpt4的api生成回答
        """
        model_generate_args:dict=self.config.get("model_generate_args")
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,  # 使用 GPT-4 模型
                max_tokens=model_generate_args.get("max_new_tokens",512),  # 设置生成回答的最大长度
                temperature=model_generate_args.get("temperature",0.2),  # 控制生成文本的创造性
                n=1,  # 生成一个回答
                stop=None  # 停止生成的条件，例如指定的符号或字符串
            )
            # 从响应中获取生成的文本
            generated_text = response.choices[0].message.content
            
            return generated_text

        except Exception as e:
            print(f"Error occurred when using api to generate response")
            traceback.print_exc()
            return None
        
    def generate_gpt4_response(self,messages:list,max_tokens=512)->str:
        """
        ## 函数作用
        调用gpt4的api生成回答
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,  # 使用 GPT-4 模型
                max_tokens=max_tokens,  # 设置生成回答的最大长度
                temperature=0.7,  # 控制生成文本的创造性
                n=1,  # 生成一个回答
                stop=None  # 停止生成的条件，例如指定的符号或字符串
            )
            # 从响应中获取生成的文本
            generated_text = response.choices[0].message.content
            
            return generated_text

        except Exception as e:
            print(f"调用 GPT-4 生成回答时出错: {e}")
            return None
        
    def _inference_deepspeed(self,std_input_text:str)->str:
        """使用deepspeed推理单个问题"""
        # 传统的推理方式，直接使用 DeepSpeed 初始化后的模型
        inputs = self.tokenizer(std_input_text, **self.tokenizer_generate_args)
        
        # 使用 DeepSpeed 的模型进行推理
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], 
                                            **self.model_generate_args)

        result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.remove_prompt:
            result_text = self._remove_prompt(std_input_text, result_text)
        return result_text
    
    def _inference_vllm(self,std_input_text:str)->str:
        output = self.model.generate([std_input_text], sampling_params=self.sampling_params)
        return output[0].outputs[0].text
    
    def _inference_local_model_default(self,std_input_text:str)->str:
        """推理单个问题，根据配置文件决定使用pipeline还是generate"""
        chat_type=self.config["chat_type"]
        
        
        if chat_type=="pipeline":
            output = self.specific_pipeline(
                std_input_text,
                **self.model_generate_args
            )
            result_text=output[0]["generated_text"]
            if self.remove_prompt:
                result_text=self._remove_prompt(std_input_text,result_text)
        elif chat_type=="classical":
            inputs = self.tokenizer(std_input_text ,**self.tokenizer_generate_args)
            if isinstance(self.device,list):
                inputs=inputs.to(self.device[0])
            else:
                inputs=inputs.to(self.device)
                
            with torch.no_grad():
                # 使用模型生成输出
                outputs = self.model.generate(
                    inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    **self.model_generate_args
                )
            
            # 只获取生成部分的 tokens
            generated_output = outputs[:, inputs["input_ids"].shape[-1]:]
            
            # 解码生成的输出
            result_text = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)
                
        else:
            result_text=""
        return result_text
        


    def run_inference(self, messages:list):
        """
        ## 函数功能
        执行模型推理并返回生成的文本
        ## 参数
        messages: 一个通过format_message方法产生的输入列表
        max_len: 输出的最大长度
        ## 返回值
        一个包含结果的字典，形式：{"Input":{input_text},"Result":{result_text}}
        """

        result_text=""
        ModelActivity.validate_messages(messages=messages)
        std_input_text = self._merge_message(messages)
        try:
            if self.chat_type=="api":
                result_text=self.generate_api_response(messages=messages)
            elif self.chat_type=="deepspeed":
                result_text=self._inference_deepspeed(std_input_text=std_input_text)
            elif self.chat_type=="vllm":
                result_text=self._inference_vllm(std_input_text=std_input_text)
            else:
                result_text=self._inference_local_model_default(std_input_text=std_input_text)
        except Exception as e:
            self.logger.error(f"Error occurred during inference")
            traceback.print_exc()
            
        result={"Input": std_input_text,"Result":result_text}

        return result



    def run_group_inference(self, messages_list:list):
        """
        ## 函数功能
        执行模型推理并返回多个问题的生成结果,处理是同时进行的，有可能导致显存溢出"
        ## 参数
        messages_list: 包含多个通过format_message方法产生的输入列表的大列表，
        max_len: 输出的最大长度
        ## 返回值
        一个包含结果的字典组成的list[dict],其长度与输入的messages_list相同\n
        每个字典形式：{"Input":{input_text},"Result":{result_text}}\n
        """
        
        
        result_texts=[]
        std_input_texts=[]
        for messages in messages_list:
            std_input_text = self._merge_message(messages=messages)
            std_input_texts.append(std_input_text)
        try:
            if self.chat_type=="api":
                for messages in messages_list:
                    ModelActivity.validate_messages(messages=messages)
                    result_texts.append(self.generate_api_response(messages=messages))
            elif self.chat_type=="deepspeed":
                result_texts=self._group_inference_deepspeed(std_input_texts=std_input_texts)
            elif self.chat_type=="vllm":
                result_texts=self._group_inference_vllm(std_input_texts=std_input_texts)
            else:
                result_texts=self._group_inference_local_model_default(std_input_texts=std_input_texts)
        except Exception as e:
            self.logger.error(f"Error occurred during inference ")
            traceback.print_exc()
            
        results = [{"Input": input_text, "Result": result_text} for input_text, result_text in zip(std_input_texts, result_texts)]
        
        return results
    


    def _group_inference_deepspeed(self,std_input_texts:list)->list:
        """使用deepspeed生成一组问题"""

        inputs = self.tokenizer(std_input_texts,**self.tokenizer_generate_args)
        # 使用 DeepSpeed 的模型进行推理
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids,
                                            attention_mask=inputs.input_ids.ne(self.tokenizer.pad_token_id),
                                            **self.model_generate_args)
        # 解码生成的结果
        generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        result_texts=self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        return result_texts
    
    def _group_inference_vllm(self,std_input_texts:list)->list:
        outputs = self.model.generate(std_input_texts, sampling_params=self.sampling_params)
    
        # 提取每个生成的文本并返回结果
        result_texts = [output.outputs[0].text for output in outputs]
        return result_texts
    
    def _group_inference_local_model_default(self,std_input_texts:list)->list:
        """生成一组问题的回答，根据配置文件选择pipeline和generate"""
        chat_type=self.config["chat_type"]
            
        # print(std_input_texts)
        # 对所有输入文本进行批量 tokenization
        # Tokenize the entire batch of inputs            
        if chat_type=="pipeline" :
            # pipeline approach
            with torch.no_grad():
                output_texts = self.specific_pipeline(
                    std_input_texts,
                    **self.model_generate_args
                )
                
            result_texts=[]
            for i in range(len(output_texts)):
                target_text=output_texts[i][0]["generated_text"]
                result_texts.append(target_text)
                if self.remove_prompt:
                    result_texts[i]=(self._remove_prompt(std_input_texts[i],result_texts[i]))
                    # print(result_texts[i]
        elif chat_type=="classical":
            inputs = self.tokenizer(std_input_texts,**self.tokenizer_generate_args)
            
            if isinstance(self.device,list):
                inputs=inputs.to(self.device[0])
            else:
                inputs=inputs.to(self.device)
            # 使用 DeepSpeed 的模型进行推理
            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids,
                                                attention_mask=inputs.input_ids.ne(self.tokenizer.pad_token_id),
                                                **self.model_generate_args)
            # 解码生成的结果
            generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            result_texts=self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        elif chat_type=="chat":
            pass

        return result_texts
    
    def get_model(self)->LLM:
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def set_chat_template(self,chat_template):
        self.tokenizer.chat_template=chat_template
        
    
    
    def calculate_ppl(self, text_segments, ppl_start_index=0, max_lengths=None):
        """
        计算从指定部分开始的困惑度，适用于任意长度的输入文本数组。

        :param text_segments: 文本段组成的字符串数组，例如 ['你好，今天天气怎么样', '天气很好'] 
        :param ppl_start_index: 从哪个索引开始计算困惑度（例如 1 表示从第 2 段文本开始计算）
        :param max_lengths: 每段文本的最大长度数组。如果未提供，则默认为 512。
        :return: 困惑度值
        """
        raise RuntimeError("PPL calculation is not supported now")
        self.model.eval()

        # 检查 max_lengths，如果没有提供，则为每段文本设置默认最大长度
        if max_lengths is None:
            max_lengths = [512] * len(text_segments)
        else:
            if len(max_lengths) != len(text_segments):
                raise ValueError("max_lengths 的长度必须与 text_segments 相同")

        if len(text_segments) <= ppl_start_index:
            raise ValueError(f"Invalid text_segments length {len(text_segments)} with start_index {ppl_start_index}")

        # 从指定的 ppl_start_index 开始的文本段
        segments = text_segments[ppl_start_index:]
        lengths = max_lengths[ppl_start_index:]

        # 过滤掉无效的段落
        valid_segments = []
        valid_lengths = []
        for i, (segment, max_len) in enumerate(zip(segments, lengths), start=ppl_start_index):
            if not segment or not isinstance(segment, str):
                print(f"Invalid segment at index {i}: {segment}")
                continue  # 跳过无效的段落
            valid_segments.append(segment)
            valid_lengths.append(max_len)

        if not valid_segments:
            raise ValueError("没有有效的文本段可用于计算困惑度。")

        # 分别处理每个有效的文本段，收集 input_ids
        input_ids_list = []
        for segment, max_len in zip(valid_segments, valid_lengths):
            try:
                # 使用 tokenizer 处理每个文本段，确保返回为张量
                inputs = self.tokenizer(
                    segment, 
                    return_tensors="pt", 
                    padding='max_length', 
                    truncation=True, 
                    max_length=max_len
                )
                input_ids = inputs['input_ids']  # 这是一个 torch.Tensor
            except Exception as e:
                print(f"Tokenizer 处理文本时出错: {e}")
                continue  # 跳过处理出错的段落

            # 转移到设备
            input_ids = input_ids.to(self.device)
            input_ids_list.append(input_ids)

        if not input_ids_list:
            raise ValueError("没有有效的 input_ids 被生成。请检查输入的 text_segments。")

        # 将 input_ids_list 在批次维度拼接
        input_ids = torch.cat(input_ids_list, dim=0)

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(input_ids)

        # 对齐 logits 和 input_ids，确保对每个标记的预测是基于前一个标记
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        # 计算损失
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        # 计算困惑度
        perplexity = torch.exp(loss)

        return perplexity.item()
    
    
    







