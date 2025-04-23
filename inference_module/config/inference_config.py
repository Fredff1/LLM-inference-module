# config/full_config.py

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Union, List



# -------------------------------
# GenerationParams 对应之前 format_model_generation_params
# 默认值参照 HuggingFace 的生成参数（如 max_new_tokens、temperature、top_p、top_k 等），
# 同时支持额外参数（additional）用于覆盖默认设置。
# -------------------------------
@dataclass
class GenerationParams:
    max_new_tokens: int = 1024         # 与 Hugging Face 默认生成长度一致
    temperature: float = 1.0          # 温度，默认1.0（不改变多样性），与原函数一致
    top_p: float = 1.0                # Nucleus sampling 截断概率，1.0表示不裁剪
    top_k: int = 50                   # Top-K 筛选大小，原函数默认50
    do_sample: bool = True            # 是否采样，原函数默认True以防止警告
    num_beams: int = 1                # Beam 数量，1表示不采用beam search
    length_penalty: float = 1.0       # 长度惩罚，默认1.0
    use_cache: bool = True            # 缓存设置，默认True
    repetition_penalty: float = 1.0   # 重复惩罚，默认1.0
    additional: Dict[str, Any] = field(default_factory=dict)  # 用于扩展其他 HuggingFace 参数

    def to_dict(self) -> Dict[str, Any]:
        # 合并 asdict() 输出和 additional 参数，additional优先覆盖默认参数
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}

# -------------------------------
# TokenizerParams 对应之前 format_tokenizer_params
# 默认参数与 HuggingFace 分词器调用一致: padding, add_special_tokens, return_tensors 等
# -------------------------------
@dataclass
class TokenizerParams:
    padding: bool = True
    add_special_tokens: bool = True
    return_tensors: str = "pt"
    additional: Dict[str, Any] = field(default_factory=dict)  # 用于扩展其他分词器参数

    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}

# -------------------------------
# ModelInitParams 对应之前 format_model_init_args
# 默认值参考 HuggingFace 的 from_pretrained 初始化设置
# -------------------------------
@dataclass
class ModelInitParams:
    trust_remote_code: bool = True   # 默认信任远程代码，与原函数相同
    torch_dtype: Union[str, None] = "auto"  # "auto" 表示自动选择数据类型
    cache_dir: Optional[str] = None   # 缓存目录，默认为None
    additional: Dict[str, Any] = field(default_factory=dict)  # 其他初始化参数

    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}
    
@dataclass
class TokenizerInitParams:
    padding_side: str = "left" 
    trust_remote_code: bool = True 
    cache_dir: str = None,
    additional: Dict[str, Any] = field(default_factory=dict)  
    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}
    

# -------------------------------
# VLLMParams 对应之前 format_vllm_args
# 默认值参考 vLLM 库对加载参数的要求：tensor_parallel_size, gpu_memory_utilization, dtype, enforce_eager, enable_chunked_prefill, max_num_seqs
# -------------------------------
@dataclass
class VLLMParams:
    tensor_parallel_size: Union[int, str] = "auto"  # 可为 "auto" 或具体整数
    gpu_memory_utilization: float = 0.8             # 默认 0.8
    dtype: str = "bfloat16"                         # 默认 bfloat16，与部分 vLLM 示例对应；如需要float16，根据实际情况调整
    enforce_eager: bool = True                      # 是否强制使用 eager 模式
    enable_chunked_prefill: bool = False            # 分块预填充，默认False
    max_num_seqs: int = 8                           # 默认值为8
    additional: Dict[str, Any] = field(default_factory=dict)  # 额外参数

    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}
    
@dataclass
class SamplingParams:
    max_tokens: int = 1024                # 对应原代码中的 max_tokens
    temperature: float = 1.0              # 温度，默认设置为 1.0，可根据需求调整
    top_p: float = 1.0                    # 累计概率阈值，默认1.0
    top_k: int = 50                       # top_k 采样的范围，-1 可以理解为不限制（根据实际需求设置）
    repetition_penalty: float = 1.0       # 重复惩罚，默认1.0
    presence_penalty: float = 0.0         # 出现惩罚
    frequency_penalty: float = 0.0        # 频率惩罚
    additional: Dict[str, Any] = field(default_factory=dict)  # 扩展字段

    def to_dict(self) -> Dict[str, Any]:
        # 合并 asdict() 输出和 additional，确保 additional 中的值可以覆盖默认值
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}
    
@dataclass
class DeviceConfig:
    """
    封装推理器设备配置参数，可以用于 classical、pipeline、vllm、api 模式，
    这里直接写入 inference 相关配置中。
    
    注意：
      - 对于 classical/pipeline 模式：
            如果 device_id 为 int：指定 "device": torch.device
            如果 device_id 为 list：采用 "device_map": "auto" 与 max_memory 控制
      - 对于 vLLM/api 模式：
            一般不需要设置或者设置为空
    """
    device_id: Union[int, List[int]]  # 单个或多个 GPU id
    max_memory: Any  = None           # 可接受的形式：dict 或字符串，例如 {"0": "13GiB", "1": "13GiB"}
    
    def to_dict(self) -> Dict[str, Any]:
        # 根据 device_id 类型判断返回对应配置
        if isinstance(self.device_id, list):
            return {"device_map": "auto", "max_memory": self.max_memory}
        else:
            return {"device": f"cuda:{self.device_id}"}
        
@dataclass
class ApiConfig:
    api_key: str = None       # API 模式时使用的密钥
    url: str = None           # API 请求地址
    max_retries :int = 3
    backoff_factor :float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        # 合并 asdict() 输出和 additional，确保 additional 中的值可以覆盖默认值
        params = asdict(self)
        return {**params}
    

# -------------------------------
# FullConfig 整体配置数据类
#
# 该类统一管理模型配置，包括模型名称、路径、chat_type 以及生成、分词器、初始化、vLLM参数等。
#
# 对照原先的 build_full_config 函数，以下字段：
#   - model_generate_args 对应 GenerationParams.to_dict()
#   - tokenizer_generate_args 对应 TokenizerParams.to_dict()
#   - model_init_args 对应 ModelInitParams.to_dict()
#
# api_key、url 等用于 API 模式的扩展字段。
# -------------------------------
@dataclass
class InferenceConfig:
    model_name: str                     # 模型名称
    model_path: str                     # 模型所在根路径（通常后续与 model_name 拼接）
    chat_type: str = "classical"        # 推理模式，示例支持： "api", "vllm", "pipeline", "classical"
    model_type:str = 'auto'
    device_config: DeviceConfig = None
    generation_params: GenerationParams = field(default_factory=GenerationParams)
    tokenizer_params: TokenizerParams = field(default_factory=TokenizerParams)
    model_init_params: ModelInitParams = field(default_factory=ModelInitParams)
    tokenizer_init_params:TokenizerInitParams = field(default_factory=TokenizerInitParams)
    vllm_params: Optional[VLLMParams] = None  # 仅 vLLM 模式下使用
    sampling_params: Optional[SamplingParams] = None  # 如果使用 vLLM 或需要单独控制采样，则传入
    api_config: Optional[ApiConfig] = None
    apply_chat_template: bool = False    # 是否应用chat模板，用于部分特殊用途
    log_dir:str = None

    def to_dict(self) -> Dict[str, Any]:
        config = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "chat_type": self.chat_type,
            "model_type":self.model_type,
            "model_generate_args": self.generation_params.to_dict(),
            "tokenizer_generate_args": self.tokenizer_params.to_dict(),
            "tokenizer_init_args": self.tokenizer_init_params.to_dict(),
            "model_init_args": self.model_init_params.to_dict(),
            "apply_chat_template": self.apply_chat_template,
            "log_dir":self.log_dir,
        }
        if self.vllm_params:
            config["vllm_args"] = self.vllm_params.to_dict()
        if self.sampling_params:
            config["sampling_params"] = self.sampling_params.to_dict()
        if self.api_config:
            config["api_config"] = self.api_config.to_dict()
        
        if self.device_config:
            config['device_config'] = self.device_config.to_dict()
            config['device_id']=self.device_config.device_id
        
        if self.chat_type == "api":
            if self.api_config is None or config["api_config"].get("api_key",None) is None or config["api_config"].get("url") is None:
                raise ValueError("api key and url must be provided for api chat")
        elif self.chat_type == "vllm":
             if self.vllm_params is None or self.sampling_params is None:
                raise ValueError("vllm_params and sampling_params must be provided for vllm chat")
        elif self.chat_type in ["classical","pipeline"]:
            if self.device_config is None:
                raise ValueError("Hugging face inference must provide device config")

        return config

# -------------------------------
# 使用示例说明：
#
# 假设原来通过 build_full_config 来构造配置，现在可以这样写：
#
# from config.full_config import FullConfig, GenerationParams
#
# config_obj = FullConfig(
#     model_name="gpt_model",
#     model_path="/home/workspace/Models/",
#     chat_type="classical",
#     generation_params=GenerationParams(max_new_tokens=512, temperature=0.7),
#     api_key="your_api_key",       # 若 chat_type 为 "api" 则必填
#     url="https://api.example.com/v1"
# )
#
# full_config_dict = config_obj.to_dict()
# 这样生成的字典，与以前的 build_full_config 输出保持一致，同时与 Hugging Face/vLLM 等库的参数对照无误。
# -------------------------------
