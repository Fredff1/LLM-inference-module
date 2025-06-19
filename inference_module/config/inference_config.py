# config/full_config.py

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Union, List


@dataclass
class GenerationParams:
    max_new_tokens: int = 1024         
    temperature: float = 1.0          
    top_p: float = 1.0               
    top_k: int = 50                   
    do_sample: bool = True            
    num_beams: int = 1                
    length_penalty: float = 1.0       
    use_cache: bool = True            
    repetition_penalty: float = 1.0   
    additional: Dict[str, Any] = field(default_factory=dict)  

    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}

@dataclass
class TokenizerParams:
    padding: bool = True
    add_special_tokens: bool = True
    return_tensors: str = "pt"
    additional: Dict[str, Any] = field(default_factory=dict)  

    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}


@dataclass
class ModelInitParams:
    trust_remote_code: bool = True  
    torch_dtype: Union[str, None] = "auto"  
    cache_dir: Optional[str] = None   
    additional: Dict[str, Any] = field(default_factory=dict)  

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
    


@dataclass
class VLLMParams:
    tensor_parallel_size: Union[int, str] = "auto"  
    gpu_memory_utilization: float = 0.8            
    dtype: str = "bfloat16"                         
    enforce_eager: bool = True                      
    enable_chunked_prefill: bool = False            
    max_num_seqs: int = 8                          
    additional: Dict[str, Any] = field(default_factory=dict)  # 额外参数

    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}
    
@dataclass
class VllmSamplingParams:
    max_tokens: int = 1024                
    temperature: float = 1.0              
    top_p: float = 1.0                   
    top_k: int = 50                       
    repetition_penalty: float = 1.0       
    presence_penalty: float = 0.0         
    frequency_penalty: float = 0.0        
    additional: Dict[str, Any] = field(default_factory=dict)  

    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        additional = params.pop("additional")
        return {**params, **additional}
    
@dataclass
class DeviceConfig:
    """
    封装推理器设备配置参数，可以用于 classical、pipeline、vllm、api 模式，

    """
    device_id: Union[int, List[int]]  
    max_memory: Any  = None           
    
    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.device_id, list):
            return {"device_map": "auto", "max_memory": self.max_memory}
        else:
            return {"device": f"cuda:{self.device_id}"}
        
@dataclass
class ApiConfig:
    api_key: str = None       
    url: str = None           
    max_retries :int = 3
    backoff_factor :float = 1.0
    max_concurrent_requests: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        params = asdict(self)
        return {**params}
    

@dataclass
class InferenceConfig:
    model_name: str                     
    model_path: str = ""                     
    chat_type: str = "classical"       
    model_type:str = 'auto'
    device_config: DeviceConfig = None
    generation_params: GenerationParams = field(default_factory=GenerationParams)
    tokenizer_params: TokenizerParams = field(default_factory=TokenizerParams)
    model_init_params: ModelInitParams = field(default_factory=ModelInitParams)
    tokenizer_init_params:TokenizerInitParams = field(default_factory=TokenizerInitParams)
    vllm_params: Optional[VLLMParams] = None  
    sampling_params: Optional[VllmSamplingParams] = None  
    api_config: Optional[ApiConfig] = None
    apply_chat_template: bool = True   
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


