# inference/vllm.py

from transformers import AutoTokenizer
from inference_module.inference.base import BaseInference
from inference_module.token_handler.token_handler import handle_missing_tokens

try:
    from vllm import LLM ,SamplingParams 
except ImportError:
    print("Vllm is disabled because it is not installed")

class VLLMInference(BaseInference):
    def initialize(self) -> None:
        """
        初始化 vLLM 模型与分词器：
          - 加载 tokenizer（基于配置中的 model_path 和 tokenizer 参数）
          - 从配置中获取 vllm_args，处理 tensor_parallel_size 参数（"auto" 时自动根据 device_id 数目设置）
          - 实例化 LLM 模型，并将 tokenizer 传入
          - 初始化采样参数
        """
        
        
        model_full_path = self.config["model_path"] + self.config["model_name"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_full_path,
            **self.config.get("tokenizer_init_args")
        )
        
        vllm_args = self.config.get("vllm_args")
        default_tensor_parallel_size = 1
        if isinstance(self.config.get("device_id"), list):
            default_tensor_parallel_size = len(self.config.get("device_id"))
        
        tensor_parallel_size = vllm_args.pop("tensor_parallel_size", "auto")
        if tensor_parallel_size == "auto":
            tensor_parallel_size = default_tensor_parallel_size
        vllm_args["tensor_parallel_size"] = tensor_parallel_size
        
        self.model = LLM(
            model_full_path,
            tokenizer_mode="auto",
            **vllm_args
        )
        
        self.model.set_tokenizer(self.tokenizer)
        self._init_sampling_params()
        handle_missing_tokens(self)
    
    def _init_sampling_params(self):
        """
        初始化采样参数。这里假设采样参数已经在配置中设置，
        若不存在则可以设定默认值。该方法将设置 self.sampling_params，
        后续生成时将作为参数传入 self.model.generate。
        """
        sampling_conf = self.config.get("sampling_params", {
            "max_tokens": 1024,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "sampling_times": 1
        })
        self.sampling_params = SamplingParams(**sampling_conf)

    def run(self, input_content: str) -> str:
        """
        单条推理接口：对单个输入文本进行生成推理。
        调用 vllm 的 generate 方法，并返回生成的文本。
        """
        outputs = self.model.generate([input_content], sampling_params=self.sampling_params)
        result_text = outputs[0].outputs[0].text
        return result_text

    def run_batch(self, input_contents: list[str]) -> list:
        """
        批量推理接口：对多个输入文本进行生成推理，返回文本结果列表。
        """
        outputs = self.model.generate(input_contents, sampling_params=self.sampling_params)
        result_texts = [output.outputs[0].text for output in outputs]
        return result_texts
    
    def validate_input(self, input):
        # from inference_module.utils.message_utils import format_message
        # tpl = self.config.get("apply_chat_template", False)
        # 纯文本
        if isinstance(input, str):
            return input, "single"
        if isinstance(input, list) and input and all(isinstance(x, str) for x in input):
            return input, "list"

        
        raise ValueError(
            "Classical/VLLM 模式下，只支持 str 或 List[str]；"
        )
