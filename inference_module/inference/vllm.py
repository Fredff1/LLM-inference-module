# inference/vllm.py

from transformers import AutoTokenizer
from inference_module.inference.base import BaseInference
from inference_module.token_hanlder.token_hanlder import handle_missing_tokens

# 假设 vllm 库已经安装并导入（若未安装，请先安装 vllm 库）
try:
    from vllm import LLM ,SamplingParams # vllm 模型类
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
        
        
        # 拼接模型完整路径（假设 config 中 model_path 与 model_name 配置好）
        model_full_path = self.config["model_path"] + self.config["model_name"]
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_full_path,
            **self.config.get("tokenizer_init_args")
        )
        
        # 获取 vllm 加载参数，从配置中获取或使用默认值（此处参考以前的 format_vllm_args）
        vllm_args = self.config.get("vllm_args")
        default_tensor_parallel_size = 1
        # 如果 device_id 是一个列表，则设定 tensor_parallel_size 为列表的长度
        if isinstance(self.config.get("device_id"), list):
            default_tensor_parallel_size = len(self.config.get("device_id"))
        
        # 处理 tensor_parallel_size 参数：如果为 "auto"，则替换为默认值
        tensor_parallel_size = vllm_args.pop("tensor_parallel_size", "auto")
        if tensor_parallel_size == "auto":
            tensor_parallel_size = default_tensor_parallel_size
        vllm_args["tensor_parallel_size"] = tensor_parallel_size
        
        # 实例化 LLM 模型（vllm库的接口），注意 tokenizer_mode 通常设置为 "auto"
        self.model = LLM(
            model_full_path,
            tokenizer_mode="auto",
            **vllm_args
        )
        
        
        # 将加载好的分词器设置到 vllm 模型中
        self.model.set_tokenizer(self.tokenizer)
        # 初始化采样参数，假设采样参数在配置中传入，例如 config["sampling_params"]
        self._init_sampling_params()
        handle_missing_tokens(self)
    
    def _init_sampling_params(self):
        """
        初始化采样参数。这里假设采样参数已经在配置中设置，
        若不存在则可以设定默认值。该方法将设置 self.sampling_params，
        后续生成时将作为参数传入 self.model.generate。
        """
        # 如果配置中有采样参数，直接使用；否则设置默认值
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
        # 这里直接存储字典，可进一步转换成 vllm 所需的 SamplingParams 对象
        self.sampling_params = SamplingParams(**sampling_conf)

    def run(self, input_content: str) -> str:
        """
        单条推理接口：对单个输入文本进行生成推理。
        调用 vllm 的 generate 方法，并返回生成的文本。
        """
        # vllm 的 generate 接口要求传入 list 格式的输入，因此构造列表
        outputs = self.model.generate([input_content], sampling_params=self.sampling_params)
        # outputs 的结构假设为列表，取第一条结果，并提取 outputs[0].outputs[0].text
        result_text = outputs[0].outputs[0].text
        return result_text

    def run_batch(self, input_contents: list[str]) -> list:
        """
        批量推理接口：对多个输入文本进行生成推理，返回文本结果列表。
        """
        outputs = self.model.generate(input_contents, sampling_params=self.sampling_params)
        # 遍历每个生成结果，提取文本
        result_texts = [output.outputs[0].text for output in outputs]
        return result_texts
