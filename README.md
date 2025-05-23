# LLM 推理模块

一个简单的模块化、可扩展的框架，用于在多种后端上运行大语言模型（LLM）推理：

* 本地 Hugging Face 模型
* vLLM 引擎
* OpenAI API

## 功能特性

* **统一配置**：使用 `FullConfig` 数据类管理模型名称、路径、生成参数、分词器参数、初始化参数、vLLM 参数及 API 参数。
* **设备管理**：通过 `DeviceConfig` 支持单 GPU 或多 GPU 设置，自动配置 `device_map` 和 `max_memory`。
* **消息工具**：`format_message` 构造对话消息，`merge_messages` 根据需要合并消息列表。
* **模块化推理引擎**：`classical`（Hugging Face）、`vllm`、`api` 三种推理器实现，统一继承 `BaseInference`。
* **顶层封装**：`ModelInference` 类提供infer接口，支持单独推理和batch推理，提供标准chat_message格式化函数format_message
* **日志功能**：通过 `utils/logger.py` 初始化日志，记录运行信息和配置详情。
* **简易工具**: 提供`JsonUtil`和`DataUtil`读写json/jsonl文件并简单处理数据

## 安装

```bash
# 克隆仓库
git clone https://github.com/Fredff1/LLM-inference-module.git
cd LLM-inference-module
# 安装依赖
pip install -r requirements.txt
```

> **可选安装:**
>
> * vLLM 支持：`pip install vllm

## 目录结构

``` text
llm-inference-module/
├── config/             # 配置数据类
│   └── inference_config.py # DeviceConfig, InferenceConfig等配置以及总配置
│
├── utils/              # 通用工具
│   ├── logger.py       # 日志初始化与记录
│   ├── data_util       # DataUtil类，提供一些基本的数据操作函数
│   ├── json_util       # JsonUtil类，提供一些基本的Json操作函数
│   └── message_utils.py# 消息构造与合并函数
│
├── inference/          # 推理器实现
│   ├── base.py         # BaseInference 抽象基类
│   ├── classical.py    # ClassicalInference（HF 本地推理）
│   ├── vllm.py         # VLLMInference（vLLM 推理）
│   ├── api.py          # APIInference（OpenAI API 推理）
│   └── factory.py      # create_inference_engine 工厂函数
│
├── token_handler/ 
│   └── token_handler.py # 自动补全一些模型的token
│
├── model_inference.py    # 顶层封装
│
├── gpu_manager/
│   └── gpu_manager.py  # gpu管理类
│
└── examples/           # 使用示例脚本
    └── example.py      # 快速入门示例
```

## 快速开始

```python
from inference_module.model_inference import ModelInference
from inference_module.gpu_manager.gpu_manager import GPUManager
from inference_module.config.inference_config import *
from inference_module.utils.json_util import JsonUtil
from inference_module.utils.data_util import DataUtil

# 1. 构建配置对象
config =InferenceConfig(
        model_name="qwen2.5-0.5B",
        model_path="/mnt/data/models/", #别忘了拖尾的/
        model_type="qwen",
        chat_type=args.chat_type,
        device_config=None,
        generation_params=GenerationParams(
            max_new_tokens=1024,
            temperature=1.0,
            do_sample=True,
            repetition_penalty=1.0),
        tokenizer_params=TokenizerParams(), # 一般默认配置使用tokenizer即可
        tokenizer_init_params=TokenizerInitParams(), # 一般默认配置加载tokenizer即可
        model_init_params=ModelInitParams(), # 一般默认配置加载模型即可，使用vllm时不会生效
        vllm_params=VLLMParams(
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ), # vllm加载时的参数
        sampling_params=VllmSamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty
        ), # vllm的推理配置
        api_config=ApiConfig(
            api_key=args.api_key,
            url=args.url,
        ), # api配置
        apply_chat_template=True
        # 是否使用chat_template True会在推理的时候自动将转化message为text
    )

# 2. 初始化推理实例
inference = ModelInference(config, task_name="示例任务")

# 3. 单条推理
result = inference.infer("给我讲一个关于猫的笑话。")
print(result)

# 4. 批量推理
prompts = ["请翻译：Hello World","写一首关于春天的诗"]
results = inference.infer(prompts)
for r in results:
    print(r)
```

## 配置说明

### InferenceConfig：核心配置类，主要字段：

* `model_name`: 模型名称（如 gpt2、llama3 等）
* `model_path`: 模型路径,为到达模型文件夹的**前一级**目录
* `chat_type`: 推理模式（"classical"/"vllm"/"api"）
* `model_type`: 模型类型（如qwen/llama）
* `device_config` 设备配置
* `generation_params`: 生成参数 (max\_new\_tokens, temperature, top\_p 等)
* `tokenizer_params`: 分词器参数 (padding, return\_tensors 等)
* `tokenizer_init_params` 分词器加载参数（padding）
* `model_init_params`: 模型加载参数 (trust\_remote\_code, cache\_dir 等)
* `vllm_params`: vLLM 特有加载参数 用于加载LLM类
* `sampling_params`: vLLM 采样参数 VllmSamplingParam,不要和vllm的SamplingParam混淆
* `api_config`, API模式的配置
* `apply_chat_template`: 是否对消息应用模板
* `log_dir`: 日志记录的路径

---

### GenerationParams: 模型生成参数

* `max_new_tokens` 最多添加的token
* `temperature` 推理温度
* `top_p` Nucleus sampling 截断概率，默认1
* `top_k` Top-K 筛选大小，原函数默认50
* `num_beams` beam_search的参数 默认1，不启用
* `length_penalty` 长度惩罚，默认1.0
* `use_cache` 启用缓存，一般设置为True
* `repetition_penalty` 重复惩罚，默认1.0，不启用重复惩罚
* `repetition_penalty` 额外参数，不同模型可能支持额外的生成参数

---
  
### TokenizerParams：分词器生成时的参数

* `padding`（bool）  
  是否对 batch 中的样本进行 padding。
* `add_special_tokens`（bool）  
  是否在序列前后加入特殊标记（如 BOS/EOS）。
* `return_tensors`（str）  
  返回的张量格式，通常为 `"pt"`（PyTorch）。
* `additional`（Dict[str, Any]）  
  额外参数，用于扩展其他 Hugging Face 分词器的可选字段。

---

### TokenizerInitParams：分词器初始化时的参数

* `padding_side`（str）  
  填充方向，`"left"` 或 `"right"`。
* `trust_remote_code`（bool）  
  是否信任远程仓库中的自定义 tokenizer 代码。
* `cache_dir`（Optional[str]）  
  本地缓存目录，默认 `None`。
* `additional`（Dict[str, Any]）  
  额外参数，用于扩展其他初始化选项。

---

### ModelInitParams：模型加载时的参数

* `trust_remote_code`（bool）  
  是否信任远程仓库中的自定义模型代码。
* `torch_dtype`（Union[str, None]）  
  加载模型时使用的浮点类型，如 `"auto"`、`"float16"` 等。
* `cache_dir`（Optional[str]）  
  本地缓存目录，默认 `None`。
* `additional`（Dict[str, Any]）  
  额外参数，用于扩展 Hugging Face `from_pretrained` 的其他可选字段。

---

### VLLMParams：vLLM 加载时的参数

* `tensor_parallel_size`（int 或 `"auto"`）  
  张量并行大小，`"auto"` 时自动设为可用 GPU 数量。
* `gpu_memory_utilization`（float）  
  目标 GPU 内存利用率，通常 `0.8` 代表 80%。
* `dtype`（str）  
  数据类型，如 `"bfloat16"` 或 `"float16"`。
* `enforce_eager`（bool）  
  是否强制使用 Eager 模式加载。
* `enable_chunked_prefill`（bool）  
  是否开启分块预填充以减少显存峰值。
* `max_num_seqs`（int）  
  最大并发序列数。
* `additional`（Dict[str, Any]）  
  额外参数，用于扩展 vLLM 的其他接口字段。

---

### VllmSamplingParams：vLLM 采样参数

* `max_tokens`（int）  
  最多生成的 Token 数，原代码默认 1024。
* `temperature`（float）  
  采样温度，默认 1.0（越高越随机）。
* `top_p`（float）  
  核心采样（Nucleus sampling）的累计概率阈值，默认 1.0。
* `top_k`（int）  
  Top-K 采样规模，默认 50。
* `repetition_penalty`（float）  
  重复惩罚因子，默认 1.0（不惩罚）。
* `presence_penalty`（float）  
  新内容惩罚，默认 0.0。
* `frequency_penalty`（float）  
  频率惩罚，默认 0.0。
* `additional`（Dict[str, Any]）  
  额外字段，可传入 vLLM `SamplingParams` 支持的其他参数。

---

### DeviceConfig：设备配置

* `device_id`（int 或 List[int]）  
  指定单 GPU ID 或多 GPU ID 列表。
* `max_memory`（Any）  
  在多 GPU 模式下，为每张卡指定 `max_memory`（如 `{"0":"13GiB","1":"13GiB"}`）。
* **to_dict() 结果**  
  * 单 GPU：`{"device":"cuda:<id>"}`  
  * 多 GPU：`{"device_map":"auto","max_memory":<max_memory>}`

---

### ApiConfig：API 模式配置

* `api_key`（str）  
  OpenAI 或其他云 API 的访问密钥。
* `url`（str）  
  API 请求的基础 URL。
* `max_retries`（int）  
  最大重试次数，默认 3。
* `backoff_factor`（float）  
  重试的指数退避系数，默认 1.0。

## 日志

日志同时输出到控制台和 `utils/log/` 目录下的文件，同时支持自动日志目录。通过 `init_logging(task_name, log_dir, level)` 调整配置。

## 示例脚本

`example.py` 包含更完整的使用示例

---

**许可证**：MIT （虽然也没人会用就是了）
