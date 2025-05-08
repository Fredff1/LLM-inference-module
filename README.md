# LLM 推理模块

一个模块化、可扩展的框架，用于在多种后端上运行大语言模型（LLM）推理：

* 本地 Hugging Face 模型
* vLLM 引擎
* OpenAI API

## 功能特性

* **统一配置**：使用 `FullConfig` 数据类管理模型名称、路径、生成参数、分词器参数、初始化参数、vLLM 参数及 API 参数。
* **设备管理**：通过 `DeviceConfig` 支持单 GPU 或多 GPU 设置，自动配置 `device_map` 和 `max_memory`。
* **消息工具**：`format_message` 构造对话消息，`merge_messages` 根据需要合并消息列表。
* **模块化推理引擎**：`classical`（Hugging Face）、`vllm`、`api` 三种推理器实现，统一继承 `BaseInference`。
* **顶层封装**：`ModelInference` 类提供一致的单条（`run_inference`）和批量（`run_group_inference`）推理接口。
* **日志功能**：通过 `utils/logger.py` 初始化日志，记录运行信息和配置详情。

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/llm-inference-module.git
cd llm-inference-module

# 安装依赖
pip install -r requirements.txt
```

> **可选安装:**
>
> * vLLM 支持：`pip install vllm

## 目录结构

```
llm-inference-module/
├── config/             # 配置数据类
│   ├── full_config.py  # FullConfig 及子配置类
│   └── inference_config.py # DeviceConfig, InferenceConfig
│
├── utils/              # 通用工具
│   ├── logger.py       # 日志初始化与记录
│   └── message_utils.py# 消息构造与合并函数
│
├── inference/          # 推理器实现
│   ├── base.py         # BaseInference 抽象基类
│   ├── classical.py    # ClassicalInference（HF 本地推理）
│   ├── vllm.py         # VLLMInference（vLLM 推理）
│   ├── api.py          # APIInference（OpenAI API 推理）
│   └── factory.py      # create_inference_engine 工厂函数
│
├── model_inference/    # 顶层封装
│   └── model_inference.py # ModelInference 类
│
└── examples/           # 使用示例脚本
    └── example.py      # 快速入门示例
```

## 快速开始

```python
from config.full_config import FullConfig
from model_inference.model_inference import ModelInference

# 1. 构建配置对象
config = FullConfig(
    model_name="gpt2",
    model_path="gpt2",
    chat_type="classical",       # 可选 "vllm" 或 "api"
    apply_chat_template=False     # 是否使用聊天模板拼接
)

# 2. 初始化推理实例
infer = ModelInference(config, task_name="示例任务")

# 3. 单条推理
result = infer.run_inference("给我讲一个关于猫的笑话。")
print(result)

# 4. 批量推理
prompts = ["请翻译：Hello World","写一首关于春天的诗"]
results = infer.run_group_inference(prompts)
for r in results:
    print(r)
```

## 配置说明

* **FullConfig**：核心配置类，主要字段：

  * `model_name`: 模型名称（如 gpt2、llama3 等）
  * `model_path`: 模型路径或 Hugging Face 名称
  * `chat_type`: 推理模式（"classical"/"vllm"/"api"）
  * `generation_params`: 生成参数 (max\_new\_tokens, temperature, top\_p 等)
  * `tokenizer_params`: 分词器参数 (padding, return\_tensors 等)
  * `model_init_params`: 模型加载参数 (trust\_remote\_code, cache\_dir 等)
  * `vllm_params`: vLLM 特有加载参数
  * `sampling_params`: vLLM 采样参数
  * `api_key`, `url`: API 模式下的密钥和请求地址
  * `apply_chat_template`: 是否对消息应用模板

* **DeviceConfig**: 单独在 InferenceConfig 中管理GPU：

  * `device_id`: 整数或整数列表
  * `max_memory`: dict 或字符串，用于 `device_map` 模式

## 日志

日志同时输出到控制台和 `utils/log/` 目录下的文件，通过 `init_logging(task_name, log_dir, level)` 调整配置。

## 示例脚本

`examples/example.py` 包含更完整的使用示例，可直接运行查看效果。

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交并推送分支
4. 提交 PR 并描述变更内容

---

**许可证**：MIT
