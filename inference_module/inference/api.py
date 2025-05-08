# inference/api.py

import traceback
import unicodedata
import random
import time
from inference_module.inference.base import BaseInference
from typing import Dict, Any, List
from openai import OpenAI  # 确保已安装 openai 库

class APIInference(BaseInference):
    def initialize(self) -> None:
        """
        初始化 API 模式：
          - 根据配置创建 OpenAI 客户端
          - 设置模型名称等参数
        注意：API 模式不加载本地模型与分词器，因此不需要 device 管理配置
        """
        api_config=self.config["api_config"]
        self.api_config=api_config
        self.client = OpenAI(
            api_key=api_config.get("api_key"),
            base_url=api_config.get("url")
        )
        self.model_name = self.config["model_name"]

        

    def run(self, input_content:List[Dict[str, Any]]) -> str:
        """
        单条推理接口：对输入文本构造一个消息列表，并调用 API 获取生成文本
        :param input_text: 输入提示文本
        :return: 生成回答的文本
        """
        
        return self._generate_api_response(input_content)

    def run_batch(self, input_contents: List) -> List[str]:
        """
        批量推理接口：对多个输入文本分别调用 API 生成回答
        此处采用顺序调用 API 的方式，也可以考虑并发调用
        :param input_texts: 多条输入提示文本组成的列表
        :return: 每条输入对应的生成文本结果列表
        """
        results = []
        for content in input_contents:
            result = self.run(content)
            results.append(result)
        return results
    
    
    def validate_input(self, input):
        from inference_module.utils.message_utils import format_message

        # 仅接受 dict 或 List[dict]

        if isinstance(input, list) and input and all(isinstance(x, dict) for x in input):
            return input,"single"
        elif isinstance(input, list) and input and all(isinstance(x, list) for x in input):
            return input,"list"
        # 如果有人传 str，我们不自动 format——直接报错
        raise ValueError("API 模式只支持标准消息格式")
    
    def decode_unicode_response(content):
        content = unicodedata.normalize('NFKC', content)
        return content

    def _generate_api_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        内部方法，用于调用 API 生成回答。参数设置参考原来的 generate_api_response 函数：
          - 使用 self.config 中的 model_generate_args（如 max_new_tokens, temperature, 等）
        :param messages: 消息列表，每条消息包含 role 和 content
        :return: API 返回的生成文本
        """
        model_generate_args: Dict[str, Any] = self.config.get("model_generate_args", {})
        args: Dict[str, Any] = {
        "model":       self.model_name,
        "messages":    messages,
        "temperature": model_generate_args.get("temperature", 0.2),
        # ...
        }
        # 只在配置里明确写了 max_tokens 时，才传给 API
        if "max_tokens" in model_generate_args:
            args["max_tokens"] = model_generate_args["max_tokens"]
        else:
            args["max_tokens"]=2048
        # 只在配置里写了 stop 且不为 None 时传
        stop = model_generate_args.get("stop")
        if stop:
            args["stop"] = stop
            
        max_retries    = self.api_config.get("api_max_retries", 3)
        backoff_factor = self.api_config.get("api_backoff_base", 1.0)  # 秒
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**args)
                # 提取文本
                if isinstance(resp, str):
                    return resp
                return resp.choices[0].message.content

            except Exception as e:
                # 记录错误
                msg = f"[API attempt {attempt}/{max_retries}] Error: {e}"
                if self.logger:
                    self.logger.warning(msg, exc_info=True)
                else:
                    print(msg)
                    traceback.print_exc()

                # 如果已经是最后一次，返回空字符串或抛出
                if attempt == max_retries:
                    final_msg = f"API Failed after repeating {max_retries} times, aborting."
                    if self.logger:
                        self.logger.error(final_msg)
                    else:
                        print(final_msg)
                    return ""

                # 计算指数退避时间 + 随机抖动
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                # 可加入微小随机抖动，防止雪崩
                jitter = sleep_time * 0.1
                time_to_sleep = sleep_time + (jitter * (2 * random.random() - 1))
                if self.logger:
                    self.logger.info(f"Sleeping {time_to_sleep:.2f}s before retry…")
                time.sleep(time_to_sleep)
                # 进入下一次重试
                continue

