# inference/api.py

import traceback
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
        self.client = OpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("url")
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

    def _generate_api_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        内部方法，用于调用 API 生成回答。参数设置参考原来的 generate_api_response 函数：
          - 使用 self.config 中的 model_generate_args（如 max_new_tokens, temperature, 等）
        :param messages: 消息列表，每条消息包含 role 和 content
        :return: API 返回的生成文本
        """
        model_generate_args: Dict[str, Any] = self.config.get("model_generate_args", {})
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,  # 使用指定模型，例如 GPT-4
                max_tokens=model_generate_args.get("max_new_tokens", 512),
                temperature=model_generate_args.get("temperature", 0.2),
                n=1,                   # 生成一个回答
                stop=model_generate_args.get("stop", None)
            )
            # 从响应中提取生成文本，假设返回格式中 choices[0].message.content 包含回答文本
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            if self.logger:
                self.logger.error("Error occurred when using API to generate response", exc_info=True)
            else:
                print("Error occurred when using API to generate response")
                traceback.print_exc()
            return ""
