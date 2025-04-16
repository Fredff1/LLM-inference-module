# utils/message_utils.py

from typing import List, Union, Dict, Any
from argparse import ArgumentError

def format_message(user_prompt: Union[str, List[str]],
                   sys_prompt: str = "",
                   assist_prompt: str = "") -> List[Dict[str, Any]]:
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
        raise ArgumentError("不合法的用户输入，你必须指定一个有效的用户输入作为 prompt")
    
    # 如果 user_prompt 为列表，则对每个文本构造单独的消息
    if isinstance(user_prompt, list):
        for prompt in user_prompt:
            messages.append({"role": "user", "content": prompt})
    else:
        messages.append({"role": "user", "content": user_prompt})
    
    if assist_prompt:
        messages.append({"role": "assistant", "content": assist_prompt})
    
    return messages


def merge_messages(messages: List[Dict[str, Any]], 
                   apply_chat_template: bool,
                   tokenizer: Any = None) -> str:
    """
    合并消息列表为最终推理使用的文本输入。
    
    参数：
      - messages: 预先格式化的消息列表，每条消息包含 role 与 content。
      - apply_chat_template: 是否应用聊天模板。
      - tokenizer: 分词器实例，如果需要调用 tokenizer.apply_chat_template，则必须传入该参数。
    
    返回：
      - str: 最终合并后的文本输入。
    
    说明：
      1. 当 apply_chat_template 为 False 或推理模式为 API 时，采用简单拼接形式。
      2. 当 apply_chat_template 为 True 时，必须提供 tokenizer，
         将调用 tokenizer.apply_chat_template 方法得到合并结果。
    """
    # 首先进行简单的角色验证（也可以在另一个 util 函数中实现完整验证）
    if not messages or not all("role" in msg and "content" in msg for msg in messages):
        raise ValueError("消息列表格式错误，必须包含 role 与 content")
    
    if not apply_chat_template:
        # 简单拼接，将每个消息按 role: content 格式合并成字符串
        output_text = ""
        for msg in messages:
            output_text += f"{msg['role']}\n{msg['content']}\n"
        return output_text
    else:
        # 如果需要应用模板，则必须提供 tokenizer，并调用其 apply_chat_template 方法
        if tokenizer is None:
            raise ValueError("启用了 apply_chat_template，但未提供 tokenizer 实例")
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
