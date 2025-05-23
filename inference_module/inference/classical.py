# inference/classical.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference_module.inference.base import BaseInference
from typing import List

from inference_module.token_hanlder.token_hanlder import handle_missing_tokens



class ClassicalInference(BaseInference):
    

    
    def initialize(self) -> None:
        model_path = self.config["model_path"] + self.config["model_name"]
        model_init_args = self.config.get("model_init_args")
        tokenizer_init_args = self.config.get("tokenizer_init_args")
        device_config = self.config.get('device_config')
        
        if "device" in device_config:
            device_config = {}
        self.model = AutoModelForCausalLM.from_pretrained(model_path,  **device_config,**model_init_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_init_args)
        
        self.device_id = self.config.get('device_id')
        if isinstance(self.device_id,int):
            self.model.to(self.device_id)
        elif isinstance(self.device_id,list):
            pass
        else:
            raise ValueError("Invalid device id type.")
        handle_missing_tokens(self)
            

    def run(self, input_content: str) -> str:
        tokenizer_args = self.config.get("tokenizer_generate_args", {})
        model_generate_args = self.config.get("model_generate_args", {})
        inputs = self.tokenizer(input_content, **tokenizer_args)
        device = self.device_id if isinstance(self.device_id,int) else self.device_id[0]
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], 
                                          attention_mask=inputs["attention_mask"],
                                          **model_generate_args)
        generated_output = outputs[:, inputs["input_ids"].shape[-1]:]
        result_text = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)
        return result_text

    def run_batch(self, input_contents: List[str]) -> List[str]:
        tokenizer_args = self.config.get("tokenizer_generate_args", {})
        model_generate_args = self.config.get("model_generate_args", {})
        device = self.device_id if isinstance(self.device_id,int) else self.device_id[0]
        inputs = self.tokenizer(input_contents, **tokenizer_args)

        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.input_ids.ne(self.tokenizer.pad_token_id),
                **model_generate_args)
        generated_outputs = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
        results = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        return results
    
    def validate_input(self, input):
        from inference_module.utils.message_utils import format_message
        tpl = self.config.get("apply_chat_template", False)

        # 纯文本
        if isinstance(input, str):
            return input, "single"
        if isinstance(input, list) and input and all(isinstance(x, str) for x in input):
            return input, "list"

        
        raise ValueError(
            "Classical/VLLM 模式下，只支持 str 或 List[str]；"
        )
