from inference_module.inference.base import BaseInference

from typing import Optional,Dict,List

class MockInference(BaseInference):
    

    def initialize(self):
        pass

    def run(self, input_content: str) -> str:
        return f"Mock[{input_content}]"

    def run_batch(self, input_contents: List[str]) -> List[str]:
        return [
                self.run(i)
                for i in input_contents
            ]
    
    def validate_input(self, input_content):
        if isinstance(input_content,list):
            return input_content, "list"
        else:
            return input_content, "single"
