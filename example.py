from inference_module.model_inference import ModelInference
from inference_module.gpu_manager.gpu_manager import GPUManager
from inference_module.config.inference_config import *
from inference_module.utils.json_util import JsonUtil
from inference_module.utils.data_util import DataUtil


def vanilla_infer():
    config = InferenceConfig(
        model_name="qwen2.5-0.5B",
        model_path="/mnt/data/models/",
        chat_type="classical",
        device_config=DeviceConfig(0,40),
        generation_params=GenerationParams(),
        tokenizer_params=TokenizerParams(),
        model_init_params=ModelInitParams(),
        apply_chat_template=False,
    )
    
    model_infer = ModelInference(config,"test",GPUManager(2))
    result = model_infer.infer(["介绍一下C++是什么样的编程语言","介绍一下Java是什么样的编程语言"])
    print(f"Infer:{result}")
    
def api_infer():
    config = InferenceConfig(
            model_name="qwen2.5-0.5B",
            generation_params=GenerationParams(),
            api_config=ApiConfig(
                api_key="sk-your-api-key",
                url = "http//api.com"
            )
        )
    
    model_infer = ModelInference(config,"test",GPUManager(2))
    result = model_infer.infer(["介绍一下C++是什么样的编程语言","介绍一下Java是什么样的编程语言"])
    print(f"Infer:{result}")
    
def vllm_infer():
    config = InferenceConfig(
        model_name="qwen2.5-0.5B",
        model_path="/mnt/data/models/",
        chat_type="classical",
        device_config=DeviceConfig(0,40),
        generation_params=GenerationParams(),
        tokenizer_params=TokenizerParams(),
        model_init_params=ModelInitParams(),
        vllm_params=VLLMParams(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        ),
        sampling_params=VllmSamplingParams(
            max_tokens=1024,
            temperature=1.0
        ),
        apply_chat_template=False,
    )
    model_infer = ModelInference(config,"test",GPUManager(2))
    result = model_infer.infer(["介绍一下C++是什么样的编程语言","介绍一下Java是什么样的编程语言"])
    print(f"Infer:{result}")

def main():
    vanilla_infer()
    api_infer()
    vllm_infer()

if __name__ == "__main__":
    main()