from inference_module.model_inference import ModelInference
from inference_module.gpu_manager.gpu_manager import GPUManager
from inference_module.config.inference_config import *
from inference_module.utils.json_util import JsonUtil
from inference_module.utils.data_util import DataUtil

input_single="Say Hi~"

input_list=["介绍一下C++是什么样的编程语言","介绍一下Java是什么样的编程语言"]


apply_chat_template=True

def print_result(input):
    if isinstance(input,str):
        print(input)
    elif isinstance(input,list):
        for msg in input:
            print(msg)

def vanilla_infer():
    config = InferenceConfig(
        model_name="llama32-3B-ins",
        model_path="/root/co-teaching/Model/BaseModel/",
        chat_type="classical",
        device_config=DeviceConfig(0,40),
        generation_params=GenerationParams(),
        tokenizer_params=TokenizerParams(),
        model_init_params=ModelInitParams(),
        apply_chat_template=apply_chat_template,
    )
    
    model_infer = ModelInference(config,"test",GPUManager(2))
    
    # 基础的单/多推理
    result = model_infer.infer(input_list)
    print_result(result)
    result = model_infer.infer(input_single)
    print_result(result)
    
    # 有chat_template的推理
    if apply_chat_template:
        messages=[model_infer.format_message(content) for content in input_list]
        inputs = model_infer.apply_chat_template(messages)
        result=model_infer.infer(inputs)
        print_result(result)
        message=model_infer.format_message(input_single)
        input=model_infer.apply_chat_template(message)
        result=model_infer.infer(input)
        print_result(result)
        
    
def api_infer():
    api_conflg = ApiConfig(
                api_key="sk-apikey",
                url = "https://api-platform"
            )
    config = InferenceConfig(
            model_name="gpt-4o",
            generation_params=GenerationParams(),
            chat_type="api",
            api_config= api_conflg
        )
    
    model_infer = ModelInference(config,"test",GPUManager(2))
     # 基础的单/多推理
    result = model_infer.infer([model_infer.format_message(content) for content in input_list])
    print_result(result)
    result = model_infer.infer(model_infer.format_message(input_single))
    print_result(result)
    
def vllm_infer():
    config = InferenceConfig(
        model_name="llama32-3B-ins",
        model_path="/root/co-teaching/Model/BaseModel/",
        chat_type="vllm",
        device_config=DeviceConfig(0,40),
        generation_params=GenerationParams(),
        tokenizer_params=TokenizerParams(),
        model_init_params=ModelInitParams(),
        vllm_params=VLLMParams(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5
        ),
        sampling_params=VllmSamplingParams(
            max_tokens=1024,
            temperature=1.0
        ),
        apply_chat_template=apply_chat_template,
    )
    model_infer = ModelInference(config,"test",GPUManager(2))
    # 基础的单/多推理
    result = model_infer.infer(input_list)
    print_result(result)
    result = model_infer.infer(input_single)
    print_result(result)
    
    # 有chat_template的推理
    if apply_chat_template:
        messages=[model_infer.format_message(content) for content in input_list]
        inputs = model_infer.apply_chat_template(messages)
        result=model_infer.infer(inputs)
        print_result(result)
        message=model_infer.format_message(input_single)
        input=model_infer.apply_chat_template(message)
        result=model_infer.infer(input)
        print_result(result)

def main():
    # vanilla_infer()
    # vllm_infer()
    api_infer()

if __name__ == "__main__":
    main()