from model_inference import ModelInference
from config.inference_config import *
from gpu_manager.gpu_manager import GPUManager

def main():
    config = InferenceConfig(
        model_name="qwen2.5-0.5B",
        model_path="D:\\Files\\Models\\",
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

if __name__ == "__main__":
    main()