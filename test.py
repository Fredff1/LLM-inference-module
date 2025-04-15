from inference_module.model_activity import ModelActivity
from inference_module.gpu_manager import GPUManager
from inference_module.utility import Util
from inference_module.config import Config

import multiprocessing


def main():
    # config = ModelActivity.build_vllm_config(
    #     model_path="/cpfs01/projects-HDD/cfff-da33bbc17f45_HDD/zjz_24110240108/workspace/Models/Llama-3.1-8B-Instruct",
    #     vllm_args=ModelActivity.format_vllm_args(
    #         gpu_memory_utilization=0.9
    #     )
    # )
    # model_infer = ModelActivity(0,config,GPUManager(60),"test_vllm")
    
    # reuslt = model_infer.run_inference(ModelActivity.format_message(user_prompt="什么是神经网络"))
    
    # print(reuslt)
    import multiprocessing

def worker(rank):
    print(f"Worker {rank} start")
    import cv2
    print(f"Worker {rank} done")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    procs = []
    for i in range(2):  # 模拟 tensor_parallel_size=2
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


# if __name__ == "__main__":
#     main()