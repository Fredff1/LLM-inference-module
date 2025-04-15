from inference_module.model_activity import ModelActivity
from inference_module.gpu_manager import GPUManager
from inference_module.utility import Util
from inference_module.config import Config

import tqdm
import argparse
import ast
import traceback




def main():
    args=parse_args()
    generate_question(args)

def parse_args():
    parser = argparse.ArgumentParser(description="Model inference script with arguments for configuration and execution.")

    # 必需参数
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file (JSON format).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results (JSONL format).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    

    # 可选参数
    parser.add_argument("--model_name", type=str, default="default",help="Name of the model to use.")
   
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory to use.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation.")
    parser.add_argument("--sampling_times", type=int, default=1, help="Number of sampling iterations for generation.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling during generation (default: False).")
    parser.add_argument("--chat_type", type=str, default="vllm", choices=["vllm", "classical", "api", "pipeline"],
                        help="Type of chat interface to use.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for external services (if applicable).")
    parser.add_argument("--url", type=str, default=None, help="URL for the API service (if applicable).")
    parser.add_argument("--task_name",default=None,help="The task name of the infer")
    return parser.parse_args()



def generate_question(args):
    data=Util.read_jsonlines(file_path=args.data_path)
    prompts=Config.load_config("/home/workspace/fuziche/data_produce/prompts.yml")
    
    
    
    
    user_prompt_template:str=prompts["generate_answer_1"]
    
    manager=GPUManager(5)
    vllm_args=ModelActivity.format_vllm_args(
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    sample_para=ModelActivity.format_sampling_params(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sampling_times=args.sampling_times
    )

    model_generate_para=ModelActivity.format_model_generation_params(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample
    )
    
    config=ModelActivity.build_full_config(
        model_path=args.model_path,
        model_name=args.model_name,
        chat_type=args.chat_type,
        api_key=args.api_key,
        url=args.url,
        sampling_params=sample_para,
        vllm_args=vllm_args,
        model_generate_args=model_generate_para
    )
    
    
    
    model_infer=ModelActivity(
        manager.get_all_avai_gpu_id(),
        config=config,
        gpu_manager=manager,
        task_name=args.task_name
        )
    
    for data_chunk in tqdm.tqdm(data, desc="processing batch", unit="batch"):
        input_messages_list = []
        try:
            # 生成模型输入并记录 QA 来源
            for i, data_block in enumerate(data_chunk["QA"]):
                system_prompt = prompts["system_1"]
                
                user_prompt = user_prompt_template.format_map({
                    
                    "Context": list(data_chunk["Context"].values())[0],
                    "Example_context_0": data_chunk["Few_shot"][0]["文档片段"],
                    "Example_q_0": data_chunk["Few_shot"][0]["问题"],
                    "Example_a_0": data_chunk["Few_shot"][0]["答案"],
                    "Question": data_block["question"]
                })
                input_messages_list.append(
                    ModelActivity.format_message(
                        user_prompt=user_prompt,
                        sys_prompt=system_prompt
                    )
                )
            

            # 调用模型进行推理
            results = model_infer.run_group_inference(input_messages_list)
            full_result_list=[]
            model_name=config.get("model_name")
            
            current_data=data_chunk

            list_to_add=current_data["QA"]
            for j in range(len(results)):
                list_to_add[j][f"{model_name}"]=results[j]["Result"]
            list_to_add=Util.remove_dict_with_none(list_to_add)
                
            full_result_list.append({
                "Q_Model_name":current_data["Q_Model_name"],
                "Context_Id":current_data["Context_Id"],
                "Source_file":current_data["Source_file"],
                "Context":current_data["Context"],
                "Few_shot":current_data["Few_shot"],
                "QA":list_to_add
            })

            # 写入结果
            Util.write_jsonlines(file_path=args.output_path, data=full_result_list)
        except Exception as e:
            traceback.print_exc()
                
            


if __name__=="__main__":   
    main()