from inference_module.model_activity import ModelActivity
from inference_module.gpu_manager import GPUManager
from inference_module.utility import Util
from inference_module.config import Config

import tqdm
import argparse
import ast
import traceback

def formate_dict_list_input(data:list[dict]):
    """适用于标准输出的提取文本 context_data.jsonl

    Args:
        data (list[dict]): 待处理
    """
    resutl_dict_list=list()
    for data_chunk in data:
        title=data_chunk["title"]
        contents=data_chunk["content"]
        docu_format=data_chunk["format"]
        
        
        
        srs_file_name=str(title+'.'+docu_format)
        for content in contents:
            resutl_dict_list.append(
                {
                    "Source_file":srs_file_name,
                    "Context":content
                }
            )
    Util.add_index_for_dict_lists(resutl_dict_list)
    return resutl_dict_list


def main():
    args=parse_args()
    generate_question(args)

def parse_args():
    parser = argparse.ArgumentParser(description="Model inference script with arguments for configuration and execution.")

    # 必需参数
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file (JSON format).")
    parser.add_argument("--few_shot_path", type=str, required=True, help="Path to the few-shot examples file (JSON format).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results (JSONL format).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    

    # 可选参数
    parser.add_argument("--model_name", type=str, default="default",help="Name of the model to use.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing data.")
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

def extract_list_from_output(output: str) -> list:
    """
    从字符串中提取并解析为列表类型，并移除其中的 `...`。
    """
    # 移除换行符并去掉多余空格，确保字符串格式正确
    cleaned_text = output.strip()

    try:
        # 使用 ast.literal_eval 解析为 Python 对象
        parsed_list = ast.literal_eval(cleaned_text)

        # 如果解析结果是列表，移除所有的 ellipsis
        if isinstance(parsed_list, list):
            filtered_list = [item for item in parsed_list if item is not ...]
            return filtered_list
        else:
            # print(f"解析结果不是列表类型，返回空列表。解析结果: {parsed_list}")
            return []
    except (ValueError, SyntaxError) as e:
        # print(f"解析出错: {e}，返回空列表。原始输出: {output}")
        return []

def generate_question(args):
    data=Util.read_json(file_path=args.data_path)
    prompts=Config.load_config("/home/workspace/fu_ziche/data_produce/prompts.yml")
    few_shot_data:list[dict]=Util.read_json(file_path=args.few_shot_path)
    
    data=formate_dict_list_input(data)
    
    split_data:list[list[dict]]=Util.split_data_as_batch(data=data,batch_size=args.batch_size)
    user_prompt_template:str=prompts["generate_q_1"]
    
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
        task_name=args.task_name)
    
    for data_chunk in tqdm.tqdm(split_data,desc="processing batch",unit="batch"):
        input_messages_list=[]
        try:
            selected_example_list=Util.random_sample_from_list(few_shot_data,len(data_chunk))
            for i in range(len(data_chunk)):
                data_block=data_chunk[i]
                
                user_prompt=user_prompt_template.format_map({
                        "Context":data_block["Context"],
                        "Source_file":data_block["Source_file"],
                        "Example_context_0":selected_example_list[i]["文档片段"],
                        "Example_query_0":selected_example_list[i]["问题"],
                    })
                system_prompt=prompts["system_1"]
                input_messages_list.append(
                    ModelActivity.format_message(
                        user_prompt=user_prompt,
                        sys_prompt=system_prompt
                    )
                )
            results=model_infer.run_group_inference(input_messages_list)
            full_result_list=[]
            for i in range(len(data_chunk)):
                current_data=data_chunk[i]
                result=results[i]["Result"]
                result_list=extract_list_from_output(result)
                result_processed=[]
                it_idx=0
                for it in result_list:
                    result_processed.append(
                        {   
                            "index":it_idx,
                            "question":it
                        }
                    )
                    it_idx+=1
                    
                full_result_list.append({
                    "Q_Model_name":config.get("model_name","None"),
                    "Context_Id":current_data["Index"],
                    "Source_file":current_data["Source_file"],
                    "Context":current_data["Context"],
                    "Few_shot":[selected_example_list[i]],
                    "QA":result_processed
                })

            Util.write_jsonlines(file_path=args.output_path,data=full_result_list)
        except :
            traceback.print_exc()
                
            


if __name__=="__main__":   
    main()