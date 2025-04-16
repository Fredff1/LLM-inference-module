from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizer,PreTrainedModel,pipeline,GenerationConfig
import torch
import io
import json
import jsonlines
import re
import random
import os
import yaml

    

#######################################
# 修改这个文件的时候，请同步所有的修改
#######################################

class Util:
    
    @staticmethod
    def add_index_for_dict_lists(data:list[dict],start_index=0)->None:
        """给一个列表中的字典添加Index键"""
        index=start_index
        for data_chunk in data:
            data_chunk["Index"]=index
            index+=1
            
    @staticmethod
    def merge_two_dicts(data_1:list[dict],data_2:list[dict])->list[dict]:
        """将两个字典列表合并为一个，对于相同的键，只保留第一个参数中对应的值"""
        result_data=[]
        for sft_data, base_data in zip(data_1, data_2):
            # 过滤 sft_data 中的键，不在 base_data 中的键值对
            filtered_dict = {k: v for k, v in sft_data.items() if k not in base_data}
            
            # 复制 base_data 并更新过滤后的键值对
            target_data = base_data.copy()  # 使用 copy() 以避免修改原始字典
            target_data.update(filtered_dict)  # 使用 update 而不是 append
            
            # 将更新后的数据添加到 data_to_score 列表中
            result_data.append(target_data)
        return result_data
    
    @staticmethod
    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: dict,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
    ):
        """
        Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        
        """
        # 添加特殊标记到tokenizer中，并调整模型的embedding层大小。
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        # 如果添加了新的特殊标记，则对输入和输出的embedding进行初始化。
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        print("enbedding层调整完成")
        
    
    @staticmethod
    def handle_missing_tokens_qwen(tokenizer: PreTrainedTokenizer,model: PreTrainedModel,model_path:str):
        model.eval()
        model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
        model.generation_config.eos_token_id = [151645, 151643]
        model.generation_config.pad_token_id = 151643
        model.generation_config.do_sample = False
        
    @staticmethod
    def handle_missing_tokens_and_resize_embeddings(tokenizer: PreTrainedTokenizer,
                                                    model: PreTrainedModel,model_type="llama"):
        """
        ## 函数功能
        LLama适用的补全缺失id的
        (1) 处理缺失的标记并调整embedding
        """
        # 定义了一些用于标记的默认字符串。
        IGNORE_INDEX = -100
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        print("正在补全token")
        special_tokens_dict = dict()
        #定义特殊标记的字典。
        if tokenizer.pad_token is None:
            print("检测到pad_token缺失")
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            print("检测到eos_token缺失")
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            print("检测到bos_token缺失")
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            print("检测到unk_token缺失")
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None :
            model.config.eos_token_id = tokenizer.eos_token_id
        
        print("已将缺失标记设置为默认值，正在自动调整模型")
        #调整tokenizer和模型的embedding层大小。
        Util.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        print("模型已通过检测")
        
    @staticmethod
    def validate_file(file:str|io.IOBase, mode: str):
        """确保传入对象是一个IO对象，如果不是，则自动打开"""
        if not isinstance(file, io.IOBase):
            file = open(file, mode=mode)
        return file
    
    @staticmethod
    def read_json(file_path:str,mode="r"):
        """读取 JSON 文件"""
        with open(file_path, mode, encoding='utf-8') as file:
            print(f"\n从{file_path}读入需要的数据")
            return json.load(file)

    @staticmethod
    def write_json(file_path:str,data:list[dict],mode="w"):
        """写入 JSON 文件"""
        with open(file_path, mode, encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            print(f"\n已经将数据写入{file_path}")
            
    @staticmethod
    def read_json_from_answer(text:str)->list[dict] | None:
        """读取回答中的所有json格式的内容，以字典列表形式输出"""
        # 删除原始的 input_text 以避免重复
        # print(f"生成文本{generated_text}")
        # print(f"输入长度为{len(input_text)}")
        # print("------------------------------------------------------")
        # 匹配 JSON 内容
        all_matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        if len(all_matches) >= 1:
            json_match=all_matches[0]
        else:
            json_match=None
        json_list=[]
        if json_match:
            for each_mach in all_matches:
                try:
                    # 将字符串解析为 JSON 对象
                    json_data = json.loads(each_mach)
                    # print("匹配到的 JSON 内容")
                    json_list.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 出错: {e},跳过这一项")
                    
        else:
            print("未找到有效的 JSON 格式内容")
        return json_list
    
    @staticmethod
    def read_jsonlines(file_path,mode="r"):
        data = []
        with open(file_path, mode, encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        print(f"\n成功从{file_path}读入数据")
        return data
    @staticmethod
    def write_jsonlines(file_path, data,mode="a"):
        with open(file_path, mode, encoding="utf-8") as w:
            for i in data:
                json.dump(i, w, ensure_ascii=False)
                w.write('\n')
        print(f"\n成功向{file_path}写入数据")
                
    
    @staticmethod
    def random_sample_from_list(data_list:list[dict], sample_size:int):
        """
        从list[dict]中随机抽取指定数量的元素，并返回一个新的list[dict]。
        
        :param data_list: 原始的list[dict]
        :param sample_size: 需要抽取的元素数量
        :return: 随机抽取的list[dict]
        """
        if sample_size > len(data_list):
            sample_size=len(data_list)
            print(f"抽取序列不能长于列表{len(data_list)}")
        
        return random.sample(data_list, sample_size)
    
    @staticmethod
    def read_prompt_templates():
        """
            请读取yml中的prompt template文件作为输入
            
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 拼接相对路径，假设YAML文件保存在名为'config'的文件夹下
        yaml_file_path = os.path.join(current_dir, 'yml', 'prompt_template.yml')
        
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            
        return data["templates"]
    
    @staticmethod
    def split_data_as_batch(data:list[dict],batch_size:int=16)->list[list[dict]]:
        start_index=0
        result=[]
        while start_index<len(data):
            end_index=min(start_index+batch_size,len(data))
            data_chunk=data[start_index:end_index]
            result.append(data_chunk)
            start_index+=batch_size
            
        return result
    
    @staticmethod
    def remove_key_from_dict_list(dict_list, key_to_remove):
        """
        从字典列表中删除指定的键及其对应的值。
        在原来的dict_list上直接修改
        
        :param dict_list: 字典组成的列表
        :param key_to_remove: 需要删除的键
        :return: 删除键后的新字典列表
        """
        # 遍历字典列表，删除每个字典中指定的键
        for dictionary in dict_list:
            if key_to_remove in dictionary:
                del dictionary[key_to_remove]
        return dict_list
    
    @staticmethod
    def getCurrentPythonDir(sub_path:str=None) -> str:
        """
        从当前python脚本正在运行的路径开始寻找文件。
        如果提供了sub_path参数，将返回该子路径的完整路径。

        :param sub_path: 可选的子路径(不要包含开始的/以免被识别为绝对路径)
        :return: 当前脚本所在路径或包含子路径的完整路径
        """
        # 获取当前 Python 脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(current_dir,sub_path)

        # 如果提供了sub_path，拼接当前目录和子路径
        if sub_path:
            return os.path.join(current_dir, sub_path)

        return current_dir
    @staticmethod
    def remove_dict_with_none(data_list: list[dict]) -> list[dict]:
        return [item for item in data_list if all(value is not None for value in item.values())]

    @staticmethod
    def load_config(file_path='/home/workspace/fu_ziche/Code/LLM_QA/yml/config.yml'):
        """加载 YAML 配置文件，使用绝对路径（没有提供则为默认的config文件）"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

def test_remove_dict_with_none():
    # 示例
    data = [
        {"name": "Alice", "age": 25, "city": "New York"},
        {"name": "Bob", "age": None, "city": "Los Angeles"},
        {"name": "Charlie", "age": 30, "city": "Chicago"},
        {"name": None, "age": 22, "city": "Miami"}
    ]

    filtered_data = Util.remove_dict_with_none(data)
    print(filtered_data)
        
    

def test_prompt_template():
    data=Util.read_prompt_templates()
    print(data)
    
if __name__=="__main__":
    print("开始测试Util的工作")
    # test_prompt_template()
    test_remove_dict_with_none()
    Util.read_jsonlines("/home/workspace/fu_ziche/Code/LLM_QA/Data/generated_answers/base_answers/base_answer_2.jsonl")
    Util.read_json("/home/workspace/fu_ziche/Code/LLM_QA/Data/input_questions/eval_dataset.jsonl")
    print("Pass Test")