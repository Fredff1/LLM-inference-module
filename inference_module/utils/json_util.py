import json
import jsonlines
import re

class JsonUtil:

    @staticmethod
    def read_json(file_path:str,mode="r"):
        """读取 JSON 文件"""
        with open(file_path, mode, encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def write_json(file_path:str,data:list[dict],mode="w"):
        """写入 JSON 文件"""
        with open(file_path, mode, encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

            
    @staticmethod
    def read_json_from_answer(text:str,log_error:bool = False)->list[dict] | None:
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
                    if log_error:
                        print(f"解析 JSON 出错: {e},跳过这一项")
                    
        else:
            if log_error:
                print("未找到有效的 JSON 格式内容")
        return json_list
    
    @staticmethod
    def read_first_json_from_answer(text:str,log_error:bool = False)->dict | None:
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
                    if log_error:
                        print(f"解析 JSON 出错: {e},跳过这一项")
                    
        else:
            if log_error:
                print("未找到有效的 JSON 格式内容")
        if len(json_list)>0:
            return json_list[0]
        else:
            return None        

    @staticmethod
    def read_jsonlines(file_path,mode="r"):
        data = []
        with open(file_path, mode, encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        return data

    @staticmethod
    def write_jsonlines(file_path, data,mode="a"):
        with open(file_path, mode, encoding="utf-8") as w:
            for i in data:
                json.dump(i, w, ensure_ascii=False)
                w.write('\n')
