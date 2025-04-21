import random

class DataUtil:
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