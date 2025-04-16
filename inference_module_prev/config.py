# modules/config.py
import yaml
import argparse

class Config:
    @staticmethod
    def load_config(file_path='/home/workspace/fu_ziche/Code/LLM_QA/yml/config.yml'):
        """加载 YAML 配置文件，使用绝对路径（没有提供则为默认的config文件）"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
        
    @staticmethod
    def load_config_special(file_name="config.yaml"):
        """加载 YAML 配置文件,使用默认路径，在默认路径下寻找目标名的文件"""
        default_path_prefix='/home/workspace/fu_ziche/Code/LLM_QA/yml/'
        with open(default_path_prefix+file_name, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
        
    @staticmethod
    def parse_args(*args):
        """
        解析命令行参数，优先级高于配置文件
        ## 这个函数目前没有实现！请勿使用。
        """
        pass
    
