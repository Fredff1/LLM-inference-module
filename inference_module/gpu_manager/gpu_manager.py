# modules/gpu_manager.py
import torch
import os

class GPUManager:
    def __init__(self,required_memory_gb=10.0, ignored_gpus:int|list[int]=None):
        self.available_gpus = []
        self.ignored_gpus = ignored_gpus if ignored_gpus is not None else []  # Default is not to ignore any GPUs
        self.required_memory_gb=required_memory_gb
        self.refresh_available_gpus(self.required_memory_gb)
        
    def set_max_memory_usage(self,proportion=0.9):
        for id in self.available_gpus:
            dev=self.get_device_from_id(id)
            torch.cuda.set_per_process_memory_fraction(proportion,device=dev)
    
    def get_gpu_free_memory_GB(self,gpu_id:int):
        """根据gpu的id得到他目前可用显存大小"""
        result = os.popen(f"nvidia-smi --id={gpu_id} --query-gpu=memory.free --format=csv,noheader,nounits").read().strip()
        free_memory_mb = int(result)
        free_memory_gb = free_memory_mb / 1024
        return free_memory_gb

    def get_max_gpu_count(self):
        """得到设备上所有可用gpu数量"""
        return torch.cuda.device_count()
    
    def get_max_memory_map(self,device_id:int|list[int])->dict[int,str]:
        """根据内置的所需最小显存，设置一个memroy_map,其中不可用的设备的最大显存使用将被设置为0"""
        max_memory=dict()
        for i in range(self.get_max_gpu_count()):
            max_memory[i]=f"{0}GiB"
            
        if isinstance(device_id,int):
            target_max_mem=self.get_gpu_free_memory_GB(device_id)*0.9
            max_memory[device_id]=f"{target_max_mem}GiB"   
        elif isinstance(device_id,list):
            for id in device_id:
                target_max_mem=self.get_gpu_free_memory_GB(id)*0.9
                max_memory[id]=f"{target_max_mem}GiB"
        return max_memory
            
    
    def refresh_available_gpus(self, required_memory_gb,max_mem_proportion=0.9):
        """根据内存需求更新可用 GPU 列表"""

        self.available_gpus.clear()
        num_gpus = torch.cuda.device_count()
        
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            all_gpus = [int(id) for id in cuda_visible_devices.split(",")]
        else:
            all_gpus = list(range(torch.cuda.device_count()))
        for gpu_id in all_gpus:
            if gpu_id not in self.ignored_gpus and self.check_gpu_memory(gpu_id, required_memory_gb):
                self.available_gpus.append(gpu_id)
        # print("-------------------------------------------------------")
        # print("可用GPU已经更新,包括以下GPU")
        # for gpu_id in self.available_gpus:
        #     print("GPU ",gpu_id,"目前可用")
        # print("-------------------------------------------------------")
        if not self.available_gpus:
            raise RuntimeError("No sufficient GPUs")

    def check_gpu_memory(self, gpu_id, required_memory_gb):
        """根据输入的gpu id(从0开始)检查某个 GPU 是否有足够的可用内存"""
        free_memory_gb=self.get_gpu_free_memory_GB(gpu_id)
        
        flag=free_memory_gb > required_memory_gb
        if flag:
            print(gpu_id,"号GPU有空余显存",free_memory_gb,"GB,需要显存",required_memory_gb,"GB,目前可用")
        else:
            print(gpu_id,"号GPU有空余显存",free_memory_gb,"GB,需要显存",required_memory_gb,"GB,目前不可用")
        
        return flag

    def get_next_available_gpu_id(self):
        """获取下一个可用的 GPU的id ，返回设备的id(从0开始)"""
        if not self.available_gpus:
            raise RuntimeError("No available GPU.")
        return self.available_gpus.pop(0)  

    def get_next_available_gpu(self):
        """获取下一个可用的 GPU 设备，返回完整设备名"""
        if not self.available_gpus:
            raise RuntimeError("No available GPU.")
        id=self.available_gpus.pop(0)
        return self.get_device_from_id(id) 
    
    def get_device_from_id(self,gpu_id):
        """根据GPU的id获得具体设备名"""
        if isinstance(gpu_id,int):
            device = torch.device(f"cuda:{gpu_id}")
        elif isinstance(gpu_id,list):
            device=[]
            for id in gpu_id:
                device.append(torch.device(f"cuda:{id}"))
        else :
            raise(ValueError("不合法的设备"))
        return device
    
    def get_all_avai_gpu(self)->list:
        """得到一个包含所有可用gpu设备名的列表
        ### 注意：不是id列表"""
        dev_list=[]
        for id in self.available_gpus:
            dev_list.append(self.get_device_from_id(id))
        return dev_list
    
    def get_all_avai_gpu_id(self)->list[int]:
        """得到一个包含所有可用gpu的id的列表"""
        return self.available_gpus
    
    def add_ignored_gpus(self, gpu_ids:int|list[int]):
        if isinstance(gpu_ids, int):
            self.ignored_gpus.append(gpu_ids)
        elif isinstance(gpu_ids, list):
            self.ignored_gpus.extend(gpu_ids)
        # print(f"当前忽略的 GPU 列表: {self.ignored_gpus}")
        self.refresh_available_gpus(self.required_memory_gb)
        
    def remove_ignored_gpus(self, gpu_ids:int|list[int]):
        if isinstance(gpu_ids, int):
            self.ignored_gpus.remove(gpu_ids)
        elif isinstance(gpu_ids, list):
            for gpu_id in gpu_ids:
                self.ignored_gpus.remove(gpu_id)
        # print(f"已更新忽略的 GPU 列表: {self.ignored_gpus}")
        self.refresh_available_gpus(self.required_memory_gb)


