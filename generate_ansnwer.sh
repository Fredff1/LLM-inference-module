#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 输入数据路径
DATA_PATH="/home/workspace/data/New_Data/网络管控/网络管控_questions.jsonl"

OUTPUT_PATH="/home/workspace/data/New_Data/网络管控/网络管控_qa.jsonl"

# 模型配置
MODEL_PATH="/root/Models/Qwen2.5-72B-Instruct"
MODEL_NAME="default"
CHAT_TYPE="vllm"

# 参数配置

TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
MAX_TOKENS=1024
TEMPERATURE=0
SAMPLING_TIMES=1
DO_SAMPLE_FLAG="--do_sample"  # 设置采样标志（留空表示不使用采样）
TASK_NAME="网络管控_72b_q"

# API 配置（如有需要）
API_KEY="default"
URL="default"

# 运行脚本
/opt/miniconda3/envs/vllm/bin/python /home/workspace/fuziche/data_produce/generate_answer.py \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --chat_type "$CHAT_TYPE" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --sampling_times $SAMPLING_TIMES \
    $DO_SAMPLE_FLAG \
    --api_key "$API_KEY" \
    --url "$URL" \
    --task_name "$TASK_NAME"
