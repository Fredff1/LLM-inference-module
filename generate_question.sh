#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

# 输入数据路径
DATA_PATH="/home/workspace/Data/ICT/New_Data/SDWAN/ICT_SDWAN_output.json"
FEW_SHOT_PATH="/home/workspace/fu_ziche/data_produce/useful_QA.json"
OUTPUT_PATH="/home/workspace/Data/ICT/New_Data/SDWAN/SDWAN_questions_1.jsonl"

# 模型配置
MODEL_PATH="/home/workspace/Models/Qwen2.5-72B-Instruct"
MODEL_NAME="default"
CHAT_TYPE="vllm"

# 参数配置
BATCH_SIZE=64
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
MAX_TOKENS=2048
TEMPERATURE=0.4
SAMPLING_TIMES=1
DO_SAMPLE_FLAG="--do_sample"  # 设置采样标志（留空表示不使用采样）
TASK_NAME="q_SDWAN_questions"

# API 配置（如有需要）
API_KEY="default"
URL="default"

# 运行脚本
/opt/miniconda3/bin/python /home/workspace/fu_ziche/data_produce/generate_question.py \
    --data_path "$DATA_PATH" \
    --few_shot_path "$FEW_SHOT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --chat_type "$CHAT_TYPE" \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --sampling_times $SAMPLING_TIMES \
    $DO_SAMPLE_FLAG \
    --api_key "$API_KEY" \
    --url "$URL" \
    --task_name "$TASK_NAME"
