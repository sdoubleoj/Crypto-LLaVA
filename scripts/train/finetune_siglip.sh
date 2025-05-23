#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=5
export NCCL_DEBUG=INFO 

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600

LLM_VERSION="Qwen/Qwen2.5-1.5B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

MY_DATA_PATH="data/instruction/train/merged_train_v0.2.json"
MY_IMAGE_FOLDER="Crypto-LLaVA/"
MID_RUN_NAME="finetune-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}"
OUTPUT_DIR="Crypto-LLaVA/checkpoints/${MID_RUN_NAME}"

echo "RUN NAME: ${MID_RUN_NAME}"

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="cryptollava-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

CKPT_PATH=$LLM_VERSION

torchrun --nproc_per_node=1 \
    llava/train/train_mem.py \
    --lora_enable True \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${MY_DATA_PATH} \
    --image_folder ${MY_IMAGE_FOLDER} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 384)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa
