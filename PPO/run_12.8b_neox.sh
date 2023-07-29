#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_ZERO_STAGE="--actor_zero_stage 0"
CRITIC_ZERO_STAGE="--critic_zero_stage 0"

OUTPUT="./output"

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=0
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=0
fi

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=5e-4
Critic_Lr=5e-6

mkdir -p $OUTPUT

deepspeed --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --local_rank 4 \
   --actor_model_name_or_path Trofish/KULLM-SFT-v1 \
   --critic_model_name_or_path Trofish/KULLM-SFT-v1 \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 8 \
   --per_device_mini_train_batch_size 8 \
   --generation_batch_numbers 1 \
   --release_inference_cache \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 16 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} \
   --actor_lora_dim 128 \
   --critic_lora_dim 128 \
   --print_answers \
   --disable_actor_dropout \
   --enable_hybrid_engine \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --enable_tensorboard \
   --tensorboard_path /content/drive/MyDrive/output/tensorboard.log \
   --output_dir $OUTPUT \
    2>&1 | tee "$OUTPUT/training.log"
