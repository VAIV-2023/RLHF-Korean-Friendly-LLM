#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT


deepspeed --num_gpus 1 main.py --model_name_or_path EleutherAI/polyglot-ko-1.3b \
   --data_path local/jsonfile --data_split 0,1,0 \
   --output_dir ./model-output --seed 1 \
   --lora_dim 8 --gradient_checkpointing \
   --per_device_train_batch_size 16 \
   --num_padding_at_beginning 0 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log

#   --data_path local/jsonfile --data_split 1,5,5 \