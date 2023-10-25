#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 run_llm_generation.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset_name paniniDot/sci_lay \
        --dataset_subset MBIO \
        --split test \
        --max_source_length 200  \
        --output_dir output_test_llm \
        --guidelines "keep it simple"
