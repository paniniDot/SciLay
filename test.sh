#!/bin/bash

N=3

#VARS=("A" "all" "B" "C" "CB" "I" "MBIO" "NC" "OTHER" "PLB" "PLCB" "PLGEN" "PLNTD" "PLPAT" "SD")

MODELS=("facebook/bart-large" "ccdv/lsg-bart-base-4096" "ccdv/lsg-pegasus-large-4096")

SOURCE_LENGTHS=(1024 4096 2048)

for ((i=0; i<N; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 run_generation.py \
        --logging online \
        --do_train \
        --do_eval \
        --do_predict \
        --output_dir output_test \
        --dataset_name paniniDot/sci_lay \
        --subset_name all \
        --task_name full_to_technical_summarization \
        --model_name_or_path "${MODELS[i]}" \
        --log_level error \
        --gradient_accumulation_steps 1 \
        --max_source_length ${SOURCE_LENGTHS[i]} \
        --max_target_length 512 \
        --generation_max_length 512 \
        --num_train_epochs 4 \
        --learning_rate 5e-5 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --gradient_checkpointing \
        --load_best_model_at_end \
        --predict_with_generate \
        --overwrite_cache \
        --metric_for_best_model eval_rouge1 \
        --save_total_limit 1 \
        --num_beams 5 \
        --generation_num_beams 2 \
        --group_by_length \
        --sortish_sampler \
        --weight_decay 0.01 \
        --label_smoothing_factor 0.1 \
        --include_inputs_for_metrics \
        --remove_unused_columns \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        #--max_eval_samples 10 \
        #--max_train_samples 100 \
        #--max_test_samples 10 &
    wait
done
