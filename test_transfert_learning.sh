#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 run_generation_transfert_learning.py \
    --logging disabled \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir output_test_tl \
    --dataset_name paniniDot/sci_lay \
    --subset_name MBIO \
    --task_name full_to_lay_transfert_summarization \
    --model_name_or_path ccdv/lsg-bart-large-4096 \
    --log_level error \
    --gradient_accumulation_steps 1 \
    --max_source_length 4096 \
    --max_target_length 512 \
    --generation_max_length 512 \
    --num_train_epochs 3 \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --fp16

