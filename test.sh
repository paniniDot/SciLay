#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 run_generation.py \
    --logging disabled \
    --perform_retrieval \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir output_test \
    --dataset_name paniniDot/sci_lay \
    --task_name text_summarization \
    --model_name_or_path facebook/bart-large \
    --log_level error \
    --gradient_accumulation_steps 1 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --generation_max_length 512 \
    --num_train_epochs 5 \
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
    #--max_test_samples 10 
