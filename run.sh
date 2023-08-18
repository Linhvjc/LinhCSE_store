#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 \
# TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface/transformers \
# HF_DATASETS_CACHE=$SCRATCH/RankCSE/data/ \
# HF_HOME=$SCRATCH/.cache/huggingface \
# XDG_CACHE_HOME=$SCRATCH/.cache \
# TRANSFORMERS_OFFLINE=1 \
# HF_DATASETS_OFFLINE=1 \
python train.py \
    --model_name_or_path vinai/phobert-base-v2 \
    --train_file data/wiki_all_2018.txt \
    --output_dir runs/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --load_best_model_at_end \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    --first_teacher_name_or_path VoVanPhuc/sup-SimCSE-VietNamese-phobert-base \
    --second_teacher_name_or_path keepitreal/vietnamese-sbert \
    --distillation_loss listmle \
    --alpha_ 0.33 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05 \
    --num_sample_train 100000