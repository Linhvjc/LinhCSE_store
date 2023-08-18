#!/bin/bash
python evaluation.py \
    --model_name_or_path ./runs/scratch-listmle-bert-base-uncased \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test