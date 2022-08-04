#!/bin/bash

python3 main.py --pname baselines --run_mode local \
                 --algo rise-rep --dataset computers --bbm dnn --bbm_path computers_dnn_ce.ckpt \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100 \
                 --run_eval True \
                 --run_id 1