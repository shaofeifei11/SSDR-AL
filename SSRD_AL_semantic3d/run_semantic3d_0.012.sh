#!/bin/bash
reg_strength=0.012
python -u partition/compute_superpoint_semantic3d.py --reg_strength ${reg_strength} > record_log/superpoint_distribution.log 2>&1

# 21
python -u ssdr_create_baseline.py --gpu 2 --dataset semantic3d --epoch 50 --lr_decay 0.90 --reg_strength ${reg_strength} > record_log/semantic3d_ssdr_log_baseline-50-0.90_${reg_strength}_2.txt 2>&1  # miou 0.725
python -u ssdr_create_seed.py --gpu 0 --dataset semantic3d --seed_percent 0.008 --reg_strength ${reg_strength} > record_log/semantic3d_ssdr_log_seed_${reg_strength}_2.txt 2>&1  # 4468

#
python -u ssdr_main_semantic3d.py --dataset semantic3d --reg_strength ${reg_strength} --t 4 --gpu 0 --round 2 --sampler random --oracle_mode dominant --min_size 5 >> record_log/semantic3d_ssdr_log_t4-random-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_semantic3d.py --dataset semantic3d --reg_strength ${reg_strength} --t 4 --gpu 2 --round 2 --sampler T --point_uncertainty_mode entropy --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5 >> record_log/semantic3d_ssdr_log_t4-entropy-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_semantic3d.py --dataset semantic3d --reg_strength ${reg_strength} --t 4 --gpu 3 --round 2 --sampler T --point_uncertainty_mode lc --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/semantic3d_ssdr_log_t4-lc-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_semantic3d.py --dataset semantic3d --reg_strength ${reg_strength} --t 4 --gpu 2 --round 2 --sampler T --point_uncertainty_mode sb --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/semantic3d_ssdr_log_t4-sb-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_semantic3d.py --dataset semantic3d --reg_strength ${reg_strength} --t 4 --gpu 3 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/semantic3d_ssdr_log_t4-sb-clsbal-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_semantic3d.py --dataset semantic3d --reg_strength ${reg_strength} --t 4 --gpu 1,2 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --gcn_fps 1 --uncertainty_mode WetSU --oracle_mode NAIL --threshold 0.9 --min_size 5  > record_log/semantic3d_ssdr_log_t4-sb-clsbal-gcn_fps-WetSU-NAIL-0.9-5_${reg_strength}.txt 2>&1 &
