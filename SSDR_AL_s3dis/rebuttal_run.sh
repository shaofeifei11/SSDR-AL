#!/bin/bash
# todo 使用新的 seed
reg_strength=0.008
# 104
python -u sff_create_seed.py --gpu 0 --seed_percent 0.005 --reg_strength ${reg_strength} > record_log/rebuttal_log_seed_${reg_strength}.txt 2>&1

# 105
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000000 --gpu 0 --round 2 --sampler random --oracle_mode dominant --min_size 5 >> record_log/ssdr_log_t10000000-random-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000000 --gpu 0 --round 2 --sampler T --point_uncertainty_mode entropy --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000000-entropy-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000000 --gpu 3 --round 2 --sampler T --point_uncertainty_mode lc --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000000-lc-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000000 --gpu 2 --round 2 --sampler T --point_uncertainty_mode sb --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000000-sb-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u sff_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000000 --gpu 1 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/rebuttal_log_t10000000-sb-clsbal-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u sff_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000000 --gpu 0 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertainty_mode WetSU --gcn_fps 1 --oracle_mode NAIL --threshold 0.9 --min_size 5  >> record_log/rebuttal_log_t10000000-sb-clsbal-WetSU-gcn_fps-NAIL-0.9-5_${reg_strength}.txt 2>&1 &



