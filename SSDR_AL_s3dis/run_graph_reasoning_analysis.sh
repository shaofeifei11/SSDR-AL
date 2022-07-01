#!/bin/bash
reg_strength=0.008

# 104
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 0 --round 2 --gcn_number 2 --gcn_top 0 --sampler T --point_uncertainty_mode sb --classbal 2 --gcn_fps 1 --uncertainty_mode WetSU --oracle_mode NAIL --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-gcn_fps-WetSU-NAIL-0.9-5-2-0_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 1 --round 2 --gcn_number 3 --gcn_top 0 --sampler T --point_uncertainty_mode sb --classbal 2 --gcn_fps 1 --uncertainty_mode WetSU --oracle_mode NAIL --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-gcn_fps-WetSU-NAIL-0.9-5-3-0_${reg_strength}.txt 2>&1 &

# 105
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 1 --round 2 --gcn_number 1 --gcn_top 100 --sampler T --point_uncertainty_mode sb --classbal 2 --gcn_fps 1 --uncertainty_mode WetSU --oracle_mode NAIL --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-gcn_fps-WetSU-NAIL-0.9-5-1-100_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 2 --round 2 --gcn_number 2 --gcn_top 100 --sampler T --point_uncertainty_mode sb --classbal 2 --gcn_fps 1 --uncertainty_mode WetSU --oracle_mode NAIL --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-gcn_fps-WetSU-NAIL-0.9-5-2-100_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 3 --round 2 --gcn_number 3 --gcn_top 100 --sampler T --point_uncertainty_mode sb --classbal 2 --gcn_fps 1 --uncertainty_mode WetSU --oracle_mode NAIL --threshold 0.9 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-gcn_fps-WetSU-NAIL-0.9-5-3-100_${reg_strength}.txt 2>&1 &


