#!/bin/bash
reg_strength=0.008


#run 106
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 0 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertainty_mode WetSU --gcn_fps 1 --oracle_mode NAIL --threshold 1.0 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-WetSU-gcn_fps-NAIL-1.0-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 1 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertainty_mode WetSU --gcn_fps 1 --oracle_mode NAIL --threshold 0.8 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-WetSU-gcn_fps-NAIL-0.8-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 2 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertainty_mode WetSU --gcn_fps 1 --oracle_mode NAIL --threshold 0.7 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-WetSU-gcn_fps-NAIL-0.7-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 3 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertainty_mode WetSU --gcn_fps 1 --oracle_mode NAIL --threshold 0.6 --min_size 5  >> record_log/ssdr_log_t10000-sb-clsbal-WetSU-gcn_fps-NAIL-0.6-5_${reg_strength}.txt 2>&1 &


