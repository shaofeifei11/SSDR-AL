#!/bin/bash
reg_strength=0.012
python -u ssdr_main_semantic3d.py --reg_strength ${reg_strength} --t 200 --gpu 3 --round 2 --sampler T --point_uncertainty_mode lc --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/semantic3d_ssdr_log_t200-lc-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_semantic3d.py --reg_strength ${reg_strength} --t 200 --gpu 2 --round 2 --sampler T --point_uncertainty_mode sb --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/semantic3d_ssdr_log_t200-sb-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &

python -u ssdr_main_semantic3d.py --reg_strength ${reg_strength} --t 201 --gpu 0 --round 2 --sampler T --point_uncertainty_mode lc --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/semantic3d_ssdr_log_t201-lc-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
python -u ssdr_main_semantic3d.py --reg_strength ${reg_strength} --t 201 --gpu 2 --round 2 --sampler T --point_uncertainty_mode sb --classbal 0 --uncertainty_mode mean --oracle_mode dominant --threshold 0.9 --min_size 5  >> record_log/semantic3d_ssdr_log_t201-sb-mean-dominant-0.9-5_${reg_strength}.txt 2>&1 &
