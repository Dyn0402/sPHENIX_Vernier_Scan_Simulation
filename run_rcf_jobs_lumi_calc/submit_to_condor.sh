#!/bin/bash

# run make_job_list.py then condor_submit submit_vernier_parameter_scan.job
python make_job_list.py
condor_submit submit_lumi_calc.job
echo "Jobs submitted to condor"