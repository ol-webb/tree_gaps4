#!/bin/sh
# Grid Engine options
#$ -N xgboost_tuning  # Job name
#$ -cwd               # Use the current working directory
#$ -l h_rt=24:00:00   # Runtime limit of 24 hours
#$ -l h_vmem=16G      # Memory limit of 16 GB per core
#$ -pe sharedmem 4    # Request 4 cores

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python module (make sure this matches the Python version you need)
module load anaconda/2020.02

# Run the Python script
python3 xgboost4_eddie.py
