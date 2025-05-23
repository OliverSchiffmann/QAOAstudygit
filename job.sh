#!/bin/bash
 
#SBATCH --job-name=test_job
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:1:0
#SBATCH --mem=100M
#SBATCH --account=eeme036064
 
#SBATCH --array=1-2
 
#SBATCH --output=/dev/null

#SBATCH --error=/dev/null

# Add the modules you need
module add languages/python/3.12.3
 
# My need cd to folder in you home dir
cd qaoaSim
 
# Activate the venv you have already created
source IBMQSimvenv/bin/activate
 
# Run the script passing in the task id value
python main.py ${SLURM_ARRAY_TASK_ID}