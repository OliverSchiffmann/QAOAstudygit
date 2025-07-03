#!/bin/bash
 
#SBATCH --job-name=qaoaOptimisationSim
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:0:0
#SBATCH --mem=100M
#SBATCH --account=eeme036064
 
#SBATCH --array=1-5
 
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