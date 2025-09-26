#!/bin/bash
 
#SBATCH --job-name=QAOA_TSP_IBM
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --account=eeme036064
 
#SBATCH --array=1-100
 
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Add the modules
module add languages/python/3.12.3
 
# cd to folder in you home dir
cd $HOME/qaoaSim
 
# Activate the venv 
source IBMQSimvenv/bin/activate

# Run the script passing in theproblem type taks id qaoa layers and simulator to use (NOISY or IDEAL)
python QAOA_IBM_Sim.py --problem_type TSP --instance_id ${SLURM_ARRAY_TASK_ID} --num_layers 1 --simulator IDEAL