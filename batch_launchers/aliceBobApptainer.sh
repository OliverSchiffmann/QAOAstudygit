#!/bin/bash
 
#SBATCH --job-name=QAOA_Knapsack_ALICEBOB
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --mem=100M
#SBATCH --account=eeme036064
 
#SBATCH --array=1-1--
 
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Add the modules
module add apptainer
 
# cd to folder, might need to change this to be the blueApp folder
cd $HOME/qaoaSim

# Run the script passing in the task id and the QUBO file
# python QAOA_ALICEBOB_Sim.py --problem_type Knapsack --instance_id ${SLURM_ARRAY_TASK_ID}
$ apptainer run --env "PROBLEM_TYPE=Knapsack" "INSTANCE_ID=1" test.silf
