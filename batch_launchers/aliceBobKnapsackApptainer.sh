#!/bin/bash
 
#SBATCH --job-name=QAOA_Knapsack_ALICEBOB
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=28
#SBATCH --time=60:0:0
#SBATCH --account=eeme036064
 
#SBATCH --array=1-100
 
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Add the modules
module load apptainer
 
# cd to folder
cd $HOME/qaoaSim/blueAppStuff

# Run the script passing in the task id and the QUBO file
apptainer run --env "PROBLEM_TYPE=Knapsack,INSTANCE_ID=${SLURM_ARRAY_TASK_ID}" ALICEBOBSim.silf
