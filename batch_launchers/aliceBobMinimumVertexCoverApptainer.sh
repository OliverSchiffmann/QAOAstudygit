#!/bin/bash
 
#SBATCH --job-name=QAOA_MVC_ALICEBOB
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=60:0:0
#SBATCH --mem=150G
#SBATCH --account=eeme036064
 
#SBATCH --array=1-5
 
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Add the modules
module load apptainer
 
# cd to folder
cd $HOME/qaoaSim/blueAppStuff

# Run the script passing in the task id and the QUBO file
apptainer run --env "PROBLEM_TYPE=MinimumVertexCover,INSTANCE_ID=${SLURM_ARRAY_TASK_ID},NUM_LAYERS=1" ALICEBOBSim.silf
