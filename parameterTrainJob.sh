#!/bin/bash
 
#SBATCH --job-name=qaoaOptimisationSim
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --mem=100M
#SBATCH --account=eeme036064
 
#SBATCH --array=1-100
 
#SBATCH --output=/dev/null

#SBATCH --error=/dev/null

# Add the modules you need
module add languages/python/3.12.3
 
# My need cd to folder in you home dir
cd qaoaSim
 
# Activate the venv you have already created
source IBMQSimvenv/bin/activate
 
# Run the script passing in the task id and the QUBO file
python mainParameterTrain.py "isingBatches/batch_Ising_data_TSP_9q_.json" ${SLURM_ARRAY_TASK_ID}
