#!/bin/bash
 
#SBATCH --job-name=QAOA_for_TSP_with_IBM
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --mem=100M
#SBATCH --account=eeme036064
 
#SBATCH --array=1-100
 
#SBATCH --output=/dev/null

#SBATCH --error=/dev/null

# Add the modules
module add languages/python/3.12.3
 
# cd to folder in you home dir
cd qaoaSim
 
# Activate the venv 
source IBMQSimvenv/bin/activate
 
# Run the script passing in the task id and the QUBO file
python TSP_QAOA_IBM.py "isingBatches/batch_Ising_data_TSP_9q_.json" ${SLURM_ARRAY_TASK_ID}
