#!/bin/bash

#SBATCH --job-name=QAOA_TSP_IBM
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --account=eeme036064

#SBATCH --array=1-400  # Total jobs = (num_instances * num_layer_configs)

#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# --- Parameter Definitions ---
# Define the number of different instances you are testing
numInstances=100
# Define an array of layer values you want to test
declare -a layerValues=(1 2 3 4)

# --- Task Logic ---
# Calculate the zero-based task index
taskIndex=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate the instanceId (from 1 to 100)
instanceId=$((taskIndex % numInstances + 1))

# Calculate the index for the layerValues array
layerIndex=$((taskIndex / numInstances))

# Get the number of layers for this specific job
numLayers=${layerValues[$layerIndex]}

echo "Running task ${SLURM_ARRAY_TASK_ID}: Instance ID = ${instanceId}, Layers = ${numLayers}"

# --- Job Execution ---
# Add the modules
module add languages/python/3.12.3

# cd to folder in your home dir
cd $HOME/qaoaSim

# Activate the venv
source IBMQSimvenv/bin/activate

# Run the script passing in the calculated parameters
python QAOA_IBM_Sim.py \
    --problem_type TSP \
    --instance_id ${instanceId} \
    --num_layers ${numLayers} \
    --simulator NOISY
