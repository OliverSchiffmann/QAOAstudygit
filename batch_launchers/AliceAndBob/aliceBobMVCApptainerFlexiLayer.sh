#!/bin/bash

#SBATCH --job-name=QAOA_MVC_ALICEBOB
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:0:0
#SBATCH --mem=40G
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
module load apptainer

# cd to folder in your home dir
cd $HOME/qaoaSim/blueAppStuff

apptainer run --env "PROBLEM_TYPE=MinimumVertexCover,INSTANCE_ID=${instanceId},NUM_LAYERS=${numLayers}" ALICEBOBSim.silf
