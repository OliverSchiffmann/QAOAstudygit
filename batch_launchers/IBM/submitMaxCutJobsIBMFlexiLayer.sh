#!/bin/bash

#SBATCH --job-name=QAOA_MaxCut_IBM
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --mem=40G
#SBATCH --account=eeme036064

# --- Reducing the array size ---
# Total 100 instances / 10 instances per job = 10 jobs per layer config
# 10 jobs * 5 layer configs = 50 total jobs
#SBATCH --array=1-50

#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# --- Parameter Definitions ---
numInstancesTotal=100
instancesPerJob=10 # Number of python scripts to run per job
declare -a layerValues=(1 2 3 4 20)
numLayerConfigs=${#layerValues[@]}

# --- Task Logic ---
# Calculate which group of 10 instances this job is responsible for
jobIndex=$((SLURM_ARRAY_TASK_ID - 1))
instanceGroupIndex=$((jobIndex % (numInstancesTotal / instancesPerJob) ))

# Calculate the start and end instance IDs for this job's loop
startInstanceId=$((instanceGroupIndex * instancesPerJob + 1))
endInstanceId=$((startInstanceId + instancesPerJob - 1))

# Calculate the layer configuration for this job
layerIndex=$((jobIndex / (numInstancesTotal / instancesPerJob) ))
numLayers=${layerValues[$layerIndex]}

echo "Running SLURM task ${SLURM_ARRAY_TASK_ID}: Processing instances ${startInstanceId}-${endInstanceId} with Layers = ${numLayers}"

# --- Job Execution ---
module add languages/python/3.12.3
cd $HOME/qaoaSim
source IBMQSimvenv/bin/activate

# Loop through the assigned instances
for (( instanceId=${startInstanceId}; instanceId<=${endInstanceId}; instanceId++ )); do
    echo "--> Starting Python script for instance ${instanceId}"
    python QAOA_IBM_Sim.py \
        --problem_type MaxCut \
        --instance_id ${instanceId} \
        --num_layers ${numLayers} \
        --simulator IDEAL 
done
