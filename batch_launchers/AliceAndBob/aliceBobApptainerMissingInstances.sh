#!/bin/bash

#SBATCH --job-name=QAOA_missing_instance
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=55:0:0
#SBATCH --mem=120G
#SBATCH --account=eeme036064

# Total missing jobs
#SBATCH --array=1-13

#SBATCH --output=logs/job_REPAIR_%A_%a.out
#SBATCH --error=logs/job_REPAIR_%A_%a.err

# --- Parameter Definitions ---
# Paste your missing instance IDs for each layer into these arrays.
# NOTE: These are already filled from your provided list.
declare -a instancesForLayer1=()
declare -a instancesForLayer2=()
declare -a instancesForLayer3=()
declare -a instancesForLayer4=(65 66 69 71 72 73 75 76 79 80 82 95 96)

# --- Master Array Construction ---
# This section builds two parallel arrays (allTaskLayers and allTaskInstances)
# to map the SLURM_ARRAY_TASK_ID to a specific job configuration.

declare -a allTaskLayers=()
declare -a allTaskInstances=()

# Add Layer 1 jobs
for instanceId in "${instancesForLayer1[@]}"; do
    allTaskLayers+=("1")
    allTaskInstances+=("${instanceId}")
done

# Add Layer 2 jobs
for instanceId in "${instancesForLayer2[@]}"; do
    allTaskLayers+=("2")
    allTaskInstances+=("${instanceId}")
done

# Add Layer 3 jobs
for instanceId in "${instancesForLayer3[@]}"; do
    allTaskLayers+=("3")
    allTaskInstances+=("${instanceId}")
done

# Add Layer 4 jobs
for instanceId in "${instancesForLayer4[@]}"; do
    allTaskLayers+=("4")
    allTaskInstances+=("${instanceId}")
done

# --- Task Logic ---
# Calculate the zero-based task index
taskIndex=$((SLURM_ARRAY_TASK_ID - 1))

# Get the parameters for this specific job from the master arrays
instanceId=${allTaskInstances[$taskIndex]}
numLayers=${allTaskLayers[$taskIndex]}

# Check if parameters were found (good for debugging)
if [ -z "${instanceId}" ] || [ -z "${numLayers}" ]; then
    echo "Error: Could not find parameters for task index ${taskIndex}."
    exit 1
fi

echo "Running task ${SLURM_ARRAY_TASK_ID} (Index ${taskIndex}): Instance ID = ${instanceId}, Layers = ${numLayers}"

# --- Job Execution ---
# Add the modules
module load apptainer

# cd to folder in your home dir
cd $HOME/qaoaSim/blueAppStuff

apptainer run --env "PROBLEM_TYPE=MinimumVertexCover,INSTANCE_ID=${instanceId},NUM_LAYERS=${numLayers}" ALICEBOBSim.silf