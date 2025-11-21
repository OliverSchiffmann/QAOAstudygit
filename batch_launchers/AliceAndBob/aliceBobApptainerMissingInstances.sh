#!/bin/bash

# ==============================================================================
# SLURM Job Repair Script for QAOA Minimum Vertex Cover
# ==============================================================================
# This script is designed to re-run specific simulation jobs that failed or were
# missing from a larger batch execution.
#
# Unlike the standard bulk submission script, this does not iterate through a
# continuous range of instances. Instead, it allows the user to manually specify
# lists of "missing" instance IDs for each layer depth (p=1, 2, 3, 4).
#
# It constructs "Master Arrays" to map a linear SLURM task ID to these specific,
# disjoint parameter combinations.
#
# Targeted Problem: Minimum Vertex Cover
# Total Jobs: 6 (Calculated based on the specific entries in the arrays below)
# ==============================================================================

#SBATCH --job-name=QAOA_missing_instance
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=55:0:0
#SBATCH --mem=120G
#SBATCH --account=eeme036064

# Total missing jobs
#SBATCH --array=1-6
#
# Docstring:
# The array range corresponds to the total count of missing jobs manually
# listed below (1+1+1+3 = 6). This must be updated manually if the number
# of missing instances changes.

#SBATCH --output=logs/job_REPAIR_%A_%a.out
#SBATCH --error=logs/job_REPAIR_%A_%a.err

# --- Parameter Definitions ---
# Paste your missing instance IDs for each layer into these arrays.
# NOTE: These are already filled from your provided list.
declare -a instancesForLayer1=(12)
declare -a instancesForLayer2=(33)
declare -a instancesForLayer3=(2)
declare -a instancesForLayer4=(65 66 69)
#
# Docstring:
# These arrays contain the specific Instance IDs that need to be re-run for
# each layer depth.

# --- Master Array Construction ---
# This section builds two parallel arrays (allTaskLayers and allTaskInstances)
# to map the SLURM_ARRAY_TASK_ID to a specific job configuration.
#
# Docstring:
# Since SLURM array IDs are sequential integers, we must "flatten" the 
# separate layer-specific lists into two linear master arrays.
# - allTaskLayers: Stores the 'p' value (1, 2, 3, or 4) for the job.
# - allTaskInstances: Stores the instance ID for the job.

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
#
# Docstring:
# Converts the 1-based SLURM array task ID into a 0-based index to access
# the master arrays constructed above.

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
#
# Docstring:
# Executes the simulation for the recovered parameters.
# Note: The PROBLEM_TYPE is set to 'MinimumVertexCover' here.