#!/bin/bash

# ==============================================================================
# SLURM Job Array Script for QAOA Repetitions (IBM Ideal Simulator)
# ==============================================================================
# This script is used to execute a specific QAOA configuration multiple times
# (repetitions) to collect statistical data on optimization variability.
#
# Configuration for this batch:
# - Problem Type: MaxCut
# - Instance ID: 1 (Fixed)
# - QAOA Layers (p): 1 (Fixed)
# - Simulator: IDEAL (Noiseless AerSimulator)
# - Total Repetitions: 5
#
# The SLURM array task ID is used to assign a unique repetition ID to each run.
# ==============================================================================

#SBATCH --job-name=QAOA_IBM_repeats
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --account=eeme036064

#SBATCH --array=1-5  # Total jobs = (num_instances * num_layer_configs)
#
# Docstring:
# Defines the SLURM job array range. The size (1-5) corresponds to the
# five desired repetitions for this specific configuration.

#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# --- Parameter Definitions ---
# Define the number of different instances you are testing
numRepititions=5
#
# Docstring:
# The total number of times the simulation should be repeated (5).

# Define an array of layer values you want to test
declare -a layerValues=(1)
#
# Docstring:
# The specific QAOA depth (p=1) to be tested across all repetitions.

# --- Task Logic ---
# Calculate the zero-based task index
taskIndex=$((SLURM_ARRAY_TASK_ID - 1))
#
# Docstring:
# Converts the 1-based SLURM ID to a 0-based index.

# Calculate the instanceId (from 1 to 100)
repititionId=$((taskIndex % numRepititions + 1))
#
# Docstring:
# Determines the unique repetition ID (1 to 5) for the current job.

# Calculate the index for the layerValues array
layerIndex=$((taskIndex / numRepititions))
#
# Docstring:
# Calculates the index for the layerValues array (will always be 0 here).

# Get the number of layers for this specific job
numLayers=${layerValues[$layerIndex]}
#
# Docstring:
# Retrieves the layer count (p=1) from the array.

echo "Running task ${SLURM_ARRAY_TASK_ID}: Instance ID = ${repititionId}, Layers = ${numLayers}"

# --- Job Execution ---
# Add the modules
module add languages/python/3.12.3

# cd to folder in your home dir
cd $HOME/qaoaSim

# Activate the venv
source IBMQSimvenv/bin/activate
#
# Docstring:
# Activates the dedicated Python virtual environment.

# Run the script passing in the calculated parameters, simulator options: IDEAL or NOISY
python QAOA_IBM_Sim_repeat_instance.py \
    --problem_type MaxCut \
    --instance_id 1 \
    --repitition_id ${repititionId} \
    --num_layers ${numLayers} \
    --simulator IDEAL
#
# Docstring:
# Executes the Python simulation script. Note the fixed 'MaxCut', 'instance_id 1',
# and 'IDEAL' simulator setting, with the varying 'repititionId'.