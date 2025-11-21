#!/bin/bash

# ==============================================================================
# SLURM Job Array Script for QAOA Benchmarking using Alice and Bob's Apptainer
# ==============================================================================
# This script is designed to run a large number of QAOA simulation jobs on an
# HPC cluster using the SLURM workload manager's array feature.
#
# Each job in the array systematically tests a combination of:
# 1. A unique problem instance ID (1 to 100).
# 2. A specific number of QAOA layers (p), defined in 'layerValues'.
#
# It uses the 'apptainer' container runtime to ensure a consistent execution
# environment for the 'ALICEBOBSim.silf' simulation executable.
#
# The total number of jobs in the array is (numInstances * numLayers).
# ==============================================================================

#SBATCH --job-name=QAOA_ALICEBOB
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:0:0
#SBATCH --mem=40G
#SBATCH --account=eeme036064

#SBATCH --array=1-400  # Total jobs = (num_instances * num_layer_configs)
#
# Docstring:
# Defines the SLURM job array range. The total array size (400) is calculated
# as 100 instances * 4 layer configurations. Each task ID (1 to 400) corresponds
# to a unique (instanceId, numLayers) pair.

#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# --- Parameter Definitions ---
# Define the number of different instances you are testing
numInstances=100
#
# Docstring:
# The total number of unique problem instances to be tested.

# Define an array of layer values you want to test
declare -a layerValues=(1 2 3 4)
#
# Docstring:
# An array defining the specific number of QAOA layers (p) that will be tested
# for each instance.

# --- Task Logic ---
# Calculate the zero-based task index
taskIndex=$((SLURM_ARRAY_TASK_ID - 1))
#
# Docstring:
# Converts the 1-based SLURM array task ID into a 0-based index for internal
# array calculations.

# Calculate the instanceId (from 1 to 100)
instanceId=$((taskIndex % numInstances + 1))
#
# Docstring:
# Determines the specific problem instance ID for the current job. This uses
# the modulo operator on the task index to cycle through instance IDs 1 to 100.

# Calculate the index for the layerValues array
layerIndex=$((taskIndex / numInstances))
#
# Docstring:
# Determines the index within the 'layerValues' array, which cycles every
# 'numInstances' tasks. This ensures blocks of 100 jobs all use the same 'p' value.

# Get the number of layers for this specific job
numLayers=${layerValues[$layerIndex]}
#
# Docstring:
# Retrieves the number of layers (p) for the current job from the 'layerValues'
# array using the calculated 'layerIndex'.

echo "Running task ${SLURM_ARRAY_TASK_ID}: Instance ID = ${instanceId}, Layers = ${numLayers}"

# --- Job Execution ---
# Add the modules
module load apptainer

# cd to folder in your home dir
cd $HOME/qaoaSim/blueAppStuff

apptainer run --env "PROBLEM_TYPE=Knapsack,INSTANCE_ID=${instanceId},NUM_LAYERS=${numLayers}" ALICEBOBSim.silf
#
# Docstring:
# Executes the QAOA simulation using the Apptainer container runtime.
# Critical simulation parameters (PROBLEM_TYPE, INSTANCE_ID, NUM_LAYERS) are
# passed into the container environment using the --env flag.
# Options for PROBLEM_TYPE are "TSP", "Knapsack", "MaxCut" and "MinimumVertexCover".