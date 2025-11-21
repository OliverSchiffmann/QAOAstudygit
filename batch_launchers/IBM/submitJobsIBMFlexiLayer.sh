#!/bin/bash

# ==============================================================================
# SLURM Job Array Script for QAOA (IBM Simulator)
# ==============================================================================
# This script configures a SLURM job array to benchmark the QAOA algorithm
# on problem instances using a Python-based IBM Quantum simulator.
#
# Distinct from the previous scripts, this workflow:
# 1. Loads a specific Python module (3.12.3).
# 2. Activates a Python virtual environment (IBMQSimvenv).
# 3. Executes a Python script ('QAOA_IBM_Sim.py') with command-line arguments.
#
# It iterates through 100 instances across 4 layer depths (p=1, 2, 3, 4).
# The simulation is explicitly flagged as 'NOISY'.
# ==============================================================================

#SBATCH --job-name=QAOA_IBM
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=bp1-compute196,bp1-compute150
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:0:0
#SBATCH --account=eeme036064

#SBATCH --array=1-400  # Total jobs = (num_instances * num_layer_configs)
#
# Docstring:
# Defines the job array range (1 to 400).
# Mappings:
# - Tasks 1-100:   Layer p=1, Instances 1-100
# - Tasks 101-200: Layer p=2, Instances 1-100
# - Tasks 201-300: Layer p=3, Instances 1-100
# - Tasks 301-400: Layer p=4, Instances 1-100

#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# --- Parameter Definitions ---
# Define the number of different instances you are testing
numInstances=100
#
# Docstring:
# The total number of problem instances to iterate through per layer configuration.

# Define an array of layer values you want to test
declare -a layerValues=(1 2 3 4)
#
# Docstring:
# The specific QAOA depth values (p) to be tested.

# --- Task Logic ---
# Calculate the zero-based task index
taskIndex=$((SLURM_ARRAY_TASK_ID - 1))
#
# Docstring:
# Converts the 1-based SLURM ID to a 0-based index for array arithmetic.

# Calculate the instanceId (from 1 to 100)
instanceId=$((taskIndex % numInstances + 1))
#
# Docstring:
# Determines the instance ID for the current job using modulo arithmetic.

# Calculate the index for the layerValues array
layerIndex=$((taskIndex / numInstances))
#
# Docstring:
# Determines which layer value to use based on the block the task falls into.

# Get the number of layers for this specific job
numLayers=${layerValues[$layerIndex]}
#
# Docstring:
# Retrieves the actual 'p' value from the layerValues array.

echo "Running task ${SLURM_ARRAY_TASK_ID}: Instance ID = ${instanceId}, Layers = ${numLayers}"

# --- Job Execution ---
# Add the modules
module add languages/python/3.12.3

# cd to folder in your home dir
cd $HOME/qaoaSim

# Activate the venv
source IBMQSimvenv/bin/activate
#
# Docstring:
# Activates the dedicated Python virtual environment containing the necessary
# quantum simulation libraries (e.g., Qiskit) dependencies.

# Run the script passing in the calculated parameters
# Options for simulator: IDEAL, NOISY
# Options for problem_type: TSP, MaxCut, Knapsack, MinimumVertexCover
python QAOA_IBM_Sim.py \
    --problem_type Knapsack \
    --instance_id ${instanceId} \
    --num_layers ${numLayers} \
    --simulator NOISY
#
# Docstring:
# Executes the Python simulation script.
# Arguments:
# --problem_type: Specifies the optimization problem (Knapsack).
# --instance_id:  The unique ID of the problem instance to solve.
# --num_layers:   The depth of the QAOA circuit.
# --simulator:    Specifies the backend mode ('NOISY' implies a noise model is applied).