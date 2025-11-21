QAOA Performance Comparison Study

This repository contains the source code, simulation scripts, and analysis tools used in the research paper: "Comparing Quantum Approximate Optimization Algorithm (QAOA) Performance Across Different Problem Classes" (Pending Publication).

Overview

The goal of this study was to investigate how the performance of QAOA varies across different combinatorial optimization problems—MaxCut, Knapsack, Minimum Vertex Cover (MVC), and Traveling Salesperson Problem (TSP)—when executed on various quantum backends (Ideal Simulators, Noisy Simulators, and Hardware Providers).

This codebase facilitates:

Generation of random Ising/QUBO problem instances.

Exact solving of instances to establish ground truth (global optima).

Execution of QAOA circuits on IBM, IonQ, and Alice & Bob providers.

Benchmarking against classical and quantum annealing (D-Wave).

Analysis of solution landscapes and HPC resource consumption.

Repository Structure

The codebase is organized into the following functional groups:

1. Configuration & Utilities

config.py: Central configuration file defining problem specifications (qubit counts, file slugs) and provider backend details.

helpers.py: Core utility library. Handles Ising energy calculation, Hamiltonian construction (Cost/Mixer), initial state preparation, and results serialization.

2. Problem Generation & Ground Truth

batchIsingGenerator.py: Uses openqaoa to generate batches of 100 random instances for TSP, Knapsack, MVC, and MaxCut, saving them as JSON.

dimodExactIsingSolver.py: Uses dimod.ExactSolver to brute-force solve every generated instance. This data is required to calculate the normalized Performance Scores used in the analysis.

3. QAOA Simulation (Execution)

IBM Quantum:

QAOA_IBM_Sim.py: The main script for running batches on IBM Aer simulators (Ideal and Noisy/FakeTorino).

IBMQPUTiming.py: Utility for estimating QPU execution time. Found within sub_experiemnts/QPUTiming

IonQ:

QAOA_IONQ_Sim.py: Handles execution on IonQ backends, featuring robust error handling for API connection stability and parallel processing via ProcessPoolExecutor.

Alice & Bob:

QAOA_ALICEBOB_Sim.py: Script adapted for the specific requirements of the Alice & Bob provider.

D-Wave (Annealing Comparison):

anneal_dwave.py: runs Simulated Annealing on the instances for performance comparison.

4. HPC & Containers

ALICEBOBSim.def: Apptainer/Singularity definition file for creating the containerized environment used in HPC jobs.

aliceBobApptainerFlexiLayer.sh: Example SLURM batch script for submitting large job arrays (e.g., Knapsack benchmark).

aliceBobApptainerMissingInstances.sh: SLURM script for re-running specific dropped/failed jobs.

5. Data Processing & Analysis

individual_results_merger.py: Consolidates individual JSON result files into master dataset files. Performs integrity checks to ensure all 100 instances are present.

totalCPUMaxMem.py: Parses SLURM logs to calculate classical HPC resource usage (CPU time and RAM). Found in sub_experiemnts/HPCResource

IBMQTimePlotting.py: Estimates total QPU time based on optimization loop counts.

merged_results_plotter_handcrafter.py: This plots the results of repeatedly solving the same problem instance to investigate variation in solution quality.

6. Visualization

merged_results_plotter.py: The primary plotting tool. Generates boxplots (Approximation Ratios) and bar charts (Success Counts) aggregating the full 100-instance batch.

costAndValidityLandscape.py: Deep-dive tool. Visualizes the cost and validity of every possible bitstring ($2^N$) for a single instance.

Installation & Requirements

This project requires Python 3.10+.

Clone the repository:

git clone [https://github.com/yourusername/qaoa-performance-study.git](https://github.com/yourusername/qaoa-performance-study.git)
cd QAOASTUDYGIT


Install dependencies. Key libraries include:

qiskit, qiskit-ibm-runtime, qiskit-ionq, qiskit-aer

openqaoa (for problem generation)

dimod, dwave-ocean-sdk (for exact solving and annealing)

pandas, matplotlib, networkx

Environment Variables:
Create a .env file in the root directory to store your API keys:

IBM_API_TOKEN="your_ibm_token"
IBM_INSTANCE_CRN="your_crn"
IONQ_API_TOKEN="your_ionq_token"


Usage Workflow

To reproduce the study results, follow this pipeline:

Step 1: Generate Problems

Generate the batch of 100 instances for a specific problem class.

python batchIsingGenerator.py


Note: Modify the desiredProblemType variable in the script to switch between MaxCut, TSP, etc.

Step 2: Solve for Ground Truth

Calculate global optima for normalization.

python dimodExactIsingSolver.py


Step 3: Run QAOA Simulations

You can run simulations locally or submit them to a cluster.

Example (IBM Noisy Simulation):

python QAOA_IBM_Sim.py \
    --problem_type Knapsack \
    --instance_id 1 \
    --num_layers 2 \
    --simulator NOISY


Example (IonQ Parallel Simulation):

python QAOA_IONQ_Sim.py


Note: Configure the instance range inside the script's __main__ block.

Step 4: Merge Results

After simulations complete, consolidate the individual JSON outputs.

python individual_results_merger.py


Step 5: Visualize Results

Generate the final comparative plots.

Performance Distribution (Boxplot):

python merged_results_plotter.py \
    --simulators IBM_IDEAL IBM_NOISY \
    --depths 1 1 \
    --plot_type boxplot


Success Counts (Bar Chart):

python merged_results_plotter.py \
    --simulators IBM_IDEAL IBM_NOISY \
    --depths 1 1 \
    --plot_type barchart


Citation

If you use this code in your research, please cite:

[Placeholder for Citation / BibTeX]

License

[Placeholder for License, e.g., MIT]