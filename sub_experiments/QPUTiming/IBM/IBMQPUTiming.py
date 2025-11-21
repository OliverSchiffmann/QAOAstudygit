# ==============================================================================
# IBM QPU Circuit Submission Script
# ==============================================================================
# This script is designed to measure the execution time and latency of a specific
# QAOA circuit configuration on an IBM Quantum backend (ibm_torino).
#
# It performs the following steps:
# 1. Parses command-line arguments (problem type, instance ID, layers, number of samples).
# 2. Loads the necessary Ising model data and constructs the Hamiltonian (Cost and Mixer).
# 3. Builds the QAOAAnsatz circuit using a fixed initial parameter guess (linear ramp).
# 4. Connects to the Qiskit Runtime Service using environment variables.
# 5. Submits the transpiled circuit multiple times to the 'ibm_torino' backend
#    using the EstimatorV2 primitive.
#
# Execution Example (for MaxCut, instance 1, p=1, submitted once):
# python IBMQPUTiming.py --problem_type MaxCut --instance_id 1 --num_layers 1 --num_samples 1
# ==============================================================================
import argparse
import numpy as np
import os
import sys

scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.abspath(os.path.join(scriptDir, "../../../"))
if projectRoot not in sys.path:
    sys.path.append(projectRoot)

from config import problem_configs
from helpers import (
    load_ising_and_build_hamiltonian,
    build_mixer_hamiltonian,
    create_inital_state,
)

from dotenv import load_dotenv

envPath = os.path.join(projectRoot, ".env")
load_dotenv(dotenv_path=envPath)


# Packages for quantum stuff
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    QiskitRuntimeService,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# ///////////  Unique Functions  //////////
def submit_circuit(params, ansatz, estimator, cost_hamiltonian, problem_type, reps_p):
    """
    Constructs a Pub, submits the QAOA circuit to the IBM Estimator, and tags the job.

    The job is submitted as a single execution request (Pub) to the Qiskit Runtime
    service for the purpose of execution time measurement.

    Args:
        params (list): The list of fixed QAOA angles (betas and gammas).
        ansatz (QuantumCircuit): The transpiled QAOA circuit.
        estimator (EstimatorV2): The configured Qiskit Runtime Estimator instance.
        cost_hamiltonian (SparsePauliOp): The problem's cost Hamiltonian.
        problem_type (str): The name of the optimization problem (e.g., 'Knapsack').
        reps_p (int): The number of QAOA layers (p).
    """
    # Apply layout to the Hamiltonian to match the circuit's layout after transpilation
    transpiledHamil = cost_hamiltonian.apply_layout(ansatz.layout)
    # Define the Public Unit (Pub) for the Estimator
    pub = (ansatz, transpiledHamil, params)
    # Create tags for easy job identification and querying
    tag1 = str(problem_type).replace(" ", "_")  # e.g., 'bin_packing' or 'maximum_cut'
    tag2 = f"p={reps_p}"  # e.g., 'p=2'
    job_tags_list = [tag1, tag2]
    print(f"Submitting job with tags: {job_tags_list}")

    # Submit the job
    job = estimator.run([pub])
    jobID = job.job_id()
    # Explicitly update tags on the job object
    job.update_tags(new_tags=job_tags_list)
    print(f"Successfully updated tags for job {jobID}.")


def setup_configuration():
    """
    Handles script configuration by parsing command-line arguments.

    Retrieves necessary configuration variables for the simulation, including
    the problem type, instance ID, layer count, and number of submission samples.

    Returns:
        tuple: A tuple containing:
            - problem_type (str)
            - instance_of_interest (int)
            - num_layers (int)
            - ising_file_name (str): Full path to the Ising data file.
            - problem_file_name_tag (str)
            - num_samples (int): Number of times to submit the circuit.
    """
    fileDirectory = os.path.join(projectRoot, "isingBatches")

    parser = argparse.ArgumentParser(
        description="Run a QAOA simulation for a specific problem class and instance."
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        required=True,
        help="The problem class to run (e.g., TSP, Knapsack).",
    )
    parser.add_argument(
        "--instance_id",
        type=int,
        required=True,
        help="The instance ID from the batch file to solve.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=True,
        help="The number of QAOA layers to build.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="The number of times to submit the circuit.",
    )

    args = parser.parse_args()
    problem_type = args.problem_type
    instance_of_interest = args.instance_id
    num_layers = args.num_layers
    num_samples = args.num_samples

    # Select the configuration based on the determined problem_type
    try:
        selectedConfig = problem_configs[problem_type]
    except KeyError:
        raise ValueError(
            f"Error: '{problem_type}' is not a valid problem type. "
            f"Available options are: {list(problem_configs.keys())}"
        )

    # Construct the final filename
    problem_file_name_tag = selectedConfig["file_slug"]
    ising_file_name = os.path.join(
        fileDirectory, f"batch_Ising_data_{problem_file_name_tag}.json"
    )

    return (
        problem_type,
        instance_of_interest,
        num_layers,
        ising_file_name,
        problem_file_name_tag,
        num_samples,
    )


if __name__ == "__main__":
    # ////////////      Config.    ///////////
    (
        problemType,
        instanceOfInterest,
        repsP,
        isingFileName,
        problemFileNameTag,
        numSamples,
    ) = setup_configuration()

    # --- IBM Authentication and Backend Setup ---
    IBMApiToken = os.environ.get("IBM_API_TOKEN")
    IBMInstanceCRN = os.environ.get("IBM_INSTANCE_CRN")
    # Initialize the Qiskit Runtime service
    service = QiskitRuntimeService(
        channel="ibm_cloud", token=IBMApiToken, instance=IBMInstanceCRN
    )
    backend_name = "ibm_torino"
    backend = service.backend(backend_name)

    print(
        f"Problem Type: {problemType}, Instance ID: {instanceOfInterest}, Ising model file name: {isingFileName}"
    )

    print(problemFileNameTag)
    # /// training ///
    # --- cost hamiltonian ---
    costHamil, numQubits, isingTerms, weightCapacity = load_ising_and_build_hamiltonian(
        isingFileName, instanceOfInterest
    )
    print(f"Problem class is: {problemType}")
    if problemType == "Knapsack":
        print(f"Capacity of this knapsack is: {weightCapacity}")

    print(f"Quadratic and linear terms of the Ising model are: {isingTerms}")

    # --- mixer ---
    mixerHamil = build_mixer_hamiltonian(numQubits, problemType)
    print(mixerHamil)

    # --- inital state ---
    initialCircuit = create_inital_state(numQubits, problemType, weightCapacity)
    print(initialCircuit)

    # --- QAOA Ansatz ---
    # Construct the QAOA circuit using the cost and mixer Hamiltonians
    qaoaKwargs = {
        "cost_operator": costHamil,
        "reps": repsP,
        "initial_state": initialCircuit,
        "mixer_operator": mixerHamil,
    }
    circuit = QAOAAnsatz(**qaoaKwargs)
    # Add measurements for execution on real hardware
    circuit.measure_all()
    # Transpile the circuit for the target backend (ibm_torino)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit = pm.run(circuit)

    # linear ramp schedule
    # Define a fixed initial parameter guess (linear ramp schedule)
    initial_betas = np.linspace(np.pi, 0, repsP, endpoint=False).tolist()
    initial_gammas = np.linspace(0, np.pi, repsP, endpoint=False).tolist()
    initial_params = initial_betas + initial_gammas

    # starting training loop
    objective_func_vals = []
    numOptimisations = 0
    # Initialize the Estimator with the target backend
    estimator = Estimator(mode=backend)
    # Submit the circuit multiple times for sampling
    for i in range(numSamples):
        submit_circuit(
            initial_params, candidate_circuit, estimator, costHamil, problemType, repsP
        )
    print(f"Submitted {numSamples} jobs to backend {backend_name}.")
