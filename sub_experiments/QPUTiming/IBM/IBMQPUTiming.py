# a script to submit a circuit to UBMQ and collect the average time for execution
# python IBMQPUTiming.py --problem_type MaxCut --instance_id 1 --num_layers 1 --num_samples 1
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
    save_single_result,
    build_mixer_hamiltonian,
    create_inital_state,
)

from dotenv import load_dotenv

envPath = os.path.join(projectRoot, ".env")
load_dotenv(dotenv_path=envPath)


# Packages for quantum stuff
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
    QiskitRuntimeService,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# ///////////  Unique Functions  //////////
def submit_circuit(params, ansatz, estimator, cost_hamiltonian, problem_type, reps_p):
    transpiledHamil = cost_hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, transpiledHamil, params)
    tag1 = str(problem_type).replace(" ", "_")  # e.g., 'bin_packing' or 'maximum_cut'
    tag2 = f"p={reps_p}"  # e.g., 'p=2'
    job_tags_list = [tag1, tag2]
    print(f"Submitting job with tags: {job_tags_list}")

    job = estimator.run([pub])
    jobID = job.job_id()
    job.update_tags(new_tags=job_tags_list)
    print(f"Successfully updated tags for job {jobID}.")


def setup_configuration():
    """
    Handles script configuration by parsing command-line arguments.

    Returns:
        tuple: A tuple containing:
            - problem_type (str)
            - instanceOfInterest (int)
            - isingFileName (str)
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

    IBMApiToken = os.environ.get("IBM_API_TOKEN")
    IBMInstanceCRN = os.environ.get("IBM_INSTANCE_CRN")
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
    qaoaKwargs = {
        "cost_operator": costHamil,
        "reps": repsP,
        "initial_state": initialCircuit,
        "mixer_operator": mixerHamil,
    }
    circuit = QAOAAnsatz(**qaoaKwargs)
    circuit.measure_all()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit = pm.run(circuit)

    # linear ramp schedule
    initial_betas = np.linspace(np.pi, 0, repsP, endpoint=False).tolist()
    initial_gammas = np.linspace(0, np.pi, repsP, endpoint=False).tolist()
    initial_params = initial_betas + initial_gammas

    # starting training loop
    objective_func_vals = []
    numOptimisations = 0
    estimator = Estimator(mode=backend)
    for i in range(numSamples):
        submit_circuit(
            initial_params, candidate_circuit, estimator, costHamil, problemType, repsP
        )
    print(f"Submitted {numSamples} jobs to backend {backend_name}.")
