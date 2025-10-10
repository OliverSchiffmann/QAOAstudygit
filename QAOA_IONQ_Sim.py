import argparse
import os
import time
import numpy as np
from scipy.optimize import minimize
from config import problem_configs
from helpers import (
    load_ising_and_build_hamiltonian,
    save_single_result,
    build_mixer_hamiltonian,
    create_inital_state,
)
from dotenv import load_dotenv
from requests.exceptions import ConnectionError
from concurrent.futures import as_completed, ProcessPoolExecutor


# Packages for quantum stuff
from qiskit.circuit.library import QAOAAnsatz
from qiskit.providers import JobStatus
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ionq import IonQProvider

load_dotenv()  # Load environment variables from .env file


# ////////// Classes ////////////
class QAOACallback:
    """A thread-safe class to hold the state of the optimization callback."""

    def __init__(self, ansatz, estimator, costHamiltonian):
        self.ansatz = ansatz
        self.estimator = estimator
        self.costHamiltonian = costHamiltonian
        self.numOptimisations = 0
        self.objectiveFuncVals = []

    def cost_func_estimator(self, params, instance_id):
        """A robust cost function that handles network errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                transpiledHamil = self.costHamiltonian.apply_layout(self.ansatz.layout)
                pub = (self.ansatz, transpiledHamil, params)
                # print(self.ansatz)
                job = self.estimator.run([pub])
                results = job.result()[0]
                cost = results.data.evs
                costFloat = float(np.real(cost))
                self.objectiveFuncVals.append(costFloat)
                self.numOptimisations += 1
                print(
                    f"(Instance: {instance_id}) Optimization step {self.numOptimisations}"
                )
                return costFloat

            except ConnectionError as e:
                print(f"Network error on attempt {attempt + 1}: {e}. Retrying...")
                time.sleep(2)

            except Exception as e:
                # Catch any other potential errors from the job
                print(f"An unexpected job error occurred: {e}. Penalizing this step.")
                break

        # If all retries fail, penalize this parameter set
        print(
            f"All network attempts failed for instance {instance_id}. Returning infinity."
        )
        return float("inf")

    def cost_function_wrapper(
        self, params, instance_id
    ):  # required because ionq cant handle 0.0 angle roation gates, gateset=native should avoid this problem but just in case
        epsilon = 1e-9
        safe_params = np.copy(params)
        # Find where parameters are exactly 0 and replace them with epsilon
        safe_params[safe_params == 0] = epsilon

        return self.cost_func_estimator(safe_params, instance_id)


# ///////////  Unique Functions  //////////
def setup_configuration():
    """
    Handles script configuration by parsing command-line arguments.

    Returns:
        tuple: A tuple containing:
            - problem_type (str)
            - instanceOfInterest (int)
            - isingFileName (str)
    """
    FILEDIRECTORY = "isingBatches"

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
    args = parser.parse_args()
    problem_type = args.problem_type
    instance_of_interest = args.instance_id

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
    ising_file_name = f"{FILEDIRECTORY}/batch_Ising_data_{problem_file_name_tag}.json"

    return problem_type, instance_of_interest, ising_file_name, problem_file_name_tag


def runSingleSimulation(args):
    """
    Runs the complete QAOA optimization for a single problem instance.
    This function is designed to be called by a thread.
    """
    # Unpack the arguments for this specific run
    problemType, instanceOfInterest = args
    print(f"STARTING: {problemType} instance {instanceOfInterest}")

    # --- Configuration for this instance ---
    config = problem_configs[problemType]
    problemFileNameTag = config["file_slug"]
    isingFileName = f"isingBatches/batch_Ising_data_{problemFileNameTag}.json"

    # --- Variables ---
    INDIVIDUAL_RESULTS_FOLDER = "individual_results_warehouse"
    reps_p = 3  # Number of QAOA layers
    simType = "IDEAL"  # options: 'IDEAL','NOISY'

    # --- Backend Setup ---
    ionqApiToken = os.environ.get("IONQ_API_TOKEN")
    provider = IonQProvider(token=ionqApiToken)
    backendSimulator = provider.get_backend("ionq_simulator", gateset="native")

    ionqDevice = (
        "aria-1"  # MUST COMMENT OUT TO ENABLE CORRECT FILENAMING IF USING NOISELESS
    )
    if simType == "NOISY":
        backendSimulator.set_options(noise_model=ionqDevice)  # for noisy simulation

    backendSimulator.options.ionq_compiler_synthesis = True

    # --- Training ---
    costHamil, numQubits, isingTerms, weightCapacity = load_ising_and_build_hamiltonian(
        isingFileName, instanceOfInterest
    )
    mixerHamil = build_mixer_hamiltonian(numQubits, problemType)
    initialCircuit = create_inital_state(numQubits, problemType, weightCapacity)

    qaoaKwargs = {
        "cost_operator": costHamil,
        "reps": reps_p,
        "initial_state": initialCircuit,
        "mixer_operator": mixerHamil,
    }
    circuit = QAOAAnsatz(**qaoaKwargs)
    circuit.measure_all()
    pm = generate_preset_pass_manager(
        optimization_level=1, backend=backendSimulator
    )  # level 1 as IONQ is fully connected and they recommend 0 or 1
    candidate_circuit = pm.run(circuit)

    initialBetas = np.linspace(np.pi, 0, reps_p, endpoint=False).tolist()
    initialGammas = np.linspace(0, np.pi, reps_p, endpoint=False).tolist()
    initialParams = initialBetas + initialGammas

    # --- Training Loop ---
    estimator = Estimator(mode=backendSimulator)

    # Create an instance of our thread-safe callback handler
    qaoaCallback = QAOACallback(candidate_circuit, estimator, costHamil)

    trainResult = minimize(
        qaoaCallback.cost_function_wrapper,
        initialParams,
        args=(instanceOfInterest,),
        method="COBYLA",
        tol=1e-3,
        options={"maxiter": 500},
    )
    # --- Sampling ---
    optimizedCircuit = candidate_circuit.assign_parameters(trainResult.x)
    job = backendSimulator.run(optimizedCircuit, shots=10000)
    job_id = job.job_id()

    print(f"(Instance: {instanceOfInterest}) Submitted sampling job with ID: {job_id}")

    while job.status() in [JobStatus.QUEUED, JobStatus.INITIALIZING, JobStatus.RUNNING]:
        time.sleep(2)
    print(f"(Instance: {instanceOfInterest}) Final job status: {job.status().name}")

    if job.status() == JobStatus.DONE:
        completeJob = backendSimulator.retrieve_job(job_id)
        dist = completeJob.get_counts()
    sortedDist = sorted(dist.items(), key=lambda item: item[1], reverse=True)

    # --- Saving Results ---
    if simType == "NOISY":
        outputFilenameUnique = (
            f"{problemFileNameTag}{ionqDevice}_p{reps_p}_num_{instanceOfInterest}.json"
        )
    elif simType == "IDEAL":
        outputFilenameUnique = f"{problemFileNameTag}{backendSimulator.name}_p{reps_p}_num_{instanceOfInterest}.json"

    runMetadata = {"qaoaLayers": reps_p, "backend_name": backendSimulator.name}
    currentRunData = {
        "instance_id": instanceOfInterest,
        "sampled_distribution": sortedDist,
        "num_training_loops": qaoaCallback.numOptimisations,  # Get the count from the callback
        "final_training_cost": trainResult.fun,
        "optimal_params": trainResult.x,
    }
    dataToSave = {"metadata": runMetadata, "result": currentRunData}
    save_single_result(
        folder_path=INDIVIDUAL_RESULTS_FOLDER,
        file_name=outputFilenameUnique,
        data=dataToSave,
    )


if __name__ == "__main__":
    problemTypeToRun = (
        "MaxCut"  # options: 'TSP','Knapsack', 'MinimumVertexCover', 'MaxCut'
    )
    instancesToRun = range(1, 101)
    tasks = [(problemTypeToRun, i) for i in instancesToRun]
    maxWorkers = 100
    print(f"Starting {len(tasks)} simulations using up to {maxWorkers} threads...")

    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        # submit() returns a Future object for each task
        futureToTask = {
            executor.submit(runSingleSimulation, task): task for task in tasks
        }

        # as_completed() yields futures as they finish
        for future in as_completed(futureToTask):
            originalTask = futureToTask[future]
            try:
                # .result() will raise any exception that happened in the thread
                result = future.result()
                print(f"Task {originalTask} completed successfully.")
            except Exception as exc:
                # This block will now catch and print the error!
                print(f"Task {originalTask} generated an exception: {exc}")
