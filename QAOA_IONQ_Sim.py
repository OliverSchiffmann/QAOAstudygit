import argparse
import os
import json
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from config import problem_configs
from dotenv import load_dotenv
from requests.exceptions import ConnectionError
from concurrent.futures import as_completed, ProcessPoolExecutor


# Packages for quantum stuff
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.providers import JobStatus
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    QiskitRuntimeService,
    SamplerV2 as Sampler,
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


class NumpyArrayEncoder(json.JSONEncoder):
    """A JSON Encoder that can handle numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# ///////////  Functions  //////////
def load_ising_and_build_hamiltonian(file_path, instance_id):
    """
    Loads Ising terms and weights from a JSON file.
    Determines the number of qubits from the terms and constructs
    the Hamiltonian as a Qiskit SparsePauliOp.
    """

    with open(file_path, "r") as f:
        all_isings_data = json.load(f)  # Assumes this loads a list of dicts

    selected_ising_data = None
    # Find the desired ising model within list
    for ising_instance in all_isings_data:
        if (
            ising_instance["instance_id"] == instance_id
        ):  # Assumes 'instance_id' exists and is correct
            selected_ising_data = ising_instance
            break

    terms = selected_ising_data["terms"]
    weights = selected_ising_data["weights"]
    problem_type = selected_ising_data.get("problem_type")
    print(
        f"(Instance: {instance_id}) Problem type found from ising data: {problem_type}"
    )

    pauli_list = []
    num_qubits = 0

    # Find the max number of qubits by finding the biggest index of ising variables
    all_indices = []
    for term_group in terms:
        for idx in term_group:
            all_indices.append(idx)
    num_qubits = max(all_indices) + 1

    for term_indices, weight in zip(terms, weights):
        paulis_arr = ["I"] * num_qubits
        if len(term_indices) == 1:  # Linear term
            paulis_arr[term_indices[0]] = "Z"
        elif len(term_indices) == 2:  # Quadratic term
            paulis_arr[term_indices[0]] = "Z"
            paulis_arr[term_indices[1]] = "Z"

        pauli_list.append(
            ("".join(paulis_arr)[::-1], weight)
        )  # how from_list works here: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparsePauliOp
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    if problem_type == "knapsack":
        weight_capacity = selected_ising_data.get("weight_capacity")
        return hamiltonian, num_qubits, terms, weight_capacity
    return hamiltonian, num_qubits, terms, None


def save_single_result(folder_path, file_name, data):
    """Saves a single data dictionary to a JSON file. No locking or reading needed."""
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyArrayEncoder)


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


def build_mixer_hamiltonian(num_qubits, problem_type, instance_id):
    if problem_type == "TSP":
        print(f"(Instance: {instance_id}) Building mixer Hamiltonian for TSP...")
        if num_qubits != 9:
            raise ValueError("TSP mixer Hamiltonian only works for exactly 9 qubits.")
        # Each city must be visited once (rows in a 3x3 grid)
        city_constraints = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # Each time step can only have one city (columns in a 3x3 grid)
        time_constraints = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        # Combine all constraint groups
        constraints = city_constraints + time_constraints
        pauli_list = []
        for group in constraints:
            # Create pairs of all qubits within the constrained group
            for qubit_pair in combinations(group, 2):
                # Create the XX term
                xx_pauli = ["I"] * num_qubits
                xx_pauli[qubit_pair[0]] = "X"
                xx_pauli[qubit_pair[1]] = "X"
                # Add to the list (in Qiskit's reversed order) with a coefficient of 1.0
                pauli_list.append(("".join(xx_pauli)[::-1], 1.0))

                # Create the YY term
                yy_pauli = ["I"] * num_qubits
                yy_pauli[qubit_pair[0]] = "Y"
                yy_pauli[qubit_pair[1]] = "Y"
                pauli_list.append(("".join(yy_pauli)[::-1], 1.0))
        mixer_hamiltonian = SparsePauliOp.from_list(pauli_list)
        return mixer_hamiltonian
    elif problem_type == "Knapsack":
        print(f"(Instance: {instance_id}) Building mixer Hamiltonian for Knapsack...")
        pauli_list = []
        # Add standard X-mixer terms for all ITEM qubits (indices 3 to 8)
        item_indices = range(3, num_qubits)
        for i in item_indices:
            x_pauli = ["I"] * num_qubits
            x_pauli[i] = "X"
            pauli_list.append(("".join(x_pauli)[::-1], 1.0))

        # Add X-mixer terms for ONLY the specified slack variables (only flipping first as most optimal knapsacks are first)
        restricted_slack_indices = [0]
        for i in restricted_slack_indices:
            x_pauli = ["I"] * num_qubits
            x_pauli[i] = "X"
            pauli_list.append(("".join(x_pauli)[::-1], 1.0))

        mixer_hamiltonian = SparsePauliOp.from_list(pauli_list)
        return mixer_hamiltonian
    elif problem_type == "MinimumVertexCover":
        print(
            f"(Instance: {instance_id}) Building Mixer Hamiltonian for Minimum Vertex Cover..."
        )

        # edges = []
        # for term in terms:
        #     # A quadratic term (representing an edge in the original problem graph) is a list of two indices
        #     if len(term) == 2:
        #         edges.append(tuple(term))

        # pauli_list = []

        # for i in range(num_qubits):
        #     x_pauli = ["I"] * num_qubits
        #     x_pauli[i] = "X"
        #     pauli_list.append(("".join(x_pauli)[::-1], 1.0))

        # for qubit_pair in edges:
        #     # Create the XX term
        #     xx_pauli = ["I"] * num_qubits
        #     xx_pauli[qubit_pair[0]] = "X"
        #     xx_pauli[qubit_pair[1]] = "X"
        #     pauli_list.append(("".join(xx_pauli)[::-1], 1.0))

        #     # Create the YY term
        #     yy_pauli = ["I"] * num_qubits
        #     yy_pauli[qubit_pair[0]] = "Y"
        #     yy_pauli[qubit_pair[1]] = "Y"
        #     pauli_list.append(("".join(yy_pauli)[::-1], 1.0))

        # mixer_hamiltonian = SparsePauliOp.from_list(pauli_list) # This was an attempt at an edge based mixer but ti didnt seem much better

        pauli_list = []
        for i in range(num_qubits):
            # Create an X operator on the i-th qubit
            x_pauli = ["I"] * num_qubits
            x_pauli[i] = "X"
            # Add to the list (in Qiskit's reversed order) with a coefficient of 1.0
            pauli_list.append(("".join(x_pauli)[::-1], 1.0))
        mixer_hamiltonian = SparsePauliOp.from_list(pauli_list)

        return mixer_hamiltonian
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def create_inital_state(num_qubits, problem_type, weight_capacity=None):
    """
    Creates an initial state circuit for the given number of qubits and problem type.
    """
    initial_circuit = QuantumCircuit(num_qubits)

    if problem_type == "TSP":
        # starting with simplest obvious scenario, city 0 at time 0, city 1 at time 1, city 2 at time 2
        initial_circuit.x([0, 4, 8])
    elif problem_type == "Knapsack":
        initial_circuit.x([3])

    elif problem_type == "MinimumVertexCover":
        # initial_circuit.h(range(num_qubits))
        initial_circuit.x(
            [0, 1, 2, 3, 4, 5, 6, 7]
        )  # qubits take value 1 to represent node inclusion in set, last qubit (8) left as 0 to represent inital state with all but one node in the cover set
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    return initial_circuit


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
    INDIVIDUAL_RESULTS_FOLDER = "individual_results"
    reps_p = 20  # Number of QAOA layers

    # --- Backend Setup ---
    ionqApiToken = os.environ.get("IONQ_API_TOKEN")
    provider = IonQProvider(token=ionqApiToken)
    backendSimulator = provider.get_backend("ionq_simulator", gateset="native")
    backendSimulator.options.ionq_compiler_synthesis = True

    # --- Training ---
    costHamil, numQubits, isingTerms, weightCapacity = load_ising_and_build_hamiltonian(
        isingFileName, instanceOfInterest
    )
    mixerHamil = build_mixer_hamiltonian(numQubits, problemType, instanceOfInterest)
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
    outputFilenameUnique = (
        f"{problemFileNameTag}{backendSimulator.name}_num_{instanceOfInterest}.json"
    )
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
    problemTypeToRun = "Knapsack"  # options: 'TSP','Knapsack', 'MinimumVertexCover'
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
