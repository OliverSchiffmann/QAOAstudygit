import sys
import os
from filelock import FileLock
import json
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

# Packages for quantum stuff
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    QiskitRuntimeService,
    SamplerV2 as Sampler,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import (
    FakeBrisbane,
    FakeSherbrooke,
    FakeTorino,
)


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
    return hamiltonian, num_qubits, problem_type


def cost_func_estimator(params, ansatz, estimator, cost_hamiltonian):
    global numOptimisations
    transpiledHamil = cost_hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, transpiledHamil, params)

    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs

    cost_float = float(np.real(cost))
    objective_func_vals.append(cost_float)

    numOptimisations = numOptimisations + 1

    return cost_float


def build_mixer_hamiltonian(constraints, num_qubits):
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


def save_results_to_json(
    results_folder, metadata, run_result, file_name="results.json"
):
    """
    Saves QAOA run results to a JSON file, handling concurrent writes.

    If the file doesn't exist, it's created with metadata and the first result.
    If it exists, the new result is appended to the 'results' list.
    A file lock is used to prevent race conditions from batch jobs.

    Args:
        results_folder (str): The name of the folder to save results in.
        metadata (dict): A dictionary with overall run info like 'qaoaLayers'.
        run_result (dict): A dictionary containing the results of this specific run.
        file_name (str): The name of the JSON file to save to.
    """
    # Ensure the target directory exists
    os.makedirs(results_folder, exist_ok=True)
    file_path = os.path.join(results_folder, file_name)

    # Use a lock file to prevent race conditions during file access
    lock_path = file_path + ".lock"
    lock = FileLock(lock_path)

    with lock:
        try:
            # If file exists, read it, append the new result, and write back
            with open(file_path, "r") as f:
                data = json.load(f)
            data["results"].append(run_result)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is empty/corrupt, create the initial structure
            data = {"metadata": metadata, "results": [run_result]}

        # Write the updated data back to the file
        with open(file_path, "w") as f:
            # Use the custom encoder for NumPy arrays and indent for readability
            json.dump(data, f, indent=4, cls=NumpyArrayEncoder)


if __name__ == "__main__":
    # debugging variables
    # instanceOfInterest = 6

    # //////////    Variables    //////////
    reps_p = 20
    backend_simulator = AerSimulator()
    # backend_simulator = AerSimulator.from_backend(FakeTorino())
    instanceOfInterest = int(sys.argv[2])
    FILEDIRECTORY = "isingBatches"
    isingFileName = FILEDIRECTORY + "/batch_Ising_data_TSP_9q_.json"
    outputFilename = "TSP_QAOA_IBM_batch_results.json"

    # /// training ///
    # create cost hamiltonian for ising model
    costHamil, numQubits, problemType = load_ising_and_build_hamiltonian(
        isingFileName, instanceOfInterest
    )

    # create mixer hamiltonian specific to TSP
    city_constraints = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]  # Each city must be visited once (rows in a 3x3 grid)
    time_constraints = [
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
    ]  # Each time step can only have one city (columns in a 3x3 grid)
    all_constraint_groups = city_constraints + time_constraints
    mixerHamil = build_mixer_hamiltonian(all_constraint_groups, numQubits)

    # qaoa circuit building
    initialCircuit = QuantumCircuit(numQubits)
    initialCircuit.x([0, 4, 8])
    qaoaKwargs = {
        "cost_operator": costHamil,
        "reps": reps_p,
        "initial_state": initialCircuit,
        "mixer_operator": mixerHamil,
    }
    circuit = QAOAAnsatz(**qaoaKwargs)
    circuit.measure_all()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend_simulator)
    candidate_circuit = pm.run(circuit)

    # creating random inital parameters
    num_params = 2 * reps_p
    initial_betas = (np.random.rand(reps_p) * np.pi).tolist()
    initial_gammas = (np.random.rand(reps_p) * (np.pi)).tolist()
    initial_params = initial_betas + initial_gammas

    # starting training loop
    objective_func_vals = []
    numOptimisations = 0
    estimator = Estimator(mode=backend_simulator)
    trainResult = minimize(
        cost_func_estimator,
        initial_params,
        args=(candidate_circuit, estimator, costHamil),
        method="COBYLA",  # Using COBYLA for gradient free optimization also fast
        tol=1e-3,
        options={"maxiter": 1000},
    )
    print(trainResult.x, trainResult.fun, numOptimisations)

    # /// Sampling ///
    # Assigning the optimized parameters to the circuit
    optimized_circuit = candidate_circuit.assign_parameters(trainResult.x)

    # setting backend for sampling
    sampler = Sampler(mode=backend_simulator)
    sampler.options.default_shots = 1000

    # collecting distribution
    sampleResult = sampler.run([optimized_circuit]).result()
    dist = sampleResult[0].data.meas.get_counts()
    sortedDist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    print("Distribution:", sortedDist)

    # /// Saving results ///
    run_metadata = {"qaoaLayers": reps_p, "backend_name": backend_simulator.name}
    current_run_data = {
        "instance_id": instanceOfInterest,
        "sampled_distribution": sortedDist,
        "num_training_loops": numOptimisations,
        "final_training_cost": trainResult.fun,
        "optimal_params": trainResult.x,
    }
    save_results_to_json(
        results_folder=FILEDIRECTORY,
        metadata=run_metadata,
        run_result=current_run_data,
        file_name=outputFilename,
    )
