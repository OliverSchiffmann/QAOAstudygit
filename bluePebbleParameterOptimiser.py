import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Packages for quantum stuff
from qiskit.quantum_info import SparsePauliOp
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
)  # For simulation with realistic noise

file_name = "QUBO_batches/batch_QUBO_data_MaxCut_6q_.json"


def load_qubo_and_build_hamiltonian(file_path, instance_id):
    """
    Loads QUBO terms, weights, and constant from a JSON file.
    Determines the number of qubits from the terms and constructs
    the Hamiltonian as a Qiskit SparsePauliOp.
    """

    with open(file_path, "r") as f:
        all_qubos_data = json.load(f)  # Assumes this loads a list of dicts

    selected_qubo_data = None
    # Find the dictionary for the specified instance_id
    for qubo_instance in all_qubos_data:
        if (
            qubo_instance["instance_id"] == instance_id
        ):  # Assumes 'instance_id' exists and is correct
            selected_qubo_data = qubo_instance
            break

    terms = selected_qubo_data["terms"]
    weights = selected_qubo_data["weights"]
    constant = selected_qubo_data.get("constant", 0.0)
    problemType = selected_qubo_data.get("problem_type")

    pauli_list = []
    num_qubits = 0

    if terms:
        # Flatten the list of lists and filter out empty sublists or non-integer elements
        all_indices = []
        for term_group in terms:
            for idx in term_group:
                all_indices.append(idx)
        num_qubits = max(all_indices) + 1

    for term_indices, weight in zip(terms, weights):
        if not term_indices or not all(isinstance(idx, int) for idx in term_indices):
            # Skip if term_indices is empty or contains non-integers
            continue

        paulis_arr = ["I"] * num_qubits
        if len(term_indices) == 1:  # Linear term
            paulis_arr[term_indices[0]] = "Z"
        elif len(term_indices) == 2:  # Quadratic term
            paulis_arr[term_indices[0]] = "Z"
            paulis_arr[term_indices[1]] = "Z"

        pauli_list.append(("".join(paulis_arr)[::-1], weight))

    if (
        not pauli_list and num_qubits > 0
    ):  # No valid Pauli terms were created, but num_qubits > 0
        cost_hamiltonian = SparsePauliOp(
            ["I"] * num_qubits, [0]
        )  # Zero operator on n_qubits
    elif not pauli_list and num_qubits == 0:
        cost_hamiltonian = SparsePauliOp(
            "I", [0]
        )  # Placeholder for 1 qubit if everything is empty
    else:
        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)

    return cost_hamiltonian, constant, num_qubits, problemType


def cost_func_estimator(
    params,
    ansatz,
    estimator,
    cost_hamiltonian_logical,
    constant_offset,
    backend_total_qubits=127,
):  # removed default for backend_total_qubits
    global numOptimisations
    prepared_observable = cost_hamiltonian_logical.apply_layout(ansatz.layout)
    pub = (ansatz, prepared_observable, [params])

    job = estimator.run(pubs=[pub])
    results = job.result()[0]
    cost = results.data.evs[0]

    cost_float = float(np.real(cost)) + constant_offset
    objective_func_vals.append(cost_float)

    numOptimisations = numOptimisations + 1
    # Your desired print format:
    print(
        f"Params: {params}, Cost: {cost_float}, Optimisation Round: {numOptimisations}"
    )

    return cost_float


# variables
instanceIndex = 5
reps_p = 2


cost_hamiltonian, constant_offset, num_qubits, problem_type = (
    load_qubo_and_build_hamiltonian(file_name, instanceIndex)
)
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps_p)
circuit.measure_all()

backend_simulator = AerSimulator()  # Ideal simulator
# backend_simulator = AerSimulator.from_backend(FakeBrisbane()) # Simulator with IBM_Brisbane specific noise model
# backend_simulator = AerSimulator.from_backend(FakeSherbrooke()) # Simulator with IBM_Sherbroke specific noise model
pm = generate_preset_pass_manager(optimization_level=3, backend=backend_simulator)
candidate_circuit = pm.run(circuit)

num_params = 2 * reps_p
initial_betas = (np.random.rand(reps_p) * np.pi).tolist()
initial_gammas = (np.random.rand(reps_p) * (np.pi)).tolist()
initial_params = initial_betas + initial_gammas

objective_func_vals = []
numOptimisations = 0

estimator = Estimator(mode=backend_simulator)
print("Starting optimization...")
result = minimize(
    cost_func_estimator,
    initial_params,
    args=(candidate_circuit, estimator, cost_hamiltonian, constant_offset),
    method="COBYLA",
    tol=1e-3,
    options={"maxiter": 1000},  # Adjust as needed
)
print("Optimization Result:")
print(result.x)
