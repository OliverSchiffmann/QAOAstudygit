import sys
import json
import numpy as np
from scipy.optimize import minimize
import time

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
)  # For simulation with realistic noise

# //////////    Variables    //////////
reps_p = 20
backend_simulator = AerSimulator()
# backend_simulator = AerSimulator.from_backend(FakeTorino())


# //////////    Functions    //////////
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


def cost_func_estimator(
    params, ansatz, estimator, cost_hamiltonian_logical
):  # removed default for backend_total_qubits
    global numOptimisations
    prepared_observable = cost_hamiltonian_logical.apply_layout(ansatz.layout)
    pub = (ansatz, prepared_observable, [params])

    job = estimator.run(pubs=[pub])
    results = job.result()[0]
    cost = results.data.evs[0]

    cost_float = float(np.real(cost))
    objective_func_vals.append(cost_float)

    numOptimisations = numOptimisations + 1

    return cost_float


if __name__ == "__main__":
    file_name = sys.argv[1]
    task_id = sys.argv[2]
    instanceIndex = int(task_id)

    cost_hamiltonian, num_qubits, problem_type = load_ising_and_build_hamiltonian(
        file_name, instanceIndex
    )
    initialCircuit = None
    if problem_type == "tsp":
        initialCircuit = QuantumCircuit(num_qubits)
        initialCircuit.x(
            [0, 4, 8]
        )  # this is a solution for TSP but not other problem classes

    circuit = QAOAAnsatz(
        cost_operator=cost_hamiltonian, reps=reps_p, initial_state=initialCircuit
    )
    circuit.measure_all()

    if "fake" in backend_simulator.name.lower():
        simulator_name_for_file = (
            backend_simulator.name.split("(")[1].lower().replace(")", "")
        )  # e.g., "fakebrisbane"
    else:
        simulator_name_for_file = "aer_simulator_ideal"

    output_filename = f"isingBatches/Optimised_Params_{problem_type}_{num_qubits}q_on_{simulator_name_for_file}.txt"

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
    start_time = time.time()

    result = minimize(
        cost_func_estimator,
        initial_params,
        args=(candidate_circuit, estimator, cost_hamiltonian),
        method="COBYLA",
        tol=1e-3,
        options={"maxiter": 1000},  # Adjust as needed
    )
    output = result.x
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    cost = result.fun
    print("Optimization Result:")
    print(output)

    with open(output_filename, "a") as f:
        f.write(
            f"Instance: {instanceIndex}, Time: {elapsed_time:.4f}s, Number of optimisation loops: {numOptimisations}, Cost: {cost}, Params: {output}\n"
        )
