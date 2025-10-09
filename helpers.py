import json
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
import os
import numpy as np
from itertools import combinations


class NumpyArrayEncoder(json.JSONEncoder):
    """A JSON Encoder that can handle numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calculate_ising_energy(binarySolutionString, terms, weights):
    spins = []
    for bit in binarySolutionString:
        if bit == "0":
            spins.append(1)
        else:
            spins.append(-1)
    total_energy = 0
    # Iterate through the Ising model terms and weights
    for term, weight in zip(terms, weights):
        if len(term) == 1:  # Linear term (h_i * s_i)
            i = term[0]
            total_energy += weight * spins[i]
        elif len(term) == 2:  # Quadratic term (J_ij * s_i * s_j)
            i, j = term[0], term[1]
            total_energy += weight * spins[i] * spins[j]

    return total_energy


def load_ising_and_build_hamiltonian(file_path, instance_id):
    """
    Loads Ising terms and weights from a JSON file.
    Determines the number of qubits from the terms and constructs
    the Hamiltonian as a Qiskit SparsePauliOp.
    """

    with open(file_path, "r") as f:
        all_isings_data = json.load(f)  # Assumes this loads a list of dicts

    selected_ising_data = {}
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


def build_mixer_hamiltonian(num_qubits, problem_type):
    if problem_type == "TSP":
        print("Building mixer Hamiltonian for TSP...")
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
        print("Building mixer Hamiltonian for Knapsack...")
        pauli_list = []
        # Add standard X-mixer terms for all ITEM qubits (indices 3 to 8)
        item_indices = range(3, num_qubits)
        for i in item_indices:
            x_pauli = ["I"] * num_qubits
            x_pauli[i] = "X"
            pauli_list.append(("".join(x_pauli)[::-1], 1.0))

        restricted_slack_indices = [
            0,
            1,
        ]  # Add X-mixer terms for ONLY the specified slack variables
        for i in restricted_slack_indices:
            x_pauli = ["I"] * num_qubits
            x_pauli[i] = "X"
            pauli_list.append(("".join(x_pauli)[::-1], 1.0))

        mixer_hamiltonian = SparsePauliOp.from_list(pauli_list)
        return mixer_hamiltonian
    elif problem_type == "MinimumVertexCover":
        print("Building Mixer Hamiltonian for Minimum Vertex Cover...")
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
        initial_circuit.x(
            [0, 1]
        )  # empty knapsack with no items i guaranteed valid solution

    elif problem_type == "MinimumVertexCover":
        # initial_circuit.h(range(num_qubits))
        initial_circuit.x(
            [0, 1, 2, 3]
        )  # qubits take value 1 to represent node inclusion in set, include all nodes to ensure valid solution
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    return initial_circuit
