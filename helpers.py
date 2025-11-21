# ==============================================================================
# QAOA Core Helper Functions
# ==============================================================================
# This module provides critical utilities for the entire QAOA workflow, linking
# the classical combinatorial optimization problems (MaxCut, Knapsack, etc.)
# defined by their Ising models to the quantum execution environment (Qiskit).
#
# Key functionalities include:
# 1. Serialization: Enabling the saving/loading of data containing NumPy arrays.
# 2. Energy Calculation: Converting bitstrings to classical Ising energy values.
# 3. Hamiltonian Construction: Loading Ising models and dynamically building the
#    Cost (H_C) and custom Mixer (H_M) Hamiltonians as Qiskit SparsePauliOp objects.
# 4. Initial State Creation: Defining problem-specific initial quantum circuits.
# ==============================================================================
import json
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
import os
import numpy as np
from itertools import combinations


class NumpyArrayEncoder(json.JSONEncoder):
    """
    A custom JSON Encoder that handles numpy arrays by converting them to standard
    Python lists before serialization.

    This ensures that data containing optimal QAOA parameters or other NumPy
    structures can be saved correctly to JSON files.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calculate_ising_energy(binarySolutionString, terms, weights):
    """
    Calculates the classical Ising energy of a given binary solution string.

    The binary solution string is converted to spin values (+1/-1). The energy
    is computed using the linear (s_i) and quadratic (s_i * s_j) terms defined
    in the Ising model.

    Args:
        binarySolutionString (str): The solution bitstring (e.g., '1010').
        terms (list): List of qubit indices defining the terms (e.g., [[0], [1, 2]]).
        weights (list): Corresponding coefficients for the terms (e.g., [1.0, -0.5]).

    Returns:
        float: The calculated total Ising energy (cost) of the solution.
    """
    spins = []
    # Convert binary (0, 1) to Ising spins (+1, -1)
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
    Loads Ising terms and weights from a JSON file for a specific instance,
    determines the number of qubits, and constructs the Cost Hamiltonian (H_C)
    as a Qiskit SparsePauliOp.

    Args:
        file_path (str): Path to the JSON file containing all Ising models.
        instance_id (int): The unique ID of the problem instance to load.

    Returns:
        tuple: (Hamiltonian as SparsePauliOp, number of qubits, Ising terms,
                Knapsack weight capacity (or None)).
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

    # Convert Ising terms (indices) into Qiskit Pauli strings (Z or I)
    for term_indices, weight in zip(terms, weights):
        paulis_arr = ["I"] * num_qubits
        if len(term_indices) == 1:  # Linear term (Z operator on one qubit)
            paulis_arr[term_indices[0]] = "Z"
        elif len(term_indices) == 2:  # Quadratic term (ZZ operator on two qubits)
            paulis_arr[term_indices[0]] = "Z"
            paulis_arr[term_indices[1]] = "Z"

        # Note: Qiskit requires Pauli strings to be reversed (qubit 0 is rightmost)
        pauli_list.append(
            ("".join(paulis_arr)[::-1], weight)
        )  # how from_list works here: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparsePauliOp
    hamiltonian = SparsePauliOp.from_list(pauli_list)

    weight_capacity = selected_ising_data.get("weight_capacity")
    if problem_type == "knapsack" and weight_capacity is not None:
        return hamiltonian, num_qubits, terms, weight_capacity
    return hamiltonian, num_qubits, terms, None


def save_single_result(folder_path, file_name, data):
    """
    Saves a single data dictionary to a JSON file.

    It creates the output directory if it doesn't exist and uses the
    NumpyArrayEncoder to handle any NumPy array objects within the data.

    Args:
        folder_path (str): The directory where the file will be saved.
        file_name (str): The name of the JSON file.
        data (dict): The dictionary containing the results to save.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyArrayEncoder)


def build_mixer_hamiltonian(num_qubits, problem_type):
    """
    Constructs the problem-specific Mixer Hamiltonian (H_M) as a Qiskit
    SparsePauliOp.

    For MaxCut/MVC, it uses the standard X-mixer. For Knapsack and TSP,
    custom mixers are implemented to incorporate problem constraints.

    Args:
        num_qubits (int): The total number of qubits in the system.
        problem_type (str): The name of the optimization problem.

    Returns:
        SparsePauliOp: The constructed Mixer Hamiltonian.

    Raises:
        ValueError: If an unsupported problem type is requested.
    """
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
    elif problem_type in ["MinimumVertexCover", "MaxCut"]:
        print(f"Building standard X-mixer for {problem_type}...")
        pauli_list = []
        # Standard X-mixer: Sum of X operators on all qubits
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
    Creates an initial state quantum circuit for the QAOA Ansatz.

    The circuit is configured based on the problem type to potentially bias the
    initial state towards a known, valid solution space, rather than a generic
    uniform superposition (Hadamards).

    Args:
        num_qubits (int): The number of qubits in the circuit.
        problem_type (str): The name of the optimization problem.
        weight_capacity (int, optional): The knapsack capacity, if applicable.

    Returns:
        QuantumCircuit: The initialized quantum circuit.

    Raises:
        ValueError: If an unsupported problem type is requested.
    """
    initial_circuit = QuantumCircuit(num_qubits)

    if problem_type == "TSP":
        # starting with simplest obvious scenario, city 0 at time 0, city 1 at time 1, city 2 at time 2
        # This corresponds to bitstring 100 001 001 (TSP is 3x3 grid)
        initial_circuit.x([0, 4, 8])

    elif problem_type == "Knapsack":
        # Initial state guarantees the slack qubits represent an empty knapsack,
        # which is always a valid solution (zero items selected).
        initial_circuit.x(
            [0, 1]
        )  # empty knapsack with no items i guaranteed valid solution

    elif problem_type == "MinimumVertexCover":
        # The initial state is set to include all nodes, which is a trivial but
        # guaranteed valid vertex cover (all 4 problem qubits set to 1).
        # initial_circuit.h(range(num_qubits)) # Standard uniform superposition
        initial_circuit.x(
            [0, 1, 2, 3]
        )  # qubits take value 1 to represent node inclusion in set, include all nodes to ensure valid solution
    elif problem_type == "MaxCut":
        # Sets the initial state to 1000 (only qubit 0 is 1, rest 0), which is a
        # simple, obvious valid partition (node 0 in set A, rest in set B).
        initial_circuit.x(
            [0]
        )  # most obvious valid solution is 1000 since all nodes have an edge
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    return initial_circuit
