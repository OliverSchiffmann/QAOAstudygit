# ==============================================================================
# Ising Problem Batch Generator (openQAOA)
# ==============================================================================
# This script generates a large batch of random problem instances (QUBOs/Ising Models)
# for the specified combinatorial optimization problem (TSP, Knapsack, MVC, or MaxCut).
#
# It utilizes the 'openqaoa' library to create the problem objects, converts them
# into the Ising model format (terms, weights, constant), and saves them into a
# structured JSON file for consumption by the QAOA simulation scripts.
#
# Constraints handled:
# - Knapsack: Ensures the generated instance uses exactly 6 qubits.
# - MaxCut: Ensures the generated graph contains no isolated nodes.
# ==============================================================================
# %%
import numpy as np
from openqaoa.problems import TSP, Knapsack, MinimumVertexCover, MaximumCut
import json
import os
import networkx as nx


# %%
def generate_results_filename(problem_type, num_qubits):
    """
    Generates the standardized filename for the batch of Ising models.

    Args:
        problem_type (str): The name of the problem (e.g., 'Knapsack').
        num_qubits (int): The maximum number of qubits across all instances in the batch.

    Returns:
        str: The generated filename string.
    """
    if problem_type == "Knapsack":
        # Includes the number of items in the Knapsack filename
        problem_type = f"{problem_type}_{N_ITEMS}_items"
    # Replaces spaces with underscores and appends qubit count
    return f"batch_Ising_data_{str(problem_type).replace(' ', '_')}_{num_qubits}q_.json"


# %%
desiredProblemType = (
    "MaxCut"  # options: 'TSP','Knapsack', 'MinimumVertexCover', 'MaxCut'
)
# make sure you typed in that mf desiredProblemType correctly
batchSize = 100
RESULTS_FOLDER = "isingBatches"


all_isings_list = []
num_qubits = 0
for instance in range(batchSize):
    if desiredProblemType == "TSP":
        n_cities = 4  # Number of cities for TSP
        connection_probability = 1  # Probability for edge creation in gnp_random_graph
        # Ensure the graph is connected
        G = nx.generators.fast_gnp_random_graph(n=n_cities, p=connection_probability)

        min_edge_weight = 1
        max_edge_weight = 10
        rng_weights = (
            np.random.default_rng()
        )  # Use a different seed or manage seeds as needed
        for u, v in G.edges():
            # Assign random integer weights to graph edges
            weight = int(
                rng_weights.integers(
                    low=min_edge_weight, high=max_edge_weight, endpoint=True
                )
            )  # endpoint=True includes high value
            G.edges[u, v]["weight"] = weight
        # A is the penalty for not visiting a city or visiting it multiple times.
        # B is the penalty for not having a valid tour structure (e.g. sub-tours or incorrect number of cities at a position)
        B_val = 1
        A_val = 15
        # Create the TSP QUBO using the specified penalties A and B
        tsp_prob = TSP(G=G, A=A_val, B=B_val)  # Using your specified A and B penalties
        isingProb = tsp_prob.qubo

    elif desiredProblemType == "Knapsack":
        N_ITEMS = 4
        target_qubits = 6
        # Loop until an instance with the correct number of qubits is found
        while True:
            knapsack_prob = Knapsack.random_instance(n_items=N_ITEMS)
            isingProb = knapsack_prob.qubo
            if (
                isingProb.n == target_qubits
            ):  # Ensure the weight capacity is not 5 to avoid weird edge cases where slack variables cant describe weight capacity
                break

    elif desiredProblemType == "MinimumVertexCover":
        # Generate a random graph for MVC
        G = nx.generators.fast_gnp_random_graph(n=4, p=0.6)
        # Create the MVC QUBO
        mvc_prob = MinimumVertexCover(G, field=1, penalty=10)
        isingProb = mvc_prob.qubo

    elif desiredProblemType == "MaxCut":
        G = None
        # Loop until a graph with no isolated nodes (4 qubits) is generated
        while (
            G is None or len(list(nx.isolates(G))) > 0
        ):  # ensuring that there are no isolated nodes, which means there is always 4 qubits as well
            G = nx.generators.fast_gnp_random_graph(n=4, p=0.6)
        # Create the MaxCut QUBO
        maxCut_prob = MaximumCut(G)
        isingProb = maxCut_prob.qubo

    print(
        f"Generated Ising model for {desiredProblemType} instance {instance + 1} of {batchSize}",
        end="\r",
    )
    # Convert the openQAOA QUBO object to a dictionary format
    ising_dict = isingProb.asdict()
    current_ising_data = {
        "instance_id": instance + 1,  # Add an identifier for each QUBO instance
        "terms": ising_dict["terms"],
        "weights": ising_dict["weights"],
        "constant": ising_dict.get("constant", 0.0),
        "problem_type": ising_dict.get("problem_instance", {}).get(
            "problem_type", "unknown"
        ),
        # Optional: include more problem-specific details if needed for later analysis, can add the number of qubits/problem size
    }

    if desiredProblemType == "Knapsack":
        current_ising_data["weight_capacity"] = (
            knapsack_prob.weight_capacity
        )  # necessary for custom inital state generation

    all_isings_list.append(current_ising_data)

    # Track the maximum number of qubits used across all generated instances
    all_indices = []
    terms = ising_dict["terms"]
    for term_group in terms:
        for idx in term_group:
            all_indices.append(idx)
    if max(all_indices) + 1 > num_qubits:
        num_qubits = max(all_indices) + 1

# Generate the final filename based on the collected qubit count
file_name = generate_results_filename(desiredProblemType, num_qubits)
results_filename_with_path = os.path.join(RESULTS_FOLDER, file_name)

# Ensure the output directory exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Write the list of Ising models to the JSON file
with open(results_filename_with_path, "w") as f:
    f.write("[\n")  # Start of the JSON array
    for i, single_ising_data in enumerate(all_isings_list):
        ising_json_string = json.dumps(single_ising_data)
        f.write("  " + ising_json_string)

        # Add a comma separator unless it's the last item
        if i < len(all_isings_list) - 1:
            f.write(",\n")
        else:
            f.write("\n")

    f.write("]\n")

print(
    f"\nBatch of {len(all_isings_list)} Ising models saved to {results_filename_with_path}"
)
