# ==============================================================================
# QAOA Solution Space Analyzer
# ==============================================================================
# This script performs a full enumeration analysis (2^N solutions) for a single
# instance of each combinatorial optimization problem (TSP, Knapsack, MVC, MaxCut).
#
# Key functionalities:
# 1. Problem Generation: Creates one specific problem instance for each type.
# 2. Full Enumeration: Generates all 2^N possible bitstrings (solutions).
# 3. Validity Check: Determines if each bitstring is a valid solution according
#    to the classical problem constraints (e.g., meeting capacity in Knapsack).
# 4. Performance Scoring: Calculates the normalized Performance Score for every
#    solution relative to the optimal and maximum possible costs.
# 5. Visualization: Plots the resulting Performance Score distribution, colored
#    by validity, and highlights the initial quantum state bitstrings.
# ==============================================================================
import numpy as np
from openqaoa.problems import TSP, Knapsack, MinimumVertexCover, MaximumCut
import networkx as nx
import matplotlib.pyplot as plt
from config import problem_configs
from helpers import calculate_ising_energy

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 14


def generate_tsp_problem(cities):
    """
    Generates a random, fully connected Traveling Salesperson Problem (TSP) instance
    using an openQAOA 3x3 grid encoding for N cities (where N^2 = qubits).

    Args:
        cities (int): The number of cities (e.g., 4 cities uses 9 qubits).

    Returns:
        TSP: The openQAOA TSP problem object.
    """
    n_cities = cities  # Number of cities for TSP
    connection_probability = 1  # Ensure the graph is connected
    G = nx.generators.fast_gnp_random_graph(n=n_cities, p=connection_probability)

    min_edge_weight = 1
    max_edge_weight = 10
    rng_weights = (
        np.random.default_rng()
    )  # Use a different seed or manage seeds as needed
    for u, v in G.edges():
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
    prob = TSP(G=G, A=A_val, B=B_val)
    return prob


def generate_knapsack_problem(items, qubits, bannedCapacities):
    """
    Generates a Knapsack instance, looping until the QUBO representation matches
    the specified target qubit count and avoids known problematic weight capacities.

    Args:
        items (int): The number of items in the knapsack.
        qubits (int): The target number of qubits (items + slack variables).
        bannedCapacities (list): List of weight capacities to avoid.

    Returns:
        Knapsack: The openQAOA Knapsack problem object.
    """
    n_items = items
    target_qubits = qubits
    # Loop until an instance with the correct number of qubits is found
    while True:
        prob = Knapsack.random_instance(n_items=n_items)
        isingProb = prob.qubo
        if (
            isingProb.n == target_qubits
            and prob.weight_capacity not in bannedCapacities
        ):  # Ensure the weight capacity is not 5 to avoid weird edge cases where slack variables cant describe weight capacity
            break
    return prob


def generate_minimum_vertex_cover_problem(nodes):
    """
    Generates a random Minimum Vertex Cover (MVC) instance on a graph with N nodes.

    Args:
        nodes (int): The number of nodes in the graph.

    Returns:
        MinimumVertexCover: The openQAOA MVC problem object.
    """
    G = nx.generators.fast_gnp_random_graph(n=nodes, p=0.6)
    prob = MinimumVertexCover(G, field=1, penalty=10)
    return prob


def generate_max_cut_problem(nodes):
    """
    Generates a random Maximum Cut (MaxCut) instance on a graph with N nodes,
    looping to ensure the resulting graph has no isolated nodes.

    Args:
        nodes (int): The number of nodes in the graph.

    Returns:
        MaximumCut: The openQAOA MaxCut problem object.
    """
    G = None
    while (
        G is None or len(list(nx.isolates(G))) > 0
    ):  # ensuring that there are no isolated nodes, which means there is always 4 qubits as well
        G = nx.generators.fast_gnp_random_graph(n=nodes, p=0.6)
    prob = MaximumCut(G)
    return prob


def generate_all_bitstrings(length):
    """
    Generates a list of all 2^N binary strings of a specified length.

    Args:
        length (int): The length of the bitstrings (N).

    Returns:
        list: A list of all bitstrings (e.g., ['00', '01', '10', '11']).
    """
    return [bin(i)[2:].zfill(length) for i in range(2**length)]


def check_tsp_validity(bitstring, cities, qubits):
    """
    Checks if a bitstring represents a valid TSP tour based on the 3x3 encoding.

    A valid tour must satisfy two constraints simultaneously:
    1. Each city is visited exactly once.
    2. Exactly one city is visited at each time step.

    Args:
        bitstring (str): The bitstring solution.
        cities (int): The number of cities (e.g., 4).
        qubits (int): The total number of qubits (e.g., 9).

    Returns:
        bool: True if the bitstring represents a valid tour, False otherwise.
    """
    citiesToVisit = cities - 1
    # This set will store the positions (columns) where a '1' has been found.
    foundPositions = set()

    if len(bitstring) != qubits:
        return False

    for i in range(citiesToVisit):
        # Calculate the start and end index for the current chunk
        start = i * citiesToVisit
        end = start + citiesToVisit
        chunk = bitstring[start:end]

        # Constraint 1: Check that exactly one city is visited at this time step (chunk)
        if chunk.count("1") != 1:
            return False

        # Constraint 2: Check that each city is visited only once (via position)
        position = chunk.find("1")
        if position in foundPositions:
            return False

        foundPositions.add(position)
    return True


def check_knapsack_validity(
    bitString, itemWeights, weightCapacity, numQubits, cost, costUpperBound
):
    """
    Checks if a bitstring is a valid Knapsack solution.

    A valid solution must satisfy two conditions related to the QUBO encoding:
    1. The total weight of selected items does not exceed the capacity.
    2. The integer value of the slack variables exactly equals the remaining space.

    Args:
        bitString (str): The bitstring solution (slack bits first, then item bits).
        itemWeights (list): List of weights for each item.
        weightCapacity (int): The maximum allowed total weight.
        numQubits (int): Total number of qubits.
        cost (float): The energy cost (used for debug printing only).
        costUpperBound (float): A reference cost (used for debug printing only).

    Returns:
        bool: True if the solution is valid, False otherwise.
    """
    numItems = len(itemWeights)

    numSlackVariables = numQubits - numItems
    if numSlackVariables < 0:
        print("Error: numQubits cannot be less than numItems.")
        return False

    slackString = bitString[0:numSlackVariables]
    # Item bits are the last 'numItems' bits
    itemString = bitString[numSlackVariables:]

    totalWeight = 0
    for i in range(numItems):
        if itemString[i] == "1":
            totalWeight += itemWeights[i]

    # Condition 1: Check capacity constraint
    if totalWeight > weightCapacity:
        return False

    remainingSpace = weightCapacity - totalWeight

    intSlackValue = 0
    if numSlackVariables > 0:
        # Convert the binary slack string to its integer value
        reversedSlackString = slackString[
            ::-1
        ]  # needs reversing because of the way openQAOA orders bits
        intSlackValue = int(reversedSlackString, 2)
    # If numSlackVariables is 0, slackValue correctly remains 0.

    # Condition 2: The integer value of the slack bits must *exactly* match the remaining space.
    # if intSlackValue == remainingSpace or bitString == "110000":
    if intSlackValue == remainingSpace:
        return True
    else:
        return False


def check_mvc_validity(bitString, problemInfo):
    """
    Checks if a bitstring is a valid Minimum Vertex Cover (MVC).

    A solution is valid if, for every edge in the graph, at least one of its
    endpoints is included in the vertex cover (represented by a '1' bit).

    Args:
        bitString (str): The bitstring solution (node i = bit i).
        problemInfo (dict): The dictionary containing the graph structure.

    Returns:
        bool: True if the solution is a valid vertex cover, False otherwise.
    """
    try:
        edges = problemInfo["problem_instance"]["G"]["links"]
    except KeyError:
        print("Error: Could not find 'problem_instance.G.links' in the dictionary.")
        return False

    # Check the length of the bitstring
    numNodes = problemInfo["n"]
    if len(bitString) != numNodes:
        print(
            f"Error: Bitstring length ({len(bitString)}) does not match node count ({numNodes})."
        )
        return False

    # Iterate through every edge in the graph
    for edge in edges:
        node1 = edge["source"]
        node2 = edge["target"]

        # Check the failure condition:
        # If *both* nodes for this edge are '0' (not in the cover),
        # then the edge is NOT covered, and the solution is invalid.
        if bitString[node1] == "0" and bitString[node2] == "0":
            return False
    return True


def plotSolutionData(allSolutionData, highlightBitstrings, plotType="scatter"):
    """
    Plots the full performance score distribution with validity coloring,
    highlighting specific bitstrings (like the initial state) with a blue circle,
    using jitter to separate overlapping points.

    Args:
        allSolutionData (dict): Nested dictionary containing cost and validity
                                for every bitstring for every problem.
        highlightBitstrings (dict): Dictionary mapping problem names to the
                                    bitstrings that should be highlighted.
        plotType (str): Type of plot to generate (currently only 'scatter' is fully supported).
    """
    numProblems = len(allSolutionData)
    if numProblems == 0:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(4 * numProblems, 7))
    # fig.suptitle("Performance Score Distribution of All $2^N$ Solutions", fontsize=16)

    plotLabels = []

    for j, (problemName, solutions) in enumerate(allSolutionData.items()):
        print(f"--- Plotting {problemName} ---")

        if not solutions:
            print(f"Warning: No solution data found for {problemName}.")
            continue

        # Extract costs, validity, and bitstrings
        allCosts = []
        allValidity = []
        allBitstrings = []
        for bitstring, info in solutions.items():
            allCosts.append(info["cost"])
            allValidity.append(info["valid"])
            allBitstrings.append(bitstring)

        # Calculate Performance Scores
        allCosts = np.array(allCosts)
        optimumCost = np.min(allCosts)
        maximumCost = np.max(allCosts)
        energyRange = maximumCost - optimumCost

        if energyRange == 0:
            performanceScores = np.ones_like(allCosts)
        else:
            performanceScores = 1.0 - ((allCosts - optimumCost) / energyRange)

        # --- NEW: Calculate jitter ONCE for all points of the current problem ---
        numTotalSolutions = len(performanceScores)
        # Apply random Gaussian jitter to the x-axis for visual separation
        jitterValues = np.random.normal(loc=0.0, scale=0.05, size=numTotalSolutions)
        xPosition = j + 1  # The categorical index (1, 2, 3...)
        xPositionsWithJitter = xPosition + jitterValues  # Store jittered x-coordinates

        # Separate scores by validity AND track highlighted points
        validScores = []
        invalidScores = []
        validXCoords = []
        invalidXCoords = []

        highlightedScores = []
        highlightedXCoords = []

        currentHighlightStrings = highlightBitstrings.get(problemName, [])

        for i, (score, valid, bitstring) in enumerate(
            zip(performanceScores, allValidity, allBitstrings)
        ):
            xCoord = xPositionsWithJitter[
                i
            ]  # Use the pre-calculated jittered x-coordinate

            if bitstring in currentHighlightStrings:
                highlightedScores.append(score)
                highlightedXCoords.append(xCoord)  # Store its specific x-coordinate

            if valid:
                validScores.append(score)
                validXCoords.append(xCoord)  # Store its specific x-coordinate
            else:
                invalidScores.append(score)
                invalidXCoords.append(xCoord)  # Store its specific x-coordinate

        label = "MVC" if problemName == "MinimumVertexCover" else problemName
        plotLabels.append(label)

        if plotType == "violin":
            # ... (Keep this the same) ...
            pass

        elif plotType == "scatter":
            # Plot INVALID solutions in RED
            if len(invalidScores) > 0:
                ax.scatter(
                    invalidXCoords,  # Use stored X coordinates
                    invalidScores,
                    alpha=0.3,
                    s=5,
                    c="red",
                    label="Invalid" if j == 0 else "",
                )

            # Plot VALID solutions in GREEN
            if len(validScores) > 0:
                ax.scatter(
                    validXCoords,  # Use stored X coordinates
                    validScores,
                    alpha=0.5,
                    s=8,
                    c="green",
                    label="Valid" if j == 0 else "",
                )

            # Plot HIGHLIGHTED solutions in BLUE CIRCLE
            if len(highlightedScores) > 0:
                ax.scatter(
                    highlightedXCoords,  # Use the specific X coordinates for highlighted points
                    highlightedScores,
                    s=50,  # Larger size for emphasis
                    facecolors="none",  # No fill color inside the circle
                    edgecolors="blue",  # Blue outline
                    linewidths=1.5,  # Thicker line for visibility
                    label=(
                        "Inital State" if j == 0 else ""
                    ),  # Add legend label only once
                    zorder=10,  # Ensure these dots are plotted on top of the others
                )

    # --- Set axis labels and titles ---
    ax.set_xticks(np.arange(1, len(plotLabels) + 1))
    ax.set_xticklabels(plotLabels)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylabel("Performance Score")

    if plotType == "scatter":
        ax.legend(
            loc="lower left",  # Specifies the location *inside* the axes
            bbox_to_anchor=(
                0,
                0,
            ),  # Anchors the legend to the (x=0, y=0) coordinates of the axes
            shadow=False,  # Optional: Keep the legend clean
            fancybox=True,  # Optional: Gives it rounded corners
        )
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
    plt.show()


# --- Main Execution Block ---
all_solution_data = {}

# Define the specific bitstrings used as initial states for highlighting in the plot
highlightBitstrings = {
    "TSP": ["100010001"],
    "Knapsack": ["110000"],
    "MinimumVertexCover": ["1111"],
    "MaxCut": ["1000"],
}

# Iterate through all configured problem types to analyze their solution spaces
for problem_name, problem_config in problem_configs.items():
    if problem_name == "TSP":
        # TSP (9 Qubits)
        num_cities = 4
        tsp_prob = generate_tsp_problem(num_cities)
        isingProb = tsp_prob.qubo
        ising_dict = isingProb.asdict()
        print(f"TSP problem as dictionary: \n{ising_dict}")
        requiredQubits = ising_dict["n"]
        possibleBitstrings = generate_all_bitstrings(requiredQubits)

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
            # Check validity based on specific TSP constraints
            validity = check_tsp_validity(bitstring, num_cities, requiredQubits)
            stringsCostsValidity[bitstring] = {
                "cost": cost,
                "valid": validity,
            }
        print("Bitstring costs and validity for TSP:")
        for bitstring, info in stringsCostsValidity.items():
            if info["valid"]:
                print(
                    f"Bitstring: {bitstring}, Cost: {info['cost']}, Valid: {info['valid']}"
                )
        all_solution_data[problem_name] = stringsCostsValidity

    elif problem_name == "Knapsack":
        # Knapsack (6 Qubits)
        banned_weight_capacity = []  # can be used for debigging
        items = 4
        target_qubits = 6
        knapsack_prob = generate_knapsack_problem(
            items, target_qubits, banned_weight_capacity
        )
        isingProb = knapsack_prob.qubo
        ising_dict = isingProb.asdict()
        print(f"Knapsack problem as dictionary: \n{ising_dict}")
        requiredQubits = ising_dict["n"]
        possibleBitstrings = generate_all_bitstrings(requiredQubits)

        # Calculate cost of the worst valid solution (used for display/debug reference)
        worstValidCost = calculate_ising_energy(
            "110000", ising_dict["terms"], ising_dict["weights"]
        )

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
            # Check validity based on capacity and slack variable matching
            validity = check_knapsack_validity(
                bitstring,
                ising_dict["problem_instance"]["weights"],
                ising_dict["problem_instance"]["weight_capacity"],
                requiredQubits,
                cost,
                worstValidCost,
            )
            stringsCostsValidity[bitstring] = {
                "cost": cost,
                "valid": validity,
            }
        print("Bitstring costs and validity for Knapsack:")
        for bitstring, info in stringsCostsValidity.items():
            if info["valid"]:
                print(
                    f"Bitstring: {bitstring}, Cost: {info['cost']}, Valid: {info['valid']}"
                )
        all_solution_data[problem_name] = stringsCostsValidity

    elif problem_name == "MinimumVertexCover":
        # Minimum Vertex Cover (4 Qubits)
        mvc_prob = generate_minimum_vertex_cover_problem(4)
        isingProb = mvc_prob.qubo
        ising_dict = isingProb.asdict()
        print(f"Minimum Vertex Cover problem as dictionary: \n{ising_dict}")
        requiredQubits = ising_dict["n"]
        possibleBitstrings = generate_all_bitstrings(requiredQubits)

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
            # Check validity against MVC edge coverage constraint
            validity = check_mvc_validity(bitstring, ising_dict)
            stringsCostsValidity[bitstring] = {
                "cost": cost,
                "valid": validity,
            }
        print("Bitstring costs and validity for MVC:")
        for bitstring, info in stringsCostsValidity.items():
            if info["valid"]:
                print(
                    f"Bitstring: {bitstring}, Cost: {info['cost']}, Valid: {info['valid']}"
                )
        all_solution_data[problem_name] = stringsCostsValidity

    elif problem_name == "MaxCut":
        # Max Cut (4 Qubits)
        maxcut_prob = generate_max_cut_problem(4)
        isingProb = maxcut_prob.qubo
        ising_dict = isingProb.asdict()
        print(f"Max Cut problem as dictionary: \n{ising_dict}")
        requiredQubits = ising_dict["n"]
        possibleBitstrings = generate_all_bitstrings(requiredQubits)

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
            # MaxCut is unconstrained; all bitstrings are trivially valid
            validity = True  # All bitstrings are valid for MaxCut
            stringsCostsValidity[bitstring] = {
                "cost": cost,
                "valid": validity,
            }
        print("Bitstring costs and validity for Max Cut:")
        for bitstring, info in stringsCostsValidity.items():
            if info["valid"]:
                print(
                    f"Bitstring: {bitstring}, Cost: {info['cost']}, Valid: {info['valid']}"
                )
        all_solution_data[problem_name] = stringsCostsValidity

print("\n--- Generating Plot ---")
plotSolutionData(all_solution_data, highlightBitstrings, plotType="scatter")
