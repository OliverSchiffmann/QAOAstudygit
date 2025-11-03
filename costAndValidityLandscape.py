import numpy as np
from openqaoa.problems import TSP, Knapsack, MinimumVertexCover, MaximumCut
import networkx as nx
import matplotlib.pyplot as plt
from config import problem_configs
from helpers import calculate_ising_energy

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def generate_tsp_problem(cities):
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
    G = nx.generators.fast_gnp_random_graph(n=nodes, p=0.6)
    prob = MinimumVertexCover(G, field=1, penalty=10)
    return prob


def generate_max_cut_problem(nodes):
    G = None
    while (
        G is None or len(list(nx.isolates(G))) > 0
    ):  # ensuring that there are no isolated nodes, which means there is always 4 qubits as well
        G = nx.generators.fast_gnp_random_graph(n=nodes, p=0.6)
    prob = MaximumCut(G)
    return prob


def generate_all_bitstrings(length):
    return [bin(i)[2:].zfill(length) for i in range(2**length)]


def check_tsp_validity(bitstring, cities, qubits):
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

        if chunk.count("1") != 1:
            return False

        position = chunk.find("1")
        if position in foundPositions:
            return False

        foundPositions.add(position)
    return True


def check_knapsack_validity(
    bitString, itemWeights, weightCapacity, numQubits, cost, costUpperBound
):
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

    # The integer value of the slack bits must *exactly* match
    # the calculated remaining space.

    if cost < costUpperBound:
        # debugging print statements
        print(f"Bitstring: {bitString}, cost: {cost}")
        print(f"item weights: {itemWeights}")
        print(f"weight capacity: {weightCapacity}")
        print(f"total weight: {totalWeight}")
        print(f"remaining space: {remainingSpace}")
        print(f"int slack value: {intSlackValue}")
        print(f"item string: {itemString}")
        print(f"slack string: {slackString}")

    if intSlackValue == remainingSpace or bitString == "110000":

        return True
    else:
        return False


def check_mvc_validity(bitString, problemInfo):
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


def plotSolutionData(allSolutionData, plotType="scatter"):
    """
    Plots the full performance score distribution with validity coloring.
    """
    # --- Set up a single plot ---
    numProblems = len(allSolutionData)
    if numProblems == 0:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(4 * numProblems, 7))
    fig.suptitle("Performance Score Distribution of All $2^N$ Solutions", fontsize=16)

    plotLabels = []

    # --- Main Loop: One column per problem type ---
    # Use enumerate to get an index (j) for plotting
    for j, (problemName, solutions) in enumerate(allSolutionData.items()):

        print(f"--- Plotting {problemName} ---")

        if not solutions:
            print(f"Warning: No solution data found for {problemName}.")
            continue

        # Extract costs and validity
        allCosts = []
        allValidity = []
        for info in solutions.values():
            allCosts.append(info["cost"])
            allValidity.append(info["valid"])

        # --- Calculate Performance Scores ---
        allCosts = np.array(allCosts)
        optimumCost = np.min(allCosts)
        maximumCost = np.max(allCosts)
        energyRange = maximumCost - optimumCost

        if energyRange == 0:
            performanceScores = np.ones_like(allCosts)
        else:
            performanceScores = 1.0 - ((allCosts - optimumCost) / energyRange)

        # --- Separate scores by validity ---
        validScores = []
        invalidScores = []
        for score, valid in zip(performanceScores, allValidity):
            if valid:
                validScores.append(score)
            else:
                invalidScores.append(score)

        # Store the label for the x-axis
        label = "MVC" if problemName == "MinimumVertexCover" else problemName
        plotLabels.append(label)

        # --- Generate the selected plot type ---
        xPosition = j + 1  # The categorical index (1, 2, 3...)

        if plotType == "violin":
            # Note: This won't show validity, but it's here for completeness
            parts = ax.violinplot(
                performanceScores, [xPosition], showmeans=False, showmedians=True
            )
            # Customize the median line
            parts["cmedians"].set_edgecolor("red")
            parts["cmedians"].set_linewidth(2)

        elif plotType == "scatter":
            # Plot INVALID solutions in RED
            numInvalid = len(invalidScores)
            if numInvalid > 0:
                jitterInvalid = np.random.normal(loc=0.0, scale=0.05, size=numInvalid)
                xInvalid = xPosition + jitterInvalid
                ax.scatter(
                    xInvalid,
                    invalidScores,
                    alpha=0.3,
                    s=5,
                    c="red",
                    label="Invalid" if j == 0 else "",
                )

            # Plot VALID solutions in GREEN
            numValid = len(validScores)
            if numValid > 0:
                jitterValid = np.random.normal(loc=0.0, scale=0.05, size=numValid)
                xValid = xPosition + jitterValid
                ax.scatter(
                    xValid,
                    validScores,
                    alpha=0.5,
                    s=8,
                    c="green",
                    label="Valid" if j == 0 else "",
                )

    # --- Set axis labels and titles ---
    # ax.set_title("Performance Score by Problem")
    ax.set_xticks(np.arange(1, len(plotLabels) + 1))
    ax.set_xticklabels(plotLabels)
    ax.set_ylim(0.9, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylabel("Performance Score")

    # Add a legend
    if plotType == "scatter":
        ax.legend()

    # --- Show the final plot ---
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
    plt.show()


all_solution_data = {}

for problem_name, problem_config in problem_configs.items():
    if problem_name == "TSP":
        num_cities = 4
        tsp_prob = generate_tsp_problem(num_cities)
        isingProb = tsp_prob.qubo
        ising_dict = isingProb.asdict()
        print(f"TSP problem as dictionary: \n{ising_dict}")
        requiredQubits = ising_dict["n"]
        possibleBitstrings = generate_all_bitstrings(requiredQubits)
        # print(f"Possible bitstrings for TSP with {requiredQubits} qubits:")
        # print(possibleBitstrings)

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
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
        # print(f"Possible bitstrings for Knapsack with {requiredQubits} qubits:")
        # print(possibleBitstrings)

        worstValidCost = calculate_ising_energy(
            "110000", ising_dict["terms"], ising_dict["weights"]
        )

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
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
        mvc_prob = generate_minimum_vertex_cover_problem(4)
        isingProb = mvc_prob.qubo
        ising_dict = isingProb.asdict()
        print(f"Minimum Vertex Cover problem as dictionary: \n{ising_dict}")
        requiredQubits = ising_dict["n"]
        possibleBitstrings = generate_all_bitstrings(requiredQubits)
        # print(
        #    f"Possible bitstrings for Minimum Vertex Cover with {requiredQubits} qubits:"
        # )
        # print(possibleBitstrings)

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
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
        maxcut_prob = generate_max_cut_problem(4)
        isingProb = maxcut_prob.qubo
        ising_dict = isingProb.asdict()
        print(f"Max Cut problem as dictionary: \n{ising_dict}")
        requiredQubits = ising_dict["n"]
        possibleBitstrings = generate_all_bitstrings(requiredQubits)
        # print(f"Possible bitstrings for Max Cut with {requiredQubits} qubits:")
        # print(possibleBitstrings)

        stringsCostsValidity = {}
        for bitstring in possibleBitstrings:
            cost = calculate_ising_energy(
                bitstring, ising_dict["terms"], ising_dict["weights"]
            )
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
plotSolutionData(all_solution_data, plotType="scatter")
