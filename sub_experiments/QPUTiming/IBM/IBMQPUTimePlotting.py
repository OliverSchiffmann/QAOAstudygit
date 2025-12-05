import json
import os
import sys
import matplotlib.pyplot as plt

# ==============================================================================
# QPU Time Estimation Analysis Script
# ==============================================================================
# This script estimates the total Quantum Processing Unit (QPU) usage required
# for the study based on simulation data.
#
# Since access to actual quantum hardware is time-constrained, this script uses
# results from "fake_torino" noisy simulations to count the number of optimization
# loops (iterations) the optimizer required to converge.
#
# It applies a constant time factor (avgQPUTimePerLoop = 12s) to these loop
# counts to estimate the physical runtime.
#
# Key outputs:
# 1. Printed total estimated QPU time (in seconds and minutes).
# 2. Boxplots showing the distribution of estimated QPU times per problem class
#    across different QAOA depths (p=1, 2, 3, 4).
# ==============================================================================

scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.abspath(os.path.join(scriptDir, "../../../"))
if projectRoot not in sys.path:
    sys.path.append(projectRoot)

from config import problem_configs


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 14

mergedResultsDir = os.path.join(projectRoot, "merged_results_warehouse")

# Constant estimated duration for a single optimization loop on hardware
avgQPUTimePerLoop = 12  # in seconds, obtained from previous experiments


def get_num_optimisation_loops(depth):
    """
    Retrieves the number of optimization loops performed for each problem class
    at a specific QAOA depth.

    It reads the merged JSON result files for the 'fake_torino' backend and
    extracts the 'num_training_loops' field for every instance.

    Args:
        depth (int): The QAOA layer depth (p) to analyze.

    Returns:
        dict: A dictionary where keys are problem names (e.g., 'Knapsack') and
              values are lists of integers representing the loop counts for
              each instance of that problem.
    """
    num_loops = {problem: [] for problem in problem_configs.keys()}

    depthSlug = f"p{depth}_" if depth is not None else ""

    for problemName, problemConfig in problem_configs.items():
        problemFileSlug = problemConfig["file_slug"]

        # Construct the path to the merged results file
        mergedFilePath = os.path.join(
            mergedResultsDir,
            f"MERGED_results_{problemFileSlug}aer_simulator_from(fake_torino)_{depthSlug}.json",
        )

        try:
            with open(mergedFilePath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(
                f"WARNING: Results file not found for {problemName}, on fake_torino, depth={depth}. Skipping."
            )
            continue

        for instance in data["results"]:
            num_loops[problemName].append(instance["num_training_loops"])

    return num_loops


if __name__ == "__main__":
    # --- Data Collection ---
    # Collect loop counts for depths 1 through 4
    numLoops = {}
    depths = [1, 2, 3, 4]
    for depth in depths:
        numLoops[depth] = get_num_optimisation_loops(depth)
        print(f"Total loops for depth={depth}: {numLoops[depth]}")

    simQPUTimes = {}

    # --- Time Estimation ---
    # Convert raw loop counts to estimated seconds using the constant factor
    # mulitple every value in numLoops by avgQPUTimePerLoop
    for depth in numLoops:
        simQPUTimes[depth] = {}
        for problem in numLoops[depth]:
            simQPUTimes[depth][problem] = [
                loops * avgQPUTimePerLoop for loops in numLoops[depth][problem]
            ]
    # print(simQPUTimes)

    # --- Totals Calculation ---
    # Summing all QPU times to get total estimated QPU time per category
    totalEstimatedQPUTimes = {}
    for depth in simQPUTimes:
        print(f"Simulated QPU times for depth({depth}): {simQPUTimes[depth]}")
        totalEstimatedQPUTimes[depth] = {}
        for problem in simQPUTimes[depth]:
            totalEstimatedQPUTimes[depth][problem] = sum(simQPUTimes[depth][problem])
    # print(
    #     "Total Estimated QPU Times (s) per depth and problem:", totalEstimatedQPUTimes
    # )

    # Calculate and print the grand total across all experiments
    totalQPUTime = 0
    for depth in totalEstimatedQPUTimes:
        print(
            f"total estimated qpu times for depth({depth}): {totalEstimatedQPUTimes[depth]}"
        )
        for problem in totalEstimatedQPUTimes[depth]:
            totalQPUTime += totalEstimatedQPUTimes[depth][problem]
    print(f"Overall Total Estimated QPU Time (s): {totalQPUTime}")
    print(f"Overall Total Estimated QPU Time (mins): {totalQPUTime / 60}")

    # --- Plotting ---
    # Create a subplot grid (1 row, 4 columns) for each depth
    fig, axes = plt.subplots(
        1, len(depths), figsize=(6 * len(depths), 7), squeeze=False
    )

    # fig.suptitle("Estimated QPU Time Using Fake Torino", fontsize=16)
    yLabel = "Estimated QPU Time / s"

    for i in depths:
        ax = axes[0, i - 1]
        originalLabels = list(problem_configs.keys())
        # Shorten labels for cleaner plotting (MinimumVertexCover -> MVC)
        plotLabels = [
            "MVC" if label == "MinimumVertexCover" else label
            for label in originalLabels
        ]

        ax.set_title(f"(p={i})")
        if i == 1:
            ax.set_ylabel(yLabel)
        else:
            ax.tick_params(axis="y", labelleft=False)

        plotData = [simQPUTimes[i][problem] for problem in originalLabels]
        if any(len(d) > 0 for d in plotData):
            ax.boxplot(plotData, tick_labels=plotLabels, patch_artist=True)
            ax.set_ylim(0, 1200)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.88, wspace=0.05)
    plt.show()
