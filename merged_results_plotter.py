# ==============================================================================
# QAOA Full Results Plotter
# ==============================================================================
# This script analyzes the aggregated results from all 100 problem instances
# for various QAOA configurations (Problem Type, Provider, and Depth).
#
# It supports generating two main types of comparative plots:
# 1. Boxplots: Visualizing the distribution of the Performance Score (Approximation Ratio).
# 2. Bar Charts: Displaying the Success Count (number of optimal solutions found).
#
# The script is designed to compare multiple configurations (Simulators/Depths)
# side-by-side in subplots.
#
# Example Usage:
# python merged_results_plotter.py --simulators IBM_IDEAL IBM_IDEAL IBM_IDEAL IBM_IDEAL IBM_IDEAL --depths 1 2 3 4 20 --plot_type boxplot
# python merged_results_plotter.py --simulators IBM_IDEAL IBM_NOISY IONQ_IDEAL IONQ_NOISY --depths 4 4 4 4 --plot_type barchart
# ==============================================================================
import json
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import problem_configs, provider_configs
from helpers import calculate_ising_energy

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

MERGED_RESULTS_DIR = "merged_results_warehouse"
SUPPLEMENTARY_DATA_DIR = "isingBatches"


def preload_supplementary_data():
    """
    Loads all exact solutions (optimal/maximum costs) and Ising models
    (terms and weights) into memory for quick access.

    This data is essential for calculating the normalized performance score
    without hitting disk repeatedly during the calculation loop.

    Returns:
        dict: Structured dictionary containing solutions and ising models
              indexed by problem name and instance ID.
    """
    print("Pre-loading supplementary data (solutions and Ising models)...")
    preloadedData = {}
    for problemName, problemConfig in problem_configs.items():
        problemFileSlug = problemConfig["file_slug"]
        preloadedData[problemName] = {"solutions": {}, "ising_models": {}}

        # Load exact solutions
        solutionsFilePath = os.path.join(
            SUPPLEMENTARY_DATA_DIR, f"solved*{problemFileSlug}.json"
        )
        solutionsFile = glob.glob(solutionsFilePath)
        if solutionsFile:
            with open(solutionsFile[0], "r") as f:
                solutions = json.load(f)
                for sol in solutions:
                    preloadedData[problemName]["solutions"][sol["instance_id"]] = sol

        # Load Ising models
        isingModelsFilePath = os.path.join(
            SUPPLEMENTARY_DATA_DIR, f"batch*{problemFileSlug}.json"
        )
        isingModelsFile = glob.glob(isingModelsFilePath)
        if isingModelsFile:
            with open(isingModelsFile[0], "r") as f:
                models = json.load(f)
                for model in models:
                    preloadedData[problemName]["ising_models"][
                        model["instance_id"]
                    ] = model
    print("Supplementary data loaded successfully.\n")
    return preloadedData


def calculate_performance_scores(simulatorName, depth, preloadedData):
    """
    Calculates the Performance Score for the most probable solution of every
    instance in the merged results for a given simulator and depth configuration.

    The score is normalized by the optimal and maximum cost of the instance.

    Args:
        simulatorName (str): Key from provider_configs (e.g., 'IBM_NOISY').
        depth (int or None): The QAOA layer depth (p).
        preloadedData (dict): Dictionary containing exact solutions and Ising models.

    Returns:
        dict: A dictionary where keys are problem names and values are lists
              of calculated performance scores (one score per instance).
    """
    performanceScores = {problem: [] for problem in problem_configs.keys()}

    providerFileSlug = provider_configs[simulatorName]["file_slug"]
    depthSlug = f"p{depth}_" if depth is not None else ""

    for problemName, problemConfig in problem_configs.items():
        problemFileSlug = problemConfig["file_slug"]

        # Construct the path to the merged results file (containing all 100 instances)
        mergedFilePath = os.path.join(
            MERGED_RESULTS_DIR,
            f"MERGED_results_{problemFileSlug}{providerFileSlug}{depthSlug}.json",
        )

        try:
            with open(mergedFilePath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(
                f"WARNING: Results file not found for {problemName}, {simulatorName}, depth={depth}. Skipping."
            )
            continue

        # Access preloaded data for this problem class
        solutionsData = preloadedData[problemName]["solutions"]
        isingModelsData = preloadedData[problemName]["ising_models"]

        for runInstance in data["results"]:
            instanceId = runInstance["instance_id"]

            # Get data from preloaded dictionaries
            solutionInfo = solutionsData.get(instanceId)
            isingModel = isingModelsData.get(instanceId)

            if not solutionInfo or not isingModel:
                continue

            # Get the result bitstring with the highest count
            mostProbableBitstring = runInstance["sampled_distribution"][0][0]
            mostProbableSolution = mostProbableBitstring[::-1]  # Reverse the bitstring

            optimumCost = solutionInfo["cost"]
            maximumCost = solutionInfo["max_cost"]

            isingTerms = isingModel["terms"]
            isingWeights = isingModel["weights"]

            # Calculate the cost of the found solution using the helper function
            mostProbableSolutionCost = calculate_ising_energy(
                mostProbableSolution, isingTerms, isingWeights
            )

            # Calculate the normalized performance score
            energyRange = maximumCost - optimumCost
            performance = 1 - (
                (mostProbableSolutionCost - optimumCost) / energyRange
                if energyRange != 0
                else 0
            )
            performanceScores[problemName].append(performance)

    return performanceScores


def calculate_success_counts(simulatorName, depth, preloadedData):
    """
    Calculates the number of times QAOA found an optimal solution for each
    problem class (Success Count).

    A success is defined as the most probable bitstring matching one of the
    known global optimal solutions.

    Args:
        simulatorName (str): Key from provider_configs.
        depth (int or None): The QAOA layer depth (p).
        preloadedData (dict): Dictionary containing exact solutions.

    Returns:
        dict: A dictionary where keys are problem names and values are the
              total number of successful runs (out of 100).
    """
    successCounts = {problem: 0 for problem in problem_configs.keys()}
    providerFileSlug = provider_configs[simulatorName]["file_slug"]
    depthSlug = f"p{depth}_" if depth is not None else ""

    for problemName, problemConfig in problem_configs.items():
        problemFileSlug = problemConfig["file_slug"]
        mergedFilePath = os.path.join(
            MERGED_RESULTS_DIR,
            f"MERGED_results_{problemFileSlug}{providerFileSlug}{depthSlug}.json",
        )
        try:
            with open(mergedFilePath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            # This warning is handled by the performance score function
            continue

        solutionsData = preloadedData[problemName]["solutions"]

        for runInstance in data["results"]:
            instanceId = runInstance["instance_id"]
            solutionInfo = solutionsData.get(instanceId)
            if not solutionInfo:
                continue

            mostProbableBitstring = runInstance["sampled_distribution"][0][0]
            mostProbableSolution = mostProbableBitstring[::-1]

            # Convert correct Ising solutions (spin strings '+1,-1') to binary strings ('0', '1') for comparison
            correctSolutionsIsing = solutionInfo["solutions"]
            correctSolutionsBinary = [
                s.replace("-1", "1").replace("+1", "0").replace(",", "")
                for s in correctSolutionsIsing
            ]

            # Check if the sampled solution matches any of the known optimal binary strings
            if mostProbableSolution in correctSolutionsBinary:
                successCounts[problemName] += 1
    return successCounts


def main():
    """
    Main execution function: parses arguments, loads data, calculates scores/counts, and plots results.
    """
    parser = argparse.ArgumentParser(
        description="Generate comparative plots for QAOA results."
    )
    parser.add_argument(
        "--simulators",
        nargs="+",
        required=True,
        help="List of simulator names (keys from provider_configs in config.py).",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        required=True,
        type=int,
        help="List of QAOA depths (use -1 for default(20)/None depth).",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="boxplot",
        choices=["boxplot", "barchart"],
        help="Type of plot to generate: 'boxplot' for performance or 'barchart' for success counts.",
    )

    args = parser.parse_args()
    simulatorNames = args.simulators
    # Convert depth argument '-1' to None to handle legacy p=20 naming convention
    depths = [None if d == -1 else d for d in args.depths]
    plotType = args.plot_type

    # Validation check
    if len(simulatorNames) != len(depths):
        raise ValueError("The number of simulators must match the number of depths.")

    preloadedData = preload_supplementary_data()
    numPairings = len(simulatorNames)
    # Set up the figure for subplots (one column per simulator/depth pairing)
    fig, axes = plt.subplots(
        1, numPairings, figsize=(6 * numPairings, 7), squeeze=False
    )

    # --- Set titles and labels based on plot type ---
    if plotType == "boxplot":
        fig.suptitle("QAOA Performance Distribution by Problem Class", fontsize=16)
        yLabel = "Performance Score"
    else:  # barchart
        fig.suptitle("QAOA Success Count by Problem Class", fontsize=16)
        yLabel = "Optimal Solutions Found"

    # Iterate through all required simulator/depth pairs to generate plots
    for i, (simName, depth) in enumerate(zip(simulatorNames, depths)):
        ax = axes[0, i]
        print(f"--- Generating plot for {simName} at depth p={depth or 'default'} ---")

        originalLabels = list(problem_configs.keys())
        # Shorten MinimumVertexCover for cleaner plot labels
        plotLabels = [
            "MVC" if label == "MinimumVertexCover" else label
            for label in originalLabels
        ]

        # Handle title display for depth=None (legacy p=20)
        ax.set_title(f"{simName} (p={depth or '20'})")
        if i == 0:
            ax.set_ylabel(yLabel)
        else:
            # Hide Y-axis labels on subsequent subplots for clarity
            ax.tick_params(axis="y", labelleft=False)

        # --- Generate the selected plot type ---
        if plotType == "boxplot":
            scores = calculate_performance_scores(simName, depth, preloadedData)
            plotData = [scores[problem] for problem in originalLabels]
            # Plot only if data was actually found
            if any(len(d) > 0 for d in plotData):
                ax.boxplot(plotData, labels=plotLabels, patch_artist=True)
                # Set consistent y-limits for comparison (0 to 1, plus buffer)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data found", ha="center", va="center")

        elif plotType == "barchart":
            counts = calculate_success_counts(simName, depth, preloadedData)
            plotData = [counts[problem] for problem in originalLabels]
            if sum(plotData) > 0:
                # Generate bars
                bars = ax.bar(plotLabels, plotData)
                # Set y-limit slightly above 100 since there are 100 instances
                ax.set_ylim(0, 105)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                # Add count labels on top of each bar
                ax.bar_label(bars, padding=3)
            else:
                ax.text(0.5, 0.5, "No data found", ha="center", va="center")

    # Adjust layout to prevent overlap
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.88, wspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
