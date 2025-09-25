# example usage for two plots \/
# python merged_results_plotter.py --simulators IONQ_NOISY IBM_IDEAL --depths 2 -1 --plot_type barchart

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
    """Loads all exact solutions and Ising models into memory for quick access."""
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
    Calculates the performance scores for all problem classes for a given
    simulator and depth.
    """
    performanceScores = {problem: [] for problem in problem_configs.keys()}

    providerFileSlug = provider_configs[simulatorName]["file_slug"]
    depthSlug = f"p{depth}_" if depth is not None else ""

    for problemName, problemConfig in problem_configs.items():
        problemFileSlug = problemConfig["file_slug"]

        # Construct the path to the merged results file
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

            mostProbableBitstring = runInstance["sampled_distribution"][0][0]
            mostProbableSolution = mostProbableBitstring[::-1]  # Reverse the bitstring

            optimumCost = solutionInfo["cost"]
            maximumCost = solutionInfo["max_cost"]

            isingTerms = isingModel["terms"]
            isingWeights = isingModel["weights"]

            mostProbableSolutionCost = calculate_ising_energy(
                mostProbableSolution, isingTerms, isingWeights
            )

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
    problem class.
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
            # This warning is printed in the main loop, no need to repeat
            continue

        solutionsData = preloadedData[problemName]["solutions"]

        for runInstance in data["results"]:
            instanceId = runInstance["instance_id"]
            solutionInfo = solutionsData.get(instanceId)
            if not solutionInfo:
                continue

            mostProbableBitstring = runInstance["sampled_distribution"][0][0]
            mostProbableSolution = mostProbableBitstring[::-1]

            # Convert correct Ising solutions to binary strings for comparison
            correctSolutionsIsing = solutionInfo["solutions"]
            correctSolutionsBinary = [
                s.replace("-1", "1").replace("+1", "0").replace(",", "")
                for s in correctSolutionsIsing
            ]

            if mostProbableSolution in correctSolutionsBinary:
                successCounts[problemName] += 1
    return successCounts


def main():
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
    depths = [None if d == -1 else d for d in args.depths]
    plotType = args.plot_type

    if len(simulatorNames) != len(depths):
        raise ValueError("The number of simulators must match the number of depths.")

    preloadedData = preload_supplementary_data()
    numPairings = len(simulatorNames)
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

    for i, (simName, depth) in enumerate(zip(simulatorNames, depths)):
        ax = axes[0, i]
        print(f"--- Generating plot for {simName} at depth p={depth or 'default'} ---")

        originalLabels = list(problem_configs.keys())
        plotLabels = [
            "MVC" if label == "MinimumVertexCover" else label
            for label in originalLabels
        ]

        ax.set_title(f"{simName} (p={depth or '20'})")
        if i == 0:
            ax.set_ylabel(yLabel)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # --- Generate the selected plot type ---
        if plotType == "boxplot":
            scores = calculate_performance_scores(simName, depth, preloadedData)
            plotData = [scores[problem] for problem in originalLabels]
            if any(len(d) > 0 for d in plotData):
                ax.boxplot(plotData, labels=plotLabels, patch_artist=True)
                ax.set_ylim(0.4, 1.1)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data found", ha="center", va="center")

        elif plotType == "barchart":
            counts = calculate_success_counts(simName, depth, preloadedData)
            plotData = [counts[problem] for problem in originalLabels]
            if sum(plotData) > 0:
                bars = ax.bar(plotLabels, plotData)
                ax.set_ylim(0, 105)  # Assuming max 100 instances
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                # Add count labels on top of each bar
                ax.bar_label(bars, padding=3)
            else:
                ax.text(0.5, 0.5, "No data found", ha="center", va="center")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.88, wspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
