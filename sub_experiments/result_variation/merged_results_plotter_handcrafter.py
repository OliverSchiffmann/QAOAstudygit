# example usage
# python merged_results_plotter_handcrafter.py --simulators IBM_NOISY IBM_NOISY IBM_NOISY IBM_NOISY --depths 1 2 3 4 --plot_type boxplot --instance_id 1


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

MERGED_RESULTS_DIR = "merged_repeat_instance_results"
SUPPLEMENTARY_DATA_DIR = "/Users/kv18799/Github/QAOAstudygit/isingBatches"


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


def calculate_performance_scores(simulatorName, depth, preloadedData, instance):
    """
    Calculates the performance scores for all problem classes for a given
    simulator, depth and problem instance.
    """
    performanceScores = {problem: [] for problem in problem_configs.keys()}

    providerFileSlug = provider_configs[simulatorName]["file_slug"]
    depthSlug = f"p{depth}_" if depth is not None else ""

    for problemName, problemConfig in problem_configs.items():
        problemFileSlug = problemConfig["file_slug"]

        # Construct the path to the merged results file
        mergedFilePath = os.path.join(
            MERGED_RESULTS_DIR,
            f"MERGED_results_{problemFileSlug}{providerFileSlug}{depthSlug}inst_{instance}.json",
        )

        try:
            with open(mergedFilePath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(
                f"WARNING: Results file not found for {problemName}, {simulatorName}, depth={depth}, instance ID = {instance}. Skipping."
            )
            continue

        # Access preloaded data for this problem class
        solutionsData = preloadedData[problemName]["solutions"]
        isingModelsData = preloadedData[problemName]["ising_models"]
        solutionInfo = solutionsData.get(instance)
        isingModel = isingModelsData.get(instance)

        optimumCost = solutionInfo["cost"]
        maximumCost = solutionInfo["max_cost"]

        isingTerms = isingModel["terms"]
        isingWeights = isingModel["weights"]

        for repitition in data["results"]:
            repititionID = repitition["repitition_id"]

            mostProbableBitstring = repitition["sampled_distribution"][0][0]
            mostProbableSolution = mostProbableBitstring[::-1]  # Reverse the bitstring

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
        choices=["boxplot", "violin"],
        help="Type of plot to generate.",
    )
    parser.add_argument(
        "--instance_id",
        type=int,
        default=1,
        help="The specific problem instance ID to plot.",
    )

    args = parser.parse_args()
    simulatorNames = args.simulators
    depths = [None if d == -1 else d for d in args.depths]
    plotType = args.plot_type
    instanceIdToPlot = args.instance_id

    if len(simulatorNames) != len(depths):
        raise ValueError("The number of simulators must match the number of depths.")

    preloadedData = preload_supplementary_data()
    numPairings = len(simulatorNames)
    fig, axes = plt.subplots(
        1, numPairings, figsize=(6 * numPairings, 7), squeeze=False
    )

    # --- Set titles and labels based on plot type ---
    if plotType == "boxplot":
        fig.suptitle(
            f"Result Variability for Single Instance (ID = {instanceIdToPlot})",
            fontsize=16,
        )
        yLabel = "Performance Score"
    else:
        print(f"Plot type '{plotType}' is not implemented.")
        return

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

        if plotType == "boxplot":
            scores = calculate_performance_scores(
                simName, depth, preloadedData, instanceIdToPlot
            )
            plotData = [scores[problem] for problem in originalLabels]
            if any(len(d) > 0 for d in plotData):
                ax.boxplot(plotData, labels=plotLabels, patch_artist=True)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data found", ha="center", va="center")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.88, wspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
