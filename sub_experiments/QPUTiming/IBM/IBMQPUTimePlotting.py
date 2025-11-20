import argparse
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.abspath(os.path.join(scriptDir, "../../../"))
if projectRoot not in sys.path:
    sys.path.append(projectRoot)

from config import problem_configs
from helpers import (
    load_ising_and_build_hamiltonian,
    save_single_result,
    build_mixer_hamiltonian,
    create_inital_state,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

mergedResultsDir = os.path.join(projectRoot, "merged_results_warehouse_test")

avgQPUTimePerLoop = 12  # in seconds, obtained from previous experiments


def get_num_optimisation_loops(depth):
    """
    Calculates the performance scores for all problem classes for a given
    simulator and depth.
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
    numLoops = {}
    depths = [1, 2, 3, 4]
    for depth in depths:
        numLoops[depth] = get_num_optimisation_loops(depth)

    simQPUTimes = {}

    # mulitple every value in numLoops by avgQPUTimePerLoop
    for depth in numLoops:
        simQPUTimes[depth] = {}
        for problem in numLoops[depth]:
            simQPUTimes[depth][problem] = [
                loops * avgQPUTimePerLoop for loops in numLoops[depth][problem]
            ]
    # print(simQPUTimes)

    # Summing all QPU times to get total estimated QPU time
    totalEstimatedQPUTimes = {}
    for depth in simQPUTimes:
        totalEstimatedQPUTimes[depth] = {}
        for problem in simQPUTimes[depth]:
            totalEstimatedQPUTimes[depth][problem] = sum(simQPUTimes[depth][problem])
    # print(
    #     "Total Estimated QPU Times (s) per depth and problem:", totalEstimatedQPUTimes
    # )
    totalQPUTime = 0
    for depth in totalEstimatedQPUTimes:
        for problem in totalEstimatedQPUTimes[depth]:
            totalQPUTime += totalEstimatedQPUTimes[depth][problem]
    print(f"Overall Total Estimated QPU Time (s): {totalQPUTime}")
    print(f"Overall Total Estimated QPU Time (mins): {totalQPUTime / 60}")

    # Plotting
    fig, axes = plt.subplots(
        1, len(depths), figsize=(6 * len(depths), 7), squeeze=False
    )

    fig.suptitle("Estimated QPU Time Using Fake Torino", fontsize=16)
    yLabel = "Estimated QPU Time / s"

    for i in depths:
        ax = axes[0, i - 1]
        originalLabels = list(problem_configs.keys())
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
            ax.set_ylim(0, 1000)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.88, wspace=0.05)
    plt.show()
