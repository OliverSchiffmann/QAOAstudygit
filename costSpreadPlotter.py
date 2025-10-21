#!/usr/bin/env python

"""
Plots the full performance score distribution for all 2^N possible solutions
to a set of Ising models.

Example Usage:
python costSpreadPlotter.py --instances 1 2 --plot_type violin
python costSpreadPlotter.py --instances 1 2 --plot_type scatter
"""

import dimod
import json
import numpy as np
import matplotlib.pyplot as plt
from config import problem_configs
import os
import glob
import argparse

# --- Set Matplotlib Font ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# --- Constants ---
FILE_DIRECTORY = "isingBatches/"


# --- Helper Functions ---


def load_ising(filePath):
    """Loads an Ising model JSON file."""
    with open(filePath, "r") as f:
        allIsingsData = json.load(f)
    return allIsingsData


def convert_to_dimod_ising(allIsingsData):
    """Converts the script's JSON format to dimod's bias format."""
    dimodIsing = {}
    for instance in allIsingsData:
        instanceId = instance["instance_id"]
        terms = instance["terms"]
        weights = instance["weights"]

        linearBias = {}
        quadraticBias = {}

        for term, weight in zip(terms, weights):
            if len(term) == 1:
                variable = f"x{term[0]}"
                linearBias[variable] = weight
            elif len(term) == 2:
                var1 = f"x{term[0]}"
                var2 = f"x{term[1]}"
                quadraticBias[(var1, var2)] = weight

        dimodIsing[instanceId] = (linearBias, quadraticBias)
    return dimodIsing


# --- New Function to Pre-load All Models ---


def preload_ising_models():
    """
    Loads all Ising models from all problem-type files into a
    nested dictionary for fast access.
    """
    print("Pre-loading all Ising models...")
    preloadedIsingData = {}

    for problemName, config in problem_configs.items():
        preloadedIsingData[problemName] = {}
        fileSlug = config["file_slug"]
        searchPattern = os.path.join(FILE_DIRECTORY, f"batch*{fileSlug}.json")

        try:
            # Find the batch file for this problem type
            isingBatchFile = glob.glob(searchPattern)[0]

            # Load and convert the data
            isings = load_ising(isingBatchFile)
            dimodIsings = convert_to_dimod_ising(isings)

            # Store it in the nested dictionary
            preloadedIsingData[problemName] = dimodIsings

        except IndexError:
            print(f"Warning: No Ising batch file found for {problemName}")

    print("Ising models loaded successfully.\n")
    return preloadedIsingData


# --- Main Plotting Function ---


def main():
    parser = argparse.ArgumentParser(
        description="Plot full solution performance landscapes for Ising models."
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        required=True,
        type=int,
        help="List of instance IDs to plot (e.g., 1 2 3).",
    )
    # <<< MODIFIED: Added plot_type argument >>>
    parser.add_argument(
        "--plot_type",
        type=str,
        default="violin",
        choices=["violin", "scatter"],
        help="Type of plot to generate: 'violin' for distribution or 'scatter' for discrete points.",
    )
    args = parser.parse_args()
    instanceIds = args.instances
    plotType = args.plot_type

    # Load all models into memory
    preloadedIsingData = preload_ising_models()

    # Initialize the exact solver
    solver = dimod.ExactSolver()

    # --- Set up the plot ---
    numInstances = len(instanceIds)
    fig, axes = plt.subplots(
        1, numInstances, figsize=(6 * numInstances, 7), squeeze=False
    )

    fig.suptitle("Performance Score Distribution of All $2^N$ Solutions", fontsize=16)

    # --- Main Loop: One subplot per instance ---
    for i, instanceId in enumerate(instanceIds):
        ax = axes[0, i]
        print(f"--- Processing Instance {instanceId} ---")

        plotData = []
        plotLabels = []

        # --- Inner Loop: One violin/scatter per problem type ---
        for problemName, config in problem_configs.items():

            problemIsings = preloadedIsingData.get(problemName)
            if not problemIsings:
                continue

            biases = problemIsings.get(instanceId)
            if not biases:
                print(f"Warning: No data for {problemName} instance {instanceId}.")
                continue

            linearBias, quadraticBias = biases

            print(f"Solving {problemName} (Instance {instanceId})...")
            sampleset = solver.sample_ising(linearBias, quadraticBias)

            allEnergies = sampleset.record.energy

            optimumCost = np.min(allEnergies)
            maximumCost = np.max(allEnergies)
            energyRange = maximumCost - optimumCost

            if energyRange == 0:
                performanceScores = np.ones_like(allEnergies)
            else:
                performanceScores = 1.0 - ((allEnergies - optimumCost) / energyRange)

            plotData.append(performanceScores)
            label = "MVC" if problemName == "MinimumVertexCover" else problemName
            plotLabels.append(label)

        # --- Filter out empty data arrays (common to both plots) ---
        filteredPlotData = []
        filteredLabels = []
        for label, data in zip(plotLabels, plotData):
            if len(data) > 0:
                filteredPlotData.append(data)
                filteredLabels.append(label)

        if not filteredPlotData:
            ax.text(
                0.5,
                0.5,
                f"No data found for instance {instanceId}",
                ha="center",
                va="center",
            )
            continue

        # <<< MODIFIED: Added plot_type logic >>>

        # --- Generate the selected plot type ---
        if plotType == "violin":
            parts = ax.violinplot(filteredPlotData, showmeans=False, showmedians=True)
            # Customize the median line
            parts["cmedians"].set_edgecolor("red")
            parts["cmedians"].set_linewidth(2)

        elif plotType == "scatter":
            for j, scores in enumerate(filteredPlotData):
                # x_position is the categorical index (1, 2, 3...)
                x_position = j + 1
                numPoints = len(scores)

                # Add jitter: small random noise to spread points horizontally
                jitter = np.random.normal(loc=0.0, scale=0.05, size=numPoints)
                x_values = x_position + jitter

                # Plot with low opacity and small size
                ax.scatter(x_values, scores, alpha=0.3, s=5, c="blue")

        # --- Set axis labels and titles (common to both plots) ---
        ax.set_title(f"Instance {instanceId}")
        ax.set_xticks(np.arange(1, len(filteredLabels) + 1))
        ax.set_xticklabels(filteredLabels)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        if i == 0:
            ax.set_ylabel("Performance Score")
        else:
            ax.tick_params(axis="y", labelleft=False)

    # --- Show the final plot ---
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.88, wspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
