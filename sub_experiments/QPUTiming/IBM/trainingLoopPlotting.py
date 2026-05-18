import json
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.abspath(os.path.join(scriptDir, "../../../"))
if projectRoot not in sys.path:
    sys.path.append(projectRoot)

from config import problem_configs, provider_configs

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 18

mergedResultsDir = os.path.join(projectRoot, "merged_results_warehouse")


def get_training_loops(depth, sim):
    num_loops = {problem: [] for problem in problem_configs.keys()}
    depthSlug = f"p{depth}_" if depth is not None else ""
    # print(f"depth: {depthSlug}, \n sim:{sim}")
    for problemName, problemConfig in problem_configs.items():
        problemFileSlug = problemConfig["file_slug"]
        mergedFilePath = os.path.join(
            mergedResultsDir,
            f"MERGED_results_{problemFileSlug}{sim}{depthSlug}.json",
        )
        try:
            with open(mergedFilePath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(
                f"WARNING: Results file not found for {problemName}, on {sim}, depth={depth}. Skipping."
            )
            continue
        for instance in data["results"]:
            num_loops[problemName].append(instance["num_training_loops"])

    return num_loops


if __name__ == "__main__":
    numLoops = defaultdict(dict)
    depths = [1, 2, 3, 4]
    for depth in depths:
        for providerName, providerConfig in provider_configs.items():
            providerFileSlug = providerConfig["file_slug"]
            numLoops[depth][providerFileSlug] = get_training_loops(
                depth, providerFileSlug
            )
    print(numLoops)

    #     fig, axes = plt.subplots(
    #         len(depths),
    #         len(provider_configs.items()),
    #         figsize=(6 * len(provider_configs.items()), 7 * len(depths)),
    #         squeeze=False,
    #     )
    #     yLabel = "Training Loops"

    #     for depth in depths:
    #         i = 0
    #         for providerName, providerConfig in provider_configs.items():
    #             providerFileSlug = providerConfig["file_slug"]
    #             ax = axes[depth - 1, i]
    #             providerColour = provider_configs[providerName]["colour"]
    #             originalLabels = list(problem_configs.keys())
    #             plotLabels = [
    #                 "MVC" if label == "MinimumVertexCover" else label
    #                 for label in originalLabels
    #             ]
    #             ax.set_title(f"{providerName} (p={depth})", fontsize=18)
    #             if i == 0:
    #                 ax.set_ylabel(yLabel)
    #             else:
    #                 # Hide Y-axis labels on subsequent subplots for clarity
    #                 ax.tick_params(axis="y", labelleft=False)
    #             i += 1
    #             plotData = [
    #                 numLoops[depth][providerFileSlug][problem] for problem in originalLabels
    #             ]
    #             if any(len(d) > 0 for d in plotData):
    #                 ax.boxplot(
    #                     plotData,
    #                     labels=plotLabels,
    #                     patch_artist=True,
    #                     boxprops=dict(facecolor=providerColour),
    #                     medianprops=dict(color="red", linewidth=1.5),
    #                 )
    #                 # Set consistent y-limits for comparison (0 to 1, plus buffer)
    #                 ax.set_ylim(0, 100)
    #                 ax.grid(axis="y", linestyle="--", alpha=0.7)
    #             else:
    #                 ax.text(0.5, 0.5, "No data found", ha="center", va="center")
    #     fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.88, wspace=0.05)
    #     plt.show()

    # ... (numLoops data gathering remains the same) ...

    numDepths = len(depths)
    numProviders = len(provider_configs)

    fig, axes = plt.subplots(
        numDepths,
        numProviders,
        figsize=(6 * numProviders, 5 * numDepths),
        squeeze=False,
    )

    # 1. Centered Y-Label for the whole figure
    fig.supylabel("Training Loops", x=0.015)

    for depthIndex, depth in enumerate(depths):
        for providerIndex, (providerName, providerConfig) in enumerate(
            provider_configs.items()
        ):
            providerFileSlug = providerConfig["file_slug"]
            ax = axes[depthIndex, providerIndex]

            providerColour = provider_configs[providerName]["colour"]
            originalLabels = list(problem_configs.keys())
            plotLabels = [
                "MVC" if label == "MinimumVertexCover" else label
                for label in originalLabels
            ]

            # 3. Provider Titles: Only on the top row
            if depthIndex == 0:
                ax.set_title(providerName, fontsize=18)

            # 2. X-axis Labels: Only on the bottom row
            if depthIndex == numDepths - 1:
                ax.set_xticklabels(plotLabels, rotation=-45, ha="left")
            else:
                ax.set_xticklabels([])

            # 4. Depth Labels: On the right side of each row
            if providerIndex == numProviders - 1:
                # Add a label to the right of the last plot in each row
                ax.set_ylabel(
                    f"p = {depth}",
                    rotation=-90,
                    labelpad=25,
                )
                ax.yaxis.set_label_position("right")

            # Clean up inner tick marks for a shared-look grid
            ax.tick_params(axis="y", labelleft=(providerIndex == 0))

            plotData = [
                numLoops[depth][providerFileSlug][problem] for problem in originalLabels
            ]

            if any(len(d) > 0 for d in plotData):
                ax.boxplot(
                    plotData,
                    labels=(
                        plotLabels
                        if depthIndex == numDepths - 1
                        else [""] * len(plotLabels)
                    ),
                    patch_artist=True,
                    boxprops=dict(facecolor=providerColour),
                    medianprops=dict(color="red", linewidth=1.5),
                )
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_ylim(0, 105)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data found", ha="center", va="center")

    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.11, top=0.965, wspace=0.05)
    # --- Factor of Increase in Training Loops: Depth 1 to Depth 4 ---
    allLoopsDepth1 = []
    allLoopsDepth4 = []

    for providerName, providerConfig in provider_configs.items():
        providerFileSlug = providerConfig["file_slug"]
        for problem in problem_configs.keys():
            allLoopsDepth1.extend(numLoops[1][providerFileSlug][problem])
            allLoopsDepth4.extend(numLoops[4][providerFileSlug][problem])

    meanDepth1 = sum(allLoopsDepth1) / len(allLoopsDepth1) if allLoopsDepth1 else 0
    meanDepth4 = sum(allLoopsDepth4) / len(allLoopsDepth4) if allLoopsDepth4 else 0
    factor = meanDepth4 / meanDepth1 if meanDepth1 > 0 else 0

    print("\n" + "=" * 60)
    print("TRAINING LOOP FACTOR OF INCREASE: DEPTH 1 -> DEPTH 4")
    print("=" * 60)
    print(f"  Mean training loops at depth p=1: {meanDepth1:.2f}")
    print(f"  Mean training loops at depth p=4: {meanDepth4:.2f}")
    print(f"  Factor of increase (p=4 / p=1):   {factor:.2f}x")
    print("=" * 60 + "\n")

    # --- IBM/IonQ vs Alice & Bob Training Loop Comparison ---
    ibmIonqProviders = {"IBM_IDEAL", "IBM_NOISY", "IONQ_IDEAL", "IONQ_NOISY"}
    aliceBobProviders = {"ALICEBOB"}

    ibmIonqLoops = {depth: [] for depth in depths}
    aliceBobLoops = {depth: [] for depth in depths}

    for providerName, providerConfig in provider_configs.items():
        providerFileSlug = providerConfig["file_slug"]
        for depth in depths:
            for problem in problem_configs.keys():
                loops = numLoops[depth][providerFileSlug][problem]
                if providerName in ibmIonqProviders:
                    ibmIonqLoops[depth].extend(loops)
                elif providerName in aliceBobProviders:
                    aliceBobLoops[depth].extend(loops)

    print("=" * 60)
    print("IBM/IONQ vs ALICE & BOB: MEAN TRAINING LOOPS PER DEPTH")
    print("=" * 60)
    print(
        f"  {'Depth':<10} {'IBM/IonQ Mean':>15} {'Alice&Bob Mean':>15} {'Difference':>12} {'% Reduction':>12}"
    )
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*12} {'-'*12}")
    for depth in depths:
        ibmIonqMean = (
            sum(ibmIonqLoops[depth]) / len(ibmIonqLoops[depth])
            if ibmIonqLoops[depth]
            else 0
        )
        aliceBobMean = (
            sum(aliceBobLoops[depth]) / len(aliceBobLoops[depth])
            if aliceBobLoops[depth]
            else 0
        )
        difference = ibmIonqMean - aliceBobMean
        pctReduction = (difference / ibmIonqMean * 100) if ibmIonqMean > 0 else 0
        print(
            f"  {f'p={depth}':<10} {ibmIonqMean:>15.2f} {aliceBobMean:>15.2f} {difference:>12.2f} {pctReduction:>11.1f}%"
        )
    print("=" * 60 + "\n")

    plt.show()
