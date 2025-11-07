import json
import os
import glob
from config import problem_configs, provider_configs

# Configuration
SOURCE_DIR = "individual_results_warehouse"
FINAL_OUTPUT_DIR = "merged_results_warehouse_test"


def merge_all_problem_classes():
    # 'None' is used as a placeholder for filenames that do not specify a depth.
    qaoaDepths = [None, 1, 2, 3, 4, 20]
    completeResultSets = []
    individualResultsTracker = {}
    for problemName, problemConfig in problem_configs.items():
        for providerName, providerConfig in provider_configs.items():
            for depth in qaoaDepths:
                if (
                    depth is None
                ):  # still needed as ideal tsp results for depth 20 use legacy naming convention
                    depthSlug = ""
                    depthDescription = "legacy naming convention for p20"
                else:
                    depthSlug = f"p{depth}_"
                    depthDescription = f"p={depth}"

                print(
                    f"--- Processing {problemName} from {providerName} at {depthDescription} ---"
                )

                problemFileSlug = problemConfig["file_slug"]
                providerFileSlug = providerConfig["file_slug"]

                # Build the search pattern dynamically based on the current depth
                searchPattern = os.path.join(
                    SOURCE_DIR,
                    f"{problemFileSlug}{providerFileSlug}{depthSlug}num_*.json",
                )
                individualFiles = glob.glob(searchPattern)

                print(f"Found {len(individualFiles)} result files to merge.")
                if len(individualFiles) == 100:
                    completeResultSets.append((problemName, providerName, depth))

                if not individualFiles:
                    continue

                allResults = []
                metadata = {}

                collectedInstanceIDs = []

                for filePath in individualFiles:
                    with open(filePath, "r") as f:
                        data = json.load(f)

                    allResults.append(data["result"])

                    if not metadata:
                        metadata = data["metadata"]

                    collectedInstanceIDs.append(
                        data["result"]["instance_id"]
                    )  # keep ttrack of the instance ids i have collected

                missingInstanceIDs = []
                for i in range(1, 101):  # find missing instance ids
                    if i not in collectedInstanceIDs:
                        missingInstanceIDs.append(i)

                individualResultsTracker[(problemName, providerName, depth)] = (
                    len(individualFiles),
                    missingInstanceIDs,
                )

                allResults.sort(key=lambda x: x["instance_id"])

                finalData = {"metadata": metadata, "results": allResults}

                if depth is None:
                    depthSlug = "p20_"

                finalOutputFilename = f"MERGED_results_{problemFileSlug}{providerFileSlug}{depthSlug}.json"
                finalPath = os.path.join(FINAL_OUTPUT_DIR, finalOutputFilename)

                os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

                with open(finalPath, "w") as f:
                    json.dump(finalData, f, indent=4)

                print(
                    f"Successfully merged {len(allResults)} results into {finalPath}\n"
                )

    for problemName, problemConfig in problem_configs.items():
        for providerName, providerConfig in provider_configs.items():
            for depth in qaoaDepths:
                if (problemName, providerName, depth) not in completeResultSets:
                    if (depth is None or depth == 20) and (
                        providerName == "IBM_NOISY"
                        or providerName == "IONQ_NOISY"
                        or providerName == "ALICEBOB"
                    ):
                        continue  # skip depth 20 for noisy sims
                    else:
                        numFiles = individualResultsTracker.get(
                            (problemName, providerName, depth), (0, [])
                        )[0]
                        missing_ids = individualResultsTracker.get(
                            (problemName, providerName, depth), (0, [])
                        )[1]
                        print(
                            f"WARNING: {problemName} incomplete for {providerName} at depth {depth}, found: {numFiles} files"
                        )
                        if missing_ids:
                            print(f"Missing instance IDs: {missing_ids}")


if __name__ == "__main__":
    merge_all_problem_classes()
