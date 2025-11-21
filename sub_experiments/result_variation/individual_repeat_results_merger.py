# ==============================================================================
# QAOA Individual Result Merging Utility
# ==============================================================================
# This script consolidates results from repeated individual QAOA simulation runs
# into single, structured JSON files for easier downstream analysis and plotting.
#
# It iterates over all defined problem types, quantum providers/simulators,
# and QAOA layer depths (p), searching for all repeated runs of a specific,
# hardcoded problem instance ID (currently ID 2).
#
# Inputs: Individual JSON files located in the SOURCE_DIR.
# Outputs: Consolidated MERGED_results JSON files in the FINAL_OUTPUT_DIR.
# ==============================================================================
import json
import os
import glob
from config import problem_configs, provider_configs

# Configuration
SOURCE_DIR = "individual_repeat_instance_results"
FINAL_OUTPUT_DIR = "merged_repeat_instance_results"


def merge_all_problem_classes():
    """
    Iterates through all problem, provider, and depth combinations to merge
    individual result files for a specific instance ID.

    The merged file contains a list of 'result' objects, one for each repetition,
    along with the common 'metadata'. Handles special case for p=20 results
    using a legacy naming convention.
    """
    # 'None' is used as a placeholder for filenames that do not specify a depth.
    qaoaDepths = [None, 1, 2, 3, 4, 20]
    # NOTE: This script is currently hardcoded to merge results only for instance 2.
    problem_instance_id = 2

    for problemName, problemConfig in problem_configs.items():
        for providerName, providerConfig in provider_configs.items():
            for depth in qaoaDepths:
                if (
                    depth is None
                ):  # still needed as ideal tsp results for depth 20 use legacy naming convention
                    # Custom slug handling for the legacy p=20 files that didn't include the 'pXX_' slug
                    depthSlug = ""
                    depthDescription = "legacy naming convention for p20"
                else:
                    # Standard naming convention
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
                    f"{problemFileSlug}{providerFileSlug}{depthSlug}inst_{problem_instance_id}_repeat_*.json",
                )
                individualFiles = glob.glob(searchPattern)
                if not individualFiles:
                    print(
                        f"No result files found for pattern: {searchPattern}. Skipping.\n"
                    )
                    continue

                print(f"Found {len(individualFiles)} result files to merge.")

                allResults = []
                metadata = {}

                # Read and aggregate data from all found individual files
                for filePath in individualFiles:
                    with open(filePath, "r") as f:
                        data = json.load(f)

                    allResults.append(data["result"])

                    # Capture metadata from the first file found
                    if not metadata:
                        metadata = data["metadata"]

                # Sort results by instance ID (though this script only uses one ID)
                allResults.sort(key=lambda x: x["instance_id"])

                finalData = {"metadata": metadata, "results": allResults}

                # Fix the depth slug for the output filename if the legacy depth was processed
                if depth is None:
                    depthSlug = "p20_"

                # Construct the final output filename
                finalOutputFilename = f"MERGED_results_{problemFileSlug}{providerFileSlug}{depthSlug}inst_{problem_instance_id}.json"
                finalPath = os.path.join(FINAL_OUTPUT_DIR, finalOutputFilename)

                # Ensure the output directory exists
                os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

                # Write the consolidated data to the final JSON file
                with open(finalPath, "w") as f:
                    json.dump(finalData, f, indent=4)

                print(
                    f"Successfully merged {len(allResults)} results into {finalPath}\n"
                )


if __name__ == "__main__":
    merge_all_problem_classes()
