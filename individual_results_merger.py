import json
import os
import glob
from config import problem_configs, provider_configs

# Configuration
SOURCE_DIR = "individual_results_warehouse"
FINAL_OUTPUT_DIR = "merged_results_warehouse"


def merge_all_problem_classes():
    # 'None' is used as a placeholder for filenames that do not specify a depth.
    qaoaDepths = [None, 1, 2, 3, 4]

    for problemName, problemConfig in problem_configs.items():
        for providerName, providerConfig in provider_configs.items():
            for depth in qaoaDepths:
                if depth is None:
                    depthSlug = ""
                    depthDescription = "default depth (20)"
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
                if not individualFiles:
                    print(
                        f"No result files found for pattern: {searchPattern}. Skipping.\n"
                    )
                    continue

                print(f"Found {len(individualFiles)} result files to merge.")

                allResults = []
                metadata = {}

                for filePath in individualFiles:
                    with open(filePath, "r") as f:
                        data = json.load(f)

                    allResults.append(data["result"])

                    if not metadata:
                        metadata = data["metadata"]

                allResults.sort(key=lambda x: x["instance_id"])

                finalData = {"metadata": metadata, "results": allResults}

                finalOutputFilename = f"MERGED_results_{problemFileSlug}{providerFileSlug}{depthSlug}.json"
                finalPath = os.path.join(FINAL_OUTPUT_DIR, finalOutputFilename)

                os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

                with open(finalPath, "w") as f:
                    json.dump(finalData, f, indent=4)

                print(
                    f"Successfully merged {len(allResults)} results into {finalPath}\n"
                )


if __name__ == "__main__":
    merge_all_problem_classes()
