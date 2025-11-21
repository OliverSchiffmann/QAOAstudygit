# ==============================================================================
# Global Optima Solver (Exact Enumeration)
# ==============================================================================
# This script performs a full enumeration of the solution space for every Ising
# model instance in the batch files using dimod.ExactSolver.
#
# Its purpose is to accurately determine the absolute minimum energy (optimal cost)
# and maximum energy (worst possible cost) for normalization of QAOA results.
#
# Input: Reads batch Ising JSON files from 'isingBatches/'.
# Output: Creates new 'solved_batch_Ising_data_...json' files containing the list
#         of optimal solutions, optimal cost, and max cost for every instance.
# ==============================================================================
# %%
# use classicalvenv
import dimod
import json
import numpy as np
from config import problem_configs
import os
import glob


# %%
# ////////// variables //////////
FILEDIRECTORY = "isingBatches/"


# %%
def loadIsing(filePath):
    """
    Loads an Ising model JSON file containing a list of instance dictionaries.

    Args:
        filePath (str): Path to the JSON file.

    Returns:
        list: List of dictionaries, each representing an Ising problem instance.
    """
    with open(filePath, "r") as f:
        allIsingsData = json.load(f)
    return allIsingsData


def saveSolutionsToJson(solutionsDict, outputFilepath):
    """
    Formats and saves the global optimal solutions and energy bounds for a
    problem class to a new JSON file.

    Args:
        solutionsDict (dict): Dictionary mapping instance IDs to their
                              solutions list and max cost.
        outputFilepath (str): The path and filename for the output JSON file.
    """
    outputList = []
    for instanceId, data in solutionsDict.items():
        solutionsList = data["solutions_list"]
        maxCost = data["max_cost"]
        solutions = [sol[0] for sol in solutionsList]

        # The cost is the same for all solutions of an instance, so grab the first one
        cost = solutionsList[0][1]

        formattedEntry = {
            "instance_id": instanceId,
            "solutions": solutions,  # Stored as a list of spin strings (+1, -1)
            "cost": float(cost),
            "max_cost": maxCost,
        }

        outputList.append(formattedEntry)

    with open(outputFilepath, "w") as f:
        json.dump(outputList, f, indent=2)


def convertToDimodIsing(allIsingsData):
    """
    Converts the script's custom Ising JSON format (terms/weights) into
    Dimod's standard bias format (linear and quadratic biases).

    Args:
        allIsingsData (list): List of Ising dictionaries loaded from file.

    Returns:
        dict: Nested dictionary mapping instance ID to a tuple of
              (linearBias, quadraticBias) dictionaries.
    """
    dimodIsing = {}
    for instance in allIsingsData:
        instanceId = instance["instance_id"]
        terms = instance["terms"]
        weights = instance["weights"]

        linearBias = {}
        quadraticBias = {}

        for term, weight in zip(terms, weights):
            if len(term) == 1:
                # Linear term (single variable)
                variable = f"x{term[0]}"
                linearBias[variable] = weight
            elif len(term) == 2:
                # Quadratic term (two variables/interaction)
                var1 = f"x{term[0]}"
                var2 = f"x{term[1]}"
                quadraticBias[(var1, var2)] = weight

        dimodIsing[instanceId] = (linearBias, quadraticBias)
    return dimodIsing


def extractLowestEnergySolutions(allSolutions):
    """
    Extracts all degenerate ground state solutions from a dimod SampleSet,
    formatting them into a list of (solutionString, energy) tuples.

    Args:
        allSolutions (dimod.SampleSet): The result set from dimod.ExactSolver.

    Returns:
        list: A list of tuples [(solution_string, energy)] for all ground states.
    """
    results = []
    # The sampleset is sorted by energy, so the first record is the lowest
    lowestEnergy = allSolutions.first.energy
    for datum in allSolutions.data():
        if datum.energy > lowestEnergy:
            # Stop once we pass the ground state energy
            break

        # Sort variables by index (e.g., x0, x1, x2) for consistent string ordering
        sortedKeys = sorted(datum.sample.keys(), key=lambda v: int(v[1:]))
        variableValues = []
        for key in sortedKeys:
            val = datum.sample[key]
            # Format the spin values for output (e.g., +1 or -1)
            if val > 0:
                variableValues.append(f"+{val}")
            else:
                variableValues.append(str(val))

        variableString = ",".join(variableValues)
        energy = float(datum.energy)
        results.append((variableString, energy))

    return results


# %%
if __name__ == "__main__":
    for problemName, config in problem_configs.items():
        print(f"--- Processing class: {problemName} ---")

        fileSlug = config["file_slug"]
        searchPattern = os.path.join(FILEDIRECTORY, f"batch*{fileSlug}.json")
        # Locate the batch file for this problem type
        isingBatchFile = glob.glob(searchPattern)[0]

        # Load and convert the Ising data to dimod BQM format
        isings = loadIsing(isingBatchFile)
        dimodIsings = convertToDimodIsing(isings)

        # Initialize the exact solver for full enumeration
        solver = dimod.ExactSolver()

        globalOptima = {}
        # Iterate over every instance in the batch
        for instance, biases in dimodIsings.items():
            # Get linear and quadratic biases for the current instance
            I = dimodIsings[instance]
            # Solve exactly
            sampleset = solver.sample_ising(I[0], I[1])
            # Extract all solutions corresponding to the minimum energy
            instanceGlobalOptima = extractLowestEnergySolutions(sampleset)
            # Find the maximum energy state (worst cost)
            maxEnergy = np.max(sampleset.record.energy)

            globalOptima[instance] = {
                "solutions_list": instanceGlobalOptima,
                "max_cost": float(maxEnergy),
            }

        print(f"Top 10 solutions for the final problem instance ({instance}) \/ \/ \/")
        print(sampleset.slice(10))
        print(sampleset)

        # Split the path into directory and filename
        directory, filename = os.path.split(isingBatchFile)

        # Construct the new path by adding 'solved_' to the filename
        outputFile = os.path.join(directory, f"solved_{filename}")
        saveSolutionsToJson(globalOptima, outputFile)
