# ==============================================================================
# D-Wave Quantum Annealing (QA) Execution Script
# ==============================================================================
# This script is designed to run a specific Ising model instance using the
# D-Wave framework, enabling comparison with QAOA results.
#
# It supports two primary execution modes via the '--simulator' argument:
# 1. 'simulatedAnnealing': Uses a local classical sampler (Path Integral Annealing).
# 2. 'quantumAnnealing': Connects to a remote D-Wave QPU via the DWaveSampler,
#    using EmbeddingComposite to handle graph mismatches.
#
# Execution Example:
# python anneal_dwave.py --problem_type MaxCut --instance_id 1 --simulator quantumAnnealing
# ==============================================================================
import dimod
import argparse
import json
from config import problem_configs
from dwave.samplers import PathIntegralAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite


def setup_configuration():
    """
    Handles script configuration by parsing command-line arguments.

    Parses required arguments: problem_type, instance_id, and simulator.
    It validates the problem type against `problem_configs` and constructs
    the filename for the Ising data.

    Returns:
        tuple: A tuple containing:
            - problemType (str)
            - instanceOfInterest (int)
            - isingFileName (str): Path to the Ising data file.
            - problemFileNameTag (str)
            - simulator (str): The execution backend identifier.
    """

    parser = argparse.ArgumentParser(
        description="Run a Quantum Annealing simulation for a specific problem class and instance."
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        required=True,
        help="The problem class to run (e.g., TSP, Knapsack).",
    )
    parser.add_argument(
        "--instance_id",
        type=int,
        required=True,
        help="The instance ID from the batch file to solve.",
    )

    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="The simulator to use (e.g., simulatedAnnealing, quantumAnnealing).",
    )

    args = parser.parse_args()
    problem_type = args.problem_type
    instance_of_interest = args.instance_id
    simulator = args.simulator

    # Select the configuration based on the determined problem_type
    try:
        selectedConfig = problem_configs[problem_type]
    except KeyError:
        raise ValueError(
            f"Error: '{problem_type}' is not a valid problem type. "
            f"Available options are: {list(problem_configs.keys())}"
        )

    # Construct the final filename
    problem_file_name_tag = selectedConfig["file_slug"]
    ising_file_name = f"{FILEDIRECTORY}/batch_Ising_data_{problem_file_name_tag}.json"

    return (
        problem_type,
        instance_of_interest,
        ising_file_name,
        problem_file_name_tag,
        simulator,
    )


def load_ising_model(file_path, instance_id):
    """
    Loads the Ising model terms (qubit indices) and weights (coefficients)
    for the specified instance from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing all Ising models.
        instance_id (int): The unique ID of the problem instance to load.

    Returns:
        tuple: (terms [list of indices], weights [list of coefficients]).
    """

    with open(file_path, "r") as f:
        all_isings_data = json.load(f)  # Assumes this loads a list of dicts

    selected_ising_data = {}
    # Find the desired ising model within list
    for ising_instance in all_isings_data:
        if (
            ising_instance["instance_id"] == instance_id
        ):  # Assumes 'instance_id' exists and is correct
            selected_ising_data = ising_instance
            break

    terms = selected_ising_data["terms"]
    weights = selected_ising_data["weights"]
    problem_type = selected_ising_data.get("problem_type")
    print(
        f"(Instance: {instance_id}) Problem type found from ising data: {problem_type}"
    )
    return terms, weights


def create_binary_quadratic_model(terms, weights):
    """
    Creates a Dimod BinaryQuadraticModel (BQM) from the raw Ising terms and weights.

    The BQM uses the 'SPIN' variable type required for Ising models in D-Wave.

    Args:
        terms (list): List of qubit indices (single for linear, pair for quadratic).
        weights (list): Corresponding coefficients.

    Returns:
        dimod.BinaryQuadraticModel: The BQM ready for sampling.
    """

    bqm = dimod.BinaryQuadraticModel(
        {}, {}, 0.0, dimod.SPIN
    )  # ising mdoels so need spin var type, ignore constant

    for term_indices, weight in zip(terms, weights):
        if len(term_indices) == 1:  # Linear term (h_i * s_i)
            bqm.add_variable(term_indices[0], weight)
        elif len(term_indices) == 2:  # Quadratic term (J_ij * s_i * s_j)
            bqm.add_interaction(term_indices[0], term_indices[1], weight)

    return bqm


if __name__ == "__main__":
    # //////////    Variables    //////////
    FILEDIRECTORY = "isingBatches"
    INDIVIDUAL_RESULTS_FOLDER = "individual_results_warehouse"

    # ////////////      Config.    ///////////
    (
        problemType,
        instanceOfInterest,
        isingFileName,
        problemFileNameTag,
        backend_identifier,
    ) = setup_configuration()

    # --- Sampler Selection ---
    if backend_identifier == "simulatedAnnealing":
        # Use a classical sampler (Path Integral) for local simulation
        sampler = PathIntegralAnnealingSampler()
        print("Using PathIntegralAnnealingSampler (Simulated)")
    elif backend_identifier == "quantumAnnealing":
        # Connect to D-Wave QPU and use EmbeddingComposite for automatic minor embedding
        sampler = EmbeddingComposite(DWaveSampler())
        print("Using DWaveSampler via EmbeddingComposite (Quantum Annealing)")
    else:
        raise ValueError(f"Unknown backend identifier: {backend_identifier}")

    # --- Model Preparation ---
    terms, weights = load_ising_model(isingFileName, instanceOfInterest)
    bqm = create_binary_quadratic_model(terms, weights)

    # --- Execution ---
    sampleset = sampler.sample(bqm)

    # --- Output ---
    print(sampleset)
