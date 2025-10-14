import argparse
import numpy as np
from scipy.optimize import minimize
from config import problem_configs
from helpers import (
    load_ising_and_build_hamiltonian,
    save_single_result,
    build_mixer_hamiltonian,
    create_inital_state,
)

# Packages for quantum stuff
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_alice_bob_provider import AliceBobLocalProvider


# ///////////  Unique Functions  //////////
def cost_func_estimator(params, ansatz, estimator, pass_manager, cost_hamiltonian):
    global numOptimisations
    boundAnsatz = ansatz.assign_parameters(params)
    transpiledAnsatz = pass_manager.run(boundAnsatz)
    # print(transpiledAnsatz)
    # transpiledHamil = cost_hamiltonian.apply_layout(transpiledAnsatz.layout)
    pub = (transpiledAnsatz, cost_hamiltonian)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs

    cost_float = float(np.real(cost))
    objective_func_vals.append(cost_float)

    numOptimisations = numOptimisations + 1
    print(f"Optimization step {numOptimisations}, Cost: {cost_float}")

    return cost_float


def setup_configuration():
    """
    Handles script configuration by parsing command-line arguments.

    Returns:
        tuple: A tuple containing:
            - problem_type (str)
            - instanceOfInterest (int)
            - isingFileName (str)
    """
    FILEDIRECTORY = "isingBatches"

    parser = argparse.ArgumentParser(
        description="Run a QAOA simulation for a specific problem class and instance."
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
        "--num_layers",
        type=int,
        required=True,
        help="The number of QAOA layers to build.",
    )
    args = parser.parse_args()
    problem_type = args.problem_type
    instance_of_interest = args.instance_id
    num_layers = args.num_layers

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
        num_layers,
        ising_file_name,
        problem_file_name_tag,
    )


if __name__ == "__main__":
    # //////////    Variables    //////////
    FILEDIRECTORY = "isingBatches"
    INDIVIDUAL_RESULTS_FOLDER = "individual_results"
    provider = AliceBobLocalProvider()
    backend_simulator = provider.get_backend(
        "EMU:15Q:LOGICAL_EARLY"
    )  # options are 'EMU:40Q:LOGICAL_NOISELESS', EMU:15Q:LOGICAL_EARLY

    # ////////////      Config.    ///////////
    problemType, instanceOfInterest, reps_p, isingFileName, problemFileNameTag = (
        setup_configuration()
    )

    print(
        f"Problem Type: {problemType}, Instance ID: {instanceOfInterest}, Ising model file name: {isingFileName}"
    )

    print(problemFileNameTag)

    # /// training ///
    # --- cost hamiltonian ---
    costHamil, numQubits, isingTerms, weightCapacity = load_ising_and_build_hamiltonian(
        isingFileName, instanceOfInterest
    )
    print(f"Problem class is: {problemType}")
    if problemType == "Knapsack":
        print(f"Capacity of this knapsack is: {weightCapacity}")

    print(f"Quadratic and linear terms of the Ising model are: {isingTerms}")

    # --- mixer ---
    mixerHamil = build_mixer_hamiltonian(numQubits, problemType)
    print(mixerHamil)

    # --- inital state ---
    initialCircuit = create_inital_state(numQubits, problemType, weightCapacity)
    print(initialCircuit)

    # --- QAOA Ansatz ---
    qaoaKwargs = {
        "cost_operator": costHamil,
        "reps": reps_p,
        "initial_state": initialCircuit,
        "mixer_operator": mixerHamil,
    }
    circuit = QAOAAnsatz(**qaoaKwargs)
    circuit.measure_all()

    print("Generating pass manager...")
    passManager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend_simulator,
    )
    estimator = Estimator(mode=backend_simulator)
    # trying linear ramp schedule again in case
    initial_betas = np.linspace(np.pi, 0, reps_p, endpoint=False).tolist()
    initial_gammas = np.linspace(0, np.pi, reps_p, endpoint=False).tolist()
    initial_params = initial_betas + initial_gammas

    # starting training loop
    objective_func_vals = []
    numOptimisations = 0

    print("Starting optimsation loop...")
    trainResult = minimize(
        cost_func_estimator,
        initial_params,
        args=(circuit, estimator, passManager, costHamil),
        method="COBYLA",  # Using COBYLA for gradient free optimization also fast
        tol=1e-3,
        options={"maxiter": 500},
    )
    print(trainResult.x, trainResult.fun, numOptimisations)

    finalBoundCircuit = circuit.assign_parameters(trainResult.x)
    optimized_circuit = passManager.run(finalBoundCircuit)

    # /// Sampling ///
    # setting backend for sampling
    sampler = Sampler(mode=backend_simulator)
    # sampler.options.default_shots = 10000

    # collecting distribution
    sampleResult = sampler.run([optimized_circuit], shots=10000).result()
    dist = sampleResult[0].data.meas.get_counts()
    sortedDist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    print("Distribution:", sortedDist[0:5])  # print top 5 results

    # /// Saving results ///
    output_filename_unique = f"{problemFileNameTag}{backend_simulator.name}_p{reps_p}_num_{instanceOfInterest}.json"

    run_metadata = {"qaoaLayers": reps_p, "backend_name": backend_simulator.name}
    current_run_data = {
        "instance_id": instanceOfInterest,
        "sampled_distribution": sortedDist,
        "num_training_loops": numOptimisations,
        "final_training_cost": trainResult.fun,
        "optimal_params": trainResult.x,
    }

    # Combine metadata and the result for this single run
    data_to_save = {"metadata": run_metadata, "result": current_run_data}

    # Call the new, simple save function
    save_single_result(
        folder_path=INDIVIDUAL_RESULTS_FOLDER,
        file_name=output_filename_unique,
        data=data_to_save,
    )
