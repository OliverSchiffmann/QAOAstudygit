# ==============================================================================
# QAOA Optimization and Simulation Script (ALICEBOB Provider)
# ==============================================================================
# This script executes the Quantum Approximate Optimization Algorithm (QAOA)
# using the Alice & Bob Qiskit provider, typically running inside an Apptainer
# container (see ALICEBOBSim.def).
#
# It performs the full variational optimization cycle:
# 1. Configuration: Reads problem (e.g., Knapsack), instance ID, and layer depth (p).
# 2. Hamiltonian Construction: Loads the Ising model and builds the Hamiltonians.
# 3. Optimization: Uses the classical optimizer **COBYLA** to find optimal QAOA angles,
#    sending the circuit to the AliceBobLocalProvider's Estimator primitive at each step.
# 4. Sampling: Samples the final optimized circuit using the Sampler primitive.
# 5. Output: Saves all metrics (cost, parameters, loops) to a uniquely named JSON file.
#
# Execution Environment: EMU:15Q:LOGICAL_EARLY simulator backend via AliceBobLocalProvider.
# ==============================================================================
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
    """
    The objective function called by the classical optimizer (COBYLA) for ALICEBOB.

    This version explicitly binds parameters and runs the pass manager (transpilation)
    *inside* the optimization loop.

    Args:
        params (list): The current QAOA angles (gammas and betas).
        ansatz (QuantumCircuit): The untranspiled QAOA circuit template.
        estimator (EstimatorV2): The configured Qiskit Estimator instance (ALICEBOB backend).
        pass_manager (PassManager): The pre-configured Qiskit transpilation pipeline.
        cost_hamiltonian (SparsePauliOp): The problem's cost Hamiltonian.

    Returns:
        float: The real part of the expectation value (the cost).
    """
    global numOptimisations
    # Bind the current parameters to the ansatz template
    boundAnsatz = ansatz.assign_parameters(params)
    # Transpile the bound circuit for the ALICEBOB backend
    transpiledAnsatz = pass_manager.run(boundAnsatz)
    # print(transpiledAnsatz)
    # transpiledHamil = cost_hamiltonian.apply_layout(transpiledAnsatz.layout) # Commented out, layout applied implicitly by Estimator

    # Define the Public Unit (Pub): circuit and Hamiltonian
    pub = (transpiledAnsatz, cost_hamiltonian)

    # Run the Estimator job
    job = estimator.run([pub])
    results = job.result()[0]
    # Extract the expectation value (evs)
    cost = results.data.evs

    # Convert complex result to a real float
    cost_float = float(np.real(cost))
    objective_func_vals.append(cost_float)

    # Update and print optimization status
    numOptimisations = numOptimisations + 1
    print(f"Optimization step {numOptimisations}, Cost: {cost_float}")

    return cost_float


def setup_configuration():
    """
    Handles script configuration by parsing command-line arguments.

    Parses required arguments: problem_type, instance_id, and num_layers.

    Returns:
        tuple: A tuple containing:
            - problemType (str)
            - instanceOfInterest (int)
            - numLayers (int)
            - isingFileName (str): Path to the Ising data file.
            - problemFileNameTag (str)
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

    # --- ALICEBOB Backend Setup ---
    provider = AliceBobLocalProvider()
    backend_simulator = provider.get_backend(
        "EMU:15Q:LOGICAL_EARLY"
    )  # options are 'EMU:40Q:LOGICAL_NOISELESS', EMU:15Q:LOGICAL_EARLY

    # ////////////      Config.    ///////////
    (
        problemType,
        instanceOfInterest,
        reps_p,
        isingFileName,
        problemFileNameTag,
    ) = setup_configuration()

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
    # Assemble the QAOA circuit using the cost, mixer, initial state, and depth (reps_p)
    qaoaKwargs = {
        "cost_operator": costHamil,
        "reps": reps_p,
        "initial_state": initialCircuit,
        "mixer_operator": mixerHamil,
    }
    circuit = QAOAAnsatz(**qaoaKwargs)
    # Add measurements for final state collection
    circuit.measure_all()

    print("Generating pass manager...")
    # Generate the transpilation pipeline for the target backend
    passManager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend_simulator,
    )
    # Initialize the Estimator primitive
    estimator = Estimator(mode=backend_simulator)

    # trying linear ramp schedule again in case
    # Define the initial parameter guess (linear ramp initialization)
    initial_betas = np.linspace(np.pi, 0, reps_p, endpoint=False).tolist()
    initial_gammas = np.linspace(0, np.pi, reps_p, endpoint=False).tolist()
    initial_params = initial_betas + initial_gammas

    # starting training loop
    objective_func_vals = []
    numOptimisations = 0

    print("Starting optimsation loop...")
    # --- Optimization using COBYLA ---
    trainResult = minimize(
        cost_func_estimator,
        initial_params,
        args=(
            circuit,
            estimator,
            passManager,
            costHamil,
        ),  # Note: PassManager is an explicit argument here
        method="COBYLA",  # Using COBYLA for gradient free optimization also fast
        tol=1e-3,
        options={"maxiter": 500},
    )
    print(trainResult.x, trainResult.fun, numOptimisations)

    # Apply optimal parameters to the circuit and transpile once more for sampling
    finalBoundCircuit = circuit.assign_parameters(trainResult.x)
    optimized_circuit = passManager.run(finalBoundCircuit)

    # /// Sampling ///
    # setting backend for sampling
    # Initialize the Sampler primitive
    sampler = Sampler(mode=backend_simulator)
    # sampler.options.default_shots = 10000

    # collecting distribution
    # Run the sampling job with a fixed number of shots
    sampleResult = sampler.run([optimized_circuit], shots=10000).result()
    dist = sampleResult[0].data.meas.get_counts()
    # Sort the measured bit strings by count (descending)
    sortedDist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    print("Distribution:", sortedDist[0:5])  # print top 5 results

    # /// Saving results ///
    # Create a unique filename based on all job parameters and backend slug
    output_filename_unique = f"{problemFileNameTag}{backend_simulator.name}_p{reps_p}_num_{instanceOfInterest}.json"

    # Compile metadata and the run's results
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
