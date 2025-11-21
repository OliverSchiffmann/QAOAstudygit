# ==============================================================================
# QAOA Optimization and Simulation Script (Main Batch Run)
# ==============================================================================
# This is the core Python script used to run the Quantum Approximate Optimization
# Algorithm (QAOA) study for a batch of 100 instances on the IBM Quantum simulator.
#
# It performs the full variational quantum optimization cycle:
# 1. Configuration: Reads problem type, instance ID, layer depth (p), and
#    simulator choice (IDEAL/NOISY) from command-line arguments.
# 2. Hamiltonian Construction: Loads the Ising model for the given instance and
#    builds the Cost and Mixer Hamiltonians.
# 3. Optimization: Uses the classical optimizer **COBYLA** to minimize the cost
#    function by iteratively calling the Qiskit Estimator primitive.
# 4. Sampling: Samples the final optimized circuit using the Qiskit Sampler primitive.
# 5. Output: Saves all metrics (final cost, optimal parameters, optimization steps,
#    and distribution) to a uniquely named JSON file, typically indexed by instance ID.
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
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import (
    FakeTorino,
)


# ///////////  Unique Functions  //////////
def cost_func_estimator(params, ansatz, estimator, cost_hamiltonian):
    """
    The objective function called by the classical optimizer (COBYLA).

    This function calculates the expectation value of the cost Hamiltonian
    with respect to the QAOA ansatz, parameterized by 'params'. It also tracks
    and prints the optimization progress.

    Args:
        params (list): The current QAOA angles (gammas and betas).
        ansatz (QuantumCircuit): The transpiled QAOA circuit.
        estimator (EstimatorV2): The configured Qiskit Estimator instance.
        cost_hamiltonian (SparsePauliOp): The problem's cost Hamiltonian.

    Returns:
        float: The real part of the expectation value (the cost).
    """
    global numOptimisations
    # Apply layout to the Hamiltonian to match the circuit's layout after transpilation
    transpiledHamil = cost_hamiltonian.apply_layout(ansatz.layout)
    # Define the Public Unit (Pub) for the Estimator
    pub = (ansatz, transpiledHamil, params)

    # Run the Estimator job
    job = estimator.run([pub])
    results = job.result()[0]
    # Extract the expectation value (evs)
    cost = results.data.evs

    # Convert complex result to a real float
    cost_float = float(np.real(cost))
    # Track the history of cost values
    objective_func_vals.append(cost_float)

    # Update and print optimization status
    numOptimisations = numOptimisations + 1
    print(f"Optimization step {numOptimisations}, Cost: {cost_float}")

    return cost_float


def setup_configuration():
    """
    Handles script configuration by parsing command-line arguments.

    Parses required arguments: problem_type, instance_id, num_layers, and simulator.
    It validates the problem type and constructs the filename for the Ising data.

    Returns:
        tuple: A tuple containing:
            - problem_type (str)
            - instance_of_interest (int)
            - num_layers (int)
            - ising_file_name (str): Path to the Ising data file.
            - problem_file_name_tag (str)
            - simulator (str): 'IDEAL' or 'NOISY'.
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

    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="The simulator to use.",
    )

    args = parser.parse_args()
    problem_type = args.problem_type
    instance_of_interest = args.instance_id
    num_layers = args.num_layers
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
        num_layers,
        ising_file_name,
        problem_file_name_tag,
        simulator,
    )


if __name__ == "__main__":
    # //////////    Variables    //////////
    FILEDIRECTORY = "isingBatches"
    INDIVIDUAL_RESULTS_FOLDER = "individual_results_warehouse"

    # ////////////      Config.    ///////////
    (
        problemType,
        instanceOfInterest,
        reps_p,
        isingFileName,
        problemFileNameTag,
        backend_identifier,
    ) = setup_configuration()

    # --- Backend Selection ---
    # Configure the Qiskit AerSimulator based on the command-line argument
    if backend_identifier == "IDEAL":
        backend_simulator = AerSimulator()
        print(f"Backend: {backend_simulator.name}")
    elif backend_identifier == "NOISY":
        # Use AerSimulator initialized with the noise model of a real device (FakeTorino)
        backend_simulator = AerSimulator.from_backend(FakeTorino())
        print(f"Backend: {backend_simulator.name}")
    else:
        raise ValueError(f"Unknown backend identifier: {backend_identifier}")

    print(
        f"Problem Type: {problemType}, Instance ID: {instanceOfInterest}, Ising model file name: {isingFileName}"
    )

    print(problemFileNameTag)

    # /// training ///
    # --- cost hamiltonian ---
    # Load the instance data and construct the Cost Hamiltonian (H_C)
    costHamil, numQubits, isingTerms, weightCapacity = load_ising_and_build_hamiltonian(
        isingFileName, instanceOfInterest
    )
    print(f"Problem class is: {problemType}")
    if problemType == "Knapsack":
        print(f"Capacity of this knapsack is: {weightCapacity}")

    print(f"Quadratic and linear terms of the Ising model are: {isingTerms}")

    # --- mixer ---
    # Build the Mixer Hamiltonian (H_M)
    mixerHamil = build_mixer_hamiltonian(numQubits, problemType)
    print(mixerHamil)

    # --- inital state ---
    # Create the initial quantum state circuit
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
    # Add measurements to the circuit
    circuit.measure_all()
    # Transpile the circuit for the selected simulator
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend_simulator)
    candidate_circuit = pm.run(circuit)

    # linear ramp schedule
    # Define the initial parameter guess (linear ramp initialization)
    initial_betas = np.linspace(np.pi, 0, reps_p, endpoint=False).tolist()
    initial_gammas = np.linspace(0, np.pi, reps_p, endpoint=False).tolist()
    initial_params = initial_betas + initial_gammas

    # starting training loop
    objective_func_vals = []
    numOptimisations = 0
    # Initialize the Estimator primitive with the configured simulator
    estimator = Estimator(mode=backend_simulator)
    # --- Optimization using COBYLA ---
    trainResult = minimize(
        cost_func_estimator,
        initial_params,
        args=(candidate_circuit, estimator, costHamil),
        method="COBYLA",  # Using COBYLA for gradient free optimization also fast
        tol=1e-3,
        options={"maxiter": 500},
    )
    print(trainResult.x, trainResult.fun, numOptimisations)

    # /// Sampling ///
    # Assigning the optimized parameters to the circuit
    optimized_circuit = candidate_circuit.assign_parameters(trainResult.x)

    # setting backend for sampling
    # Initialize the Sampler primitive with the configured simulator
    sampler = Sampler(mode=backend_simulator)

    # collecting distribution
    # Run the sampling job with a fixed number of shots
    sampleResult = sampler.run([optimized_circuit], shots=10000).result()
    dist = sampleResult[0].data.meas.get_counts()
    # Sort the measured bit strings by count (descending)
    sortedDist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    print("Distribution:", sortedDist[0:5])  # print top 5 results

    # /// Saving results ///
    # Create a unique filename based on all job parameters
    output_filename_unique = f"{problemFileNameTag}{backend_simulator.name}_p{reps_p}_num_{instanceOfInterest}.json"  # CREATE A UNIQUE FILENAME FOR THIS JOB'S RESULT

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
