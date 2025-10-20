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
    global numOptimisations
    transpiledHamil = cost_hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, transpiledHamil, params)

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
        "--repitition_id",
        type=int,
        required=True,
        help="The repeat number of this run.",
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
    repitition_id = args.repitition_id
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
        repitition_id,
    )


if __name__ == "__main__":
    # //////////    Variables    //////////
    FILEDIRECTORY = "isingBatches"  # note that this works for the HPC bc the file structure is different but not locally
    INDIVIDUAL_RESULTS_FOLDER = "individual_repeat_instance_results"

    # ////////////      Config.    ///////////
    (
        problemType,
        instanceOfInterest,
        reps_p,
        isingFileName,
        problemFileNameTag,
        backend_identifier,
        repitition_num,
    ) = setup_configuration()

    if backend_identifier == "IDEAL":
        backend_simulator = AerSimulator()
        print(f"Backend: {backend_simulator.name}")
    elif backend_identifier == "NOISY":
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
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend_simulator)
    candidate_circuit = pm.run(circuit)

    # linear ramp schedule
    initial_betas = np.linspace(np.pi, 0, reps_p, endpoint=False).tolist()
    initial_gammas = np.linspace(0, np.pi, reps_p, endpoint=False).tolist()
    initial_params = initial_betas + initial_gammas

    # starting training loop
    objective_func_vals = []
    numOptimisations = 0
    estimator = Estimator(mode=backend_simulator)
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
    sampler = Sampler(mode=backend_simulator)

    # collecting distribution
    sampleResult = sampler.run([optimized_circuit], shots=10000).result()
    dist = sampleResult[0].data.meas.get_counts()
    sortedDist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    print("Distribution:", sortedDist[0:5])  # print top 5 results

    # /// Saving results ///
    output_filename_unique = f"{problemFileNameTag}{backend_simulator.name}_p{reps_p}_inst_{instanceOfInterest}_repeat_{repitition_num}.json"  # CREATE A UNIQUE FILENAME FOR THIS JOB'S RESULT

    run_metadata = {"qaoaLayers": reps_p, "backend_name": backend_simulator.name}
    current_run_data = {
        "instance_id": instanceOfInterest,
        "repitition_id": repitition_num,
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
