# This file contains the configurations for each problem tackled using the QAOA
problem_configs = {
    "TSP": {"file_slug": "TSP_9q_", "qubits": 9},
    "Knapsack": {"file_slug": "Knapsack_6_items_9q_", "qubits": 9, "items": 6},
    "MinimumVertexCover": {"file_slug": "MinimumVertexCover_9q_", "qubits": 9},
    # Can easily add more problem classes here in the future
    # 'MaxCut': { ... }
}

provider_configs = {
    "IONQ_IDEAL": {"file_slug": "ionq_simulator_"},
    "IBM_IDEAL": {"file_slug": "aer_simulator_"},
    "IONQ_NOISY": {"file_slug": "aria-1_"},
    "IBM_NOISY": {"file_slug": "aer_simulator_from(fake_torino)_"},
    "ALICEBOB": {"file_slug": "EMU:40Q:LOGICAL_EARLY_"},
}
