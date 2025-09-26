# This file contains the configurations for each problem tackled using the QAOA
problem_configs = {
    "TSP": {"file_slug": "TSP_9q_", "qubits": 9},
    "Knapsack": {"file_slug": "Knapsack_4_items_6q_", "qubits": 6, "items": 4},
    "MinimumVertexCover": {"file_slug": "MinimumVertexCover_4q_", "qubits": 4},
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
