# This file contains the configurations for each problem tackled using the QAOA
problem_configs = {
    "TSP": {"file_slug": "TSP_9q_", "qubits": 9},
    "Knapsack": {"file_slug": "Knapsack_6_items_9q_", "qubits": 9, "items": 6},
    "MinimumVertexCover": {"file_slug": "MinimumVertexCover_9q_", "qubits": 9},
    # Can easily add more problem classes here in the future
    # 'MaxCut': { ... }
}

provider_configs = {
    "IONQ": {"file_slug": "ionq_simulator"},
    "IBM": {"file_slug": "aer_simulator"},
    "ALICEBOB": {"file_slug": "EMU:40Q:LOGICAL_NOISELESS"},
}
