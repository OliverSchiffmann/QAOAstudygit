# a script to submit a circuit to UBMQ and collect the average time for execution
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
