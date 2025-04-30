# website that this example was taken from: https://learning.quantum.ibm.com/tutorial/quantum-approximate-optimization-algorithm
# This does submit to IBM usccessfully, but it seems to run many many jobs in the session, unsure whne it would finish if left unattended.
# Packages \/\/\/
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Packages for quantum stuff
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# functions
def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list


def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)

    return cost


# 1
n = 5
graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n, 1))
edge_list = [
    (0, 1, 1.0),
    (0, 2, 1.0),
    (0, 4, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0),
]
graph.add_edges_from(edge_list)
draw_graph(graph, node_size=600, with_labels=True)


# 2
max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
print("Cost Function Hamiltonian:", cost_hamiltonian)

# 3
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
circuit.measure_all()
circuit.draw("mpl")
print(circuit.parameters)

# 4
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="1b0ab941b010112fea6054ef44befc6d05ea3eca559a8b7b9dcaf68adcac0a64b6f24eafce9f69b96e6bf67eb223e27c7085ab8da9e12a0ba54f10c48616effa",
    overwrite=True,
    set_as_default=True,
)
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(min_num_qubits=127)
print(backend)


# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
candidate_circuit = pm.run(circuit)
candidate_circuit.draw("mpl", fold=False, idle_wires=False)

# 5 Setting the parameters for the QAOA cost and mixer hamiltonian
initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [initial_gamma, initial_beta, initial_gamma, initial_beta]

# 6 finding the optimal parameters for the QAOA circuit using a 'estimator'
objective_func_vals = []  # Global variable
with Session(
    backend=backend, max_time=25
) as session:  # maxtime specified in seconds (int)
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    print(result)


# plt.figure(figsize=(12, 6))
# plt.plot(objective_func_vals)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")


# plt.show(block=False)
# plt.pause(1)
# input()
# plt.close()
