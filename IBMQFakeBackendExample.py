import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Packages for quantum stuff
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


# functions \/\/\/\/\/
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


def cost_func_estimator(params, ansatz, estimator):  # Removed hamiltonian argument
    # The Hamiltonian information should ideally be accessed via the ansatz
    # or passed if strictly needed, but Estimator needs the observable.
    # Rebuild observable based on ansatz qubits if needed, or use pre-built one
    # Assuming cost_hamiltonian is defined globally/accessible here for simplicity
    # apply_layout might be needed if the estimator doesn't handle it automatically
    # based on the transpiled ansatz. Let's assume estimator + transpiled ansatz handles layout.
    isa_observable = cost_hamiltonian.apply_layout(
        ansatz.layout
    )  # Apply layout based on *transpiled* ansatz

    pub = (ansatz, isa_observable, [params])  # Pass params in a list for V2 PUB
    job = estimator.run(pubs=[pub])  # Pass pubs as keyword arg

    # Handle potential errors during job execution
    try:
        results = job.result()[0]  # Get the first PubResult
        cost = results.data.evs[0]  # Get the first (and only) EV
    except Exception as e:
        print(f"Error getting result for params {params}: {e}")
        # Return a high cost or handle error appropriately for the optimizer
        return float("inf")

    # Ensure cost is a standard float for scipy.optimize
    cost_float = float(np.real(cost))
    objective_func_vals.append(cost_float)
    print(f"Params: {params}, Cost: {cost_float}")  # Add print for debugging
    return cost_float


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
# draw_graph(graph, node_size=600, with_labels=True)


# 2
max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
print("Cost Function Hamiltonian:", cost_hamiltonian)

# 3
# Hopefully we should be able to insert the circuit created using the openQAOA package somewhere here
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
# circuit.measure_all() #measurement not needed for estimator?
print("Ansatz Parameters:", circuit.parameters)
# circuit.draw("mpl")


# 4
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="1b0ab941b010112fea6054ef44befc6d05ea3eca559a8b7b9dcaf68adcac0a64b6f24eafce9f69b96e6bf67eb223e27c7085ab8da9e12a0ba54f10c48616effa",
    overwrite=True,
    set_as_default=True,
)
service = QiskitRuntimeService(channel="ibm_quantum")
fakeBackend = FakeBrisbane()
simulator = AerSimulator.from_backend(fakeBackend)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3, backend=fakeBackend)
candidate_circuit = pm.run(circuit)
# candidate_circuit.draw("mpl", fold=False, idle_wires=False)

# 5 Setting the parameters for the QAOA cost and mixer hamiltonian
initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [initial_gamma, initial_beta, initial_gamma, initial_beta]

# 6 finding the optimal parameters for the QAOA circuit using a 'estimator'
objective_func_vals = []  # Global variable

estimator = Estimator(mode=simulator)
estimator.options.default_shots = 1000

print("Starting optimization...")
result = minimize(
    cost_func_estimator,
    init_params,
    args=(candidate_circuit, estimator),  # Pass transpiled circuit and estimator
    method="COBYLA",
    tol=1e-2,  # Tolerance for termination
    options={"maxiter": 100},  # Set max iterations to prevent running forever
)
print("Optimization Result:")
print(result)

# You can plot the objective function values if desired
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
