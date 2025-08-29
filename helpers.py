def calculate_ising_energy(binarySolutionString, terms, weights):
    spins = []
    for bit in binarySolutionString:
        if bit == "0":
            spins.append(1)
        else:
            spins.append(-1)
    total_energy = 0
    # Iterate through the Ising model terms and weights
    for term, weight in zip(terms, weights):
        if len(term) == 1:  # Linear term (h_i * s_i)
            i = term[0]
            total_energy += weight * spins[i]
        elif len(term) == 2:  # Quadratic term (J_ij * s_i * s_j)
            i, j = term[0], term[1]
            total_energy += weight * spins[i] * spins[j]

    return total_energy
