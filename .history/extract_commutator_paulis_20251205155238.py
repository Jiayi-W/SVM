import sys
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from quantum_utils import H7, stab_01n_op

def commutator(A, B):
    return 2**(-2)*(A @ B - B @ A).simplify()

def extract_pauli_terms(n, max_order=15):
    H = H7(n).ham
    stab_sum = sum(stab_01n_op(n)).simplify()
    pauli_set = set()
    prev_size = stab_sum.size
    sizes = [prev_size]
    stab_sums = [stab_sum]
    pauli_set.update([p for p, _ in stab_sum.to_list()])
    for order in range(1, max_order+1):
        next_sum = commutator(H, stab_sums[-1]).simplify()
        size = next_sum.size
        sizes.append(size)
        stab_sums.append(next_sum)
        pauli_set.update([p for p, _ in next_sum.to_list()])
        # Stop if size is unchanged for two consecutive orders
        if size == prev_size:
            break
        prev_size = size
    return sorted(pauli_set), sizes

def main():
    pauli_dict = {}
    size_dict = {}
    for n in [6, 7, 8]:
        paulis, sizes = extract_pauli_terms(n)
        pauli_dict[n] = paulis
        size_dict[n] = sizes
        print(f"n={n}: {len(paulis)} unique Pauli strings, commutator sizes: {sizes}")
    np.save('commutator_paulis_dict.npy', pauli_dict)
    np.save('commutator_paulis_sizes.npy', size_dict)
    print("Saved commutator Pauli sets to commutator_paulis_dict.npy and sizes to commutator_paulis_sizes.npy")

if __name__ == '__main__':
    main()
