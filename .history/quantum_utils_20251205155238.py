# =============================================================================
# QUANTUM UTILITIES FOR ENTANGLEMENT WITNESS CONSTRUCTION
# =============================================================================

# Core Python
from functools import reduce
import os
import time

# Scientific Computing
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import scipy.sparse.linalg

# Quantum Computing
from qiskit.quantum_info import Statevector, SparsePauliOp

# Optimization and Visualization
import cvxpy as cp
import matplotlib.pyplot as plt


# =============================================================================
# QUANTUM HAMILTONIAN CONSTRUCTION
# =============================================================================

class Nearest_Neighbour_1d:
    """
    Constructs a 1D nearest-neighbor quantum Hamiltonian with optional periodic boundary conditions.
    
    Parameters:
    -----------
    n : int
        Number of qubits
    Jx, Jy, Jz : float
        Coupling strengths for XX, YY, ZZ interactions
    hx, hy, hz : float
        Local field strengths
    pbc : bool
        Whether to use periodic boundary conditions
    verbose : bool
        Whether to print Hamiltonian information
    rand_field : list
        Random field values for each qubit
    """
    
    def __init__(self, n: int, Jx=0, Jy=0, Jz=0, hx=0, hy=0, hz=0, pbc=False, verbose=False, rand_field=[]):
        self.n = n
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.hx, self.hy, self.hz = hx, hy, hz

        # Build interaction terms
        self.xx_tuples = [('XX', [i, i + 1], Jx) for i in range(0, n-1)]
        self.yy_tuples = [('YY', [i, i + 1], Jy) for i in range(0, n-1)]
        self.zz_tuples = [('ZZ', [i, i + 1], Jz) for i in range(0, n-1)]

        # Handle random fields
        if len(rand_field) == 0:
            self.rand_field = [0]*n
        elif len(rand_field) >= n:
            self.rand_field = rand_field[:n]
        else:
            raise ValueError(f'Length of random field should be at least {n}!')

        # Build local field terms
        self.x_tuples = [('X', [i], (self.rand_field[i]+1)*hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], (self.rand_field[i]+1)*hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], (self.rand_field[i]+1)*hz) for i in range(0, n)] 

        # Add periodic boundary conditions
        if pbc and n > 2: 
            self.xx_tuples.append(('XX', [n-1, 0], Jx))
            self.yy_tuples.append(('YY', [n-1, 0], Jy))
            self.zz_tuples.append(('ZZ', [n-1, 0], Jz))

        # Construct the full Hamiltonian
        all_terms = [*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, 
                    *self.x_tuples, *self.y_tuples, *self.z_tuples]
        self.ham = SparsePauliOp.from_sparse_list(all_terms, num_qubits=n).simplify()
        
        # Group terms for analysis
        self.xyz_group()
        self.par_group()
        
        if verbose: 
            print('The Hamiltonian: \n', self.ham)
            print('The xyz grouping: \n', self.ham_xyz)
            print('The parity grouping: \n', self.ham_par)

    def xyz_group(self):
        """
        Group Hamiltonian terms by Pauli operator types (X, Y, Z).
        
        This creates separate SparsePauliOp objects for each type of Pauli operator,
        making it easier to analyze the different components of the Hamiltonian.
        """
        # Extract X, Y, Z terms from the full Hamiltonian
        all_terms = self.ham.to_list()
        
        # Separate terms by Pauli operator type
        x_terms = []
        y_terms = []
        z_terms = []
        
        for pauli_string, coeff in all_terms:
            if 'X' in pauli_string and 'Y' not in pauli_string and 'Z' not in pauli_string:
                x_terms.append((pauli_string, coeff))
            elif 'Y' in pauli_string and 'X' not in pauli_string and 'Z' not in pauli_string:
                y_terms.append((pauli_string, coeff))
            elif 'Z' in pauli_string and 'X' not in pauli_string and 'Y' not in pauli_string:
                z_terms.append((pauli_string, coeff))
        
        # Create grouped Hamiltonians
        self.ham_x = SparsePauliOp.from_list(x_terms) if x_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        self.ham_y = SparsePauliOp.from_list(y_terms) if y_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        self.ham_z = SparsePauliOp.from_list(z_terms) if z_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        
        # Store the grouped terms
        self.ham_xyz = {
            'X': self.ham_x,
            'Y': self.ham_y, 
            'Z': self.ham_z
        }
    
    def par_group(self):
        """
        Group Hamiltonian terms by parity (even/odd).
        
        This separates terms based on whether they preserve or flip certain symmetries,
        which is useful for analyzing conserved quantities and symmetries.
        """
        # Extract all terms from the full Hamiltonian
        all_terms = self.ham.to_list()
        
        # Separate terms by parity
        even_terms = []
        odd_terms = []
        
        for pauli_string, coeff in all_terms:
            # Count the number of non-identity Pauli operators
            num_paulis = sum(1 for p in pauli_string if p != 'I')
            
            # Even parity: even number of Pauli operators
            # Odd parity: odd number of Pauli operators
            if num_paulis % 2 == 0:
                even_terms.append((pauli_string, coeff))
            else:
                odd_terms.append((pauli_string, coeff))
        
        # Create grouped Hamiltonians
        self.ham_even = SparsePauliOp.from_list(even_terms) if even_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        self.ham_odd = SparsePauliOp.from_list(odd_terms) if odd_terms else SparsePauliOp('I' * self.n, coeffs=[0])
        
        # Store the grouped terms
        self.ham_par = {
            'even': self.ham_even,
            'odd': self.ham_odd
        }

class Power_Law:
    """
    Constructs a 1D power-law quantum Hamiltonian with optional periodic boundary conditions.
    
    Parameters:
    -----------
    n : int
        Number of qubits
    """
    def __init__(self, n: int, alpha: int, Jx=0, Jy=0, Jz=0, hx=0.0, hy=0.0, hz=0, pbc=False, verbose=False):
        self.n, self.alpha = n, alpha
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.xx_tuples = [('XX', [i, j], Jx*abs(i-j)**(-alpha)) for i in range(0, n-1) for j in range(i+1, n)]
        self.yy_tuples = [('YY', [i, j], Jy*abs(i-j)**(-alpha)) for i in range(0, n-1) for j in range(i+1, n)]
        self.zz_tuples = [('ZZ', [i, j], Jz*abs(i-j)**(-alpha)) for i in range(0, n-1) for j in range(i+1, n)]
        self.x_tuples = [('X', [i], hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], hz) for i in range(0, n)] 
        if pbc: 
            # self.xx_tuples.append(('XX', [n-1, 0], Jx))
            # self.yy_tuples.append(('YY', [n-1, 0], Jy))
            # self.zz_tuples.append(('ZZ', [n-1, 0], Jz))
            raise ValueError(f'PBC is not defined!')

        self.ham = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, *self.x_tuples, *self.y_tuples, *self.z_tuples], num_qubits=n).simplify()
        if verbose: print('The Hamiltonian: \n', self.ham)

class Homogeneous_random_Heisenberg:
    def __init__(self, n: int, Jx=0, Jy=0, Jz=0, hx=0.0, hy=0.0, hz=0, pbc=False, verbose=False):
        random_list = np.random.uniform(-1, 1, int(n * (n - 1) / 2)).tolist()
        reshape_coeffs  = []
        for i in range(0, n-1):
            temp = []
            for j in range(i+1, n):
                temp.append(random_list.pop(0))
            reshape_coeffs.append(temp)

        self.n = n
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.xx_tuples = [('XX', [i, j], reshape_coeffs[i][j-i-1] ) for i in range(0, n-1) for j in range(i+1, n)]
        self.yy_tuples = [('YY', [i, j], reshape_coeffs[i][j-i-1]) for i in range(0, n-1) for j in range(i+1, n)]
        self.zz_tuples = [('ZZ', [i, j], reshape_coeffs[i][j-i-1]) for i in range(0, n-1) for j in range(i+1, n)]
        self.x_tuples = [('X', [i], hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], hz) for i in range(0, n)] 
        if pbc: 
            # self.xx_tuples.append(('XX', [n-1, 0], Jx))
            # self.yy_tuples.append(('YY', [n-1, 0], Jy))
            # self.zz_tuples.append(('ZZ', [n-1, 0], Jz))
            raise ValueError(f'PBC is not defined!')

        self.ham = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, *self.x_tuples, *self.y_tuples, *self.z_tuples], num_qubits=n).simplify()
        if verbose: print('The Hamiltonian: \n', self.ham)


# =============================================================================
# MATRIX EXPONENTIAL UTILITIES
# =============================================================================

def expH(H, t):
    """
    Compute matrix exponential exp(-i*t*H) for Hamiltonian evolution.
    
    Parameters:
    -----------
    H : array_like or SparsePauliOp
        Hamiltonian matrix
    t : float
        Time parameter
        
    Returns:
    --------
    array_like
        Matrix exponential result
    """
    if isinstance(H, csr_matrix):
        return scipy.sparse.linalg.expm(-1j * t * H)
    else:
        return scipy.linalg.expm(-1j * t * H)


# =============================================================================
# QUANTUM STATE AND STABILIZER UTILITIES
# =============================================================================

def stab_01n_op(n, mat=False):
    """
    Generate stabilizer operators for 01^n state.
    
    Parameters:
    -----------
    n : int
        Number of qubits
    mat : bool
        Whether to return matrices instead of SparsePauliOp objects
        
    Returns:
    --------
    list
        List of stabilizer operators
    """
    a = [1, -1] * (n // 2) + [1] * (n % 2)
    coe = [i * ((-1) ** (n + 1)) for i in a]
    stab_list = [
        SparsePauliOp.from_sparse_list([("Z", [i], coe[i])], n) for i in range(0, n)
    ]
    return [i.to_matrix() for i in stab_list] if mat else stab_list

def ini_state_01(n):
    """Create initial state |01...01⟩ for n qubits."""
    return Statevector.from_label("01" * (n // 2) + "0" * (n % 2))

def H7(n):
    """Create H7 Hamiltonian with specific parameters."""
    return Nearest_Neighbour_1d(n, Jx=1, Jy=1, Jz=0, hz=0.2, pbc=True)

def H5(n):
    """Create H5 Power Law Nearest Neighbour 1d Hamiltonian"""
    return Power_Law(n, alpha=2, Jx=1, Jy=1, Jz=0, hx=0, hy=0, hz=0.2, pbc=False)

def H6(n):
    """Create H6 Homogeneous random Heisenberg Hamiltonian"""
    return Homogeneous_random_Heisenberg(n, Jx=1, Jy=1, Jz=1, hx=0, hy=0, hz=0, pbc=False)

def poly_log_t(n):
    """Polylogarithmic time scaling function."""
    return 1/(2**(np.log(n)**2))


# =============================================================================
# QUANTUM EVOLUTION UTILITIES
# =============================================================================

def U_step_n(n, H, t, steps, mat=True, n_step=True):
    """
    Compute Trotterized evolution operator using second-order Trotter formula.
    
    This function works with any Hamiltonian that supports:
    - H.ham (SparsePauliOp): The Hamiltonian operator
    - Matrix multiplication (@) and simplification (.simplify())
    
    The Trotter formula approximates: U(t) ≈ [I - i*dt*H - (dt²/2)*H²]^steps
    
    Parameters:
    -----------
    n : int
        Number of qubits
    H : object
        Hamiltonian object with .ham attribute (SparsePauliOp)
        Can be Nearest_Neighbour_1d or any compatible Hamiltonian class
    t : float
        Total evolution time
    steps : int
        Number of Trotter steps (higher = more accurate, slower)
    mat : bool
        Whether to return matrix form (True) or SparsePauliOp (False)
    n_step : bool
        Whether to compute U_step^steps (True) or just U_step (False)
        
    Returns:
    --------
    array_like or SparsePauliOp
        Evolution operator U(t) ≈ exp(-i*t*H)
    """
    dt_var = t / steps
    # Create identity operator for n qubits using proper Qiskit method
    # Use explicit string construction to avoid Qiskit interception
    identity_pauli_string = ''.join(['I' for _ in range(n)])
    iden = SparsePauliOp([identity_pauli_string], coeffs=[1. + 0.j])
    
    # Second-order Trotter formula
    U_step = (
        iden
        - (1.0j * dt_var * H.ham)
        - (0.5 * (dt_var**2) * ((H.ham)@(H.ham)).simplify())
    ).simplify()

    U_step_n = U_step if not n_step else reduce(
        lambda acc, _: (acc @ U_step).simplify(), 
        range(1, steps), 
        U_step
    )

    return U_step_n.to_matrix() if mat else U_step_n

def state_evolved(n, H, t, steps, ini_state, ideal_state=False, den_mat_out=True):
    """
    Evolve quantum state under Hamiltonian.
    
    This function works with any Hamiltonian that supports:
    - H.ham (SparsePauliOp): The Hamiltonian operator
    - Matrix operations for evolution
    
    Parameters:
    -----------
    n : int
        Number of qubits
    H : object
        Hamiltonian object with .ham attribute (SparsePauliOp)
        Can be Nearest_Neighbour_1d or any compatible Hamiltonian class
    t : float
        Evolution time
    steps : int
        Number of Trotter steps (only used if ideal_state=False)
    ini_state : Statevector
        Initial quantum state
    ideal_state : bool
        Whether to use ideal evolution (exact) or Trotterized
        - True: Uses exact matrix exponential exp(-i*t*H)
        - False: Uses Trotterized approximation
    den_mat_out : bool
        Whether to return density matrix (True) or statevector (False)
        
    Returns:
    --------
    array_like or Statevector
        Evolved quantum state
    """
    if ideal_state:
        U = expH(H.ham, t)
    else:
        U = U_step_n(n, H, t, steps, mat=True)
    
    state_evolve = ini_state.evolve(U)
    
    if den_mat_out:
        return Statevector.to_operator(state_evolve).to_matrix()
    else:
        return state_evolve

def truncate_stab(stab, up=0.1):
    """
    Truncate stabilizer by removing small coefficients.
    
    This function removes terms with coefficients smaller than a threshold,
    which is useful for reducing computational complexity while maintaining
    reasonable accuracy in quantum simulations.
    
    Parameters:
    -----------
    stab : SparsePauliOp
        The stabilizer operator to truncate
    up : float, optional (default=0.1)
        Threshold for coefficient truncation as a fraction of the maximum coefficient.
        Terms with |coeff| > up * |max_coeff| are kept.
        
    Returns:
    --------
    SparsePauliOp
        Truncated stabilizer operator with small coefficients removed
    """
    terms = stab.to_list()
    max_coeff = max(np.abs(c) for _, c in terms)
    filtered = [(l, c) for l, c in terms if np.abs(c) > up * max_coeff]
    return SparsePauliOp.from_list(filtered)

def stab_sum_after(n, H, stab_sum, t, steps, approx=False, up=0.1, step_by_step=False):
    """Evolve stabilizer sum under Hamiltonian."""
    U = U_step_n(n, H, t, steps, mat=False, n_step=not step_by_step)
    stab_sum_after = reduce(
        lambda acc, _: ((U @ acc).simplify() @ U.conjugate()).simplify(), 
        range(steps) if step_by_step else [0], 
        stab_sum
    )
    
    if approx:
        stab_sum_after = truncate_stab(stab_sum_after, up)
    
    return stab_sum_after


# =============================================================================
# OPTIMIZATION AND VISUALIZATION UTILITIES
# =============================================================================

def solve_sdp_W(n, W, solver='adaptive', num_samples=10000, max_time=300, eps=1e-6):
    """
    Solve SDP to find maximum trace of W with respect to separable states.
    
    Parameters:
    -----------
    n : int
        Number of qubits
    W : array_like
        Witness operator
    solver : str
        Solver to use: 'default', 'scs', 'mosek', 'approx', 'adaptive'
    num_samples : int
        Number of samples for approximate method
    max_time : float
        Maximum time in seconds for exact solver before switching to approximate
    eps : float
        Tolerance parameter for MOSEK and SCS solvers (default: 1e-6)
        
    Returns:
    --------
    float
        Maximum trace value
    """
    d = 2
    total_dim = d ** n
    a_dim = d ** (n//2)
    b_dim = d ** (n - n//2)
    
    # Adaptive solver selection based on problem size and time constraints
    if solver == 'adaptive':
        if n >= 12:
            solver = 'approx'
            print(f"Using approximate method for n={n} (too large for exact SDP)")
        elif n >= 10:
            # Try exact first, fallback to approximate if too slow
            solver = 'exact_with_timeout'
        else:
            solver = 'default'
    
    if solver == 'exact_with_timeout':
        return solve_sdp_W_with_timeout(n, W, max_time, num_samples)
    elif solver == 'approx':
        return solve_sdp_W_approx(n, W, num_samples)
    
    rho = cp.Variable((total_dim, total_dim), hermitian=True)
    constraints = [rho >> 0]
    constraints += [cp.trace(rho) == 1]
    constraints += [cp.partial_transpose(rho, (a_dim, b_dim), 0) >> 0]
    
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(W @ rho))), constraints)
    
    if solver == 'scs':
        prob.solve(solver=cp.SCS, verbose=True, max_iters=10000, eps=eps)
    elif solver == 'mosek':
        try:
            prob.solve(solver=cp.MOSEK, verbose=True, eps=eps)
        except:
            print("MOSEK not available, using default solver")
            prob.solve()
    else:
        prob.solve()
    
    return prob.value

def solve_sdp_W_with_timeout(n, W, max_time, num_samples):
    """
    Try exact SDP first, fallback to approximate if too slow.
    Critical for entanglement detection accuracy.
    """
    print(f"Trying exact SDP for n={n} (max {max_time}s)...")
    
    start_time = time.time()
    
    try:
        # Try exact SDP with timeout
        result = solve_sdp_W(n, W, solver='default')
        elapsed_time = time.time() - start_time
        
        if elapsed_time <= max_time:
            print(f"✓ Exact SDP completed in {elapsed_time:.2f}s: {result:.6f}")
            return result
        else:
            print(f"⚠ Exact SDP too slow ({elapsed_time:.2f}s > {max_time}s)")
            print("Switching to approximate method...")
            
    except Exception as e:
        print(f"⚠ Exact SDP failed: {e}")
        print("Switching to approximate method...")
    
    # Fallback to approximate with higher sample count for better accuracy
    enhanced_samples = max(num_samples * 2, 50000)  # Use more samples for better accuracy
    print(f"Using enhanced approximate method with {enhanced_samples} samples...")
    
    return solve_sdp_W_approx(n, W, enhanced_samples)

def solve_sdp_W_approx(n, W, num_samples=10000):
    """
    Approximate SDP solution using random sampling of separable states.
    Much faster for large n, but gives approximate result.
    """
    d = 2
    total_dim = d ** n
    a_dim = d ** (n//2)
    b_dim = d ** (n - n//2)
    
    max_trace = -np.inf
    
    print(f"Running approximate SDP with {num_samples} samples...")
    
    for i in range(num_samples):
        if i % (num_samples // 10) == 0:
            print(f"Progress: {i/num_samples*100:.1f}%")
            
        # Generate random separable state
        rho_a = np.random.randn(a_dim, a_dim) + 1j * np.random.randn(a_dim, a_dim)
        rho_a = rho_a @ rho_a.conj().T
        rho_a = rho_a / np.trace(rho_a)
        
        rho_b = np.random.randn(b_dim, b_dim) + 1j * np.random.randn(b_dim, b_dim)
        rho_b = rho_b @ rho_b.conj().T
        rho_b = rho_b / np.trace(rho_b)
        
        rho_sep = np.kron(rho_a, rho_b)
        trace_val = np.real(np.trace(W @ rho_sep))
        max_trace = max(max_trace, trace_val)
    
    print(f"Approximate maximum trace: {max_trace:.6f}")
    return max_trace

def solve_sdp_W_approx_pure(n, W, num_samples=10000):
    """Generate random pure separable states"""
    d = 2
    a_dim = d ** (n//2)
    b_dim = d ** (n - n//2)
    
    max_trace = -np.inf
    
    for i in range(num_samples):
        # Generate random pure states (normalized vectors)
        psi_a = np.random.randn(a_dim) + 1j * np.random.randn(a_dim)
        psi_a = psi_a / np.linalg.norm(psi_a)
        
        psi_b = np.random.randn(b_dim) + 1j * np.random.randn(b_dim)
        psi_b = psi_b / np.linalg.norm(psi_b)
        
        # Create pure separable state |ψ_a⟩⊗|ψ_b⟩
        psi_sep = np.kron(psi_a, psi_b)
        rho_sep = np.outer(psi_sep, psi_sep.conj())  # |ψ⟩⟨ψ|
        
        trace_val = np.real(np.trace(W @ rho_sep))
        max_trace = max(max_trace, trace_val)
    
    print(f"Approximate maximum trace: {max_trace:.6f}")
    return max_trace

def sample_random_pure_separable_state(n, seed=None, return_statevec=False):
    """
    Sample a random pure separable n-qubit state by splitting the system
    into left and right parts at the middle: left has `n//2` qubits and
    right has `n - n//2` qubits. Each side is sampled from the Haar
    measure by drawing a complex normal vector and normalizing it.

    Parameters:
    -----------
    n : int
        Total number of qubits.
    seed : int or None
        Optional random seed for reproducibility.
    return_statevec : bool
        If True, also return the separable statevector `|psi_left> ⊗ |psi_right>`.

    Returns:
    --------
    rho_sep : ndarray
        Density matrix of the separable pure state (shape `(2**n, 2**n)`).
    rho_left : ndarray
        Density matrix of the left subsystem (shape `(2**(n//2), 2**(n//2))`).
    rho_right : ndarray
        Density matrix of the right subsystem (shape `(2**(n-n//2), 2**(n-n//2))`).
    psi_sep : ndarray, optional
        Statevector of the separable pure state (returned when `return_statevec=True`).

    Example:
    --------
    >>> rho_sep, rho_L, rho_R = sample_random_pure_separable_state(4, seed=42)
    >>> rho_sep.shape
    (16, 16)
    """
    if seed is not None:
        np.random.seed(seed)

    n_left = n // 2
    n_right = n - n_left

    dim_left = 2 ** n_left
    dim_right = 2 ** n_right

    # Sample Haar-random pure states by normalizing complex Gaussian vectors
    psi_left = np.random.randn(dim_left) + 1j * np.random.randn(dim_left)
    psi_left = psi_left / np.linalg.norm(psi_left)

    psi_right = np.random.randn(dim_right) + 1j * np.random.randn(dim_right)
    psi_right = psi_right / np.linalg.norm(psi_right)

    # Density matrices for each subsystem
    rho_left = np.outer(psi_left, psi_left.conj())
    rho_right = np.outer(psi_right, psi_right.conj())

    # Separable pure state density matrix via tensor product
    rho_sep = np.kron(rho_left, rho_right)

    if return_statevec:
        psi_sep = np.kron(psi_left, psi_right)
        return rho_sep, rho_left, rho_right, psi_sep

    return rho_sep

def train_linear_svm_pauli(
    n,
    rho_approx,
    num_samples=1000,
    class_weights=(1.0, 1.0),
    seed=None,
    C=1.0,
    return_model=True,
    # margins: distances will be margin/||w|| for each class (neg, pos)
    margin_neg: float = 1.0,
    margin_pos: float = 1.0,
):
    """
    Train a linear SVM using Pauli-term expectation features.

    Procedure (implements user's spec):
    1) Generate `num_samples` random separable pure states using
       `sample_random_pure_separable_state(n, seed=...)` and label them -1.
    2) Label the provided approximated evolved state `rho_approx` as +1.
    3) Extract Pauli strings from `rho_approx` (expects a `SparsePauliOp`)
       and form features for each candidate state as
           feature_j = tr(P_j @ rho_candidate)
       where P_j is the Pauli operator matrix for string j.
    4) Train a linear SVM separating the two classes with class weights
       mapped from `class_weights` tuple (b, c) -> {-1: b, +1: c}.

    Parameters:
    -----------
    n : int
        Number of qubits.
    rho_approx : SparsePauliOp or ndarray
        Approximated evolved state (preferably a `SparsePauliOp` object).
    num_samples : int
        Number of random separable samples to generate (class -1).
    class_weights : tuple (b, c)
        Class weights for labels -1 and +1 respectively.
    seed : int or None
        RNG seed for reproducibility.
    C : float
        Regularization parameter for SVM (only used in cvxpy fallback).
    return_model : bool
        If True return the trained sklearn model or cvxpy solution dict.

    Returns:
    --------
    result : dict
        A dictionary with keys:
        - 'model': trained sklearn `SVC` object or cvxpy solution dict (if sklearn not available)
        - 'pauli_strings': list of Pauli strings used as features
        - 'X': feature matrix (shape: num_samples+1, n_features)
        - 'y': labels vector

    Notes:
    ------
    - Falls back to a `cvxpy` formulation of soft-margin SVM with class-weighted
      slack penalties when `scikit-learn` is not available.
    - Features are raw expectation values (no scaling). You can standardize
      `X` before training if desired.
    """
    if seed is not None:
        np.random.seed(seed)

    # If rho_approx is a SparsePauliOp-like object, extract term list
    try:
        pauli_list = rho_approx.to_list()
    except Exception:
        raise TypeError('`rho_approx` must be a `SparsePauliOp`-like object with `to_list()`')

    # Extract unique pauli strings (order preserved)
    pauli_strings = [p for p, _ in pauli_list]
    n_features = len(pauli_strings)

    # Precompute Pauli operator matrices for each string
    P_mats = [SparsePauliOp.from_list([(p, 1)], num_qubits=n).to_matrix() for p in pauli_strings]

    # Generate separable samples and compute features
    X = np.zeros((num_samples + 1, n_features), dtype=float)
    y = np.zeros((num_samples + 1,), dtype=int)

    for i in range(num_samples):
        rho_sep = sample_random_pure_separable_state(n, seed=None)
        feats = [np.real(np.trace(P @ rho_sep)) for P in P_mats]
        X[i, :] = feats
        y[i] = -1

    # Compute features for the approximated evolved state (positive class)
    # Convert rho_approx (SparsePauliOp) to matrix
    try:
        rho_approx_mat = rho_approx.to_matrix()
    except Exception:
        # If it's already a matrix, accept it
        rho_approx_mat = np.asarray(rho_approx)

    feats_pos = [np.real(np.trace(P @ rho_approx_mat)) for P in P_mats]
    X[-1, :] = feats_pos
    y[-1] = 1

    # Class weight mapping from tuple (b, c) -> {-1: b, +1: c}
    b_weight, c_weight = class_weights
    class_weight_map = {-1: b_weight, 1: c_weight}

    # If symmetric margins requested (default 1.0,1.0) try sklearn first for speed.
    use_cvxpy_force = not (float(margin_neg) == 1.0 and float(margin_pos) == 1.0)

    if not use_cvxpy_force:
        try:
            from sklearn.svm import SVC

            clf = SVC(kernel='linear', class_weight=class_weight_map)
            clf.fit(X, y)
            result = {
                'model': clf,
                'pauli_strings': pauli_strings,
                'X': X,
                'y': y,
            }
            return result if return_model else (pauli_strings, X, y)

        except Exception:
            # fall through to cvxpy
            pass

    # Use cvxpy primal solver. This code also supports asymmetric margins
    # by using separate margin offsets `margin_pos` and `margin_neg`.
    try:
        import cvxpy as _cp
    except Exception:
        raise ImportError('scikit-learn not available and cvxpy not available for fallback. Please install one of them.')

    m, d = X.shape
    w = _cp.Variable(d)
    b_var = _cp.Variable(1)
    xi = _cp.Variable(m, nonneg=True)

    sample_weights = np.array([class_weight_map[int(lbl)] for lbl in y], dtype=float)

    constraints = []
    # Build constraints with separate margins:
    # For y==+1: X_i @ w + b >= margin_pos - xi_i
    # For y==-1: - (X_i @ w + b) >= margin_neg - xi_i   (i.e. X_i @ w + b <= -margin_neg + xi_i)
    pos_idx = [i for i, lbl in enumerate(y) if lbl == 1]
    neg_idx = [i for i, lbl in enumerate(y) if lbl == -1]

    if len(pos_idx) > 0:
        constraints.append(X[pos_idx, :] @ w + b_var >= float(margin_pos) - xi[pos_idx])
    if len(neg_idx) > 0:
        constraints.append(- (X[neg_idx, :] @ w + b_var) >= float(margin_neg) - xi[neg_idx])

    objective = 0.5 * _cp.sum_squares(w) + C * sample_weights @ xi
    prob = _cp.Problem(_cp.Minimize(objective), constraints)

    # Try solving. If solver fails, let cvxpy pick alternatives; caller can enable verbose if needed.
    prob.solve()

    w_val = np.array(w.value).reshape(-1)
    b_val = float(b_var.value)

    solution = {'w': w_val, 'b': b_val, 'status': prob.status, 'margin_neg': float(margin_neg), 'margin_pos': float(margin_pos)}
    result = {
        'model': solution,
        'pauli_strings': pauli_strings,
        'X': X,
        'y': y,
    }
    return result if return_model else (pauli_strings, X, y)


def train_rbf_svm_pauli(
    n,
    rho_approx,
    num_samples=1000,
    class_weights=(1.0, 1.0),
    seed=None,
    C=1.0,
    gamma='scale',
    return_model=True,
):
    """
    Train an RBF (Gaussian) kernel SVM using Pauli-term expectation features.

    This function generates separable samples and trains an SVM with RBF kernel,
    which can capture non-linear decision boundaries in the Pauli feature space.

    Parameters:
    -----------
    n : int
        Number of qubits.
    rho_approx : SparsePauliOp or ndarray
        Approximated evolved state (preferably a `SparsePauliOp` object).
    num_samples : int
        Number of random separable samples to generate (class -1).
    class_weights : tuple (b, c)
        Class weights for labels -1 and +1 respectively.
    seed : int or None
        RNG seed for reproducibility.
    C : float
        Regularization parameter for SVM.
    gamma : float or str
        Kernel coefficient for RBF kernel. Default 'scale' = 1/(n_features * X.var()).
    return_model : bool
        If True return the trained sklearn model or dict with results.

    Returns:
    --------
    result : dict
        A dictionary with keys:
        - 'model': trained sklearn `SVC` object with RBF kernel
        - 'pauli_strings': list of Pauli strings used as features
        - 'X': feature matrix (shape: num_samples+1, n_features)
        - 'y': labels vector
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract Pauli strings from rho_approx
    try:
        pauli_list = rho_approx.to_list()
    except Exception:
        raise TypeError('`rho_approx` must be a `SparsePauliOp`-like object with `to_list()`')

    pauli_strings = [p for p, _ in pauli_list]
    n_features = len(pauli_strings)

    # Precompute Pauli operator matrices for each string
    P_mats = [SparsePauliOp.from_list([(p, 1)], num_qubits=n).to_matrix() for p in pauli_strings]

    # Generate separable samples and compute features
    X = np.zeros((num_samples + 1, n_features), dtype=float)
    y = np.zeros((num_samples + 1,), dtype=int)

    for i in range(num_samples):
        rho_sep = sample_random_pure_separable_state(n, seed=None)
        feats = [np.real(np.trace(P @ rho_sep)) for P in P_mats]
        X[i, :] = feats
        y[i] = -1

    # Compute features for the approximated evolved state (positive class)
    try:
        rho_approx_mat = rho_approx.to_matrix()
    except Exception:
        rho_approx_mat = np.asarray(rho_approx)

    feats_pos = [np.real(np.trace(P @ rho_approx_mat)) for P in P_mats]
    X[-1, :] = feats_pos
    y[-1] = 1

    # Class weight mapping
    b_weight, c_weight = class_weights
    class_weight_map = {-1: b_weight, 1: c_weight}

    # Train RBF SVM using sklearn
    try:
        from sklearn.svm import SVC

        clf = SVC(kernel='rbf', C=C, gamma=gamma, class_weight=class_weight_map)
        clf.fit(X, y)
        result = {
            'model': clf,
            'pauli_strings': pauli_strings,
            'X': X,
            'y': y,
            'kernel': 'rbf',
            'gamma': gamma,
        }
        return result if return_model else (pauli_strings, X, y)

    except ImportError:
        raise ImportError('scikit-learn is required for RBF SVM. Please install it: pip install scikit-learn')


def train_poly_svm_pauli(
    n,
    rho_approx,
    num_samples=1000,
    class_weights=(1.0, 1.0),
    seed=None,
    C=1.0,
    degree=3,
    return_model=True,
):
    """
    Train a polynomial kernel SVM using Pauli-term expectation features.

    This function generates separable samples and trains an SVM with polynomial kernel,
    allowing for polynomial decision boundaries in the Pauli feature space.

    Parameters:
    -----------
    n : int
        Number of qubits.
    rho_approx : SparsePauliOp or ndarray
        Approximated evolved state (preferably a `SparsePauliOp` object).
    num_samples : int
        Number of random separable samples to generate (class -1).
    class_weights : tuple (b, c)
        Class weights for labels -1 and +1 respectively.
    seed : int or None
        RNG seed for reproducibility.
    C : float
        Regularization parameter for SVM.
    degree : int
        Degree of the polynomial kernel. Default 3.
    return_model : bool
        If True return the trained sklearn model or dict with results.

    Returns:
    --------
    result : dict
        A dictionary with keys:
        - 'model': trained sklearn `SVC` object with polynomial kernel
        - 'pauli_strings': list of Pauli strings used as features
        - 'X': feature matrix (shape: num_samples+1, n_features)
        - 'y': labels vector
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract Pauli strings from rho_approx
    try:
        pauli_list = rho_approx.to_list()
    except Exception:
        raise TypeError('`rho_approx` must be a `SparsePauliOp`-like object with `to_list()`')

    pauli_strings = [p for p, _ in pauli_list]
    n_features = len(pauli_strings)

    # Precompute Pauli operator matrices for each string
    P_mats = [SparsePauliOp.from_list([(p, 1)], num_qubits=n).to_matrix() for p in pauli_strings]

    # Generate separable samples and compute features
    X = np.zeros((num_samples + 1, n_features), dtype=float)
    y = np.zeros((num_samples + 1,), dtype=int)

    for i in range(num_samples):
        rho_sep = sample_random_pure_separable_state(n, seed=None)
        feats = [np.real(np.trace(P @ rho_sep)) for P in P_mats]
        X[i, :] = feats
        y[i] = -1

    # Compute features for the approximated evolved state (positive class)
    try:
        rho_approx_mat = rho_approx.to_matrix()
    except Exception:
        rho_approx_mat = np.asarray(rho_approx)

    feats_pos = [np.real(np.trace(P @ rho_approx_mat)) for P in P_mats]
    X[-1, :] = feats_pos
    y[-1] = 1

    # Class weight mapping
    b_weight, c_weight = class_weights
    class_weight_map = {-1: b_weight, 1: c_weight}

    # Train polynomial SVM using sklearn
    try:
        from sklearn.svm import SVC

        clf = SVC(kernel='poly', C=C, degree=degree, class_weight=class_weight_map)
        clf.fit(X, y)
        result = {
            'model': clf,
            'pauli_strings': pauli_strings,
            'X': X,
            'y': y,
            'kernel': 'poly',
            'degree': degree,
        }
        return result if return_model else (pauli_strings, X, y)

    except ImportError:
        raise ImportError('scikit-learn is required for polynomial SVM. Please install it: pip install scikit-learn')


def classify_state_with_svm(state, trained_result, n=None, return_score=False):
    """Classify a quantum state using a trained linear SVM result from
    `train_linear_svm_pauli`.

    Parameters:
    -----------
    state : Statevector, SparsePauliOp, or ndarray
        The quantum state to classify. Accepted formats:
        - density matrix (ndarray shape (2**n,2**n)),
        - statevector (1D ndarray of length 2**n) or `Statevector` object,
        - `SparsePauliOp` (will be converted to matrix).
    trained_result : dict
        The dictionary returned by `train_linear_svm_pauli` (must contain
        keys `'model'` and `'pauli_strings'`).
    n : int or None
        Number of qubits. If ``None``, it will be inferred from the state
        matrix/vector size.
    return_score : bool
        If True, return a tuple ``(label, score)``. Otherwise return label.

    Returns:
    --------
    label : str
        `'entangled'` for +1, `'separable'` for -1.
    score : float (optional)
        Decision function value (positive -> +1 class).
    """
    # Validate trained_result
    if not isinstance(trained_result, dict) or 'model' not in trained_result or 'pauli_strings' not in trained_result:
        raise TypeError("`trained_result` must be the dict returned by `train_linear_svm_pauli` containing 'model' and 'pauli_strings'.")

    pauli_strings = trained_result['pauli_strings']
    model = trained_result['model']

    # Helper: classify a single state-like object and return (label, score)
    def _classify_one(single_state, n_local=None):
        # Accept tuples/lists produced by sample_random_pure_separable_state
        # (e.g., (rho_sep, rho_L, rho_R)) by choosing the first suitable
        # element when necessary.
        candidate = single_state
        if isinstance(candidate, (list, tuple)):
            # pick first element that looks like a state (ndarray or qiskit object)
            found = False
            for el in candidate:
                if el is None:
                    continue
                # quick check: numpy array or qiskit types
                if isinstance(el, np.ndarray):
                    candidate = el
                    found = True
                    break
                try:
                    from qiskit.quantum_info import Statevector as _SVc, SparsePauliOp as _SPOc
                    if isinstance(el, (_SVc, _SPOc)):
                        candidate = el
                        found = True
                        break
                except Exception:
                    # qiskit not available or types not matched
                    pass
            if not found:
                # fallback: take first element
                candidate = candidate[0]

        # Convert candidate to density matrix
        rho_local = None
        try:
            from qiskit.quantum_info import Statevector as _SV
        except Exception:
            _SV = None
        try:
            from qiskit.quantum_info import SparsePauliOp as _SPO
        except Exception:
            _SPO = None

        if _SV is not None and isinstance(candidate, _SV):
            rho_local = _SV.to_operator(candidate).to_matrix()
        elif _SPO is not None and isinstance(candidate, _SPO):
            # treat SparsePauliOp as an operator representing the state
            rho_local = candidate.to_matrix()
        else:
            arr = np.asarray(candidate)
            if arr.ndim == 1:
                psi = arr
                rho_local = np.outer(psi, psi.conj())
            elif arr.ndim == 2:
                rho_local = arr
            else:
                raise TypeError('Each state must be a statevector (1D) or density matrix (2D), or a qiskit Statevector/SparsePauliOp.')

        dim_local = rho_local.shape[0]
        if n_local is None:
            n_infer = int(round(np.log2(dim_local)))
        else:
            n_infer = n_local
        if 2 ** n_infer != dim_local:
            raise ValueError(f'Inferred dimension {dim_local} is not compatible with 2**n for n={n_infer}.')

        # Build Pauli matrices for this n once (cached by closure? small cost)
        P_mats_local = [SparsePauliOp.from_list([(p, 1)], num_qubits=n_infer).to_matrix() for p in pauli_strings]

        x_local = np.array([float(np.real(np.trace(P @ rho_local))) for P in P_mats_local], dtype=float)

        # compute score & label
        if hasattr(model, 'decision_function') and hasattr(model, 'predict'):
            X_feat = x_local.reshape(1, -1)
            try:
                score_local = float(model.decision_function(X_feat)[0])
            except Exception:
                score_local = float(model.decision_function(X_feat))
            pred_local = int(model.predict(X_feat)[0])
            label_local = 'entangled' if pred_local == 1 else 'separable'
            return label_local, score_local
        else:
            if isinstance(model, dict) and 'w' in model and 'b' in model:
                w = np.asarray(model['w']).reshape(-1)
                b = float(model['b'])
                score_local = float(np.dot(w, x_local) + b)
                pred_local = 1 if score_local > 0 else -1
                label_local = 'entangled' if pred_local == 1 else 'separable'
                return label_local, score_local
            else:
                raise TypeError('Unsupported model type. Expected sklearn SVC-like object or cvxpy solution dict with keys "w","b".')

    # Handle multiple input states: list/tuple or 3D ndarray
    multiple = False
    states_iterable = None
    if isinstance(state, (list, tuple)):
        multiple = True
        states_iterable = state
    elif isinstance(state, np.ndarray) and state.ndim == 3:
        multiple = True
        states_iterable = [state[i] for i in range(state.shape[0])]

    if multiple:
        results = []
        for s in states_iterable:
            lbl, sc = _classify_one(s, n_local=n)
            if return_score:
                results.append((lbl, sc))
            else:
                results.append(lbl)
        return results

    # Single-state case
    lbl, sc = _classify_one(state, n_local=n)
    return (lbl, sc) if return_score else lbl

def plot_x_y(x, y, xlabel, ylabel, title, x_log=False, y_log=False, save_pdf=False, save_path=None):
    """Create and optionally save plots."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    
    plt.legend()
    plt.grid(True)
    
    if save_pdf:
        if save_path is None:
            save_path = f"{os.getcwd()}/{title}.pdf"
        else:
            save_path = f"{os.getcwd()}/{save_path}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        # Do not show when saving; close the figure instead
        plt.close()
    else:
        # Only show interactively when not saving
        plt.show()


# =============================================================================
# MAIN ENTANGLEMENT WITNESS CLASS
# =============================================================================

class H_stab_state_evol:
    """
    Main class for entanglement witness construction and testing.
    
    This class handles the evolution of quantum states under Hamiltonians,
    stabilizer operations, and entanglement witness construction.
    """
    
    def __init__(self, n, H, stab_list, ini_state, t, steps):
        self.n = n
        self.H = H(n)
        self.stab_list = stab_list(n)
        self.ini_state = ini_state(n)
        self.state_ideal = ini_state(n)
        self.t = t(n)
        self.steps = steps
        self.stabsum_tog = sum(self.stab_list).simplify()
        self.Wppt_tog = None
        self.Wppt_alpha_tog = None
        self.svm_linear_model = None

    def state_evol(self, ideal_state=False, den_mat_out=True, state_update=False):
        """
        Evolve the quantum state under the Hamiltonian.
        
        Parameters:
        -----------
        ideal_state : bool
            Whether to use ideal evolution (exact) or Trotterized
        den_mat_out : bool
            Whether to return density matrix or statevector
        state_update : bool
            Whether to update the stored initial state
            
        Returns:
        --------
        array_like or Statevector
            Evolved quantum state
        """
        s_evol = state_evolved(self.n, self.H, self.t, self.steps, 
                              self.ini_state, ideal_state, den_mat_out)
        if state_update:
            self.ini_state = s_evol
        return s_evol
    
    def stablt_sum_evol(self, approx=False, up=0.1, step_by_step=False, stab_sum_update=False):
        """
        Evolve the stabilizer sum under the Hamiltonian.
        
        Parameters:
        -----------
        approx : bool
            Whether to truncate small coefficients
        up : float
            Threshold for coefficient truncation
        step_by_step : bool
            Whether to evolve step by step
        stab_sum_update : bool
            Whether to update the stored stabilizer sum
            
        Returns:
        --------
        SparsePauliOp
            Evolved stabilizer sum
        """
        stab_sum = stab_sum_after(self.n, self.H, self.stabsum_tog, self.t, 
                                 self.steps, approx, up, step_by_step)
        if stab_sum_update:
            self.stabsum_tog = stab_sum
        return stab_sum

    def stablt_sum_truncate_by_commutator(self, commutator_npy='commutator_paulis_dict.npy', stab_sum_update=False):
        """
        Truncate the stored stabilizer sum by keeping only Pauli terms present
        in the precomputed commutator Pauli dictionary (for H7-derived sums).

        This is intended for use when the Hamiltonian is `Nearest_Neighbour_1d`
        (the H7 constructor). The function loads a numpy-saved dictionary
        mapping `n` -> list of Pauli strings and removes any Pauli terms from
        `self.stabsum_tog` whose Pauli string is not in that allowed set.

        Parameters:
        - commutator_npy: path to the `.npy` file produced by
          `extract_commutator_paulis.py` (default: 'commutator_paulis_dict.npy').
        - stab_sum_update: if True, update `self.stabsum_tog` with the
          truncated stabilizer sum.

        Returns:
        - truncated SparsePauliOp
        """
        import numpy as _np

        # Only apply this selective truncation for the Nearest_Neighbour_1d Hamiltonian
        if self.H.__class__.__name__ != 'Nearest_Neighbour_1d':
            # fallback: return current stabilizer sum unchanged
            return self.stabsum_tog

        # Load commutator pauli dictionary
        try:
            data = _np.load(commutator_npy, allow_pickle=True)
            if isinstance(data, _np.ndarray) and data.shape == ():
                pauli_dict = data.item()
            else:
                pauli_dict = data.tolist() if hasattr(data, 'tolist') else dict(data)
        except Exception:
            raise FileNotFoundError(f"Could not load commutator paulis from {commutator_npy}")

        # Accept integer keys or string keys
        if self.n in pauli_dict:
            allowed = set(pauli_dict[self.n])
        elif str(self.n) in pauli_dict:
            allowed = set(pauli_dict[str(self.n)])
        else:
            raise KeyError(f"No commutator pauli list for n={self.n} found in {commutator_npy}")

        # Filter terms in stabsum_tog
        try:
            current_terms = self.stabsum_tog.to_list()
        except Exception:
            current_terms = sum(self.stab_list).to_list()

        filtered = [(p, c) for p, c in current_terms if p in allowed]

        if len(filtered) == 0:
            # If filtering removed all terms, keep original to avoid empty operator
            truncated = self.stabsum_tog
        else:
            from qiskit.quantum_info import SparsePauliOp as _SPO
            truncated = _SPO.from_list(filtered)

        if stab_sum_update:
            self.stabsum_tog = truncated

        return truncated
    
    def SVM_linear(self, num_samples, class_weights, mar_neg=1.0, mar_pos=1.0):
        """Train linear SVM on normalized stabilizer sum."""
        # Normalize stabilizer sum by adding identity and scaling by 2^(-n)
        phi_appro = (self.stabsum_tog + SparsePauliOp.from_list([('I' * self.n, 1 + 0.j)])) * (2 ** (-self.n))
        self.svm_linear_model = train_linear_svm_pauli(
            self.n, phi_appro, num_samples=num_samples, 
            class_weights=class_weights, margin_neg=mar_neg, margin_pos=mar_pos
        )

    def test_EW_SVM(self):
        """Test SVM classification on the current state. Returns (label, score)."""
        state_test = Statevector.to_operator(self.ini_state).to_matrix()
        return classify_state_with_svm(state_test, self.svm_linear_model, return_score=True)
    
    def noisy_state(self,p_noise):
        if not 0.0 <= p_noise <= 1.0:
            raise ValueError("p_noise must be between 0 and 1")

        state = Statevector.to_operator(self.ini_state).to_matrix()

        dim = 2**self.n

        identity = np.eye(dim, dtype=complex) / dim
        return (1.0 - p_noise) * state + p_noise * identity

    def noisy_bit_flip(self, p_noise):
        """Apply independent bit-flip channel on each qubit with probability `p_noise`.

        The single-qubit bit-flip channel is: rho -> (1-p) rho + p X rho X.
        We apply this channel to each qubit sequentially (independent local noise).
        Returns the noisy density matrix.
        """
        if not 0.0 <= p_noise <= 1.0:
            raise ValueError("p_noise must be between 0 and 1")

        rho = Statevector.to_operator(self.ini_state).to_matrix()
        X = np.array([[0, 1], [1, 0]], dtype=complex)

        for q in range(self.n):
            # build full X on qubit q
            op = None
            for i in range(self.n):
                mat = X if i == q else np.eye(2, dtype=complex)
                op = mat if op is None else np.kron(op, mat)
            rho = (1.0 - p_noise) * rho + p_noise * (op @ rho @ op.conj().T)

        return rho

    def noisy_phase_flip(self, p_noise):
        """Apply independent phase-flip (Z) noise on each qubit: rho -> (1-p) rho + p Z rho Z."""
        if not 0.0 <= p_noise <= 1.0:
            raise ValueError("p_noise must be between 0 and 1")

        rho = Statevector.to_operator(self.ini_state).to_matrix()
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        for q in range(self.n):
            op = None
            for i in range(self.n):
                mat = Z if i == q else np.eye(2, dtype=complex)
                op = mat if op is None else np.kron(op, mat)
            rho = (1.0 - p_noise) * rho + p_noise * (op @ rho @ op.conj().T)

        return rho

    def noisy_bit_phase_flip(self, p_noise):
        """Apply independent Y (bit+phase) flips: rho -> (1-p) rho + p Y rho Y."""
        if not 0.0 <= p_noise <= 1.0:
            raise ValueError("p_noise must be between 0 and 1")

        rho = Statevector.to_operator(self.ini_state).to_matrix()
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        for q in range(self.n):
            op = None
            for i in range(self.n):
                mat = Y if i == q else np.eye(2, dtype=complex)
                op = mat if op is None else np.kron(op, mat)
            rho = (1.0 - p_noise) * rho + p_noise * (op @ rho @ op.conj().T)

        return rho

    def noisy_amplitude_damping(self, gamma):
        """Apply amplitude-damping channel with parameter `gamma` to each qubit independently.

        Amplitude damping Kraus operators for single qubit:
            K0 = [[1, 0], [0, sqrt(1-gamma)]]
            K1 = [[0, sqrt(gamma)], [0, 0]]

        We apply the single-qubit channel sequentially on each qubit so that the
        overall channel corresponds to independent amplitude damping on all qubits.
        Returns the resulting density matrix.
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")

        rho = Statevector.to_operator(self.ini_state).to_matrix()

        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)

        for q in range(self.n):
            new_rho = np.zeros_like(rho, dtype=complex)
            for K in (K0, K1):
                # build full operator acting on qubit q
                op = None
                for i in range(self.n):
                    mat = K if i == q else np.eye(2, dtype=complex)
                    op = mat if op is None else np.kron(op, mat)
                new_rho += op @ rho @ op.conj().T
            rho = new_rho

        return rho
    
    def W_PPT(self, sum=True):
        """
        Construct PPT entanglement witness.
        
        Parameters:
        -----------
        sum : bool
            Whether to use the total stabilizer sum
            
        Returns:
        --------
        array_like
            PPT witness operator
        """
        solve = solve_sdp_W
        
        if sum:
            self.Wppt_alpha_tog = solve(self.n, self.stabsum_tog.to_matrix())
            self.Wppt_tog = (self.Wppt_alpha_tog * np.eye(2**self.n) - 
                           self.stabsum_tog.to_matrix())
            return self.Wppt_tog
        else:
            self.Wppt_alpha_sep = solve(self.n, self.stabsum_sep.to_matrix())
            self.Wppt_sep = (self.Wppt_alpha_sep * np.eye(2**self.n) - 
                           self.stabsum_sep.to_matrix())
            return self.Wppt_sep

    def test_W_PPT(self, sum=True, n_round=6):
        """
        Test the PPT witness on the current state.
        
        Parameters:
        -----------
        sum : bool
            Whether to use the total stabilizer sum
        n_round : int
            Number of decimal places for rounding
            
        Returns:
        --------
        float
            Trace of witness with current state
        """
        W = self.Wppt_tog if sum else self.Wppt_sep
        state_test = Statevector.to_operator(self.ini_state).to_matrix()
        tr = np.real(np.round(np.trace(W @ state_test), n_round))
        print(f'tr_W_PPT_{"sum" if sum else "sep"}: {tr}')
        return tr
