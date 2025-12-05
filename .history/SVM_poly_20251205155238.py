from quantum_utils import (
    H7, H5,
    poly_log_t,
    H_stab_state_evol,
    stab_01n_op,
    ini_state_01,
    train_poly_svm_pauli,
    classify_state_with_svm,
)
from qiskit.quantum_info import SparsePauliOp
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def extract_svm_intercept(svm_model):
    """Extract intercept or decision boundary info from sklearn SVM model."""
    if hasattr(svm_model, 'intercept_'):
        try:
            return float(np.asarray(svm_model.intercept_).ravel()[0])
        except Exception:
            return float(svm_model.intercept_)
    return float('nan')


def generate_results_filename(kernel_type, n, limit, turns, max_round, num_samples, class_w, gamma=None, degree=None):
    """Generate descriptive filename with all configuration parameters."""
    cw_str = f"{class_w[0]:.1f}_{class_w[1]:.1f}".replace('.', 'p')
    
    filename = (
        f"svm_{kernel_type}_n{n}_limit{limit}_turns{turns}_maxround{max_round}_"
        f"num{num_samples}_cw{cw_str}"
    )
    
    if gamma is not None:
        if isinstance(gamma, str):
            filename += f"_gamma{gamma}"
        else:
            filename += f"_gamma{gamma:.2e}".replace('.', 'p').replace('e', 'e')
    
    if degree is not None:
        filename += f"_deg{degree}"
    
    filename += ".csv"
    return filename


# Configuration
n = 7
H = H7
stab_list = stab_01n_op
ini_state = ini_state_01
steps = 10
t = poly_log_t
limit = 0.1
turns = 4
max_round = 50
num_s = 1000
class_w = (1, 1)  # (separable, entangled)
poly_degree = 3  # Polynomial kernel degree

t_list = list(range(1, max_round))
results_by_step = {}

test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)

print(f"Starting Polynomial SVM evaluation (n={n}, kernel=poly, degree={poly_degree})")
print(f"Generating {num_s} separable samples per iteration")
print("=" * 60)

for a in t_list:
    print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')
    
    # Evolve state (Ideal)
    test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
    
    # Evolve stabilizer sum with truncation
    test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)
    
    if a % turns == 0:
        # Train polynomial SVM
        phi_appro = (test_ob.stabsum_tog + SparsePauliOp.from_list([('I' * n, 1 + 0.j)])) * (2 ** (-n))
        svm_model = train_poly_svm_pauli(
            n, phi_appro, num_samples=num_s, class_weights=class_w, degree=poly_degree
        )
        
        # For polynomial, manually classify the initial state
        state_test = test_ob.ini_state
        try:
            label, score = classify_state_with_svm(state_test, svm_model, n=n, return_score=True)
        except Exception as e:
            label, score = 'error', float('nan')
        
        # Store results
        results_by_step[a] = {
            'time': a * t(n),
            'trace': score,
            'size': test_ob.stabsum_tog.size,
            'alpha': extract_svm_intercept(svm_model['model']),
            'label': label,
        }

        # Check if entanglement is NOT detected (score < 0 means separable state)
        if score < 0:
            print(f'✗ Entanglement NOT detected (score={score:.4f})')
            print(f'Stopping polynomial SVM for limit={limit}')
            break

        print(f'✓ Entanglement detected (score={score:.4f}, label={label})')

print(f"\nPolynomial SVM Evolution completed. Final results:")
print(f"  Rounds completed: {len(results_by_step)}")
if results_by_step:
    final_trace = list(results_by_step.values())[-1]['trace']
    print(f"  Final trace value: {final_trace:.4f}")

# Create results directory and save data + plots
results_dir = 'result_poly'
os.makedirs(results_dir, exist_ok=True)

if results_by_step:
    # Build DataFrame from consolidated results
    df = pd.DataFrame(list(results_by_step.values()))
    
    # Generate filename with configuration
    csv_filename = generate_results_filename(
        'poly', n, limit, turns, max_round, num_s, class_w, degree=poly_degree
    )
    csv_path = os.path.join(results_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f'Saved polynomial results to {csv_path}')
    
    # Plot helper function
    def save_plot(x_col, y_col, xlabel, ylabel, title):
        plt.figure(figsize=(8, 5))
        plt.plot(df[x_col], df[y_col], marker='s', color='green')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title + f' (Poly Kernel, degree={poly_degree})')
        plt.grid(True)
        # Generate plot filename from csv filename
        plot_name = csv_filename.replace('.csv', f'_{y_col}_vs_{x_col}.png')
        png_path = os.path.join(results_dir, plot_name)
        plt.savefig(png_path, bbox_inches='tight')
        plt.close()
        print(f'  Saved plot to {png_path}')
    
    # Generate all plots
    save_plot('time', 'trace', 'time', 'SVM score', 'SVM Score vs Time')
    save_plot('time', 'size', 'time', 'stabilizer size', 'Stabilizer Size vs Time')
    save_plot('time', 'alpha', 'time', 'SVM intercept', 'SVM Intercept vs Time')

print("Polynomial SVM analysis complete!")
