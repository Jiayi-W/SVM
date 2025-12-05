from quantum_utils import H7, H5, poly_log_t, H_stab_state_evol, stab_01n_op, ini_state_01
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_entropy(state_or_dm, n):
    try:
        if isinstance(state_or_dm, Statevector):
            full_dm = Statevector.to_operator(state_or_dm).to_matrix()
        elif hasattr(state_or_dm, 'to_matrix'):
            full_dm = state_or_dm.to_matrix()
        else:
            arr = np.asarray(state_or_dm)
            if arr.ndim == 1:
                full_dm = np.outer(arr, arr.conj())
            else:
                full_dm = arr

        # trace out second half
        subsystem_size = n // 2
        subsystem_indices = list(range(0, subsystem_size))
        rdm = partial_trace(full_dm, subsystem_indices)
        return float(np.real(entropy(rdm)))
    except Exception as e:
        print(f"Warning: Could not calculate entropy: {e}")
        return float('nan')


def find_noise_tolerance_for_witness(W, ent_obj, n, noise_type='white', noise_tolerance=1e-6, max_iterations=50, compute_entropy=False):
    """Bisection search on noise strength p where Tr(W rho_noisy) <= 0.

    Supports multiple noise models via `ent_obj` methods:
      - 'white' : depolarizing (default)
      - 'bitflip'
      - 'phaseflip'
      - 'bitphase'
      - 'amplitude'
    """
    noise_data = []

    # helper to apply selected noise and return density matrix for given p
    def apply_noise(p):
        if noise_type == 'white':
            rho_clean = Statevector.to_operator(ent_obj.ini_state).to_matrix()
            dim = 2 ** n
            return (1.0 - p) * rho_clean + p * np.eye(dim, dtype=complex) / dim
        elif noise_type == 'bitflip':
            return ent_obj.noisy_bit_flip(p)
        elif noise_type == 'phaseflip':
            return ent_obj.noisy_phase_flip(p)
        elif noise_type == 'bitphase':
            return ent_obj.noisy_bit_phase_flip(p)
        elif noise_type == 'amplitude':
            return ent_obj.noisy_amplitude_damping(p)
        else:
            raise ValueError(f"Unknown noise_type '{noise_type}'")

    # test clean state
    try:
        rho_clean = Statevector.to_operator(ent_obj.ini_state).to_matrix()
        tr_clean = float(np.real(np.trace(W @ rho_clean)))
        label_clean = 1 if tr_clean <= 0 else -1
        entry = {'p_noise': 0.0, 'trace_witness': tr_clean, 'label': label_clean}
        if compute_entropy:
            try:
                entry['entropy'] = calculate_entropy(rho_clean, n)
            except Exception:
                entry['entropy'] = float('nan')
        noise_data.append(entry)
        if label_clean == -1:
            print(f"  Warning: Clean state NOT detected by witness (trace={tr_clean:.6f})")
            return noise_data, 0.0
    except Exception as e:
        print(f"  Error testing clean state: {e}")
        return noise_data, 0.0

    p_low = 0.0
    p_high = 1.0
    max_noise = 0.0

    for it in range(max_iterations):
        p_mid = (p_low + p_high) / 2.0
        rho_noisy = apply_noise(p_mid)
        tr_mid = float(np.real(np.trace(W @ rho_noisy)))
        label_mid = 1 if tr_mid <= 0 else -1
        entry = {'p_noise': p_mid, 'trace_witness': tr_mid, 'label': label_mid}
        if compute_entropy:
            try:
                entry['entropy'] = calculate_entropy(rho_noisy, n)
            except Exception:
                entry['entropy'] = float('nan')
        noise_data.append(entry)

        if p_high - p_low < noise_tolerance:
            break

        if label_mid == 1:
            max_noise = p_mid
            p_low = p_mid
            print(f"    Iter {it}: p={p_mid:.6f}, trace={tr_mid:.6f} ✓ (detected)")
        else:
            p_high = p_mid
            print(f"    Iter {it}: p={p_mid:.6f}, trace={tr_mid:.6f} ✗ (not detected)")

    return noise_data, max_noise


def main():
    import argparse

    parser = argparse.ArgumentParser(description='WPPT noise robustness with multiple noise models')
    parser.add_argument('--n', type=int, default=7, help='Number of qubits')
    parser.add_argument('--H', type=str, default='H7', choices=['H7','H5'], help='Hamiltonian name')
    parser.add_argument('--limit', type=float, default=0.03, help='Truncation limit for stabilizer sum')
    parser.add_argument('--turns', type=int, default=4, help='How often to construct witness (turns)')
    parser.add_argument('--max-round', type=int, default=20, help='Number of rounds to try')
    parser.add_argument('--steps', type=int, default=10, help='Trotter steps for evolution')
    parser.add_argument('--noise-types', nargs='+', default=['white'], choices=['white','bitflip','phaseflip','bitphase','amplitude'], help='Noise models to apply')
    parser.add_argument('--save-plots', action='store_true', default=False, help='Save plots to files (default: False)')
    parser.add_argument('--compute-entropy', action='store_true', default=False, help='Compute and record subsystem entropy (default: False)')
    args = parser.parse_args()

    # Configuration
    n = args.n
    H_name = args.H
    H = H7 if H_name == 'H7' else H5
    # If additional Hamiltonians are added to quantum_utils, extend mapping above
    steps = args.steps
    t = poly_log_t
    limit = args.limit
    turns = args.turns
    max_round = args.max_round
    noise_types = args.noise_types

    t_list = list(range(1, max_round))

    test_ob = H_stab_state_evol(n, H, stab_01n_op, ini_state_01, poly_log_t, steps)

    results = {}
    noise_results = {}
    # Prepare per-noise aggregation container: collect DataFrames for each noise type across rounds
    combined_noise_data = {nt: [] for nt in noise_types}

    # Collect summary rows for aggregated per-noise/round max-noise values
    summary_rows = []

    # Prepare results directory that encodes run parameters
    limit_str = str(limit).replace('.', 'p')
    results_subdir = f'n{n}_limit{limit_str}_turns{turns}_maxround{max_round}'
    results_dir = os.path.join('result', 'noise_wppt', results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Starting WPPT noise robustness (n={n})")
    print("=" * 60)

    for a in t_list:
        print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')
        test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)
        test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)

        if a % turns == 0:
            try:
                test_ob.W_PPT(sum=True)
                tr_tog = test_ob.test_W_PPT(sum=True)
                size_tog = test_ob.stabsum_tog.size
            except Exception as e:
                print(f"  Error constructing/testing WPPT: {e}")
                continue

            results[a] = {'time': a * t(n), 'trace': tr_tog, 'size': size_tog, 'alpha': test_ob.Wppt_alpha_tog}

            # entanglement detected if trace <= 0
            if tr_tog > 0:
                print(f'✗ Entanglement NOT detected (separable)')
                print(f'  Stopping WPPT construction for limit={limit}')
                break

            print(f'✓ Entanglement detected (trace={tr_tog:.6f})')

            # perform noise robustness on the witness operator for each requested noise model
            W = test_ob.Wppt_tog
            noise_results[a] = {}
            for nt in noise_types:
                print(f"  Performing noise robustness test for WPPT at step {a} (noise={nt})...")
                test_ob.noise_type = nt
                noise_data, max_noise = find_noise_tolerance_for_witness(W, test_ob, n, noise_type=nt, compute_entropy=args.compute_entropy)
                noise_results[a][nt] = {'noise_data': noise_data, 'max_noise': max_noise}
                print(f"  Maximum noise tolerance ({nt}): p_noise = {max_noise:.6f}")

    # Save noise_results (aggregating per-noise-type across rounds)
    for step_idx, per_noise in noise_results.items():
        for nt, step_data in per_noise.items():
            noise_data = step_data['noise_data']
            max_noise = step_data['max_noise']
            df = pd.DataFrame(noise_data)

            # annotate dataframe with round and hamiltonian before aggregation
            try:
                df['round'] = int(step_idx)
            except Exception:
                df['round'] = step_idx
            df['hamiltonian'] = H_name
            combined_noise_data.setdefault(nt, []).append(df)

            # Extract clean-state trace for summary
            try:
                clean_trace = float(df.loc[df['p_noise'] == 0.0, 'trace_witness'].iloc[0])
            except Exception:
                clean_trace = float(df['trace_witness'].iloc[0]) if 'trace_witness' in df.columns and len(df) > 0 else float('nan')

            summary_rows.append({'noise_type': nt, 'round': int(step_idx), 'max_noise': float(max_noise), 'clean_trace': clean_trace, 'hamiltonian': H_name})

            # Trace plot (only if --save-plots)
            if args.save_plots:
                plt.figure(figsize=(8, 5))
                plt.plot(df['p_noise'], df['trace_witness'], marker='o')
                plt.axhline(0, color='r', linestyle='--')
                plt.axvline(max_noise, color='g', linestyle='--', label=f'max_noise={max_noise:.4f}')
                plt.xlabel('p_noise')
                plt.ylabel('Trace(W rho_noisy)')
                plt.title(f'WPPT Trace vs Noise (step {step_idx}, noise={nt})')
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(results_dir, f'noise_wppt_{nt}_n{n}_step{step_idx}_trace.png')
                plt.savefig(plot_path, bbox_inches='tight')
                print(f'  Saved plot to {plot_path}')
                plt.close()

            # Entropy plot (only if --compute-entropy and present)
            if args.compute_entropy and args.save_plots and 'entropy' in df.columns:
                plt.figure(figsize=(8, 5))
                plt.plot(df['p_noise'], df['entropy'], marker='s', color='orange')
                plt.axvline(max_noise, color='g', linestyle='--')
                plt.xlabel('p_noise')
                plt.ylabel('Entropy')
                plt.title(f'Entropy vs Noise (step {step_idx}, noise={nt})')
                plt.grid(True)
                plot_path2 = os.path.join(results_dir, f'noise_wppt_{nt}_n{n}_step{step_idx}_entropy.png')
                plt.savefig(plot_path2, bbox_inches='tight')
                print(f'  Saved plot to {plot_path2}')
                plt.close()

    print('\nWPPT noise robustness analysis complete!')

    # Write aggregated summary CSV for quick overview
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_fname = f'noise_wppt_summary_n{n}_H{H_name}.csv'
        summary_path = os.path.join(results_dir, summary_fname)
        summary_df.to_csv(summary_path, index=False)
        print(f'Aggregated WPPT noise summary saved to {summary_path}')

    # Write combined per-noise-type CSV that aggregates all rounds
    for nt, df_list in combined_noise_data.items():
        if len(df_list) == 0:
            continue
        try:
            combined_df = pd.concat(df_list, ignore_index=True)
        except Exception:
            combined_df = pd.DataFrame([r for sub in df_list for r in sub.to_dict(orient='records')])
        limit_str = str(limit).replace('.', 'p')
        combined_fname = f'noise_wppt_combined_{nt}_n{n}_H{H_name}_limit{limit_str}_turns{turns}_maxround{max_round}.csv'
        combined_path = os.path.join(results_dir, combined_fname)
        combined_df.to_csv(combined_path, index=False)
        print(f'Aggregated WPPT combined CSV saved to {combined_path}')


if __name__ == '__main__':
    main()
