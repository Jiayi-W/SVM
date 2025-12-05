#!/usr/bin/env python3
"""
Simple noise-robustness runner that uses the commutator-filtered stabilizer
sum (for H7) to train a linear SVM and run bisection noise tests for multiple
noise models. Saves CSV and PNG outputs under `result/commutator_noise/`.

This script is intentionally lightweight and self-contained so it can be
customised further if needed.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector, partial_trace, entropy, SparsePauliOp
from quantum_utils import H7, H_stab_state_evol, stab_01n_op, ini_state_01, poly_log_t, train_linear_svm_pauli, classify_state_with_svm


def calculate_entropy(state_or_dm, n):
    try:
        if hasattr(state_or_dm, 'to_matrix'):
            dm = state_or_dm.to_matrix()
        else:
            arr = np.asarray(state_or_dm)
            if arr.ndim == 1:
                dm = np.outer(arr, arr.conj())
            else:
                dm = arr
        subsystem_size = n // 2
        rdm = partial_trace(dm, list(range(0, subsystem_size)))
        return float(np.real(entropy(rdm)))
    except Exception:
        return float('nan')


def apply_noise_from_obj(ent_obj, noise_type, p):
    if noise_type == 'white':
        return ent_obj.noisy_state(p)
    elif noise_type == 'bitflip':
        return ent_obj.noisy_bit_flip(p)
    elif noise_type == 'phaseflip':
        return ent_obj.noisy_phase_flip(p)
    elif noise_type == 'bitphase':
        return ent_obj.noisy_bit_phase_flip(p)
    elif noise_type == 'amplitude':
        return ent_obj.noisy_amplitude_damping(p)
    else:
        raise ValueError(f'Unknown noise type: {noise_type}')


def bisection_noise_for_svm(svm_model, ent_obj, n, noise_type='white', tol=1e-6, max_iter=50, compute_entropy=False):
    noise_data = []

    # test clean state
    try:
        rho_clean = Statevector.to_operator(ent_obj.ini_state).to_matrix()
        lbl, score = classify_state_with_svm(ent_obj.ini_state, svm_model, n=n, return_score=True)
        lbl_num = 1 if str(lbl).lower().startswith('ent') else -1
        entry = {'p_noise': 0.0, 'score': float(score), 'label': lbl_num}
        if compute_entropy:
            try:
                entry['entropy'] = calculate_entropy(rho_clean, n)
            except Exception:
                entry['entropy'] = float('nan')
        noise_data.append(entry)
        if score < 0:
            return noise_data, 0.0
    except Exception as e:
        print('Error testing clean state:', e)
        return noise_data, 0.0

    p_lo = 0.0
    p_hi = 1.0
    max_noise = 0.0

    for it in range(max_iter):
        p_mid = (p_lo + p_hi) / 2.0
        try:
            rho_noisy = apply_noise_from_obj(ent_obj, noise_type, p_mid)
            lbl, score = classify_state_with_svm(rho_noisy, svm_model, n=n, return_score=True)
            lbl_num = 1 if str(lbl).lower().startswith('ent') else -1
            entry = {'p_noise': p_mid, 'score': float(score), 'label': lbl_num}
            if compute_entropy:
                try:
                    entry['entropy'] = calculate_entropy(rho_noisy, n)
                except Exception:
                    entry['entropy'] = float('nan')
            noise_data.append(entry)

            if p_hi - p_lo < tol:
                break

            if score > 0:
                max_noise = p_mid
                p_lo = p_mid
                print(f"  Iter {it}: p={p_mid:.6f}, score={score:.6f} detected")
            else:
                p_hi = p_mid
                print(f"  Iter {it}: p={p_mid:.6f}, score={score:.6f} not detected")

        except Exception as e:
            print(f"  Error at p={p_mid}: {e}")
            p_hi = p_mid

    return noise_data, max_noise


def main():
    parser = argparse.ArgumentParser(description='SVM noise robustness using commutator-filtered stabilizer sum (H7)')
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--limit', type=float, default=0.03)
    parser.add_argument('--turns', type=int, default=4)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--max-round', type=int, default=20, help='Number of rounds to try')
    parser.add_argument('--noise-types', nargs='+', default=['white'], choices=['white','bitflip','phaseflip','bitphase','amplitude'])
    parser.add_argument('--comm-npy', type=str, default='commutator_paulis_dict.npy')
    parser.add_argument('--save-plots', action='store_true', default=False, help='Save plots to files (default: False)')
    parser.add_argument('--compute-entropy', action='store_true', default=False, help='Compute and record subsystem entropy (default: False)')
    args = parser.parse_args()

    n = args.n
    num_s = args.num_samples
    limit = args.limit
    turns = args.turns
    steps = args.steps
    max_round = args.max_round
    noise_types = args.noise_types
    comm_npy = args.comm_npy

    # Prepare results directory with parameters encoded
    limit_str = str(limit).replace('.', 'p')
    results_subdir = f'n{n}_limit{limit_str}_turns{turns}_maxround{max_round}_num{num_s}'
    results_dir = os.path.join('result', 'commutator_noise', results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    # Setup object and evolution
    test_ob = H_stab_state_evol(n, H7, stab_01n_op, ini_state_01, poly_log_t, steps)

    # Multi-round training loop (like SVM_linear_noise.py)
    t_list = list(range(1, max_round))
    results_by_step = {}
    noise_results = {}
    combined_data_by_noise = {nt: [] for nt in noise_types}
    summary_rows = []

    print(f"Starting Commutator-SVM with Noise Robustness Analysis (n={n}, max_round={max_round})")
    print(f"Generating {num_s} separable samples per iteration")
    print("=" * 60)

    for a in t_list:
        print(f'Round {a}/{max_round-1}: Processing {a} * 1/polylog')

        # Evolve state (Trotterized)
        test_ob.state_evol(ideal_state=True, den_mat_out=False, state_update=True)

        # Evolve stabilizer sum with truncation
        test_ob.stablt_sum_evol(approx=True, up=limit, step_by_step=True, stab_sum_update=True)

        # Train only at turns intervals
        if a % turns != 0:
            continue

        # Apply commutator-based truncation
        truncated = test_ob.stablt_sum_truncate_by_commutator(commutator_npy=comm_npy, stab_sum_update=False)
        print(f'  Truncated stabilizer sum size: {truncated.size}')

        # Prepare normalized operator for training
        phi_appro = (truncated + SparsePauliOp.from_list([('I'*n, 1.0)]) ) * (2 ** (-n))

        # Train SVM
        print(f'  Training linear SVM...')
        svm_res = train_linear_svm_pauli(n, phi_appro, num_samples=num_s, class_weights=(1.0, 1.0))

        # Test SVM on clean state
        label, score = classify_state_with_svm(test_ob.ini_state, svm_res, n=n, return_score=True)

        # Store results
        results_by_step[a] = {
            'time': a * poly_log_t(n),
            'score': score,
            'size': truncated.size,
            'label': label,
        }

        # Check if entanglement is detected
        if score < 0:
            print(f'✗ Entanglement NOT detected (score={score:.4f})')
            print(f'Stopping SVM training for limit={limit}')
            break

        print(f'✓ Entanglement detected (score={score:.4f}, label={label})')

        # Perform noise robustness test for each requested noise model
        noise_results[a] = {}
        for nt in noise_types:
            print(f"  Performing noise robustness test for model at step {a} (noise={nt})...")
            test_ob.noise_type = nt
                noise_data, max_noise = bisection_noise_for_svm(svm_res, test_ob, n, noise_type=nt, compute_entropy=args.compute_entropy)
            noise_results[a][nt] = {'noise_data': noise_data, 'max_noise': max_noise}
            print(f"  Maximum noise tolerance ({nt}): p_noise = {max_noise:.6f}")

    print(f"\nEvolution completed. Final results:")
    print(f"  Rounds completed: {len(results_by_step)}")
    if results_by_step:
        final_score = list(results_by_step.values())[-1]['score']
        print(f"  Final SVM score: {final_score:.4f}")

    # Save noise robustness results
    for step_idx, per_noise_dict in noise_results.items():
        for nt in noise_types:
            if nt not in per_noise_dict:
                continue
            step_data = per_noise_dict[nt]
            noise_data = step_data['noise_data']
            max_noise = step_data['max_noise']

            # Build DataFrame from noise test results
            df = pd.DataFrame(noise_data)
            df['round'] = int(step_idx)

            # Add to combined list for this noise type
            combined_data_by_noise.setdefault(nt, []).append(df)

            # Extract clean-state score for summary
            try:
                clean_score = float(df.loc[df['p_noise'] == 0.0, 'score'].iloc[0])
            except Exception:
                clean_score = float(df['score'].iloc[0]) if 'score' in df.columns and len(df) > 0 else float('nan')

            summary_rows.append({'noise_type': nt, 'round': int(step_idx), 'max_noise': float(max_noise), 'clean_score': clean_score})

            # Plot (optional)
            if args.save_plots:
                plt.figure(figsize=(8, 5))
                detected = df[df['label'] == 1]
                notdet = df[df['label'] == -1]
                if not detected.empty:
                    plt.scatter(detected['p_noise'], detected['score'], c='tab:blue', label='detected')
                if not notdet.empty:
                    plt.scatter(notdet['p_noise'], notdet['score'], c='tab:orange', label='not detected')
                plt.axhline(0, color='r', linestyle='--')
                plt.axvline(max_noise, color='g', linestyle='--', label=f'max={max_noise:.4f}')
                plt.xlabel('p_noise')
                plt.ylabel('SVM score')
                plt.title(f'Commutator-SVM Score vs Noise ({nt}) - Round {step_idx}')
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(results_dir, f'comm_noise_{nt}_n{n}_step{step_idx}_score.png')
                plt.savefig(plot_path, bbox_inches='tight')
                print(f'  Saved plot to {plot_path}')
                plt.close()

            # Entropy plot (only if --compute-entropy and --save-plots)
            if args.compute_entropy and args.save_plots and 'entropy' in df.columns:
                plt.figure(figsize=(8, 5))
                detected_e = df[df['label'] == 1]
                notdet_e = df[df['label'] == -1]
                if not detected_e.empty:
                    plt.scatter(detected_e['p_noise'], detected_e['entropy'], c='tab:blue', label='detected', s=60, marker='o')
                if not notdet_e.empty:
                    plt.scatter(notdet_e['p_noise'], notdet_e['entropy'], c='tab:orange', label='not detected', s=60, marker='x')
                plt.axvline(max_noise, color='g', linestyle='--', label=f'max={max_noise:.4f}')
                plt.xlabel('p_noise')
                plt.ylabel('Entropy')
                plt.title(f'Commutator-SVM Entropy vs Noise ({nt}) - Round {step_idx}')
                plt.legend()
                plt.grid(True)
                plot_path2 = os.path.join(results_dir, f'comm_noise_{nt}_n{n}_step{step_idx}_entropy.png')
                plt.savefig(plot_path2, bbox_inches='tight')
                print(f'  Saved plot to {plot_path2}')
                plt.close()

    # Write aggregated summary CSV
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_fname = f'comm_noise_summary_n{n}.csv'
        summary_path = os.path.join(results_dir, summary_fname)
        summary_df.to_csv(summary_path, index=False)
        print(f'Aggregated commutator-SVM noise summary saved to {summary_path}')

    # Write combined CSV per noise type
    for nt in noise_types:
        if nt in combined_data_by_noise and len(combined_data_by_noise[nt]) > 0:
            try:
                combined_df = pd.concat(combined_data_by_noise[nt], ignore_index=True)
            except Exception:
                combined_df = pd.DataFrame([r for sub in combined_data_by_noise[nt] for r in sub.to_dict(orient='records')])
            combined_fname = f'comm_noise_combined_{nt}_n{n}_limit{limit_str}_turns{turns}_maxround{max_round}_num{num_s}.csv'
            combined_path = os.path.join(results_dir, combined_fname)
            combined_df.to_csv(combined_path, index=False)
            print(f'Aggregated commutator-SVM combined CSV saved to {combined_path}')

    print('Done.')

if __name__ == '__main__':
    main()
