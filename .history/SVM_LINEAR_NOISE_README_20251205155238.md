# SVM Linear Noise Robustness Testing

## Overview
`SVM_linear_noise.py` implements a comprehensive noise robustness analysis pipeline for trained SVM entanglement witnesses. After each successful entanglement detection, the script uses a **bisection method** to determine the maximum noise tolerance of the trained model.

## Key Features

### 1. **Bisection Method for Noise Tolerance**
- Tests noise levels incrementally using binary search
- Finds maximum noise $p_{\text{max}}$ where model still detects entanglement
- Convergence criterion: $p_{\text{high}} - p_{\text{low}} < 10^{-6}$ or max 50 iterations
- Records decision scores at each noise level

### 2. **Data Collection Per Model**
For each trained SVM model at step `a`, the script records:
- **p_noise**: Noise level tested (from 0.0 to $p_{\text{max}}$)
- **score**: SVM decision function value (positive = entangled)
- **entropy**: Entanglement entropy of the noisy state
- **label**: Binary classification label (1 = entangled, -1 = separable)

### 3. **State Noise Model**
```python
ρ_noisy = (1 - p_noise) * ρ_evolved + p_noise * I/2^n
```
Where:
- $\rho_{\text{evolved}}$ is the evolved quantum state
- $p_{\text{noise}} \in [0, 1]$ is the noise parameter
- $I/2^n$ is the maximally mixed state

### 4. **Entropy Calculation**
For each noisy state, the script computes entanglement entropy:
1. Trace out second half of qubits: $\rho_A = \text{Tr}_B(\rho_{\text{noisy}})$
2. Calculate von Neumann entropy: $S = -\text{Tr}(\rho_A \log_2 \rho_A)$

## Output Files

### CSV Files (in `result_noise/` folder)
Filename format: `noise_linear_n{n}_limit{limit}_turns{turns}_step{step}_num{num_samples}_cw{cw_str}{margin_str}.csv`

Example: `noise_linear_n7_limit0p01_turns1_step4_num10000_cw1p0_1p0.csv`

**Columns:**
| p_noise | score | entropy | label |
|---------|-------|---------|-------|
| 0.0 | 2.543 | 1.843 | 1 |
| 0.5 | 0.127 | 1.912 | 1 |
| 0.75 | -0.034 | 2.045 | -1 |
| ... | ... | ... | ... |

### Plot Files (in `result_noise/` folder)

1. **Score vs Noise**: `noise_linear_..._score.png`
   - X-axis: Noise level $p_{\text{noise}}$
   - Y-axis: SVM score
   - Red dashed line: Decision boundary (score = 0)
   - Green dashed line: Maximum noise tolerance $p_{\text{max}}$

2. **Entropy vs Noise**: `noise_linear_..._entropy.png`
   - X-axis: Noise level $p_{\text{noise}}$
   - Y-axis: Entanglement entropy
   - Shows how state entropy increases with noise
   - Green dashed line: $p_{\text{max}}$

## Configuration Parameters

```python
n = 7                # Number of qubits
limit = 0.01         # Truncation limit for stabilizer sum
turns = 1            # Train SVM every `turns` evolution steps
max_round = 50       # Maximum evolution rounds
num_s = 10000        # Samples per SVM training
class_w = (1, 1)     # Class weights (separable, entangled)
mar_neg = 1.0        # Negative margin
mar_pos = 1.0        # Positive margin
```

## Algorithm Flow

### Main Evolution Loop
```
for a in 1..max_round:
    1. Evolve state: ρ(t + dt)
    2. Evolve stabilizer sum with truncation
    3. If a % turns == 0:
        a. Train SVM on Pauli expectation features
        b. Test on evolved state
        c. If entanglement detected (score > 0):
            - Store model metrics (time, trace, size, b_value)
            - Perform NOISE ROBUSTNESS TEST
            - Save noise results and plots
        d. Else: stop evolution
```

### Bisection Search
```
p_low = 0.0, p_high = 1.0
max_noise = 0.0

while (p_high - p_low > 1e-6) and (iterations < 50):
    p_mid = (p_low + p_high) / 2
    score = SVM.decision_function(noisy_state(p_mid))
    
    if score > 0:  # Still detects as entangled
        max_noise = p_mid
        p_low = p_mid
    else:          # Now classified as separable
        p_high = p_mid
```

## Usage

### Basic Run
```bash
cd /Users/jiayiwu/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/GitHub/SVM
python SVM_linear_noise.py
```

### Expected Output
```
Starting Linear SVM with Noise Robustness Analysis (n=7)
Generating 10000 separable samples per iteration
============================================================
Round 1/49: Processing 1 * 1/polylog
✓ Entanglement detected (score=2.543, label=1)
  Performing noise robustness test for model at step 1...
    Iteration 0: p_noise=0.500000, score=0.127 ✓ (entangled)
    Iteration 1: p_noise=0.750000, score=-0.034 ✗ (separable)
    ...
    Iteration 8: p_noise=0.613281, score=0.002 ✓ (entangled)
  Maximum noise tolerance: p_noise = 0.613281
Saved noise robustness results to result_noise/noise_linear_n7_limit0p01_turns1_step1_num10000_cw1p0_1p0.csv
  Saved plot to result_noise/noise_linear_n7_limit0p01_turns1_step1_num10000_cw1p0_1p0_score.png
  Saved plot to result_noise/noise_linear_n7_limit0p01_turns1_step1_num10000_cw1p0_1p0_entropy.png
```

## Key Functions

### `test_noise_robustness(svm_model, ent_obj, n, noise_tolerance=1e-6, max_iterations=50)`
Performs bisection search for maximum noise tolerance.

**Parameters:**
- `svm_model`: Trained SVM model (from `train_linear_svm_pauli`)
- `ent_obj`: Quantum state object with `noisy_state()` method
- `n`: Number of qubits
- `noise_tolerance`: Bisection convergence criterion
- `max_iterations`: Maximum bisection iterations

**Returns:**
- `noise_data`: List of dicts with `{p_noise, score, entropy, label}`
- `max_noise`: Maximum noise level where entanglement is detected

### `calculate_entropy(state_or_dm, n)`
Computes entanglement entropy using partial trace of first half of qubits.

**Parameters:**
- `state_or_dm`: Statevector or density matrix
- `n`: Number of qubits

**Returns:**
- Entanglement entropy (float)

### `generate_noise_filename(...)`
Creates descriptive CSV filename encoding all configuration parameters.

**Filename Pattern:**
```
noise_linear_n{n}_limit{limit}_turns{turns}_step{step}_num{samples}_cw{class_w}_m{margins}.csv
```

## Dependencies

```python
# Quantum framework
from qiskit.quantum_info import (
    SparsePauliOp, partial_trace, entropy, Statevector
)

# Utilities
from quantum_utils import (
    H7, poly_log_t, H_stab_state_evol,
    stab_01n_op, ini_state_01, classify_state_with_svm
)

# Data & visualization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

## Output Directory Structure

```
result_noise/
├── noise_linear_n7_limit0p01_turns1_step1_num10000_cw1p0_1p0.csv
├── noise_linear_n7_limit0p01_turns1_step1_num10000_cw1p0_1p0_score.png
├── noise_linear_n7_limit0p01_turns1_step1_num10000_cw1p0_1p0_entropy.png
├── noise_linear_n7_limit0p01_turns1_step4_num10000_cw1p0_1p0.csv
├── noise_linear_n7_limit0p01_turns1_step4_num10000_cw1p0_1p0_score.png
├── noise_linear_n7_limit0p01_turns1_step4_num10000_cw1p0_1p0_entropy.png
└── ...
```

## Interpretation

### Score vs Noise Plot
- **Positive slope region**: Model robustness zone
- **Crossing zero**: Transition from entanglement detection to misclassification
- **Steeper drop**: Less robust to noise

### Entropy vs Noise Plot
- **Monotonic increase**: Noise increases state entropy
- **Flatter regions**: Indicates robustness of entanglement properties
- **Steep jumps**: Critical noise transition

### CSV Analysis
- Filter by `label` column to distinguish regions:
  - `label = 1`: Successfully detected as entangled
  - `label = -1`: Misclassified as separable
- Use `score` column to rank robustness (higher score = more confident)

## Notes

1. **Bisection Efficiency**: Uses logarithmic iterations (~50) instead of linear sampling
2. **Entropy Robustness**: Entropy increases smoothly, but SVM score may drop suddenly
3. **Edge Cases**: If clean state (p_noise=0) not detected as entangled, bisection returns max_noise=0.0
4. **Numerical Stability**: All float conversions explicit to handle sklearn/cvxpy model differences

## Future Extensions

1. **Parallel bisection**: Test multiple models simultaneously
2. **Other noise models**: Amplitude damping, dephasing, bit-flip errors
3. **Robustness ranking**: Compare max_noise across different model architectures
4. **Adaptive sampling**: Increase samples near decision boundary for finer resolution
