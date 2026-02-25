import numpy as np
from fitting import logit_fit, logit_fit_ppi
from vary_expert_simulation import generate

np.random.seed(42)

# Generate synthetic data: 1000 samples, LLM correct 90% of the time
N = 1000
X, Y_true, Y_pred = generate(N, prediction_accuracy=0.9)

# Select 200 "expert" labelled samples
n_expert = 200
selected_idx = np.random.choice(N, size=n_expert, replace=False)

# Convert to boolean mask (needed for logit_fit_ppi's ~ operator)
selected = np.zeros(N, dtype=bool)
selected[selected_idx] = True

# Method 1: all true labels (ideal upper bound, impossible in practice)
coeffs_all = logit_fit(X, Y_true)
print("Coefficients (all true labels - upper bound):")
print(np.round(coeffs_all, 4))

# Method 2: expert labels only (naive baseline)
coeffs_exp = logit_fit(X[selected], Y_true[selected])
print("\nCoefficients (expert only - naive baseline):")
print(np.round(coeffs_exp, 4))

# Method 3: PPI (uses expert labels + all LLM predictions)
coeffs_ppi = logit_fit_ppi(X, Y_true, Y_pred, selected)
print("\nCoefficients (PPI):")
print(np.round(coeffs_ppi, 4))
