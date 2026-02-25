import numpy as np
import scipy
from fitting import logit_fit, logit_fit_ppi


def generate(num_samples, prediction_accuracy):
    M = 10
    means = np.zeros(M)
    covs = 0.3 * np.ones((M, M))
    covs = covs + 0.7 * np.eye(M)
    X = np.random.multivariate_normal(means, covs, size=num_samples)
    X[:,1] = (X[:,1] > scipy.stats.norm.ppf(0.8)).astype(np.float32)
    X1, X2, X3, X4, X6 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,5]
    a = 0.1 / (1.0 + np.exp(0.5 * X3 - 0.5 * X2))
    b = 1.3 * X4 / (1.0 + np.exp(-0.1 * X2))
    c = 1.5 * X4 * X6
    d = 0.5 * X1 * X2
    e = 1.3 * X1
    f = X2
    W = a + b + c + d + e + f
    Y_true = np.random.binomial(1, scipy.special.expit(W), size=num_samples).astype(float)
    X_train = np.stack([X[:,0], X[:,0] ** 2, X[:,1], X[:,3]], axis=1)
    p = np.random.binomial(1, prediction_accuracy, size=num_samples)
    Y_pred = p * Y_true + (1 - p) * (1 - Y_true)
    return X_train, Y_true, Y_pred

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
