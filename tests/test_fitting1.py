import numpy as np
import scipy
import fitting

def generate(num_samples, prediction_accuracy):
    """
    Generate the simulated data (following Appendix E)
    """
    # number of covariates
    M = 10

    # Generate the covariates
    means = np.zeros(M)
    covs = 0.3 * np.ones((M, M))
    covs = covs + 0.7 * np.eye(M)
    X = np.random.multivariate_normal(means, covs, size=num_samples)

    # Binarize X_i2
    X[:,1] = (X[:,1] > scipy.stats.norm.ppf(0.8)).astype(np.float32)

    # Generate the true outcome Y as a function of the covariates
    X1, X2, X3, X4, X6 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,5]
    a = 0.1 / (1.0 + np.exp(0.5 * X3 - 0.5 * X2))
    b = 1.3 * X4 / (1.0 + np.exp(-0.1 * X2))
    c = 1.5 * X4 * X6
    d = 0.5 * X1 * X2
    e = 1.3 * X1
    f = X2
    W = a + b + c + d + e + f
    Y_true = np.random.binomial(1, scipy.special.expit(W), size=num_samples)
    Y_true = Y_true.astype(float)

    # Make training data from true covariates
    X_train = np.stack([
        X[:,0],
        X[:,0] ** 2,
        X[:,1],
        X[:,3],
    ], axis=1)

    # Simulate the LLM annotations
    p = np.random.binomial(1, prediction_accuracy, size=num_samples)
    Y_pred = p * Y_true + (1 - p) * (1 - Y_true)

    return X_train, Y_true, Y_pred


if __name__ == "__main__":

    N = 1000
    n = 900
    X, Y_true, Y_pred = generate(N, 0.8)

    indices_ones = np.flatnonzero(Y_true)
    indices_zeros = list(set(range(N)) - set(indices_ones))

    num_ones = len(indices_ones)
    num_zeros = len(indices_zeros)

    if num_ones == 0 or num_zeros == 0:
        raise ValueError("The generated data only has one category. Rerun.")

    if num_ones <= n // 2:
        num_zeros = n - num_ones
    elif num_zeros <= n // 2:
        num_ones = n - num_zeros
    else:
        num_ones = n // 2
        num_zeros = n // 2 + n % 2

    selected_ones = np.random.choice(
        a=len(indices_ones),
        size=num_ones,
        replace=False,
    )
    selected_zeros = np.random.choice(
        a=len(indices_zeros),
        size=num_zeros,
        replace=False,
    )

    selected = np.concatenate((selected_ones, selected_zeros))

    assert selected.shape[0] == n

    print("Ran without crashing on dummy data:")

    fitting.logit_fit(X, Y_true)
    print(" - logit_fit(X, Y_true)")

    fitting.logit_fit(X[selected], Y_true[selected])
    print(" - logit_fit(X[selected], Y_true[selected])")

    fitting.logit_fit_dsl(X, Y_true, Y_pred, selected)
    print(" - logit_fit_dsl(X, Y_true, Y_pred, selected)")

    fitting.logit_fit_ppi(X, Y_true, Y_pred, selected)
    print(" - logit_fit_ppi(X, Y_true, Y_pred, selected)")
