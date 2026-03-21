library(dsl)

set.seed(42)
n <- 500
n_labeled <- 100

# Simulate data
x <- rnorm(n)
y_true <- 2 * x + rnorm(n)
y_pred <- y_true + rnorm(n, sd = 0.5)  # noisy predictions

data <- data.frame(
  y = c(y_true[1:n_labeled], rep(NA, n - n_labeled)),
  y_pred = y_pred,
  x = x
)

# Run dsl with lambda=0 (no regularization)
fit_no_reg <- dsl(
  model = "lm",
  formula = y ~ x,
  predicted_var = "y",
  prediction = "y_pred",
  data = data,
  lambda = 0
)

# Run dsl with lambda=0.01 (L2 regularization)
fit_reg <- dsl(
  model = "lm",
  formula = y ~ x,
  predicted_var = "y",
  prediction = "y_pred",
  data = data,
  lambda = 0.01
)

cat("\n=== Coefficients (lambda=0) ===\n")
print(fit_no_reg$coefficients)

cat("\n=== Coefficients (lambda=0.01) ===\n")
print(fit_reg$coefficients)

cat("\n=== Difference (lambda=0 minus lambda=0.01) ===\n")
print(fit_no_reg$coefficients - fit_reg$coefficients)

cat("\nL2 regularization is working if coefficients differ (lambda=0.01 should be shrunk toward zero).\n")
