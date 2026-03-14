# ── Native Ridge regression — base R closed-form solution ─────────────────────
#
# No external package dependency. Solves:
#   min ||y - (X w + b)||^2 + alpha * ||w||^2
# via the augmented normal equations:
#   (X_aug^T X_aug + Lambda) w_aug = X_aug^T y
# where X_aug = [1 | X] and Lambda = diag(0, alpha, ..., alpha).
#
# Supports sample_weight via sqrt-weighted form (avoids n×n weight matrix).
#
# Internal API (not exported):
#   .lm_fit(X, y, alpha, sample_weight) -> list(intercept, coef)
#   .lm_predict(fit, X)                 -> numeric vector

.lm_fit <- function(X, y, alpha = 1.0, sample_weight = NULL) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)

  # Augmented design matrix: [bias | features]
  X_aug <- cbind(1, X)

  # Penalty: bias (column 1) not regularised
  lam    <- c(0, rep(alpha, p))
  Lambda <- diag(lam)

  if (!is.null(sample_weight)) {
    sw      <- as.numeric(sample_weight)
    sw      <- sw / sum(sw) * n   # normalise to sum = n
    sw_sqrt <- sqrt(sw)
    # Scale rows — avoids materialising n×n weight matrix
    X_w <- X_aug * sw_sqrt
    y_w <- y    * sw_sqrt
    A   <- t(X_w) %*% X_w + Lambda
    b   <- t(X_w) %*% y_w
  } else {
    A <- t(X_aug) %*% X_aug + Lambda
    b <- t(X_aug) %*% y
  }

  w <- as.numeric(solve(A, b))
  list(intercept = w[1L], coef = w[-1L])
}

.lm_predict <- function(fit, X) {
  as.numeric(as.matrix(X) %*% fit$coef + fit$intercept)
}
