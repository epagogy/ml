# ── Native logistic regression -- base R stats::optim L-BFGS-B ─────────────────
#
# No external package dependency. Supports:
#   - Binary classification (sigmoid + cross-entropy + L2)
#   - Multiclass via one-vs-rest (K independent binary classifiers)
#   - L2 regularisation (C parameter, bias not regularised)
#
# Internal API (not exported):
#   .lr_fit(X, y_int, n_classes, C, sample_weight) -> fit object
#   .lr_predict(fit, X) -> integer vector (0-based class indices)
#   .lr_proba(fit, X)   -> data.frame (n x k), rows sum to 1

# ── Loss and gradient for one binary classifier ──────────────────────────────
# w[1] = bias, w[-1] = feature weights

.lr_loss <- function(w, X, y, C, sw = NULL) {
  z <- as.numeric(X %*% w[-1] + w[1])
  z <- pmax(-500, pmin(500, z))
  p <- 1 / (1 + exp(-z))
  p_safe <- pmax(1e-15, pmin(1 - 1e-15, p))
  n <- length(y)
  if (!is.null(sw)) {
    sw_norm <- sw / sum(sw) * n
    ce <- -sum(sw_norm * (y * log(p_safe) + (1 - y) * log(1 - p_safe))) / n
  } else {
    ce <- -mean(y * log(p_safe) + (1 - y) * log(1 - p_safe))
  }
  # L2 penalty -- match sklearn's C convention: effective reg per sample = 1/(C*n)
  ce + 0.5 / (C * n) * sum(w[-1]^2)
}

.lr_grad <- function(w, X, y, C, sw = NULL) {
  z <- as.numeric(X %*% w[-1] + w[1])
  z <- pmax(-500, pmin(500, z))
  p <- 1 / (1 + exp(-z))
  n <- length(y)
  if (!is.null(sw)) {
    sw_norm <- sw / sum(sw) * n
    err <- sw_norm * (p - y)
  } else {
    err <- p - y
  }
  g_bias <- mean(err)
  g_w    <- as.numeric(t(X) %*% err) / n + w[-1] / (C * n)
  c(g_bias, g_w)
}

# ── Fit one binary classifier ─────────────────────────────────────────────────

.lr_fit_binary <- function(X, y, C = 1.0, sw = NULL) {
  n_feat <- ncol(X)
  w0     <- numeric(n_feat + 1L)  # bias + weights, init at 0

  opt <- stats::optim(
    par     = w0,
    fn      = .lr_loss,
    gr      = .lr_grad,
    X = X, y = y, C = C, sw = sw,
    method  = "L-BFGS-B",
    control = list(maxit = 1000L, factr = 1e7)
  )
  list(coef = opt$par)
}

# ── Multiclass fit (OvR) ──────────────────────────────────────────────────────

.lr_fit <- function(X, y_int, n_classes, C = 1.0, sample_weight = NULL) {
  # y_int: integer vector 0-based class indices
  Xm <- as.matrix(X)
  if (n_classes == 2L) {
    y_bin  <- as.numeric(y_int == 1L)
    models <- list(.lr_fit_binary(Xm, y_bin, C = C, sw = sample_weight))
  } else {
    models <- lapply(seq_len(n_classes) - 1L, function(k) {
      y_bin <- as.numeric(y_int == k)
      .lr_fit_binary(Xm, y_bin, C = C, sw = sample_weight)
    })
  }
  list(models = models, n_classes = n_classes, C = C)
}

# ── Predict (0-based integer class indices) ───────────────────────────────────

.lr_predict <- function(fit, X) {
  proba <- .lr_proba(fit, X)
  # which.max returns 1-based → subtract 1 for 0-based
  apply(proba, 1, which.max) - 1L
}

# ── Predict probabilities (n x k data.frame) ──────────────────────────────────

.lr_proba <- function(fit, X) {
  Xm <- as.matrix(X)
  n  <- nrow(Xm)
  k  <- fit$n_classes

  if (k == 2L) {
    w  <- fit$models[[1L]]$coef
    z  <- as.numeric(Xm %*% w[-1] + w[1])
    z  <- pmax(-500, pmin(500, z))
    p1 <- 1 / (1 + exp(-z))
    proba <- cbind(1 - p1, p1)
  } else {
    proba <- matrix(0, nrow = n, ncol = k)
    for (ki in seq_len(k)) {
      w  <- fit$models[[ki]]$coef
      z  <- as.numeric(Xm %*% w[-1] + w[1])
      z  <- pmax(-500, pmin(500, z))
      proba[, ki] <- 1 / (1 + exp(-z))
    }
    # Normalise rows to sum 1 (OvR doesn't guarantee this)
    row_s <- rowSums(proba)
    row_s[row_s == 0] <- 1
    proba <- proba / row_s
  }

  as.data.frame(proba)
}
