#' Detect data drift between reference and new data
#'
#' Compares a reference dataset (typically training data) to new data using
#' per-feature statistical tests or adversarial validation.
#'
#' **Statistical method** (default): per-feature distribution tests with no
#' labels required.
#' - Numeric features: Kolmogorov-Smirnov two-sample test
#' - Categorical features: Chi-squared test on value counts
#'
#' **Adversarial method**: trains a binary classifier to distinguish reference
#' from new data. AUC near 0.5 means similar distributions; AUC near 1.0 means
#' very different distributions.
#' - `$train_scores`: per-row probability of "looks like new data" for reference
#'   rows. Use `sort(result$train_scores, decreasing = TRUE)[1:n]` to select
#'   validation rows that mirror the new distribution.
#' - `$features`: most discriminative features (temporal leakage candidates)
#'
#' Pair with [ml_shelf()] for complete monitoring: drift() detects input
#' distribution shift (label-free), shelf() detects performance degradation
#' (requires labels).
#'
#' @param reference A data.frame — reference dataset (typically training data)
#' @param new A data.frame — new data to compare against the reference
#' @param method Detection method: "statistical" (default) or "adversarial"
#' @param threshold p-value threshold for statistical method (default 0.05)
#' @param exclude Character vector of column names to skip (e.g., ID columns)
#' @param target Target column name — automatically excluded from drift analysis
#' @param seed Random seed (required for method = "adversarial")
#' @param algorithm Algorithm for adversarial classifier: "random_forest"
#'   (default) or "xgboost"
#' @returns An object of class `ml_drift_result` with:
#'   - `$shifted`: TRUE if drift detected
#'   - `$features`: named numeric — p-values (statistical) or importances (adversarial)
#'   - `$features_shifted`: character vector of drifted feature names
#'   - `$severity`: "none", "low", "medium", or "high"
#'   - `$auc`: adversarial mode only — classifier AUC
#'   - `$train_scores`: adversarial mode only — per-row reference probabilities
#' @export
#' @examples
#' s    <- ml_split(iris, "Species", seed = 42)
#' # Simulate drift by perturbing test data
#' new  <- s$test
#' new$Sepal.Length <- new$Sepal.Length + 2
#' result <- ml_drift(reference = s$train, new = new, target = "Species")
#' result$shifted
#' result$features_shifted
ml_drift <- function(reference, new, method = "statistical", threshold = 0.05,
                     exclude = NULL, target = NULL, seed = NULL,
                     algorithm = "random_forest") {
  .drift_impl(reference = reference, new = new, method = method,
              threshold = threshold, exclude = exclude, target = target,
              seed = seed, algorithm = algorithm)
}

.drift_impl <- function(reference, new, method = "statistical", threshold = 0.05,
                        exclude = NULL, target = NULL, seed = NULL,
                        algorithm = "random_forest") {
  reference <- .coerce_data(reference)
  new       <- .coerce_data(new)

  if (!method %in% c("statistical", "adversarial")) {
    config_error(paste0("method='", method, "' not recognized. Choose from: statistical, adversarial"))
  }
  if (nrow(reference) < 5L) data_error("reference must have at least 5 rows")
  if (nrow(new)       < 5L) data_error("new must have at least 5 rows")

  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # Columns to test: shared columns, minus target, minus excludes
  shared <- intersect(names(reference), names(new))
  skip   <- c(exclude, target)
  cols   <- setdiff(shared, skip)

  if (length(cols) == 0L) {
    config_error("No columns left to test after excluding target and exclude columns")
  }

  if (method == "statistical") {
    return(.drift_statistical(reference, new, cols, threshold))
  } else {
    return(.drift_adversarial(reference, new, cols, seed, algorithm, threshold))
  }
}

# ── Statistical method ─────────────────────────────────────────────────────────

.drift_statistical <- function(reference, new, cols, threshold) {
  p_values     <- numeric(length(cols))
  test_types   <- character(length(cols))
  names(p_values)   <- cols
  names(test_types) <- cols

  for (col in cols) {
    ref_col <- reference[[col]]
    new_col <- new[[col]]

    if (is.numeric(ref_col) && is.numeric(new_col)) {
      # KS two-sample test for numeric features
      res <- tryCatch({
        r <- stats::ks.test(ref_col[!is.na(ref_col)], new_col[!is.na(new_col)])
        list(p = r$p.value, type = "ks")
      }, error = function(e) list(p = 1.0, type = "ks_failed"))
      p_values[[col]]   <- res$p
      test_types[[col]] <- res$type
    } else {
      # Chi-squared test for categorical features
      res <- tryCatch({
        ref_tab <- table(factor(ref_col))
        new_tab <- table(factor(new_col, levels = names(ref_tab)))
        # Suppress warnings about low expected counts
        r <- suppressWarnings(stats::chisq.test(
          rbind(as.numeric(ref_tab), as.numeric(new_tab))
        ))
        list(p = r$p.value, type = "chisq")
      }, error = function(e) list(p = 1.0, type = "chisq_failed"))
      p_values[[col]]   <- res$p
      test_types[[col]] <- res$type
    }
  }

  # Identify shifted features — sorted alphabetically (matches Python sorted())
  shifted_feats <- sort(names(p_values)[!is.na(p_values) & p_values < threshold])
  frac_shifted  <- length(shifted_feats) / length(cols)

  # Severity thresholds match Python: <0.1 low, <0.3 medium, else high
  severity <- if      (frac_shifted == 0)      "none"
              else if (frac_shifted < 0.1)      "low"
              else if (frac_shifted < 0.3)      "medium"
              else                              "high"

  new_ml_drift_result(
    shifted          = length(shifted_feats) > 0L,
    features         = p_values,
    features_shifted = shifted_feats,
    severity         = severity,
    n_reference      = nrow(reference),
    n_new            = nrow(new),
    threshold        = threshold,
    feature_tests    = test_types,
    auc              = NULL,
    distinguishable  = length(shifted_feats) > 0L,
    train_scores     = NULL
  )
}

# ── Adversarial method ─────────────────────────────────────────────────────────

.drift_adversarial <- function(reference, new, cols, seed, algorithm, threshold) {
  if (!requireNamespace("ranger", quietly = TRUE) && algorithm == "random_forest") {
    config_error("'ranger' required for adversarial drift. Install with: install.packages('ranger')")
  }

  # Build combined dataset: reference = class 0, new = class 1
  ref_sub <- reference[, cols, drop = FALSE]
  new_sub <- new[, cols, drop = FALSE]
  ref_sub[[".label"]] <- 0L
  new_sub[[".label"]] <- 1L

  combined <- rbind(ref_sub, new_sub)

  # Simple ordinal encoding for categoricals (adversarial just needs numeric)
  for (col in cols) {
    v <- combined[[col]]
    if (!is.numeric(v)) {
      combined[[col]] <- as.integer(factor(v)) - 1L
    }
  }

  # 5-fold OOF CV to get unbiased AUC (matches Python implementation)
  n       <- nrow(combined)
  y       <- combined[[".label"]]
  X       <- combined[, cols, drop = FALSE]
  withr::local_seed(seed)
  fold_ids <- sample(rep(1:5, length.out = n))
  oof_proba <- rep(NA_real_, n)

  for (k in 1:5) {
    tr_idx  <- which(fold_ids != k)
    val_idx <- which(fold_ids == k)

    X_tr <- as.data.frame(X[tr_idx, , drop = FALSE])
    y_tr <- factor(y[tr_idx])
    X_val <- as.data.frame(X[val_idx, , drop = FALSE])

    tryCatch({
      if (algorithm == "random_forest" || algorithm == "random_forest") {
        eng <- ranger::ranger(
          x = X_tr, y = y_tr,
          num.trees  = 100L,
          probability = TRUE,
          seed       = seed + k,
          verbose    = FALSE
        )
        p <- predict(eng, data = X_val)$predictions[, 2]
      } else if (algorithm == "xgboost") {
        if (!requireNamespace("xgboost", quietly = TRUE)) next
        dm_tr  <- xgboost::xgb.DMatrix(as.matrix(X_tr), label = as.integer(as.character(y_tr)))
        dm_val <- xgboost::xgb.DMatrix(as.matrix(X_val))
        eng    <- xgboost::xgb.train(
          params  = list(objective = "binary:logistic", seed = seed, verbosity = 0L),
          data    = dm_tr, nrounds = 50L, verbose = 0L
        )
        p <- predict(eng, dm_val)
      } else {
        p <- rep(0.5, length(val_idx))
      }
      oof_proba[val_idx] <- p
    }, error = function(e) NULL)
  }

  # AUC of OOF predictions
  auc_val <- tryCatch(
    .roc_auc_binary(y, oof_proba),
    error = function(e) 0.5
  )
  if (is.na(auc_val)) auc_val <- 0.5

  # Feature importance from full model
  importances <- tryCatch({
    eng_full <- ranger::ranger(
      x = X, y = factor(y),
      num.trees   = 100L,
      probability  = TRUE,
      importance  = "impurity",
      seed        = seed,
      verbose     = FALSE
    )
    vi <- eng_full$variable.importance
    vi / max(vi, na.rm = TRUE)  # normalize to [0, 1]
  }, error = function(e) {
    rep(0, length(cols)) |> stats::setNames(cols)
  })

  # Top discriminative features (importance > 0.5 × max)
  threshold_imp   <- 0.5
  top_feats       <- names(importances)[importances > threshold_imp]
  if (length(top_feats) == 0L) top_feats <- names(importances)[order(importances, decreasing = TRUE)[1:3]]

  # Severity thresholds (Python: <0.55 none, <0.65 low, <0.80 medium, else high)
  severity <- if      (auc_val < 0.55) "none"
              else if (auc_val < 0.65) "low"
              else if (auc_val < 0.80) "medium"
              else                     "high"

  distinguishable <- auc_val > 0.6

  # train_scores: per-row probability for REFERENCE rows that they "look like new"
  n_ref        <- nrow(reference)
  ref_oof      <- oof_proba[seq_len(n_ref)]
  train_scores <- stats::setNames(ref_oof, rownames(reference))

  new_ml_drift_result(
    shifted          = distinguishable,
    features         = importances,
    features_shifted = top_feats,
    severity         = severity,
    n_reference      = nrow(reference),
    n_new            = nrow(new),
    threshold        = threshold,
    feature_tests    = stats::setNames(rep("adversarial", length(cols)), cols),
    auc              = auc_val,
    distinguishable  = distinguishable,
    train_scores     = train_scores
  )
}

# ── S3 type ────────────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_drift_result <- function(shifted, features, features_shifted, severity,
                                 n_reference, n_new, threshold, feature_tests,
                                 auc, distinguishable, train_scores) {
  structure(
    list(
      shifted          = shifted,
      features         = features,
      features_shifted = features_shifted,
      severity         = severity,
      n_reference      = n_reference,
      n_new            = n_new,
      threshold        = threshold,
      feature_tests    = feature_tests,
      auc              = auc,
      distinguishable  = distinguishable,
      train_scores     = train_scores
    ),
    class = "ml_drift_result"
  )
}

#' Print ml_drift_result
#' @param x An ml_drift_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_drift_result <- function(x, ...) {
  status <- if (x[["shifted"]]) "DRIFT DETECTED" else "STABLE"
  cat(sprintf("-- Drift [%s] --\n", status))
  cat(sprintf("  severity  : %s\n", x[["severity"]]))
  cat(sprintf("  reference : %d rows\n", x[["n_reference"]]))
  cat(sprintf("  new       : %d rows\n", x[["n_new"]]))
  n_shifted <- length(x[["features_shifted"]])
  n_total   <- length(x[["features"]])
  cat(sprintf("  shifted   : %d/%d features\n", n_shifted, n_total))
  if (!is.null(x[["auc"]])) {
    cat(sprintf("  auc       : %.4f\n", x[["auc"]]))
  }
  if (n_shifted > 0L && n_shifted <= 5L) {
    cat(sprintf("  features  : %s\n", paste(x[["features_shifted"]], collapse = ", ")))
  }
  cat("\n")
  invisible(x)
}
