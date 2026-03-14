#' Validate model against rules and/or baseline
#'
#' Three modes: (1) absolute rules, (2) regression prevention vs baseline,
#' (3) combined. Returns a structured result with pass/fail and diagnostics.
#'
#' Tolerance is absolute (not relative): a tolerance of 0.02 means 2 percentage
#' points of allowed degradation, applied uniformly across all metrics.
#'
#' @param model An `ml_model`
#' @param test Test data.frame (use `s$test`)
#' @param rules Named list of threshold strings, e.g.
#'   `list(accuracy = ">0.85", roc_auc = ">=0.90")`
#' @param baseline An `ml_model` — previous model to check for regressions
#' @param tolerance Numeric. Allowed absolute degradation (0.02 = 2pp slack).
#'   Default 0.0.
#' @returns An object of class `ml_validate_result`
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' gate <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.80"))
#' gate$passed
#' }
ml_validate <- function(model, test, rules = NULL, baseline = NULL, tolerance = 0.0) {
  .validate_impl(model = model, test = test, rules = rules,
                 baseline = baseline, tolerance = tolerance)
}

# Lower-is-better metrics (degradation = increase)
.LOWER_IS_BETTER <- c("rmse", "mae", "mse", "log_loss")

.validate_impl <- function(model, test, rules = NULL, baseline = NULL, tolerance = 0.0) {
  # Must have at least one validation mode
  if (is.null(rules) && is.null(baseline)) {
    config_error(paste0(
      "ml_validate() requires rules= and/or baseline=.\n",
      "  Use: ml_validate(model, test = data, rules = list(accuracy = '>0.85'))\n",
      "  or:  ml_validate(model, test = data, baseline = old_model)"
    ))
  }

  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")

  # Partition guard — reject unsplit or wrong-partition data
  if (.guards_active()) {
    .part <- .resolve_partition(test)
    if (is.null(.part)) {
      .guard_action(paste0(
        "ml_validate() received data without split provenance. ",
        "Split your data first: s <- ml_split(df, target, seed = 42), ",
        "then ml_validate(model, test = s$test, rules = ...). ",
        "To disable: ml_config(guards = 'off')"
      ))
    } else if (.part != "test") {
      .guard_action(paste0(
        "ml_validate() received data tagged as '", .part, "' partition. ",
        "ml_validate() requires test data (s$test). ",
        "For validation iterations, use ml_evaluate(model, s$valid)."
      ))
    }
  }

  # Baseline target mismatch check
  if (!is.null(baseline)) {
    if (inherits(baseline, "ml_tuning_result")) baseline <- baseline[["best_model"]]
    if (model[["target"]] != baseline[["target"]]) {
      config_error(paste0(
        "model target='", model[["target"]],
        "' != baseline target='", baseline[["target"]],
        "'. Both must predict the same target."
      ))
    }
  }

  test <- .coerce_data(test)

  # Compute metrics for both model and baseline
  metrics_obj  <- .score_impl(model, test)
  metrics      <- as.list(unclass(metrics_obj))

  baseline_metrics <- NULL
  if (!is.null(baseline)) {
    bl_obj       <- .score_impl(baseline, test)
    baseline_metrics <- as.list(unclass(bl_obj))
  }

  failures     <- character(0)
  improvements <- character(0)
  degradations <- character(0)

  # --- Absolute rules ---------------------------------------------------
  if (!is.list(rules) && !is.null(rules)) {
    config_error("rules must be a named list, e.g. list(accuracy = '>0.85')")
  }
  if (!is.null(rules)) {
    for (metric_name in names(rules)) {
      rule_str <- rules[[metric_name]]
      actual   <- metrics[[metric_name]]

      if (is.null(actual)) {
        failures <- c(failures, paste0(
          "Rule '", metric_name, " ", rule_str,
          "' FAILED: metric not computed for this model"
        ))
        next
      }

      parsed <- .parse_rule(rule_str)
      if (is.null(parsed)) {
        config_error(paste0(
          "Invalid rule syntax: '", rule_str,
          "'. Valid operators: >, >=, <, <="
        ))
      }

      passes <- .eval_rule(actual, parsed$op, parsed$threshold)
      if (!passes) {
        failures <- c(failures, paste0(
          "Rule '", metric_name, " ", rule_str,
          "' FAILED: actual=", round(actual, 4)
        ))
      }
    }
  }

  # --- Regression prevention (baseline comparison) ----------------------
  if (!is.null(baseline)) {
    common_metrics <- intersect(names(metrics), names(baseline_metrics))
    for (nm in common_metrics) {
      new_val  <- metrics[[nm]]
      base_val <- baseline_metrics[[nm]]
      if (is.null(new_val) || is.null(base_val)) next
      if (is.na(new_val) || is.na(base_val)) next

      lower_better <- nm %in% .LOWER_IS_BETTER
      if (lower_better) {
        degradation <- new_val - base_val   # increase = degradation
      } else {
        degradation <- base_val - new_val   # decrease = degradation
      }

      diff_str <- sprintf("%+.4f", new_val - base_val)

      if (degradation > tolerance) {
        degradations <- c(degradations, paste0(
          nm, ": ", round(base_val, 4), " -> ", round(new_val, 4),
          " (", diff_str, ") exceeds tolerance ", tolerance
        ))
      } else if (degradation < -tolerance / 2) {
        # Improvement (using half-tolerance as threshold to avoid noise)
        improvements <- c(improvements, paste0(
          nm, ": ", round(base_val, 4), " -> ", round(new_val, 4), " (", diff_str, ")"
        ))
      }
    }

    failures <- c(failures, degradations)
  }

  passed <- length(failures) == 0L

  new_ml_validate_result(
    passed           = passed,
    metrics          = unlist(metrics),
    failures         = failures,
    baseline_metrics = if (!is.null(baseline_metrics)) unlist(baseline_metrics) else NULL,
    improvements     = improvements,
    degradations     = degradations
  )
}

# ── Rule parsing ───────────────────────────────────────────────────────────────

.parse_rule <- function(rule_str) {
  rule_str <- trimws(rule_str)
  patterns <- list(
    list(pat = "^>=\\s*(.+)$", op = ">="),
    list(pat = "^<=\\s*(.+)$", op = "<="),
    list(pat = "^>\\s*(.+)$",  op = ">"),
    list(pat = "^<\\s*(.+)$",  op = "<")
  )
  for (p in patterns) {
    m <- regmatches(rule_str, regexpr(p$pat, rule_str, perl = TRUE))
    if (length(m) > 0) {
      thresh <- as.numeric(sub(p$pat, "\\1", rule_str, perl = TRUE))
      if (!is.na(thresh)) return(list(op = p$op, threshold = thresh))
    }
  }
  NULL
}

.eval_rule <- function(actual, op, threshold) {
  switch(op,
    ">"  = actual > threshold,
    ">=" = actual >= threshold,
    "<"  = actual < threshold,
    "<=" = actual <= threshold,
    FALSE
  )
}
