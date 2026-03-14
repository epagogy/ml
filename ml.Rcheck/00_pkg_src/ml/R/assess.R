#' Assess model on held-out test data (do once)
#'
#' The final exam -- separate from [ml_evaluate()] to force a conscious choice.
#' Errors if called more than once on the same model. Use `s$test` (not
#' `s$valid`) for the test data.
#'
#' @param model An `ml_model`
#' @param test Test data.frame (use `s$test`). Specify by name for clarity.
#' @returns An object of class `ml_evidence` (sealed -- not substitutable for `ml_metrics`)
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' verdict <- ml_assess(model, test = s$test)
ml_assess <- function(model, test) {
  .assess_impl(model = model, test = test)
}

.assess_impl <- function(model, test) {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")

  # Partition guard -- reject unsplit or wrong-partition data (fires BEFORE counter increment)
  if (.guards_active()) {
    .part <- attr(test, "_ml_partition")
    if (is.null(.part)) {
      partition_error(paste0(
        "ml_assess() received data without split provenance. ",
        "Split your data first: s <- ml_split(df, target, seed = 42), ",
        "then ml_assess(model, test = s$test). ",
        "Note: dplyr verbs (filter, mutate, select) strip partition tags. ",
        "Use base R subsetting or re-split after dplyr operations. ",
        "To disable: ml_config(guards = 'off')"
      ))
    } else if (.part != "test") {
      partition_error(paste0(
        "ml_assess() received data tagged as '", .part, "' partition. ",
        "ml_assess() requires test data (s$test). ",
        "For validation iterations, use ml_evaluate(model, s$valid)."
      ))
    }
  }

  # Increment assess counter FIRST, then check (Python delta P1 #8 -- increment-then-check)
  .inc_assess_count(model)
  count <- .get_assess_count(model)
  if (count > 1L) {
    cli::cli_abort(paste0(
      "ml_assess() called ", count,
      " times on same model. Repeated peeking at test data inflates apparent performance."
    ))
  }

  result <- .evaluate_impl(model, test, .guard = FALSE)
  # Wrap in sealed ml_evidence -- not substitutable for ml_metrics (Codd condition 7)
  new_ml_evidence(unclass(result), task = attr(result, "task"), time = attr(result, "time"))
}
