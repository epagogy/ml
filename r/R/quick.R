#' One-call workflow: split + screen + fit + evaluate
#'
#' The fastest path from raw data to a trained, evaluated model.
#' Screens logistic, random_forest, and xgboost, picks the best,
#' fits on training data, and evaluates on validation.
#'
#' @param data A data.frame with features and target
#' @param target Target column name
#' @param seed Random seed
#' @returns A list with \code{model} (ml_model), \code{metrics} (ml_metrics),
#'   and \code{split} (ml_split_result).
#' @export
#' @examples
#' result <- ml_quick(iris, "Species", seed = 42)
#' result$model
#' result$metrics
ml_quick <- function(data, target, seed) {
  s <- ml_split(data, target, seed = seed)

  # Screen 3 core algorithms
  algos <- c("logistic", "random_forest", "xgboost")
  avail <- .available_algorithms()
  algos <- intersect(algos, avail)

  lb <- ml_screen(s, target, seed = seed, algorithms = algos)
  best_model <- ml_best(lb)

  if (is.null(best_model)) {
    # Fallback: fit default
    best_model <- ml_fit(data = s$train, target = target, seed = seed)
  }

  metrics <- ml_evaluate(best_model, s$valid)

  list(
    model   = best_model,
    metrics = metrics,
    split   = s
  )
}
