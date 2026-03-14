#' The ml module -- all verbs accessed via ml$verb()
#'
#' Provides the module-style interface `ml$verb()` as an alternative to the
#' standard `ml_verb()` function style. Both styles are equivalent and call
#' the same underlying implementation.
#'
#' Note: `ml$fit(...)` and `ml_fit(...)` produce identical results.
#'
#' @format A locked environment with 28 verb entries.
#' @examples
#' s <- ml$split(iris, "Species", seed = 42)
#' model <- ml$fit(s$train, "Species", seed = 42)
#' ml$evaluate(model, s$valid)
#' @export
ml <- local({
  .module <- new.env(parent = emptyenv())
  .module$config        <- ml_config
  .module$split         <- .split_impl
  .module$fit           <- .fit_impl
  .module$predict       <- .predict_impl
  .module$predict_proba <- .predict_proba_impl
  .module$evaluate      <- function(model, data) .evaluate_impl(model, data, .guard = TRUE)
  .module$assess        <- .assess_impl
  .module$explain       <- .explain_impl
  .module$save          <- .save_impl
  .module$load          <- .load_impl
  .module$profile       <- .profile_impl
  .module$screen        <- .screen_impl
  .module$compare       <- .compare_impl
  .module$tune          <- .tune_impl
  .module$stack         <- .stack_impl
  .module$validate      <- .validate_impl
  .module$algorithms    <- .algorithms_impl
  .module$dataset       <- .dataset_impl
  .module$embed         <- .embed_impl
  .module$drift         <- .drift_impl
  .module$shelf         <- .shelf_impl
  .module$calibrate     <- ml_calibrate
  .module$enough        <- ml_enough
  .module$plot          <- ml_plot
  .module$report        <- ml_report
  .module$check         <- ml_check
  .module$check_data    <- ml_check_data
  .module$leak          <- ml_leak
  .module$prepare       <- ml_prepare
  .module$split_group   <- ml_split_group
  .module$split_temporal <- ml_split_temporal
  lockEnvironment(.module, bindings = FALSE)
  .module
})
