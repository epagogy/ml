#' @keywords internal
ml_error <- function(message, class = "ml_error", ...) {
  rlang::abort(message, class = c(class, "ml_error"), ...)
}

#' @keywords internal
config_error <- function(message, ...) {
  ml_error(message, class = c("config_error", "ml_error"), ...)
}

#' @keywords internal
data_error <- function(message, ...) {
  ml_error(message, class = c("data_error", "ml_error"), ...)
}

#' @keywords internal
partition_error <- function(message, ...) {
  ml_error(message, class = c("partition_error", "data_error", "ml_error"), ...)
}

#' @keywords internal
model_error <- function(message, ...) {
  ml_error(message, class = c("model_error", "ml_error"), ...)
}

#' @keywords internal
version_error <- function(message, ...) {
  ml_error(message, class = c("version_error", "ml_error"), ...)
}

# Save global RNG state, set seed, restore on function exit.
# Must be called directly in a function body (not inside lapply/tryCatch).
.local_seed <- function(seed, envir = parent.frame()) {
  old_seed <- get0(".Random.seed", envir = globalenv())
  set.seed(seed)
  do.call("on.exit", list(substitute(
    if (is.null(OLD)) {
      suppressWarnings(rm(".Random.seed", envir = globalenv()))
    } else {
      assign(".Random.seed", OLD, envir = globalenv())
    },
    list(OLD = old_seed)
  ), add = TRUE), envir = envir)
  invisible()
}
