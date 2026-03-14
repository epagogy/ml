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
