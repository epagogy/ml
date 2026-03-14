#' Save a model to disk
#'
#' Saves an `ml_model` or `ml_tuning_result` to a `.mlr` file using
#' \code{\link[base]{saveRDS}} with a version wrapper.
#'
#' @section Security:
#' \code{ml_load()} uses \code{readRDS()} internally, which can execute
#' arbitrary R code during deserialization. Never load `.mlr` files
#' from untrusted sources.
#'
#' @param model An `ml_model` or `ml_tuning_result`
#' @param path File path (recommended extension: `.mlr`)
#' @returns The normalized path, invisibly.
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' path <- file.path(tempdir(), "iris_model.mlr")
#' ml_save(model, path)
#' loaded <- ml_load(path)
#' }
ml_save <- function(model, path) {
  .save_impl(model = model, path = path)
}

.save_impl <- function(model, path) {
  if (!inherits(model, c("ml_model", "ml_tuning_result"))) {
    model_error("model must be an ml_model or ml_tuning_result")
  }
  # Normalize path (warn on path traversal patterns)
  path <- normalizePath(path, mustWork = FALSE)
  if (grepl("\\.\\.", path)) {
    cli::cli_warn("Path contains '..'. Verify this is intentional.")
  }

  wrapper <- list(
    version   = as.character(utils::packageVersion("ml")),
    type      = class(model)[[1]],
    model     = model,
    timestamp = Sys.time()
  )
  saveRDS(wrapper, file = path)
  invisible(path)
}

#' Load a model from disk
#'
#' @param path Path to a `.mlr` file saved with [ml_save()]
#' @returns An `ml_model` or `ml_tuning_result`
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' path <- file.path(tempdir(), "iris_model.mlr")
#' ml_save(model, path)
#' loaded <- ml_load(path)
#' loaded$algorithm
#' }
ml_load <- function(path) {
  .load_impl(path = path)
}

.load_impl <- function(path) {
  if (!file.exists(path)) {
    stop(paste0("File not found: ", path), call. = FALSE)
  }
  path    <- normalizePath(path, mustWork = TRUE)
  wrapper <- readRDS(path)

  if (!is.list(wrapper) || !"version" %in% names(wrapper)) {
    model_error("File does not appear to be a valid .mlr file saved by ml_save()")
  }

  saved_ver   <- tryCatch(
    numeric_version(wrapper[["version"]]),
    error = function(e) numeric_version("0.0.0")
  )
  current_ver <- utils::packageVersion("ml")

  if (saved_ver[[c(1, 1)]] != current_ver[[c(1, 1)]]) {
    version_error(paste0(
      "Model saved with ml ", wrapper[["version"]],
      ", current version is ", as.character(current_ver),
      ". Major version mismatch -- cannot load."
    ))
  }

  wrapper[["model"]]
}
