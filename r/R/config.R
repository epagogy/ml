#' Configure ml package settings
#'
#' Set global configuration for the ml package. Currently supports `guards`
#' to control partition enforcement.
#'
#' @param guards Character: `"strict"` (default) enforces split provenance —
#'   all verbs reject data not produced by [ml_split()]. `"warn"` issues
#'   warnings instead of errors (useful for migration). `"off"` disables
#'   guards for exploration/education.
#' @returns Invisibly returns the previous settings as a list.
#' @export
#' @examples
#' ml_config(guards = "off")    # disable guards
#' ml_config(guards = "warn")   # warn instead of error
#' ml_config(guards = "strict") # re-enable (default)
ml_config <- function(guards = NULL) {
  prev <- list(guards = .guards_mode())
  if (!is.null(guards)) {
    guards <- match.arg(guards, c("strict", "warn", "off"))
    options(ml.guards = guards)
  }
  invisible(prev)
}

# Internal: get current guards mode (default = "strict")
.guards_mode <- function() {
  getOption("ml.guards", default = "strict")
}

# Internal: are guards active? (TRUE for "strict" or "warn")
.guards_active <- function() {
  .guards_mode() != "off"
}

# Internal: raise or warn based on guards mode (mirrors Python _guard_action)
.guard_action <- function(message) {
  mode <- .guards_mode()
  if (mode == "off") return(invisible(NULL))
  if (mode == "warn") {
    warning(message, call. = FALSE)
    return(invisible(NULL))
  }
  partition_error(message)
}
