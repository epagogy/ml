#' Configure ml package settings
#'
#' Set global configuration for the ml package. Currently supports `guards`
#' to control partition enforcement.
#'
#' @param guards Character: `"strict"` (default) enforces split provenance --
#'   all verbs reject data not produced by [ml_split()]. `"off"` disables
#'   guards for exploration/education.
#' @returns Invisibly returns the previous settings as a list.
#' @export
#' @examples
#' ml_config(guards = "off")    # disable guards
#' ml_config(guards = "strict") # re-enable (default)
ml_config <- function(guards = NULL) {
  prev <- list(guards = .guards_mode())
  if (!is.null(guards)) {
    guards <- match.arg(guards, c("strict", "off"))
    options(ml.guards = guards)
  }
  invisible(prev)
}

# Internal: get current guards mode (default = "strict")
.guards_mode <- function() {
  getOption("ml.guards", default = "strict")
}

# Internal: are guards active?
.guards_active <- function() {
  .guards_mode() == "strict"
}
