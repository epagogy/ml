#' Verify provenance integrity of a model
#'
#' Checks provenance chain: split parameters -> training fingerprint ->
#' assess ceremony status. Catches accidental self-deception (load-assess
#' loops, test-set shopping) rather than adversarial tampering.
#'
#' @param model An `ml_model`, `ml_tuning_result`, or path to `.mlr` file
#' @returns A list with `status` ("verified"/"unverified"/"warning"),
#'   `checks`, `provenance`, and `assess_count`.
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' report <- ml_verify(model)
#' report$status
ml_verify <- function(model) {
  if (is.character(model)) {
    model <- ml_load(model)
  }
  if (inherits(model, "ml_tuning_result")) {
    model <- model[["best_model"]]
  }
  if (!inherits(model, "ml_model")) {
    return(list(
      status = "unverified",
      checks = list(list(check = "type", ok = FALSE,
                         detail = paste0("Expected ml_model, got ", class(model)[[1]]))),
      provenance = NULL,
      assess_count = 0L
    ))
  }

  provenance <- model[[".provenance"]]
  assess_count <- .get_assess_count(model)
  checks <- list()

  # Check 1: Provenance exists
  has_prov <- !is.null(provenance) && !is.null(provenance[["train_fingerprint"]])
  checks[[length(checks) + 1L]] <- list(
    check = "provenance_exists",
    ok = has_prov,
    detail = if (has_prov) "Training fingerprint recorded"
             else "No provenance -- model was not trained through ml_split -> ml_fit pipeline"
  )

  # Check 2: Split receipt
  has_receipt <- !is.null(provenance[["split_receipt"]])
  checks[[length(checks) + 1L]] <- list(
    check = "split_receipt",
    ok = has_receipt,
    detail = if (has_receipt) paste0("Split receipt: ", provenance[["split_receipt"]])
             else "No split receipt"
  )

  # Check 3: Split lineage
  has_lineage <- !is.null(provenance[["split_id"]])
  checks[[length(checks) + 1L]] <- list(
    check = "split_lineage",
    ok = has_lineage,
    detail = if (has_lineage) paste0("Split ID: ", provenance[["split_id"]])
             else "No split lineage recorded"
  )

  # Check 4: Assess ceremony
  if (assess_count == 0L) {
    assess_ok <- TRUE
    assess_detail <- "Not yet assessed -- test data untouched"
  } else if (assess_count == 1L) {
    assess_ok <- TRUE
    assess_detail <- "Assessed once -- ceremony completed correctly"
  } else {
    assess_ok <- FALSE
    assess_detail <- paste0("Assessed ", assess_count, " times -- one-shot ceremony violated")
  }
  checks[[length(checks) + 1L]] <- list(
    check = "assess_ceremony",
    ok = assess_ok,
    detail = assess_detail
  )

  # Overall status
  all_ok <- all(vapply(checks, function(c) c$ok, logical(1)))
  any_fail <- any(!vapply(checks, function(c) c$ok, logical(1)))
  status <- if (all_ok) "verified"
            else if (any_fail && has_prov) "warning"
            else "unverified"

  list(
    status = status,
    checks = checks,
    provenance = if (has_prov) provenance else NULL,
    assess_count = assess_count
  )
}
