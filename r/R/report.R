#' Generate an HTML training report
#'
#' Produces a self-contained HTML report with model metadata, evaluation
#' metrics, and feature importances. Open in any browser.
#'
#' @param model An `ml_model` from `ml_fit()`
#' @param data A data.frame for computing metrics (use validation data)
#' @param path Output file path. Default: `"model_report.html"`
#' @returns The path to the saved report (invisibly)
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
#' tmp <- tempfile(fileext = ".html")
#' ml_report(model, data = s$valid, path = tmp)
#' unlink(tmp)
ml_report <- function(model, data = NULL, path = "model_report.html") {
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) {
    model_error("Expected an ml_model object")
  }

  # Metrics
  metrics_html <- ""
  if (!is.null(data)) {
    m <- as.list(ml_evaluate(model, data))
    rows <- paste0(
      sapply(names(m), function(k) {
        sprintf("<tr><td>%s</td><td><strong>%.4f</strong></td></tr>", k, m[[k]])
      }),
      collapse = "\n"
    )
    metrics_html <- sprintf(
      "<h2>Metrics</h2><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>%s</tbody></table>",
      rows
    )
  }

  # Feature importance
  imp_html <- ""
  imp <- tryCatch(ml_explain(model), error = function(e) NULL)
  if (!is.null(imp)) {
    top_n <- min(20L, nrow(imp))
    rows  <- paste0(
      sapply(seq_len(top_n), function(i) {
        pct <- round(imp$importance[i] * 100, 1)
        bar <- paste0(rep("\u2588", max(1L, round(pct / 2))), collapse = "")
        sprintf("<tr><td>%s</td><td><code>%s</code></td><td>%.1f%%</td></tr>",
                imp$feature[i], bar, pct)
      }),
      collapse = "\n"
    )
    imp_html <- sprintf(
      "<h2>Feature Importance (top %d)</h2><table><thead><tr><th>Feature</th><th></th><th>Importance</th></tr></thead><tbody>%s</tbody></table>",
      top_n, rows
    )
  }

  # Confusion matrix (classification only)
  cm_html <- ""
  if (!is.null(data) && model$task == "classification") {
    preds  <- predict(model, newdata = data)
    actual <- data[[model$target]]
    cm     <- table(Actual = actual, Predicted = preds)
    header <- paste0("<th></th>",
                     paste0("<th>Pred: ", colnames(cm), "</th>", collapse = ""))
    body_rows <- paste0(
      sapply(rownames(cm), function(r) {
        cells <- paste0("<td>", cm[r, ], "</td>", collapse = "")
        sprintf("<tr><th>Act: %s</th>%s</tr>", r, cells)
      }),
      collapse = "\n"
    )
    cm_html <- sprintf(
      "<h2>Confusion Matrix</h2><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>",
      header, body_rows
    )
  }

  # Assemble HTML
  ts   <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  html <- sprintf('<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ml report -- %s</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 860px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1   { border-bottom: 2px solid #4f46e5; padding-bottom: 8px; }
  h2   { color: #4f46e5; margin-top: 32px; }
  table { border-collapse: collapse; width: 100%%; margin-top: 8px; }
  th, td { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }
  thead th { background: #f9fafb; font-weight: 600; }
  tr:nth-child(even) { background: #f9fafb; }
  code { font-family: monospace; color: #4f46e5; }
  .meta { color: #6b7280; font-size: 0.9em; }
</style>
</head>
<body>
<h1>ml model report</h1>
<p class="meta">Generated: %s</p>
<h2>Model</h2>
<table>
  <tr><th>Algorithm</th><td>%s</td></tr>
  <tr><th>Task</th><td>%s</td></tr>
  <tr><th>Target</th><td>%s</td></tr>
  <tr><th>Features</th><td>%d</td></tr>
  <tr><th>Seed</th><td>%s</td></tr>
</table>
%s
%s
%s
</body>
</html>',
    model$algorithm, ts,
    model$algorithm, model$task, model$target,
    length(model$features),
    if (is.null(model$seed)) "auto" else as.character(model$seed),
    metrics_html, imp_html, cm_html
  )

  writeLines(html, path)
  message(sprintf("Saved: %s", normalizePath(path)))
  invisible(normalizePath(path))
}
