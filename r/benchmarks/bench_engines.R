#!/usr/bin/env Rscript
# Engine x Algorithm Benchmark Suite (R).
#
# Measures speed, accuracy, memory, and parity across ml engines (Rust vs CRAN)
# for 9 algorithms decomposed into 4 primitives (Represent, Objective, Search, Compose).
#
# Usage:
#   Rscript benchmarks/bench_engines.R                       # tiny+small
#   Rscript benchmarks/bench_engines.R --medium              # +100K
#   Rscript benchmarks/bench_engines.R --large               # +1M
#   Rscript benchmarks/bench_engines.R --algorithm knn       # single algo
#   Rscript benchmarks/bench_engines.R --json --output engines_r.json

suppressPackageStartupMessages({
  library(ml)
  library(jsonlite)
})

`%||%` <- function(x, y) if (!is.null(x)) x else y

# ── 4-Primitive Taxonomy (locked) ─────────────────────────────────────────────

ALGO_PRIMITIVES <- list(
  random_forest = list(represent = "tree",          objective = "gini/mse",       search = "greedy",        compose = "bagging"),
  decision_tree = list(represent = "tree",          objective = "gini/mse",       search = "greedy",        compose = "none"),
  logistic      = list(represent = "linear",        objective = "cross-entropy+L2", search = "gradient",    compose = "none"),
  linear        = list(represent = "linear",        objective = "mse+L2",         search = "closed-form",   compose = "none"),
  knn           = list(represent = "instance",      objective = "none",           search = "exhaustive",    compose = "voting"),
  elastic_net   = list(represent = "linear",        objective = "mse+L1+L2",      search = "coord_descent", compose = "none"),
  naive_bayes   = list(represent = "probabilistic", objective = "log-likelihood", search = "closed-form",   compose = "none"),
  xgboost       = list(represent = "tree",          objective = "custom+L1+L2",   search = "gradient",      compose = "boosting"),
  svm           = list(represent = "kernel",        objective = "hinge",          search = "qp_solver",     compose = "none")
)

# ── Engine Matrix ─────────────────────────────────────────────────────────────

ENGINE_MATRIX <- list(
  random_forest = list(engines = c("ml", "cran"), tasks = c("classification", "regression")),
  decision_tree = list(engines = c("ml", "cran"), tasks = c("classification", "regression")),
  logistic      = list(engines = c("ml", "cran"), tasks = c("classification")),
  linear        = list(engines = c("ml", "cran"), tasks = c("regression")),
  knn           = list(engines = c("ml", "cran"), tasks = c("classification", "regression")),
  elastic_net   = list(engines = c("cran"),       tasks = c("regression")),
  naive_bayes   = list(engines = c("cran"),       tasks = c("classification")),
  xgboost       = list(engines = c("cran"),       tasks = c("classification", "regression"), optional = TRUE),
  svm               = list(engines = c("cran"),       tasks = c("classification", "regression")),
  gradient_boosting = list(engines = c("ml"),         tasks = c("classification", "regression"))
)

# ── Dataset Sizes ─────────────────────────────────────────────────────────────

SIZES <- list(
  tiny   = c(n_rows = 1000L,    n_features = 10L),
  small  = c(n_rows = 10000L,   n_features = 20L),
  medium = c(n_rows = 100000L,  n_features = 50L),
  large  = c(n_rows = 1000000L, n_features = 100L)
)

SIZE_RUNS <- list(
  tiny   = c(warmup = 3L, runs = 7L),
  small  = c(warmup = 3L, runs = 7L),
  medium = c(warmup = 1L, runs = 5L),
  large  = c(warmup = 0L, runs = 3L)
)

# ── Dataset Generation ────────────────────────────────────────────────────────

make_dataset <- function(task, n_rows, n_features, seed = 42L) {
  set.seed(seed)
  X <- matrix(rnorm(n_rows * n_features), nrow = n_rows, ncol = n_features)
  colnames(X) <- paste0("f", seq_len(n_features))
  df <- as.data.frame(X)

  if (task == "classification") {
    # Linear boundary + noise
    signal <- rowSums(X[, 1:min(5, n_features)])
    prob <- 1 / (1 + exp(-signal))
    df$target <- factor(ifelse(runif(n_rows) < prob, 1L, 0L))
  } else {
    # Linear combination + noise
    coefs <- rnorm(min(n_features %/% 2, 10))
    signal <- X[, seq_along(coefs)] %*% coefs
    df$target <- as.numeric(signal + rnorm(n_rows) * 0.5)
  }

  s <- ml_split(df, "target", seed = seed)
  list(df = df, split = s)
}

# ── Timed Run ─────────────────────────────────────────────────────────────────

run_timed <- function(fn, warmup = 3L, runs = 7L) {
  # Warmup
  for (i in seq_len(warmup)) {
    gc(verbose = FALSE)
    fn()
  }

  # Measured runs
  times <- numeric(runs)
  for (i in seq_len(runs)) {
    gc(verbose = FALSE)
    t0 <- proc.time()["elapsed"]
    fn()
    times[i] <- proc.time()["elapsed"] - t0
  }

  times <- sort(times)
  n <- length(times)
  q1_idx <- floor((n - 1) * 0.25) + 1L
  q3_idx <- floor((n - 1) * 0.75) + 1L

  list(
    median_seconds = round(median(times), 5),
    iqr_seconds    = round(times[q3_idx] - times[q1_idx], 5),
    min_seconds    = round(min(times), 5)
  )
}

# ── Quality Metrics ───────────────────────────────────────────────────────────

compute_quality <- function(model, valid_data, task) {
  metrics <- ml_evaluate(model, valid_data)
  if (task == "classification") {
    list(
      accuracy = metrics[["accuracy"]],
      roc_auc  = metrics[["roc_auc"]],
      f1       = metrics[["f1"]],
      rmse     = NULL,
      r2       = NULL
    )
  } else {
    list(
      accuracy = NULL,
      roc_auc  = NULL,
      f1       = NULL,
      rmse     = metrics[["rmse"]],
      r2       = metrics[["r2"]]
    )
  }
}

# ── One Benchmark Cell ────────────────────────────────────────────────────────

bench_one_cell <- function(algorithm, engine, task, size_name,
                            seed = 42L, split_result = NULL) {
  sz <- SIZES[[size_name]]
  n_rows <- sz[["n_rows"]]
  n_features <- sz[["n_features"]]
  sr <- SIZE_RUNS[[size_name]]
  warmup <- sr[["warmup"]]
  runs <- sr[["runs"]]
  prims <- ALGO_PRIMITIVES[[algorithm]]

  if (is.null(split_result)) {
    ds <- make_dataset(task, n_rows, n_features, seed = seed)
    split_result <- ds$split
  }

  # Map engine name: "ml" → Rust, "cran" → bypass Rust to CRAN packages
  # .fit_engine() only enables Rust when engine %in% c("auto", "ml").
  # Any other value falls through to the CRAN switch statement.
  engine_param <- if (engine == "ml") "ml" else if (engine == "cran") "cran" else "auto"

  # Fit timing
  model <- NULL
  fit_fn <- function() {
    suppressWarnings({
      model <<- ml_fit(split_result$train, "target",
                       algorithm = algorithm, seed = seed,
                       engine = engine_param)
    })
  }

  fit_stats <- tryCatch(
    run_timed(fit_fn, warmup = warmup, runs = runs),
    error = function(e) {
      return(list(median_seconds = NA, iqr_seconds = NA, min_seconds = NA,
                  error = conditionMessage(e)))
    }
  )

  if (is.null(model) || !is.null(fit_stats$error)) {
    return(skipped_cell(algorithm, engine, task, size_name, prims,
                        n_rows, n_features,
                        paste0("fit_error: ", fit_stats$error %||% "no model")))
  }

  # Predict timing
  predict_fn <- function() {
    predict(model, newdata = split_result$valid)
  }
  predict_stats <- tryCatch(
    run_timed(predict_fn, warmup = min(warmup, 2L), runs = min(runs, 5L)),
    error = function(e) list(median_seconds = -1, iqr_seconds = -1)
  )

  # Quality
  quality <- tryCatch(
    compute_quality(model, split_result$valid, task),
    error = function(e) list(accuracy = NULL, roc_auc = NULL, f1 = NULL,
                              rmse = NULL, r2 = NULL)
  )

  # Explain
  explain_works <- tryCatch(
    { ml_explain(model); TRUE },
    error = function(e) FALSE
  )

  # Throughput
  train_rows <- nrow(split_result$train)
  fit_rows_per_s <- if (!is.na(fit_stats$median_seconds) && fit_stats$median_seconds > 0)
    round(train_rows / fit_stats$median_seconds) else 0L

  c(
    list(
      algorithm       = algorithm,
      engine          = engine,
      task            = task,
      size            = size_name,
      n_rows          = n_rows,
      n_features      = n_features
    ),
    prims,
    list(
      fit_median_s    = fit_stats$median_seconds,
      fit_iqr_s       = fit_stats$iqr_seconds,
      fit_rows_per_s  = fit_rows_per_s,
      predict_median_s = predict_stats$median_seconds,
      predict_iqr_s   = predict_stats$iqr_seconds,
      accuracy        = quality$accuracy,
      roc_auc         = quality$roc_auc,
      f1              = quality$f1,
      rmse            = quality$rmse,
      r2              = quality$r2,
      explain_works   = explain_works,
      skipped         = FALSE,
      skip_reason     = NULL
    )
  )
}

skipped_cell <- function(algorithm, engine, task, size_name, prims,
                          n_rows, n_features, reason) {
  c(
    list(algorithm = algorithm, engine = engine, task = task,
         size = size_name, n_rows = n_rows, n_features = n_features),
    prims,
    list(fit_median_s = NULL, fit_iqr_s = NULL, fit_rows_per_s = NULL,
         predict_median_s = NULL, predict_iqr_s = NULL,
         accuracy = NULL, roc_auc = NULL, f1 = NULL,
         rmse = NULL, r2 = NULL,
         explain_works = NULL,
         skipped = TRUE, skip_reason = reason)
  )
}

# ── Optional Dependency Check ────────────────────────────────────────────────

optional_available <- function(algorithm) {
  pkgs <- list(xgboost = "xgboost")
  pkg <- pkgs[[algorithm]]
  if (is.null(pkg)) return(TRUE)
  requireNamespace(pkg, quietly = TRUE)
}

# ── Main Benchmark Loop ──────────────────────────────────────────────────────

run_engine_benchmark <- function(sizes, algorithms = NULL, json_only = FALSE) {
  rayon <- Sys.getenv("RAYON_NUM_THREADS", "not_set")
  if (rayon == "not_set") {
    Sys.setenv(RAYON_NUM_THREADS = "1")
    rayon <- "1"
  }

  meta <- list(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    tool      = "bench_engines.R",
    language  = "r",
    hardware  = list(
      cpu_count = parallel::detectCores(),
      ram_gb    = tryCatch(
        as.numeric(system("sysctl -n hw.memsize 2>/dev/null || free -b 2>/dev/null | awk '/Mem:/{print $2}'",
                          intern = TRUE)) / 1024^3,
        error = function(e) -1
      ),
      machine   = Sys.info()[["machine"]]
    ),
    versions  = list(
      ml     = as.character(packageVersion("ml")),
      r      = R.version.string
    ),
    rayon_num_threads = rayon,
    n_jobs            = 1L,
    notes             = c(
      "histogram CART (Rust) vs exact splits (CRAN rpart)",
      "proc.time() elapsed for timing"
    )
  )

  if (is.null(algorithms)) algorithms <- names(ENGINE_MATRIX)

  cells <- list()

  for (size_name in sizes) {
    sz <- SIZES[[size_name]]
    n_rows <- sz[["n_rows"]]
    n_features <- sz[["n_features"]]

    if (!json_only) {
      cat(sprintf("\n%s\n  SIZE: %s (%s rows x %d features)\n%s\n",
                  strrep("=", 70), toupper(size_name),
                  format(n_rows, big.mark = ","), n_features,
                  strrep("=", 70)))
    }

    # Pre-generate datasets
    datasets <- list()
    for (task in c("classification", "regression")) {
      datasets[[task]] <- make_dataset(task, n_rows, n_features, seed = 42L)
    }

    for (algorithm in algorithms) {
      spec <- ENGINE_MATRIX[[algorithm]]
      if (is.null(spec)) {
        if (!json_only) cat(sprintf("  SKIP %s: not in ENGINE_MATRIX\n", algorithm))
        next
      }

      if (isTRUE(spec$optional) && !optional_available(algorithm)) {
        if (!json_only) cat(sprintf("  SKIP %s: optional dependency not installed\n", algorithm))
        for (task in spec$tasks) {
          for (engine in spec$engines) {
            cells[[length(cells) + 1L]] <- skipped_cell(
              algorithm, engine, task, size_name,
              ALGO_PRIMITIVES[[algorithm]], n_rows, n_features,
              "optional_dependency_not_installed"
            )
          }
        }
        next
      }

      for (task in spec$tasks) {
        ds <- datasets[[task]]
        s <- ds$split

        for (engine in spec$engines) {
          # KNN large: skip
          if (algorithm == "knn" && size_name == "large") {
            if (!json_only) cat(sprintf("  SKIP %s/%s/%s/%s: O(n^2)\n",
                                        algorithm, engine, task, size_name))
            cells[[length(cells) + 1L]] <- skipped_cell(
              algorithm, engine, task, size_name,
              ALGO_PRIMITIVES[[algorithm]], n_rows, n_features,
              "knn_large_prohibitive"
            )
            next
          }

          if (!json_only) cat(sprintf("  %s/%s/%s ... ", algorithm, engine, task))

          cell <- tryCatch(
            bench_one_cell(algorithm, engine, task, size_name,
                            seed = 42L, split_result = s),
            error = function(e) {
              skipped_cell(algorithm, engine, task, size_name,
                            ALGO_PRIMITIVES[[algorithm]], n_rows, n_features,
                            paste0("error: ", conditionMessage(e)))
            }
          )
          cells[[length(cells) + 1L]] <- cell

          if (!json_only) {
            if (isTRUE(cell$skipped)) {
              cat(sprintf("SKIPPED (%s)\n", cell$skip_reason))
            } else {
              metric <- cell$roc_auc %||% cell$r2 %||% cell$accuracy %||% "—"
              cat(sprintf("%.4fs  quality=%s\n", cell$fit_median_s, metric))
            }
          }
        }
      }
    }

    gc(verbose = FALSE)
  }

  list(meta = meta, cells = cells)
}

# ── Pretty Print ──────────────────────────────────────────────────────────────

print_summary <- function(results) {
  cells <- Filter(function(c) !isTRUE(c$skipped), results$cells)
  if (length(cells) == 0L) {
    cat("No benchmark results.\n")
    return(invisible(NULL))
  }

  # Simple table
  cat(sprintf("\n%s\n  Summary\n%s\n", strrep("=", 70), strrep("=", 70)))
  cat(sprintf("  %-15s %-7s %-5s %10s %10s %10s %8s\n",
              "algorithm", "engine", "task", "fit_s", "pred_s", "quality", "search"))
  cat(sprintf("  %s\n", strrep("-", 66)))

  for (cell in cells) {
    metric <- cell$roc_auc %||% cell$r2 %||% cell$accuracy %||% NA
    cat(sprintf("  %-15s %-7s %-5s %10.4f %10.4f %10.4f %8s\n",
                cell$algorithm, cell$engine, substr(cell$task, 1, 5),
                cell$fit_median_s, cell$predict_median_s,
                if (is.null(metric)) NA else metric,
                cell$search))
  }
}

# ── CLI ───────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

sizes <- c("tiny", "small")
algorithms <- NULL
json_only <- FALSE
output_file <- NULL

i <- 1L
while (i <= length(args)) {
  if (args[i] == "--medium") {
    sizes <- c(sizes, "medium")
  } else if (args[i] == "--large") {
    sizes <- c(sizes, "medium", "large")
  } else if (args[i] == "--algorithm" && i < length(args)) {
    i <- i + 1L
    algorithms <- args[i]
  } else if (args[i] == "--json") {
    json_only <- TRUE
  } else if (args[i] == "--output" && i < length(args)) {
    i <- i + 1L
    output_file <- args[i]
  }
  i <- i + 1L
}

sizes <- unique(sizes)

results <- run_engine_benchmark(sizes, algorithms = algorithms, json_only = json_only)

if (!json_only) print_summary(results)

if (json_only || !is.null(output_file)) {
  json_str <- toJSON(results, auto_unbox = TRUE, pretty = TRUE, null = "null")
  if (!is.null(output_file)) {
    writeLines(json_str, output_file)
    if (!json_only) cat(sprintf("\nResults saved to %s\n", output_file))
  }
  if (json_only) cat(json_str, "\n")
}
