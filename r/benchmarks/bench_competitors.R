#!/usr/bin/env Rscript
# Competitor benchmark: ml wrapper vs raw CRAN packages.
#
# 3-way comparison per Rust-backed algorithm: ml(Rust) vs ml(CRAN) vs raw CRAN.
# Isolates wrapper overhead: if ml(CRAN) ≈ raw CRAN, the wrapper is zero-cost.
#
# Usage:
#   Rscript benchmarks/bench_competitors.R                    # tiny+small
#   Rscript benchmarks/bench_competitors.R --medium           # +100K
#   Rscript benchmarks/bench_competitors.R --large            # +1M
#   Rscript benchmarks/bench_competitors.R --json --output competitors.json

suppressPackageStartupMessages({
  library(ml)
  library(jsonlite)
})

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

# ── Dataset Generation (same as bench_engines.R) ─────────────────────────────

make_dataset <- function(task, n_rows, n_features, seed = 42L) {
  set.seed(seed)
  X <- matrix(rnorm(n_rows * n_features), nrow = n_rows, ncol = n_features)
  colnames(X) <- paste0("f", seq_len(n_features))
  df <- as.data.frame(X)

  if (task == "classification") {
    signal <- rowSums(X[, 1:min(5, n_features)])
    prob <- 1 / (1 + exp(-signal))
    df$target <- factor(ifelse(runif(n_rows) < prob, 1L, 0L))
  } else {
    coefs <- rnorm(min(n_features %/% 2, 10))
    signal <- X[, seq_along(coefs)] %*% coefs
    df$target <- as.numeric(signal + rnorm(n_rows) * 0.5)
  }

  s <- ml_split(df, "target", seed = seed)
  list(df = df, split = s)
}

# ── Timed Run ─────────────────────────────────────────────────────────────────

run_timed <- function(fn, warmup = 3L, runs = 7L) {
  for (i in seq_len(warmup)) { gc(verbose = FALSE); fn() }
  times <- numeric(runs)
  for (i in seq_len(runs)) {
    gc(verbose = FALSE)
    t0 <- proc.time()["elapsed"]
    fn()
    times[i] <- proc.time()["elapsed"] - t0
  }
  times <- sort(times)
  list(median_seconds = round(median(times), 5), iqr_seconds = round(IQR(times), 5))
}

# ── Raw CRAN Benchmarks ──────────────────────────────────────────────────────

raw_ranger_fit <- function(X, y, task, seed) {
  if (task == "classification") {
    ranger::ranger(x = X, y = factor(y), num.trees = 100L, seed = seed,
                   probability = TRUE, verbose = FALSE)
  } else {
    ranger::ranger(x = X, y = as.numeric(y), num.trees = 100L, seed = seed,
                   verbose = FALSE)
  }
}

raw_rpart_fit <- function(X, y, task, seed) {
  df <- as.data.frame(X)
  if (task == "classification") {
    df$.y <- factor(y)
    set.seed(seed)
    rpart::rpart(.y ~ ., data = df, method = "class")
  } else {
    df$.y <- as.numeric(y)
    set.seed(seed)
    rpart::rpart(.y ~ ., data = df, method = "anova")
  }
}

raw_lm_fit <- function(X, y) {
  # OLS via normal equations (no Ridge; closest base R analogue)
  Xm <- cbind(1, as.matrix(X))
  solve(crossprod(Xm), crossprod(Xm, y))
}

raw_nnet_fit <- function(X, y, seed) {
  df <- as.data.frame(X)
  # Scale for logistic
  means <- colMeans(df)
  sds <- apply(df, 2, sd)
  sds[sds == 0] <- 1
  df <- as.data.frame(scale(df, center = means, scale = sds))
  df$.y <- factor(y)
  suppressWarnings(nnet::multinom(.y ~ ., data = df, trace = FALSE))
}

raw_kknn_fit <- function(X, y, task, k = 5L) {
  # kknn is lazy — stores training data
  list(X = X, y = y, task = task, k = k)
}

raw_kknn_predict <- function(engine, X_test, task) {
  df_train <- as.data.frame(engine$X)
  if (task == "classification") {
    df_train$.y <- factor(engine$y)
  } else {
    df_train$.y <- as.numeric(engine$y)
  }
  df_test <- as.data.frame(X_test)
  fit <- kknn::kknn(.y ~ ., train = df_train, test = df_test, k = engine$k)
  fit$fitted.values
}

# ── Competitor Cells ──────────────────────────────────────────────────────────

COMPETITORS <- list(
  random_forest = list(
    tasks    = c("classification", "regression"),
    raw_name = "ranger",
    raw_fit  = function(X, y, task, seed) raw_ranger_fit(X, y, task, seed),
    raw_pred = function(eng, X, task) {
      p <- predict(eng, data = X)
      if (task == "regression") p$predictions
      else apply(p$predictions, 1, which.max) - 1L
    }
  ),
  decision_tree = list(
    tasks    = c("classification", "regression"),
    raw_name = "rpart",
    raw_fit  = function(X, y, task, seed) raw_rpart_fit(X, y, task, seed),
    raw_pred = function(eng, X, task) {
      df <- as.data.frame(X)
      if (task == "regression") as.numeric(predict(eng, newdata = df))
      else as.integer(as.character(predict(eng, newdata = df, type = "class")))
    }
  ),
  logistic = list(
    tasks    = c("classification"),
    raw_name = "nnet::multinom",
    raw_fit  = function(X, y, task, seed) raw_nnet_fit(X, y, seed),
    raw_pred = function(eng, X, task) {
      # Need to scale X the same way
      df <- as.data.frame(X)
      as.integer(as.character(predict(eng, newdata = df, type = "class")))
    }
  ),
  knn = list(
    tasks    = c("classification", "regression"),
    raw_name = "kknn",
    raw_fit  = function(X, y, task, seed) raw_kknn_fit(X, y, task),
    raw_pred = function(eng, X, task) raw_kknn_predict(eng, X, task)
  ),
  linear = list(
    tasks    = c("regression"),
    raw_name = "lm (normal equations)",
    raw_fit  = function(X, y, task, seed) raw_lm_fit(X, y),
    raw_pred = function(eng, X, task) {
      Xm <- cbind(1, as.matrix(X))
      as.numeric(Xm %*% eng)
    }
  )
)

bench_competitor <- function(algorithm, task, size_name, seed = 42L) {
  sz <- SIZES[[size_name]]
  n_rows <- sz[["n_rows"]]
  n_features <- sz[["n_features"]]
  sr <- SIZE_RUNS[[size_name]]
  warmup <- sr[["warmup"]]
  runs <- sr[["runs"]]

  comp <- COMPETITORS[[algorithm]]
  ds <- make_dataset(task, n_rows, n_features, seed = seed)
  s <- ds$split

  # Prepare raw data
  X_train <- s$train[, setdiff(names(s$train), "target")]
  y_train <- s$train[["target"]]
  X_valid <- s$valid[, setdiff(names(s$valid), "target")]

  results <- list()

  # 1. ml(Rust) — engine="ml"
  ml_rust_model <- NULL
  ml_rust_fit <- function() {
    suppressWarnings({
      ml_rust_model <<- ml_fit(s$train, "target", algorithm = algorithm,
                                seed = seed, engine = "ml")
    })
  }
  ml_rust_stats <- tryCatch(
    run_timed(ml_rust_fit, warmup = warmup, runs = runs),
    error = function(e) list(median_seconds = NA, iqr_seconds = NA)
  )
  results$ml_rust <- list(
    fit_s = ml_rust_stats$median_seconds,
    iqr_s = ml_rust_stats$iqr_seconds
  )

  # 2. ml(CRAN) — engine="cran" (bypasses Rust)
  ml_cran_model <- NULL
  ml_cran_fit <- function() {
    suppressWarnings({
      ml_cran_model <<- ml_fit(s$train, "target", algorithm = algorithm,
                                seed = seed, engine = "cran")
    })
  }
  ml_cran_stats <- tryCatch(
    run_timed(ml_cran_fit, warmup = warmup, runs = runs),
    error = function(e) list(median_seconds = NA, iqr_seconds = NA)
  )
  results$ml_cran <- list(
    fit_s = ml_cran_stats$median_seconds,
    iqr_s = ml_cran_stats$iqr_seconds
  )

  # 3. Raw CRAN — direct package call
  raw_model <- NULL
  raw_fit <- function() {
    raw_model <<- comp$raw_fit(X_train, y_train, task, seed)
  }
  raw_stats <- tryCatch(
    run_timed(raw_fit, warmup = warmup, runs = runs),
    error = function(e) list(median_seconds = NA, iqr_seconds = NA)
  )
  results$raw_cran <- list(
    fit_s = raw_stats$median_seconds,
    iqr_s = raw_stats$iqr_seconds,
    package = comp$raw_name
  )

  # Wrapper overhead = ml_cran / raw_cran
  if (!is.na(results$ml_cran$fit_s) && !is.na(results$raw_cran$fit_s) &&
      results$raw_cran$fit_s > 0) {
    results$wrapper_overhead <- round(results$ml_cran$fit_s / results$raw_cran$fit_s, 3)
  }

  # Rust speedup = raw_cran / ml_rust
  if (!is.na(results$ml_rust$fit_s) && !is.na(results$raw_cran$fit_s) &&
      results$ml_rust$fit_s > 0) {
    results$rust_speedup <- round(results$raw_cran$fit_s / results$ml_rust$fit_s, 2)
  }

  list(
    algorithm  = algorithm,
    task       = task,
    size       = size_name,
    n_rows     = n_rows,
    n_features = n_features,
    results    = results
  )
}

# ── Main ──────────────────────────────────────────────────────────────────────

run_competitor_benchmark <- function(sizes, json_only = FALSE) {
  rayon <- Sys.getenv("RAYON_NUM_THREADS", "not_set")
  if (rayon == "not_set") {
    Sys.setenv(RAYON_NUM_THREADS = "1")
    rayon <- "1"
  }

  meta <- list(
    timestamp         = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    tool              = "bench_competitors.R",
    language          = "r",
    rayon_num_threads = rayon,
    versions          = list(ml = as.character(packageVersion("ml")),
                              r = R.version.string)
  )

  cells <- list()

  for (size_name in sizes) {
    sz <- SIZES[[size_name]]
    if (!json_only) {
      cat(sprintf("\n%s\n  %s (%s rows x %d features)\n%s\n",
                  strrep("=", 70), toupper(size_name),
                  format(sz[["n_rows"]], big.mark = ","), sz[["n_features"]],
                  strrep("=", 70)))
    }

    for (algorithm in names(COMPETITORS)) {
      comp <- COMPETITORS[[algorithm]]
      for (task in comp$tasks) {
        # KNN large: skip
        if (algorithm == "knn" && size_name == "large") {
          if (!json_only) cat(sprintf("  SKIP %s/%s: O(n^2)\n", algorithm, task))
          next
        }

        if (!json_only) cat(sprintf("  %s/%s ... ", algorithm, task))

        cell <- tryCatch(
          bench_competitor(algorithm, task, size_name),
          error = function(e) {
            list(algorithm = algorithm, task = task, size = size_name,
                 error = conditionMessage(e))
          }
        )
        cells[[length(cells) + 1L]] <- cell

        if (!json_only && is.null(cell$error)) {
          r <- cell$results
          cat(sprintf("rust=%.4fs  cran=%.4fs  raw=%.4fs  overhead=%.2fx  speedup=%.1fx\n",
                      r$ml_rust$fit_s %||% NA,
                      r$ml_cran$fit_s %||% NA,
                      r$raw_cran$fit_s %||% NA,
                      r$wrapper_overhead %||% NA,
                      r$rust_speedup %||% NA))
        } else if (!json_only) {
          cat(sprintf("ERROR: %s\n", cell$error))
        }
      }
    }
  }

  list(meta = meta, cells = cells)
}

# ── Null coalesce ─────────────────────────────────────────────────────────────

`%||%` <- function(x, y) if (is.null(x)) y else x

# ── CLI ───────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
sizes <- c("tiny", "small")
json_only <- FALSE
output_file <- NULL

i <- 1L
while (i <= length(args)) {
  if (args[i] == "--medium") sizes <- c(sizes, "medium")
  else if (args[i] == "--large") sizes <- c(sizes, "medium", "large")
  else if (args[i] == "--json") json_only <- TRUE
  else if (args[i] == "--output" && i < length(args)) { i <- i + 1L; output_file <- args[i] }
  i <- i + 1L
}
sizes <- unique(sizes)

results <- run_competitor_benchmark(sizes, json_only = json_only)

if (json_only || !is.null(output_file)) {
  json_str <- toJSON(results, auto_unbox = TRUE, pretty = TRUE, null = "null")
  if (!is.null(output_file)) {
    writeLines(json_str, output_file)
    if (!json_only) cat(sprintf("\nResults saved to %s\n", output_file))
  }
  if (json_only) cat(json_str, "\n")
}
