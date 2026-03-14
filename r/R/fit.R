#' Fit a machine learning model
#'
#' Trains a model using cross-validation (if `data` is an `ml_split_result`
#' with folds) or holdout (if `data` is a `data.frame`). Automatically detects task type,
#' handles encoding, and records metadata for reproducibility.
#'
#' Formula interfaces are not supported. Pass the data frame and target column
#' name as a string. Unordered factors use one-hot encoding for linear models
#' and ordinal encoding for tree-based models. Ordered factors always use
#' ordinal encoding.
#'
#' @param data A `data.frame`, `ml_split_result`, or `ml_split_result` with folds
#' @param target Target column name (string)
#' @param algorithm "auto" (default), "xgboost", "random_forest", "svm", "knn",
#'   "logistic", "linear", "naive_bayes", "elastic_net"
#' @param seed Random seed. NULL (default) auto-generates and stores for
#'   reproducibility.
#' @param task "auto", "classification", or "regression"
#' @param balance Logical. If `TRUE`, applies class-weight balancing for
#'   imbalanced classification problems. Ignored for regression. Default: `FALSE`.
#' @param engine Backend engine: `"auto"` (Rust if available, else CRAN packages),
#'   `"ml"` (Rust required), or `"r"` (CRAN packages only). Default: `"auto"`.
#' @param ... Additional hyperparameters passed to the engine
#'   (e.g., `max_depth = 6`, `num.trees = 200`)
#' @returns An object of class `ml_model`
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' model$algorithm
ml_fit <- function(data, target, algorithm = "auto", seed = NULL,
                   task = "auto", balance = FALSE, engine = "auto", ...) {
  .fit_impl(data = data, target = target, algorithm = algorithm,
            seed = seed, task = task, balance = balance, engine = engine, ...)
}

.fit_impl <- function(data, target, algorithm = "auto", seed = NULL,
                      task = "auto", balance = FALSE, engine = "auto", ...) {
  # Auto-generate seed
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # Dispatch CV vs holdout
  if (inherits(data, "ml_cv_result")) {
    return(.fit_cv(data, target, algorithm, seed, task, engine = engine, ...))
  }
  if (inherits(data, "ml_split_result") && !is.null(data[["folds"]])) {
    # ml_split_result with folds — CV on dev, test sealed on split
    cv_compat <- list(
      folds  = data[["folds"]],
      data   = data[[".folds_data"]],
      target = data[[".folds_target"]]
    )
    class(cv_compat) <- "ml_cv_result"
    return(.fit_cv(cv_compat, target, algorithm, seed, task, engine = engine, ...))
  }

  # Coerce and validate data
  data <- .coerce_data(data)
  .check_duplicate_cols(data)

  # Partition guard — reject unsplit or wrong-partition data
  if (.guards_active()) {
    .part <- .resolve_partition(data)
    if (is.null(.part)) {
      .guard_action(paste0(
        "ml_fit() received data without split provenance. ",
        "Split your data first: s <- ml_split(df, target, seed = 42), ",
        "then ml_fit(s$train, target, seed = 42). ",
        "To disable: ml_config(guards = 'off')"
      ))
    } else if (!.part %in% c("train", "valid", "dev")) {
      .guard_action(paste0(
        "ml_fit() received data tagged as '", .part, "' partition. ",
        "ml_fit() accepts train, valid, or dev data. ",
        "Use: ml_fit(s$train, ...) or ml_fit(s$dev, ...)"
      ))
    }
  }

  if (!target %in% names(data)) {
    data_error(paste0(
      "target='", target, "' not found in data. Available columns: ",
      paste(names(data), collapse = ", ")
    ))
  }

  y <- data[[target]]

  # Drop NA target rows (even if split already cleaned — validate all inputs)
  na_mask <- is.na(y)
  if (any(na_mask)) {
    n_dropped <- sum(na_mask)
    cli::cli_warn(paste0("Dropped ", n_dropped, " rows with NA target."))
    data <- data[!na_mask, , drop = FALSE]
    y    <- data[[target]]
  }

  if (length(y) == 0L) data_error(paste0("Target column '", target, "' is entirely NA"))
  if (all(is.na(y)))   data_error(paste0("Target column '", target, "' is entirely NA"))

  # Detect task
  detected_task <- .detect_task(y, task)

  # Validate class count for classification
  if (detected_task == "classification") {
    n_classes <- length(unique(y[!is.na(y)]))
    if (n_classes < 2L) {
      val <- unique(y[!is.na(y)])[[1]]
      msg <- if (is.numeric(y) && (task == "auto" || task == "classification")) {
        paste0(
          "Target '", target, "' has only 1 unique value (", val, "). ",
          "For regression with a constant target pass task='regression'. ",
          "For classification, provide at least 2 distinct classes."
        )
      } else {
        paste0(
          "Target '", target, "' has only 1 unique value (", val,
          "). Need at least 2 classes."
        )
      }
      data_error(msg)
    }
  }

  # Resolve algorithm
  algo <- .resolve_algorithm(algorithm, detected_task)

  # Features (all columns except target)
  X <- data[, setdiff(names(data), target), drop = FALSE]

  # Zero-variance check
  .check_zero_variance(X)

  t_start <- proc.time()[["elapsed"]]

  # Fit preprocessing on training data
  fit_out  <- .transform_fit(X, .prepare(X, y, algorithm = algo, task = detected_task))
  X_enc    <- fit_out$X
  norm     <- fit_out$norm
  y_enc    <- .encode_target(y, norm)

  # Balance: compute sample weights for imbalanced classes
  balance_weights <- NULL
  if (isTRUE(balance)) {
    if (detected_task != "classification") {
      config_error(paste0(
        "balance=TRUE only works for classification, not regression. ",
        "For regression with skewed targets, transform the target (e.g. log)."
      ))
    }
    classes <- unique(y_enc)
    n       <- length(y_enc)
    freqs   <- table(y_enc)
    w_map   <- n / (length(classes) * freqs)
    balance_weights <- as.numeric(w_map[as.character(y_enc)])
  }

  # Fit engine — wrap in tryCatch for helpful errors
  eng <- tryCatch(
    .fit_engine(X_enc, y_enc, detected_task, seed, algo, engine = engine,
                .balance_weights = balance_weights, ...),
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("NA|NaN", msg, ignore.case = TRUE)) {
        data_error(paste0(
          "Algorithm still has NA after preprocessing. This may indicate all-NA columns. ",
          "Remove with: data[, colSums(is.na(data)) < nrow(data)]"
        ))
      }
      stop(e)
    }
  )

  t_end  <- proc.time()[["elapsed"]]
  time_s <- as.numeric(t_end - t_start)

  # Build config hash
  hash <- .make_hash(algo, detected_task, target, seed, nrow(data), names(X))

  new_ml_model(
    engine         = eng,
    task           = detected_task,
    algorithm      = algo,
    target         = target,
    features       = names(X),
    seed           = seed,
    scores_        = NULL,
    preprocessing_ = list(algorithm = algo, encoded = !is.null(norm$label_map)),
    n_train        = nrow(data),
    time           = time_s,
    hash           = hash,
    encoders       = norm,
    provenance     = .build_provenance(data)
  )
}

# ── CV fit ─────────────────────────────────────────────────────────────────────

.fit_cv <- function(cv_result, target, algorithm, seed, task, engine = "auto", ...) {
  data <- cv_result[["data"]]
  data <- .coerce_data(data)

  if (!target %in% names(data)) {
    data_error(paste0("target='", target, "' not found in data"))
  }

  y_all <- data[[target]]
  na_mask <- is.na(y_all)
  if (any(na_mask)) {
    cli::cli_warn(paste0("Dropped ", sum(na_mask), " rows with NA target."))
    data     <- data[!na_mask, , drop = FALSE]
    # Rebuild fold indices with NA rows removed
    keep_idx <- which(!na_mask)
    folds <- lapply(cv_result[["folds"]], function(f) {
      list(
        train = which(keep_idx %in% f$train),
        valid = which(keep_idx %in% f$valid)
      )
    })
  } else {
    folds <- cv_result[["folds"]]
  }

  y_all         <- data[[target]]
  detected_task <- .detect_task(y_all, task)
  algo          <- .resolve_algorithm(algorithm, detected_task)
  features      <- setdiff(names(data), target)

  t_start    <- proc.time()[["elapsed"]]
  fold_scores <- vector("list", length(folds))

  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]$train
    valid_idx <- folds[[i]]$valid

    fold_train <- data[train_idx, , drop = FALSE]
    fold_valid <- data[valid_idx, , drop = FALSE]

    X_train <- fold_train[, features, drop = FALSE]
    y_train <- fold_train[[target]]

    # Per-fold normalization: compute stats from THIS FOLD's train data ONLY
    fit_out    <- .transform_fit(X_train, .prepare(X_train, y_train, algorithm = algo, task = detected_task))
    X_tr_enc   <- fit_out$X
    fold_norm  <- fit_out$norm
    y_tr_enc   <- .encode_target(y_train, fold_norm)

    X_val_enc  <- .transform(fold_valid[, features, drop = FALSE], fold_norm)
    y_val      <- fold_valid[[target]]

    # Fit engine on fold training data
    eng_fold <- tryCatch(
      .fit_engine(X_tr_enc, y_tr_enc, detected_task, seed + i, algo, engine = engine, ...),
      error = function(e) NULL
    )
    if (is.null(eng_fold)) next

    preds_fold <- .predict_engine(eng_fold, X_val_enc, detected_task, algo)
    preds_dec  <- .decode(preds_fold, fold_norm)

    fold_scores[[i]] <- .compute_metrics(preds_dec, y_val, detected_task,
                                          eng_fold, X_val_enc, algo, fold_norm)
  }

  # Aggregate fold scores
  fold_scores <- Filter(Negate(is.null), fold_scores)
  scores <- .aggregate_fold_scores(fold_scores)

  # Refit on ALL data with a fresh .prepare()
  X_all   <- data[, features, drop = FALSE]
  fit_out <- .transform_fit(X_all, .prepare(X_all, y_all, algorithm = algo, task = detected_task))
  X_enc   <- fit_out$X
  norm    <- fit_out$norm
  y_enc   <- .encode_target(y_all, norm)

  eng <- .fit_engine(X_enc, y_enc, detected_task, seed, algo, engine = engine, ...)

  t_end  <- proc.time()[["elapsed"]]
  time_s <- as.numeric(t_end - t_start)
  hash   <- .make_hash(algo, detected_task, target, seed, nrow(data), features)

  new_ml_model(
    engine         = eng,
    task           = detected_task,
    algorithm      = algo,
    target         = target,
    features       = features,
    seed           = seed,
    scores_        = scores,
    fold_scores_   = fold_scores,
    preprocessing_ = list(algorithm = algo, encoded = !is.null(norm$label_map)),
    n_train        = nrow(data),
    time           = time_s,
    hash           = hash,
    encoders       = norm,
    provenance     = .build_provenance(data)
  )
}

# ── Metric computation ─────────────────────────────────────────────────────────

.compute_metrics <- function(preds, y_true, task, engine = NULL, X = NULL,
                              algorithm = NULL, norm = NULL) {
  if (task == "regression") {
    residuals <- as.numeric(preds) - as.numeric(y_true)
    rmse <- sqrt(mean(residuals^2, na.rm = TRUE))
    mae  <- mean(abs(residuals), na.rm = TRUE)
    ss_res <- sum(residuals^2, na.rm = TRUE)
    ss_tot <- sum((as.numeric(y_true) - mean(as.numeric(y_true), na.rm = TRUE))^2, na.rm = TRUE)
    r2 <- 1 - ss_res / max(ss_tot, .Machine$double.eps)
    return(list(rmse = rmse, mae = mae, r2 = r2))
  }

  # Classification
  preds_char <- as.character(preds)
  truth_char <- as.character(y_true)
  classes    <- sort(unique(c(preds_char, truth_char)))
  n_classes  <- length(classes)

  acc <- mean(preds_char == truth_char, na.rm = TRUE)

  if (n_classes == 2L) {
    # Binary: pos_label = alphabetically second class (sklearn convention)
    pos_label  <- sort(unique(truth_char))[[2]]
    tp <- sum(preds_char == pos_label & truth_char == pos_label)
    fp <- sum(preds_char == pos_label & truth_char != pos_label)
    fn <- sum(preds_char != pos_label & truth_char == pos_label)
    prec <- if (tp + fp == 0) 0 else tp / (tp + fp)
    rec  <- if (tp + fn == 0) 0 else tp / (tp + fn)
    f1   <- if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)

    metrics <- list(accuracy = acc, f1 = f1, precision = prec, recall = rec)

    # roc_auc via predict_proba if available
    if (!is.null(engine) && !is.null(X) && algorithm %in% .PROBA_ALGORITHMS) {
      tryCatch({
        proba_df  <- .predict_proba_engine(engine, X, task = "classification", algorithm)
        pos_proba <- proba_df[[ncol(proba_df)]]  # last column = positive class
        # normalize
        col_sums  <- rowSums(as.matrix(proba_df))
        col_sums[col_sums == 0] <- 1
        pos_proba <- proba_df[[ncol(proba_df)]] / col_sums

        truth_bin <- as.integer(truth_char == pos_label)
        metrics$roc_auc <- .roc_auc_binary(truth_bin, pos_proba)
      }, error = function(e) NULL)
    }
    return(metrics)
  }

  # Multiclass
  class_metrics <- lapply(classes, function(cl) {
    tp <- sum(preds_char == cl & truth_char == cl)
    fp <- sum(preds_char == cl & truth_char != cl)
    fn <- sum(preds_char != cl & truth_char == cl)
    support <- sum(truth_char == cl)
    prec <- if (tp + fp == 0) 0 else tp / (tp + fp)
    rec  <- if (tp + fn == 0) 0 else tp / (tp + fn)
    f1   <- if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
    list(prec = prec, rec = rec, f1 = f1, support = support)
  })
  names(class_metrics) <- classes

  supports <- vapply(class_metrics, `[[`, numeric(1L), "support")
  total_n  <- sum(supports)

  f1_macro      <- mean(vapply(class_metrics, `[[`, numeric(1L), "f1"))
  f1_weighted   <- sum(vapply(class_metrics, `[[`, numeric(1L), "f1") * supports) / total_n
  prec_macro    <- mean(vapply(class_metrics, `[[`, numeric(1L), "prec"))
  prec_weighted <- sum(vapply(class_metrics, `[[`, numeric(1L), "prec") * supports) / total_n
  rec_macro     <- mean(vapply(class_metrics, `[[`, numeric(1L), "rec"))
  rec_weighted  <- sum(vapply(class_metrics, `[[`, numeric(1L), "rec") * supports) / total_n

  list(accuracy = acc, f1_weighted = f1_weighted, f1_macro = f1_macro,
       precision_weighted = prec_weighted, precision_macro = prec_macro,
       recall_weighted = rec_weighted, recall_macro = rec_macro)
}

.roc_auc_binary <- function(y_true, y_score) {
  # Simple trapezoid AUC
  ord  <- order(y_score, decreasing = TRUE)
  y_s  <- y_true[ord]
  n_pos <- sum(y_true == 1L)
  n_neg <- sum(y_true == 0L)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  tp <- cumsum(y_s == 1L)
  fp <- cumsum(y_s == 0L)
  tpr <- tp / n_pos
  fpr <- fp / n_neg
  # Trapezoid rule: include starting point (0, 0); all vectors are length n+1
  x <- c(0, fpr)
  y <- c(0, tpr)
  sum(diff(x) * (y[-length(y)] + y[-1]) / 2)
}

.aggregate_fold_scores <- function(fold_scores) {
  if (length(fold_scores) == 0L) return(NULL)
  metric_names <- names(fold_scores[[1]])
  result <- list()
  for (nm in metric_names) {
    vals <- vapply(fold_scores, `[[`, numeric(1L), nm)
    result[[nm]] <- mean(vals, na.rm = TRUE)
  }
  result
}

# ── Helpers ────────────────────────────────────────────────────────────────────

.resolve_algorithm <- function(algorithm, task) {
  aliases <- c(
    rf = "random_forest", lr = "logistic", dt = "decision_tree",
    et = "extra_trees",   gb = "gradient_boosting", hg = "histgradient",
    nb = "naive_bayes",   en = "elastic_net"
  )
  if (algorithm %in% names(aliases)) algorithm <- aliases[[algorithm]]
  if (algorithm != "auto") return(algorithm)
  if (requireNamespace("xgboost", quietly = TRUE)) return("xgboost")
  if (requireNamespace("ranger",  quietly = TRUE)) return("random_forest")
  "logistic"
}

.make_hash <- function(algorithm, task, target, seed, n_train, features) {
  key <- paste(algorithm, task, target, seed, n_train, paste(sort(features), collapse = ","))
  digest_val <- as.hexmode(sum(utf8ToInt(key)) %% 16^8)
  sprintf("%08x", as.integer(digest_val))
}
