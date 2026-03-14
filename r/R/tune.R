#' Tune hyperparameters via random or grid search
#'
#' @param data A data.frame or `ml_split_result`
#' @param target Target column name
#' @param model An `ml_model` object (to clone algorithm from), or NULL
#' @param algorithm Algorithm name (if model is NULL)
#' @param n_trials Number of random search trials (default 20)
#' @param cv_folds Number of CV folds per trial (default 3)
#' @param method "random" (default) or "grid"
#' @param seed Random seed
#' @param params Named list of parameter ranges (overrides defaults). For numeric
#'   ranges, provide a 2-element numeric vector c(min, max). For discrete,
#'   provide a character/integer vector.
#' @returns An object of class `ml_tuning_result`
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' tuned <- ml_tune(s$train, "Species", algorithm = "xgboost", n_trials = 5, seed = 42)
#' tuned$best_params_
#' }
ml_tune <- function(data, target, model = NULL, algorithm = NULL,
                    n_trials = 20L, cv_folds = 3L, method = "random",
                    seed = NULL, params = NULL) {
  .tune_impl(data = data, target = target, model = model, algorithm = algorithm,
             n_trials = n_trials, cv_folds = cv_folds, method = method,
             seed = seed, params = params)
}

# Default hyperparameter search spaces
.TUNE_DEFAULTS <- list(
  xgboost = list(
    max_depth      = c(3L, 10L),
    eta            = c(0.01, 0.3),
    nrounds        = c(50L, 300L),
    subsample      = c(0.6, 1.0)
  ),
  gradient_boosting = list(
    n_estimators     = c(50L, 300L),
    learning_rate    = c(0.01, 0.3),
    max_depth        = c(3L, 8L),
    subsample        = c(0.5, 1.0),
    min_samples_split = c(2L, 10L),
    min_samples_leaf  = c(1L, 5L),
    reg_lambda       = c(0.001, 10.0),           # L2 regularization
    gamma            = c(0.0, 5.0),              # min split gain
    colsample_bytree = c(0.3, 1.0),             # column subsampling per tree
    min_child_weight = c(1.0, 10.0)             # min hessian per leaf
  ),
  histgradient = list(
    n_estimators     = c(50L, 300L),
    learning_rate    = c(0.01, 0.3),
    max_depth        = c(3L, 8L),
    subsample        = c(0.5, 1.0),
    min_samples_split = c(2L, 10L),
    min_samples_leaf  = c(1L, 5L),
    reg_lambda       = c(0.001, 10.0),
    gamma            = c(0.0, 5.0),
    colsample_bytree = c(0.3, 1.0),
    min_child_weight = c(1.0, 10.0)
  ),
  random_forest = list(
    num.trees        = c(100L, 500L),
    min.node.size    = c(1L, 20L),
    min_impurity_decrease = c(0.0, 0.1),         # min impurity decrease for split
    ccp_alpha        = c(0.0, 0.05)              # cost-complexity pruning
  ),
  decision_tree = list(
    max_depth        = c(3L, 20L),
    min_impurity_decrease = c(0.0, 0.1),
    ccp_alpha        = c(0.0, 0.05)
  ),
  extra_trees = list(
    num.trees        = c(100L, 500L),
    min.node.size    = c(1L, 20L),
    min_impurity_decrease = c(0.0, 0.1),
    ccp_alpha        = c(0.0, 0.05)
  ),
  logistic = list(
    # stats::glm — limited tuning
  ),
  linear = list(
    # glmnet standardizes; alpha is fixed at 0
  ),
  elastic_net = list(
    alpha = c(0.1, 0.9)
  ),
  svm = list(
    cost    = c(0.1, 10.0),
    gamma   = c(0.001, 1.0),
    kernel  = c("linear", "rbf")               # rbf only viable for n <= 5000
  ),
  knn = list(
    k       = c(3L, 15L),
    weights = c("uniform", "distance")
  ),
  naive_bayes = list(
    # laplace smoothing
    laplace = c(0, 1)
  )
)

.tune_impl <- function(data, target, model = NULL, algorithm = NULL,
                       n_trials = 20L, cv_folds = 3L, method = "random",
                       seed = NULL, params = NULL) {
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # Resolve algorithm
  if (!is.null(model)) {
    if (!inherits(model, "ml_model")) config_error("model must be an ml_model")
    algorithm <- model[["algorithm"]]
  }
  if (is.null(algorithm)) algorithm <- "xgboost"

  # Coerce data
  data <- if (inherits(data, "ml_split_result")) data[["train"]] else .coerce_data(data)

  # Partition guard — same as fit(): require split provenance
  if (.guards_active()) {
    .part <- .resolve_partition(data)
    if (is.null(.part)) {
      .guard_action(paste0(
        "ml_tune() received data without split provenance. ",
        "Split your data first: s <- ml_split(df, target, seed = 42), ",
        "then ml_tune(s$train, target, ...). ",
        "To disable: ml_config(guards = 'off')"
      ))
    } else if (!.part %in% c("train", "valid", "dev")) {
      .guard_action(paste0(
        "ml_tune() received data tagged as '", .part, "' partition. ",
        "ml_tune() accepts train, valid, or dev data. ",
        "Use: ml_tune(s$train, target, ...)"
      ))
    }
  }

  if (!target %in% names(data)) {
    data_error(paste0("target='", target, "' not found in data"))
  }

  y            <- data[[target]]
  detected_task <- .detect_task(y)
  algo         <- .resolve_algorithm(algorithm, detected_task)

  # Primary optimization metric
  primary_metric     <- if (detected_task == "classification") "accuracy" else "rmse"
  higher_is_better   <- detected_task == "classification"

  # Build search space
  search_space <- if (!is.null(params)) params else (.TUNE_DEFAULTS[[algo]] %||% list())

  # SVM kernel conditional: rbf is O(n^2) — only viable for small datasets
  if (algo == "svm" && is.null(params) && "kernel" %in% names(search_space)) {
    n_rows <- nrow(data)
    if (n_rows > 5000L) {
      search_space[["kernel"]] <- c("linear")
    }
  }

  if (length(search_space) == 0L) {
    # No tunable params — just fit with defaults
    best_model <- .fit_impl(data = data, target = target, algorithm = algo, seed = seed)
    return(new_ml_tuning_result(
      best_params = list(),
      history     = data.frame(trial = 1L, score = NA_real_, stringsAsFactors = FALSE),
      best_model  = best_model
    ))
  }

  # NaN pre-check
  X_check <- data[, setdiff(names(data), target), drop = FALSE]
  na_count <- sum(is.na(as.matrix(X_check)))
  if (na_count > 0) {
    cli::cli_warn(paste0(na_count, " NA value(s) in features."))
  }

  # Create CV folds
  withr::local_seed(seed)
  n <- nrow(data)
  fold_ids <- sample(rep(seq_len(cv_folds), length.out = n))
  cv_folds_list <- lapply(seq_len(cv_folds), function(k) {
    list(train = which(fold_ids != k), valid = which(fold_ids == k))
  })

  # Generate parameter combinations
  if (method == "grid") {
    param_grid <- .make_grid(search_space)
    trial_params <- param_grid[seq_len(min(nrow(param_grid), n_trials)), ]
    trials <- lapply(seq_len(nrow(trial_params)), function(i) as.list(trial_params[i, ]))
  } else {
    # Random search (RNG already seeded by withr::local_seed above)
    trials <- lapply(seq_len(n_trials), function(trial) {
      lapply(search_space, function(range) {
        if (length(range) == 2L && is.numeric(range)) {
          if (is.integer(range)) {
            sample(range[[1]]:range[[2]], 1L)
          } else {
            stats::runif(1L, range[[1]], range[[2]])
          }
        } else {
          sample(range, 1L)
        }
      })
    })
  }

  history <- data.frame(
    trial = seq_len(length(trials)),
    score = rep(NA_real_, length(trials)),
    stringsAsFactors = FALSE
  )

  best_score  <- if (higher_is_better) -Inf else Inf
  best_params <- trials[[1]]
  best_model  <- NULL

  for (i in seq_along(trials)) {
    trial_p <- trials[[i]]

    # CV evaluation of this trial (suppress per-fold NA warnings)
    cv_scores <- suppressWarnings(lapply(cv_folds_list, function(fold) {
      tryCatch({
        fold_train <- data[fold$train, , drop = FALSE]
        fold_valid <- data[fold$valid, , drop = FALSE]
        X_tr <- fold_train[, setdiff(names(fold_train), target), drop = FALSE]
        y_tr <- fold_train[[target]]
        fit_out  <- .transform_fit(X_tr, .prepare(X_tr, y_tr, algo, detected_task))
        X_tr_enc <- fit_out$X
        norm_f   <- fit_out$norm
        y_tr_enc <- .encode_target(y_tr, norm_f)
        eng      <- do.call(.fit_engine, c(list(X_tr_enc, y_tr_enc, detected_task, seed + i, algo),
                                            trial_p))
        X_val <- .transform(fold_valid[, setdiff(names(fold_valid), target), drop = FALSE], norm_f)
        preds <- .decode(.predict_engine(eng, X_val, detected_task, algo), norm_f)
        m     <- .compute_metrics(preds, fold_valid[[target]], detected_task)
        m[[primary_metric]]
      }, error = function(e) NA_real_)
    }))

    trial_score <- mean(unlist(cv_scores), na.rm = TRUE)
    history$score[[i]] <- trial_score

    is_better <- if (higher_is_better) {
      !is.na(trial_score) && trial_score > best_score
    } else {
      !is.na(trial_score) && trial_score < best_score
    }

    if (is_better) {
      best_score  <- trial_score
      best_params <- trial_p
    }
  }

  # Fit best model on ALL data
  best_model <- tryCatch(
    do.call(.fit_impl, c(list(data = data, target = target,
                               algorithm = algo, seed = seed),
                          best_params)),
    error = function(e) .fit_impl(data = data, target = target,
                                   algorithm = algo, seed = seed)
  )

  # Add param columns to history
  if (length(best_params) > 0) {
    for (nm in names(best_params)) {
      history[[nm]] <- vapply(trials, function(t) {
        v <- t[[nm]]
        if (is.null(v)) NA_real_ else as.numeric(v)
      }, numeric(1L))
    }
  }

  new_ml_tuning_result(
    best_params = best_params,
    history     = history,
    best_model  = best_model
  )
}

.make_grid <- function(search_space) {
  # Build grid from numeric ranges (sample 3 values per range) or discrete vectors
  grids <- lapply(names(search_space), function(nm) {
    range <- search_space[[nm]]
    if (length(range) == 2L && is.numeric(range)) {
      if (is.integer(range)) seq(range[[1]], range[[2]], length.out = 3L)
      else seq(range[[1]], range[[2]], length.out = 3L)
    } else {
      range
    }
  })
  names(grids) <- names(search_space)
  expand.grid(grids, stringsAsFactors = FALSE)
}
