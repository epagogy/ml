# ── Algorithm registry ─────────────────────────────────────────────────────────

# Algorithms that get one-hot encoding + scaling (distance/linear models)
.LINEAR_ALGORITHMS <- c("logistic", "linear", "svm", "knn", "elastic_net")

# Algorithms that need NO scaling (glmnet standardizes internally)
.NO_SCALE_ALGORITHMS <- c("linear", "elastic_net", "naive_bayes")

# Algorithms that support predict_proba
.PROBA_ALGORITHMS <- c("xgboost", "random_forest", "logistic", "svm",
                       "naive_bayes", "elastic_net", "knn", "decision_tree",
                       "extra_trees", "gradient_boosting", "adaboost")

# Algorithms with Rust backends
.RUST_ALGORITHMS <- c("linear", "logistic", "random_forest", "decision_tree", "knn",
                      "extra_trees", "gradient_boosting", "histgradient",
                      "naive_bayes", "elastic_net", "adaboost", "svm")

# ── Engine dispatch ────────────────────────────────────────────────────────────

.fit_engine <- function(X, y, task, seed, algorithm, engine = "auto", ...) {
  dots <- list(...)
  # Extract balance weights (internal param, not passed to engines)
  balance_weights <- dots[[".balance_weights"]]
  dots[[".balance_weights"]] <- NULL

  # Rust dispatch: engine="auto" or "ml" for supported algorithms
  use_rust <- engine %in% c("auto", "ml") &&
              algorithm %in% .RUST_ALGORITHMS &&
              .rust_available()

  if (identical(engine, "ml") && !.rust_available()) {
    config_error("engine='ml' requires Rust backend (cargo not found at install time)")
  }

  if (use_rust) {
    # Algorithm-task validation (same rules as CRAN engines)
    if (algorithm == "logistic" && task == "regression") {
      config_error(paste0(
        "algorithm='logistic' is classification-only. ",
        "For regression, try algorithm='linear'."
      ))
    }
    if (algorithm == "linear" && task == "classification") {
      config_error(paste0(
        "algorithm='linear' is regression-only. ",
        "For classification, try algorithm='logistic'."
      ))
    }
    # Default criterion by task: classification=gini, regression=mse
    criterion_default <- if (task == "classification") "gini" else "mse"
    criterion <- dots[["criterion"]] %||% criterion_default
    return(switch(algorithm,
      linear        = .fit_rust_linear(X, y,
                        alpha = dots[["alpha"]] %||% 1.0,
                        sample_weight = balance_weights),
      logistic      = .fit_rust_logistic(X, as.integer(factor(y)) - 1L,
                        C = dots[["C"]] %||% 1.0, max_iter = dots[["max_iter"]] %||% 1000L,
                        sample_weight = balance_weights,
                        multi_class = dots[["multi_class"]] %||% "ovr"),
      random_forest = .fit_rust_forest(X, y, task, seed,
                        n_trees = dots[["num.trees"]] %||% dots[["n_trees"]] %||% 100L,
                        sample_weight = balance_weights,
                        criterion = criterion,
                        monotone_cst = dots[["monotone_cst"]]),
      decision_tree = .fit_rust_tree(X, y, task, seed,
                        max_depth = dots[["max_depth"]] %||% 500L,
                        sample_weight = balance_weights,
                        criterion = criterion,
                        monotone_cst = dots[["monotone_cst"]]),
      extra_trees   = .fit_rust_extra_trees(X, y, task, seed,
                        n_trees = dots[["num.trees"]] %||% dots[["n_trees"]] %||% 100L,
                        sample_weight = balance_weights,
                        criterion = criterion,
                        monotone_cst = dots[["monotone_cst"]]),
      knn               = .fit_rust_knn(X, y, task,
                            k = dots[["k"]] %||% 5L),
      gradient_boosting = ,  # fall through to histgradient
      histgradient      = .fit_rust_gbt(X, y, task, seed,
                            n_estimators    = dots[["n_estimators"]] %||% 100L,
                            learning_rate   = dots[["learning_rate"]] %||% 0.1,
                            max_depth       = dots[["max_depth"]] %||% 3L,
                            min_samples_split = dots[["min_samples_split"]] %||% 2L,
                            min_samples_leaf  = dots[["min_samples_leaf"]] %||% 1L,
                            subsample       = dots[["subsample"]] %||% 1.0,
                            sample_weight   = balance_weights,
                            reg_lambda      = dots[["reg_lambda"]] %||% dots[["lambda"]] %||% 0.0,
                            gamma           = dots[["gamma"]] %||% 0.0,
                            colsample_bytree = dots[["colsample_bytree"]] %||% 1.0,
                            min_child_weight = dots[["min_child_weight"]] %||% 1.0,
                            n_iter_no_change = dots[["n_iter_no_change"]],
                            validation_fraction = dots[["validation_fraction"]] %||% 0.1),
      naive_bayes       = if (task == "regression") {
                            config_error("algorithm='naive_bayes' does not support regression.")
                          } else {
                            .fit_rust_nb(X, y, task, seed,
                              var_smoothing = dots[["var_smoothing"]] %||% 1e-9,
                              sample_weight = balance_weights)
                          },
      elastic_net       = if (task == "classification") {
                            # Rust EN is regression-only; fall through to glmnet
                            if (identical(engine, "ml")) {
                              config_error(paste0(
                                "algorithm='elastic_net' with engine='ml' is regression-only.\n",
                                "  For classification, use engine='auto' (routes to glmnet)."
                              ))
                            }
                            .fit_elastic_net(X, y, task, seed, dots, balance_weights)
                          } else {
                            .fit_rust_en(X, y, task, seed,
                              alpha     = dots[["alpha"]] %||% 1.0,
                              l1_ratio  = dots[["l1_ratio"]] %||% 0.5,
                              max_iter  = dots[["max_iter"]] %||% 1000L,
                              tol       = dots[["tol"]] %||% 1e-4,
                              sample_weight = balance_weights)
                          },
      adaboost          = .fit_rust_ada(X, y, task, seed,
                            n_estimators  = dots[["n_estimators"]] %||% 50L,
                            learning_rate = dots[["learning_rate"]] %||% 1.0),
      svm               = .fit_rust_svm(X, y, task, seed,
                              C        = dots[["C"]] %||% 1.0,
                              epsilon  = dots[["epsilon"]] %||% 0.1,
                              tol      = dots[["tol"]] %||% 1e-3,
                              max_iter = dots[["max_iter"]] %||% 1000L,
                              sample_weight = balance_weights)
    ))
  }

  # CRAN / native R dispatch (unchanged)
  switch(algorithm,
    xgboost      = .fit_xgboost(X, y, task, seed, dots, balance_weights),
    random_forest = .fit_ranger(X, y, task, seed, dots, balance_weights),
    logistic     = .fit_logistic(X, y, task, seed, dots, balance_weights),
    linear       = .fit_linear(X, y, task, seed, dots, balance_weights),
    svm          = .fit_svm(X, y, task, seed, dots, balance_weights),
    knn          = .fit_knn(X, y, task, seed, dots),
    naive_bayes   = .fit_naive_bayes(X, y, task, seed, dots),
    elastic_net   = .fit_elastic_net(X, y, task, seed, dots, balance_weights),
    decision_tree     = .fit_decision_tree(X, y, task, seed, dots, balance_weights),
    gradient_boosting = ,  # fall through
    histgradient = config_error(paste0(
      "algorithm='gradient_boosting'/'histgradient' requires the Rust backend.\n",
      "  Reinstall ml with cargo available: install.packages('ml', type='source')"
    )),
    adaboost = config_error(paste0(
      "algorithm='adaboost' requires the Rust backend.\n",
      "  Reinstall ml with cargo available: install.packages('ml', type='source')"
    )),
    config_error(paste0(
      "algorithm='", algorithm, "' not available.\n",
      "  Choose from: ", paste(.available_algorithms(), collapse = ", ")
    ))
  )
}

.predict_engine <- function(engine, X, task, algorithm) {
  # Rust engines carry a $type tag
  if (.is_rust_engine(engine)) {
    return(switch(algorithm,
      linear        = .predict_rust_linear(engine, X),
      logistic      = .predict_rust_logistic(engine, X),
      random_forest = .predict_rust_forest(engine, X, task),
      decision_tree = .predict_rust_tree(engine, X, task),
      extra_trees       = .predict_rust_extra_trees(engine, X, task),
      knn               = .predict_rust_knn(engine, X, task),
      gradient_boosting = ,  # fall through
      histgradient      = .predict_rust_gbt(engine, X, task),
      naive_bayes       = .predict_rust_nb(engine, X, task),
      elastic_net       = .predict_rust_en(engine, X, task),
      adaboost          = .predict_rust_ada(engine, X, task),
      svm               = .predict_rust_svm(engine, X, task)
    ))
  }
  switch(algorithm,
    xgboost       = .predict_xgboost(engine, X, task),
    random_forest = .predict_ranger(engine, X, task),
    logistic      = .predict_logistic(engine, X, task),
    linear        = .predict_linear(engine, X, task),
    svm           = .predict_svm(engine, X, task),
    knn           = .predict_knn(engine, X, task),
    naive_bayes   = .predict_naive_bayes(engine, X, task),
    elastic_net   = .predict_elastic_net(engine, X, task),
    decision_tree = .predict_decision_tree(engine, X, task),
    model_error(paste0("Unknown algorithm for predict: ", algorithm))
  )
}

.predict_proba_engine <- function(engine, X, task, algorithm) {
  if (task != "classification") model_error("predict_proba is for classification only")
  # Rust engines
  if (.is_rust_engine(engine)) {
    return(switch(algorithm,
      logistic      = .proba_rust_logistic(engine, X),
      random_forest = .proba_rust_forest(engine, X),
      decision_tree = .proba_rust_tree(engine, X),
      extra_trees       = .proba_rust_extra_trees(engine, X),
      knn               = .proba_rust_knn(engine, X),
      gradient_boosting = ,  # fall through
      histgradient      = .proba_rust_gbt(engine, X),
      naive_bayes       = .proba_rust_nb(engine, X),
      adaboost          = .proba_rust_ada(engine, X),
      svm               = .proba_rust_svm(engine, X),
      model_error(paste0("predict_proba not supported for Rust engine: ", algorithm))
    ))
  }
  switch(algorithm,
    xgboost       = .proba_xgboost(engine, X),
    random_forest = .proba_ranger(engine, X),
    logistic      = .proba_logistic(engine, X),
    svm           = .proba_svm(engine, X),
    naive_bayes   = .proba_naive_bayes(engine, X),
    elastic_net   = .proba_elastic_net(engine, X),
    knn           = .proba_knn(engine, X),
    decision_tree = .proba_decision_tree(engine, X),
    model_error(paste0(
      "predict_proba not supported for algorithm='", algorithm, "'.\n",
      "  Try algorithm='xgboost' or 'random_forest'."
    ))
  )
}

# ── XGBoost ────────────────────────────────────────────────────────────────────

.fit_xgboost <- function(X, y, task, seed, dots, balance_weights = NULL) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    config_error("'xgboost' required. Install with: install.packages('xgboost')")
  }
  dots[["nrounds"]]   <- dots[["nrounds"]]   %||% 100L
  dots[["max_depth"]] <- dots[["max_depth"]] %||% 6L
  dots[["eta"]]       <- dots[["eta"]]       %||% 0.3
  dots[["verbose"]]   <- dots[["verbose"]]   %||% 0L
  dots[["nthread"]]   <- dots[["nthread"]]   %||% 1L

  if (task == "classification") {
    n_class <- length(unique(y))
    if (n_class == 2L) {
      objective <- "binary:logistic"
      extra <- list()
    } else {
      objective <- "multi:softprob"
      extra <- list(num_class = n_class)
    }
    params <- c(list(objective = objective, seed = seed, verbosity = 0L),
                extra,
                dots["max_depth"], dots["eta"], dots["verbose"], dots["nthread"],
                dots[setdiff(names(dots), c("max_depth", "eta", "verbose", "nrounds",
                                             "num_class", "nthread"))])
    params <- Filter(Negate(is.null), params)
    dm <- xgboost::xgb.DMatrix(data = as.matrix(X), label = y)
    if (!is.null(balance_weights)) xgboost::setinfo(dm, "weight", balance_weights)
    xgboost::xgb.train(params = params, data = dm,
                       nrounds = dots[["nrounds"]], verbose = 0)
  } else {
    params <- c(list(objective = "reg:squarederror", seed = seed, verbosity = 0L),
                dots["max_depth"], dots["eta"], dots["nthread"],
                dots[setdiff(names(dots), c("max_depth", "eta", "verbose", "nrounds", "nthread"))])
    dm <- xgboost::xgb.DMatrix(data = as.matrix(X), label = y)
    xgboost::xgb.train(params = params, data = dm,
                       nrounds = dots[["nrounds"]], verbose = 0)
  }
}

.predict_xgboost <- function(engine, X, task) {
  dm  <- xgboost::xgb.DMatrix(data = as.matrix(X))
  raw <- predict(engine, dm)
  if (task == "regression") return(raw)
  # classification: binary returns prob vec (length n), multiclass returns flat vec (n * k)
  n <- nrow(X)
  if (length(raw) == n) {
    # Binary: raw is probabilities
    as.integer(raw >= 0.5)
  } else {
    # Multiclass: flat vector, reshape to n × k
    n_class <- length(raw) / n
    mat <- matrix(raw, nrow = n, ncol = n_class, byrow = TRUE)
    apply(mat, 1, which.max) - 1L  # 0-based integer labels
  }
}

.proba_xgboost <- function(engine, X) {
  dm  <- xgboost::xgb.DMatrix(data = as.matrix(X))
  raw <- predict(engine, dm)
  n   <- nrow(X)
  if (length(raw) == n) {
    # Binary: raw is probabilities of positive class
    proba <- cbind(1 - raw, raw)
  } else {
    n_class <- length(raw) / n
    proba <- matrix(raw, nrow = n, ncol = n_class, byrow = TRUE)
  }
  # Normalize rows to sum=1 (XGBoost floating-point drift)
  row_sums <- rowSums(proba)
  row_sums[row_sums == 0] <- 1
  as.data.frame(proba / row_sums)
}

# ── Ranger (random forest) ─────────────────────────────────────────────────────

.fit_ranger <- function(X, y, task, seed, dots, balance_weights = NULL) {
  if (!requireNamespace("ranger", quietly = TRUE)) {
    config_error("'ranger' required. Install with: install.packages('ranger')")
  }
  dots[["num.trees"]] <- dots[["num.trees"]] %||% 500L
  if (task == "classification") {
    y_factor <- if (is.factor(y)) y else factor(y)
    ranger::ranger(
      x = X, y = y_factor,
      num.trees = dots[["num.trees"]],
      seed = seed,
      probability = TRUE,
      importance = "impurity",
      verbose = FALSE,
      case.weights = balance_weights
    )
  } else {
    ranger::ranger(
      x = X, y = as.numeric(y),
      num.trees = dots[["num.trees"]],
      seed = seed,
      importance = "impurity",
      verbose = FALSE
    )
  }
}

.predict_ranger <- function(engine, X, task) {
  pred <- predict(engine, data = X)
  if (task == "regression") return(pred$predictions)
  # probability forest: take class with max prob
  pm <- pred$predictions
  label_idx <- apply(pm, 1, which.max)
  # Return 0-based integers (decode in predict.R)
  label_idx - 1L
}

.proba_ranger <- function(engine, X) {
  pred <- predict(engine, data = X)
  pm <- pred$predictions
  # normalize rows
  row_sums <- rowSums(pm)
  row_sums[row_sums == 0] <- 1
  as.data.frame(pm / row_sums)
}

# ── Logistic (native stats::optim L-BFGS-B -- no external package) ─────────────
# Implemented in R/logistic.R: .lr_fit / .lr_predict / .lr_proba
# Supports binary and multiclass (OvR). L2 regularisation only.

.fit_logistic <- function(X, y, task, seed, dots, balance_weights = NULL) {
  if (task != "classification") {
    config_error("algorithm='logistic' only supports classification. Use 'linear' for regression.")
  }
  y_int     <- as.integer(factor(y)) - 1L
  n_classes <- length(unique(y_int))
  C         <- dots[["C"]] %||% 1.0
  .lr_fit(X, y_int, n_classes, C = C, sample_weight = balance_weights)
}

.predict_logistic <- function(engine, X, task) {
  .lr_predict(engine, as.matrix(X))
}

.proba_logistic <- function(engine, X) {
  .lr_proba(engine, as.matrix(X))
}

# ── linear (native closed-form Ridge -- no external package) ───────────────────
# Implemented in R/linear.R: .lm_fit / .lm_predict
# Ridge: min ||y - Xw||^2 + alpha * ||w||^2, bias not regularised.

.fit_linear <- function(X, y, task, seed, dots, balance_weights = NULL) {
  if (task != "regression") {
    config_error("algorithm='linear' only supports regression. Use 'logistic' for classification.")
  }
  alpha <- dots[["alpha"]] %||% 1.0
  .lm_fit(X, y, alpha = alpha, sample_weight = balance_weights)
}

.predict_linear <- function(engine, X, task) {
  .lm_predict(engine, as.matrix(X))
}

# ── glmnet: elastic_net ────────────────────────────────────────────────────────

.fit_elastic_net <- function(X, y, task, seed, dots, balance_weights = NULL) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    config_error("'glmnet' required. Install with: install.packages('glmnet')")
  }
  alpha <- dots[["alpha"]] %||% 0.5
  if (task == "classification") {
    n_class <- length(unique(y))
    family <- if (n_class > 2) "multinomial" else "binomial"
    y_fac <- if (is.factor(y)) y else factor(y)
    if (!is.null(balance_weights)) {
      glmnet::glmnet(x = as.matrix(X), y = y_fac, alpha = alpha, family = family,
                     weights = balance_weights)
    } else {
      glmnet::glmnet(x = as.matrix(X), y = y_fac, alpha = alpha, family = family)
    }
  } else {
    glmnet::glmnet(x = as.matrix(X), y = as.numeric(y), alpha = alpha, family = "gaussian")
  }
}

.predict_elastic_net <- function(engine, X, task) {
  Xm <- as.matrix(X)
  if (task == "regression") {
    preds <- predict(engine, newx = Xm, type = "response")
    mid   <- ceiling(ncol(preds) / 2)
    return(as.numeric(preds[, mid]))
  }
  # multinomial (multnet class) vs binomial (lognet class)
  if (inherits(engine, "multnet")) {
    preds <- predict(engine, newx = Xm, type = "class")
    mid   <- ceiling(ncol(preds) / 2)
    as.integer(as.character(preds[, mid]))
  } else {
    preds <- predict(engine, newx = Xm, type = "response")
    mid   <- ceiling(ncol(preds) / 2)
    as.integer(as.numeric(preds[, mid]) >= 0.5)
  }
}

.proba_elastic_net <- function(engine, X) {
  Xm <- as.matrix(X)
  if (inherits(engine, "multnet")) {
    # multinomial: predict(type="response") returns n x nclass x nlambda array
    preds <- predict(engine, newx = Xm, type = "response")
    mid   <- ceiling(dim(preds)[3] / 2)
    pm    <- preds[, , mid]  # n x nclass matrix
    row_sums <- rowSums(pm)
    row_sums[row_sums == 0] <- 1
    as.data.frame(pm / row_sums)
  } else {
    preds <- predict(engine, newx = Xm, type = "response")
    mid   <- ceiling(ncol(preds) / 2)
    prob  <- as.numeric(preds[, mid])
    data.frame(class_0 = 1 - prob, class_1 = prob)
  }
}

# ── SVM (e1071) ────────────────────────────────────────────────────────────────

.fit_svm <- function(X, y, task, seed, dots, balance_weights = NULL) {
  if (!requireNamespace("e1071", quietly = TRUE)) {
    config_error("'e1071' required. Install with: install.packages('e1071')")
  }
  if (task == "classification") {
    y_factor <- if (is.factor(y)) y else factor(y)
    # e1071::svm uses class.weights (per-class), not sample weights
    cw <- NULL
    if (!is.null(balance_weights)) {
      classes <- levels(y_factor)
      cw <- vapply(classes, function(cl) balance_weights[which(y_factor == cl)[1]], numeric(1))
      names(cw) <- classes
    }
    # probability=TRUE required for predict_proba -- set at training time
    e1071::svm(x = as.matrix(X), y = y_factor, probability = TRUE,
               class.weights = cw)
  } else {
    e1071::svm(x = as.matrix(X), y = as.numeric(y), type = "eps-regression")
  }
}

.predict_svm <- function(engine, X, task) {
  pred <- predict(engine, newdata = as.matrix(X))
  if (task == "regression") return(as.numeric(pred))
  as.integer(as.character(pred))
}

.proba_svm <- function(engine, X) {
  pred <- predict(engine, newdata = as.matrix(X), probability = TRUE)
  pm   <- attr(pred, "probabilities")
  row_sums <- rowSums(pm)
  row_sums[row_sums == 0] <- 1
  as.data.frame(pm / row_sums)
}

# ── KNN (kknn) ─────────────────────────────────────────────────────────────────

.fit_knn <- function(X, y, task, seed, dots) {
  if (!requireNamespace("kknn", quietly = TRUE)) {
    config_error("'kknn' required. Install with: install.packages('kknn')")
  }
  k <- dots[["k"]] %||% 5L
  # kknn is lazy: store training data
  list(
    X_train = X,
    y_train = y,
    k       = k,
    task    = task
  )
}

.predict_knn <- function(engine, X, task) {
  df_train <- as.data.frame(engine$X_train)
  df_train$.y <- engine$y_train
  df_test  <- as.data.frame(X)
  fit <- kknn::kknn(
    formula = .y ~ .,
    train   = df_train,
    test    = df_test,
    k       = engine$k
  )
  if (task == "regression") return(fit$fitted.values)
  as.integer(as.character(fit$fitted.values))
}

.proba_knn <- function(engine, X) {
  df_train <- as.data.frame(engine$X_train)
  df_train$.y <- factor(engine$y_train)
  df_test  <- as.data.frame(X)
  fit <- kknn::kknn(
    formula = .y ~ .,
    train   = df_train,
    test    = df_test,
    k       = engine$k
  )
  pm <- fit$prob
  row_sums <- rowSums(pm)
  row_sums[row_sums == 0] <- 1
  as.data.frame(pm / row_sums)
}

# ── Naive Bayes (naivebayes) ───────────────────────────────────────────────────

.fit_naive_bayes <- function(X, y, task, seed, dots) {
  if (!requireNamespace("naivebayes", quietly = TRUE)) {
    config_error("'naivebayes' required. Install with: install.packages('naivebayes')")
  }
  if (task != "classification") {
    config_error("algorithm='naive_bayes' only supports classification.")
  }
  y_factor <- if (is.factor(y)) y else factor(y)
  naivebayes::naive_bayes(x = as.matrix(X), y = y_factor)
}

.predict_naive_bayes <- function(engine, X, task) {
  pred <- predict(engine, newdata = as.matrix(X), type = "class")
  as.integer(as.character(pred))
}

.proba_naive_bayes <- function(engine, X) {
  pm <- predict(engine, newdata = as.matrix(X), type = "prob")
  row_sums <- rowSums(pm)
  row_sums[row_sums == 0] <- 1
  as.data.frame(pm / row_sums)
}

# ── Decision Tree (rpart -- base R) ────────────────────────────────────────────

.fit_decision_tree <- function(X, y, task, seed, dots, balance_weights = NULL) {
  if (!requireNamespace("rpart", quietly = TRUE)) {
    config_error("Package 'rpart' required for algorithm='decision_tree'. Install with install.packages('rpart')")
  }
  max_depth <- dots[["max_depth"]] %||% 10L
  cp        <- dots[["cp"]]        %||% 0.01

  df <- as.data.frame(X)
  .local_seed(seed)
  if (task == "classification") {
    y_factor  <- if (is.factor(y)) y else factor(y)
    df[[".y"]] <- y_factor
    rpart::rpart(.y ~ ., data = df, method = "class",
                 weights = balance_weights,
                 control = rpart::rpart.control(maxdepth = max_depth, cp = cp))
  } else {
    df[[".y"]] <- as.numeric(y)
    rpart::rpart(.y ~ ., data = df, method = "anova",
                 control = rpart::rpart.control(maxdepth = max_depth, cp = cp))
  }
}

.predict_decision_tree <- function(engine, X, task) {
  df <- as.data.frame(X)
  if (task == "regression") {
    return(as.numeric(predict(engine, newdata = df)))
  }
  pred <- predict(engine, newdata = df, type = "class")
  as.integer(as.character(pred))
}

.proba_decision_tree <- function(engine, X) {
  df <- as.data.frame(X)
  pm <- predict(engine, newdata = df, type = "prob")
  row_sums <- rowSums(pm)
  row_sums[row_sums == 0] <- 1
  as.data.frame(pm / row_sums)
}

# ── Null coalescing helper ─────────────────────────────────────────────────────

`%||%` <- function(x, y) if (is.null(x)) y else x
