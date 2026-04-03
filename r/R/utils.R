#' List available ML algorithms
#'
#' Returns a data.frame showing which algorithms support classification and
#' regression, and which require optional packages.
#'
#' @param task Optional filter: "classification" or "regression"
#' @returns A data.frame with columns: algorithm, classification, regression,
#'   optional_dep, installed
#' @export
#' @examples
#' ml_algorithms()
#' ml_algorithms(task = "classification")
ml_algorithms <- function(task = NULL) {
  .algorithms_impl(task = task)
}

# Internal: returns sorted character vector of available algorithm names
.available_algorithms <- function() {
  avail <- "logistic"
  if (requireNamespace("xgboost",    quietly = TRUE)) avail <- c(avail, "xgboost")
  if (requireNamespace("ranger",     quietly = TRUE)) avail <- c(avail, "random_forest", "extra_trees")
  if (requireNamespace("e1071",      quietly = TRUE)) avail <- c(avail, "svm")
  if (requireNamespace("kknn",       quietly = TRUE)) avail <- c(avail, "knn")
  if (requireNamespace("glmnet",     quietly = TRUE)) avail <- c(avail, "linear", "elastic_net")
  if (requireNamespace("naivebayes", quietly = TRUE)) avail <- c(avail, "naive_bayes")
  # rpart is base R — always available
  avail <- c(avail, "decision_tree")
  # Rust backend: adds algorithms even without CRAN packages
  if (.rust_available()) {
    avail <- c(avail, "linear", "knn", "random_forest", "extra_trees",
               "naive_bayes", "elastic_net", "adaboost", "svm",
               "gradient_boosting", "histgradient")
  }
  sort(unique(avail))
}

.algorithms_impl <- function(task = NULL) {
  has_rust <- .rust_available()
  rows <- list(
    list(algorithm = "adaboost",          classification = TRUE,  regression = FALSE, optional_dep = ""),
    list(algorithm = "decision_tree",     classification = TRUE,  regression = TRUE,  optional_dep = "rpart"),
    list(algorithm = "elastic_net",       classification = FALSE, regression = TRUE,  optional_dep = "glmnet"),
    list(algorithm = "extra_trees",       classification = TRUE,  regression = TRUE,  optional_dep = ""),
    list(algorithm = "gradient_boosting", classification = TRUE,  regression = TRUE,  optional_dep = ""),
    list(algorithm = "histgradient",      classification = TRUE,  regression = TRUE,  optional_dep = ""),
    list(algorithm = "knn",               classification = TRUE,  regression = TRUE,  optional_dep = "kknn"),
    list(algorithm = "linear",            classification = FALSE, regression = TRUE,  optional_dep = "glmnet"),
    list(algorithm = "logistic",          classification = TRUE,  regression = FALSE, optional_dep = ""),
    list(algorithm = "naive_bayes",       classification = TRUE,  regression = FALSE, optional_dep = "naivebayes"),
    list(algorithm = "random_forest",     classification = TRUE,  regression = TRUE,  optional_dep = "ranger"),
    list(algorithm = "svm",               classification = TRUE,  regression = TRUE,  optional_dep = "e1071"),
    list(algorithm = "xgboost",           classification = TRUE,  regression = TRUE,  optional_dep = "xgboost")
  )

  rust_algos <- c("linear", "logistic", "random_forest", "decision_tree", "knn",
                  "extra_trees", "gradient_boosting", "histgradient",
                  "naive_bayes", "elastic_net", "adaboost", "svm")

  df <- as.data.frame(do.call(rbind, lapply(rows, function(r) {
    pkg <- r$optional_dep
    pkg_installed <- if (nchar(pkg) == 0) TRUE else requireNamespace(pkg, quietly = TRUE)
    rust_ok <- has_rust && r$algorithm %in% rust_algos
    installed <- pkg_installed || rust_ok
    engine_info <- if (rust_ok) "rust" else if (pkg_installed) "r" else "missing"
    list(
      algorithm      = r$algorithm,
      classification = r$classification,
      regression     = r$regression,
      optional_dep   = if (nchar(pkg) == 0) "(native)" else pkg,
      installed      = installed,
      engine         = engine_info
    )
  })), stringsAsFactors = FALSE)

  if (!is.null(task)) {
    if (task == "classification") df <- df[df$classification == TRUE, , drop = FALSE]
    else if (task == "regression")  df <- df[df$regression   == TRUE, , drop = FALSE]
  }

  rownames(df) <- NULL
  df
}

#' Load a built-in dataset
#'
#' Returns one of the built-in datasets. Useful for experimenting with the
#' ml API before applying it to your own data.
#'
#' Available datasets: "iris", "wine", "cancer", "diabetes", "houses",
#' "churn", "fraud"
#'
#' @param name Dataset name (string)
#' @returns A data.frame
#' @export
#' @examples
#' churn <- ml_dataset("churn")
#' head(churn)
ml_dataset <- function(name) {
  .dataset_impl(name = name)
}

.dataset_impl <- function(name) {
  switch(name,
    iris     = datasets::iris,
    wine     = .create_wine(),
    cancer   = .create_cancer(),
    diabetes = .create_diabetes(),
    houses   = .create_houses(),
    churn    = .create_churn(),
    fraud    = .create_fraud(),
    data_error(paste0(
      "Unknown dataset: '", name, "'. Choose from: iris, wine, cancer, diabetes, houses, churn, fraud"
    ))
  )
}

# ── Synthetic datasets ─────────────────────────────────────────────────────────

.create_churn <- function() {
  withr::local_seed(42L)
  n <- 1000L
  age              <- as.integer(stats::rnorm(n, 40, 12))
  monthly_charges  <- stats::runif(n, 20, 120)
  tenure_months    <- as.integer(stats::runif(n, 1, 72))
  num_products     <- sample(1L:4L, n, replace = TRUE)
  has_internet     <- sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(0.7, 0.3))
  contract_type    <- sample(c("month-to-month", "one_year", "two_year"), n,
                              replace = TRUE, prob = c(0.5, 0.3, 0.2))

  # Target: churn ~ logistic function of features
  logit <- -2 + 0.02 * (age - 40) + 0.015 * monthly_charges -
            0.03 * tenure_months + 0.1 * (num_products - 2) -
            0.5 * as.integer(has_internet) +
            ifelse(contract_type == "month-to-month", 0.8,
            ifelse(contract_type == "one_year", 0.2, -0.5))
  prob  <- 1 / (1 + exp(-logit))
  churn <- as.integer(stats::runif(n) < prob)

  data.frame(
    age             = age,
    monthly_charges = monthly_charges,
    tenure_months   = tenure_months,
    num_products    = num_products,
    has_internet    = has_internet,
    contract_type   = contract_type,
    churn           = churn,
    stringsAsFactors = FALSE
  )
}

.create_fraud <- function() {
  # 10,000 row synthetic credit card fraud dataset (2% fraud rate)
  # Matches Python _create_fraud_dataset() structure
  withr::local_seed(42L)
  n <- 10000L
  fraud_flag <- as.integer(stats::runif(n) < 0.02)  # 2% fraud rate

  amount        <- ifelse(fraud_flag == 1L,
                          stats::rlnorm(n, 6, 1),
                          stats::rlnorm(n, 4, 0.8))
  hour          <- as.integer(ifelse(fraud_flag == 1L,
                                     sample(c(0:5, 22:23), n, replace = TRUE),
                                     sample(8:20, n, replace = TRUE)))
  distance_km   <- ifelse(fraud_flag == 1L,
                          stats::rexp(n, rate = 0.005),
                          stats::rexp(n, rate = 0.02))
  n_trans_24h   <- ifelse(fraud_flag == 1L,
                          as.integer(stats::rpois(n, 8)),
                          as.integer(stats::rpois(n, 2)))
  # Category assigned by fraud class with different probabilities
  category_legit <- sample(c("groceries", "electronics", "travel", "dining", "online"),
                             n, replace = TRUE, prob = c(0.3, 0.2, 0.1, 0.25, 0.15))
  category_fraud <- sample(c("groceries", "electronics", "travel", "dining", "online"),
                             n, replace = TRUE, prob = c(0.1, 0.3, 0.3, 0.1, 0.2))
  category <- ifelse(fraud_flag == 1L, category_fraud, category_legit)

  # Inject ~5% NaN in amount and distance
  amount[sample(n, round(n * 0.05))]      <- NA_real_
  distance_km[sample(n, round(n * 0.05))] <- NA_real_

  data.frame(
    amount            = amount,
    hour              = hour,
    distance_km       = distance_km,
    n_transactions_24h = n_trans_24h,
    category          = category,
    fraud             = fraud_flag,
    stringsAsFactors  = FALSE
  )
}

.create_wine <- function() {
  # Bundled from UCI Wine dataset (178 rows, 13 features, 3 classes)
  # Using R's built-in datasets doesn't have this; generate approximate structure
  if (requireNamespace("datasets", quietly = TRUE)) {
    # R doesn't have wine natively; load from inst/extdata if available
    path <- system.file("extdata", "wine.rda", package = "ml")
    if (nchar(path) > 0 && file.exists(path)) {
      load(path)
      return(get("wine"))
    }
  }
  # Fallback: small synthetic wine-like dataset
  withr::local_seed(42L)
  n <- 178L
  data.frame(
    alcohol               = stats::rnorm(n, 13, 0.8),
    malic_acid            = abs(stats::rnorm(n, 2.3, 1.1)),
    ash                   = stats::rnorm(n, 2.4, 0.27),
    alcalinity_of_ash     = stats::rnorm(n, 19.5, 3.3),
    magnesium             = as.integer(abs(stats::rnorm(n, 99, 14))),
    total_phenols         = stats::rnorm(n, 2.3, 0.6),
    flavanoids            = stats::rnorm(n, 2.0, 1.0),
    nonflavanoid_phenols  = abs(stats::rnorm(n, 0.36, 0.12)),
    proanthocyanins       = stats::rnorm(n, 1.6, 0.57),
    color_intensity       = abs(stats::rnorm(n, 5.1, 2.3)),
    hue                   = stats::rnorm(n, 0.96, 0.23),
    od280_od315_diluted   = stats::rnorm(n, 2.6, 0.7),
    proline               = as.integer(abs(stats::rnorm(n, 746, 314))),
    target                = sample(c("class_1", "class_2", "class_3"), n,
                                    replace = TRUE, prob = c(1/3, 1/3, 1/3)),
    stringsAsFactors = FALSE
  )
}

.create_cancer <- function() {
  withr::local_seed(42L)
  n <- 569L
  data.frame(
    mean_radius          = abs(stats::rnorm(n, 14, 3.5)),
    mean_texture         = abs(stats::rnorm(n, 19, 4)),
    mean_perimeter       = abs(stats::rnorm(n, 92, 24)),
    mean_area            = abs(stats::rnorm(n, 655, 352)),
    mean_smoothness      = abs(stats::rnorm(n, 0.096, 0.014)),
    mean_compactness     = abs(stats::rnorm(n, 0.104, 0.053)),
    mean_concavity       = abs(stats::rnorm(n, 0.089, 0.080)),
    worst_radius         = abs(stats::rnorm(n, 16, 4.8)),
    worst_texture        = abs(stats::rnorm(n, 25, 6.1)),
    worst_area           = abs(stats::rnorm(n, 880, 570)),
    target               = sample(c("malignant", "benign"), n,
                                   replace = TRUE, prob = c(0.37, 0.63)),
    stringsAsFactors = FALSE
  )
}

.create_diabetes <- function() {
  withr::local_seed(42L)
  n <- 442L
  data.frame(
    age       = stats::rnorm(n, 0, 1),
    sex       = stats::rnorm(n, 0, 1),
    bmi       = stats::rnorm(n, 0, 1),
    bp        = stats::rnorm(n, 0, 1),
    s1        = stats::rnorm(n, 0, 1),
    s2        = stats::rnorm(n, 0, 1),
    s3        = stats::rnorm(n, 0, 1),
    s4        = stats::rnorm(n, 0, 1),
    s5        = stats::rnorm(n, 0, 1),
    s6        = stats::rnorm(n, 0, 1),
    target    = as.integer(abs(stats::rnorm(n, 150, 76))),
    stringsAsFactors = FALSE
  )
}

.create_houses <- function() {
  withr::local_seed(42L)
  n <- 1000L
  data.frame(
    median_income     = abs(stats::rnorm(n, 3.9, 1.9)),
    house_age         = abs(stats::rnorm(n, 28, 12)),
    avg_rooms         = abs(stats::rnorm(n, 5.4, 2.5)),
    avg_bedrooms      = abs(stats::rnorm(n, 1.1, 0.47)),
    population        = as.integer(abs(stats::rnorm(n, 1425, 1133))),
    avg_occupancy     = abs(stats::rnorm(n, 3.1, 10.4)),
    latitude          = stats::runif(n, 32, 42),
    longitude         = stats::runif(n, -124, -114),
    target            = abs(stats::rnorm(n, 2.07, 1.15)),
    stringsAsFactors  = FALSE
  )
}
