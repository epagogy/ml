# Shared test data fixtures

# Small iris split (fast, deterministic)
iris_split <- function(seed = 42L) {
  ml_split(iris, "Species", seed = seed)
}

# Small mtcars split (regression)
mtcars_split <- function(seed = 42L) {
  ml_split(mtcars, "mpg", seed = seed)
}

# Tiny dataset for fast tests (30 rows)
tiny_clf <- function() {
  set.seed(42L)
  data.frame(
    x1     = stats::rnorm(30L),
    x2     = stats::rnorm(30L),
    target = sample(c("a", "b"), 30L, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

tiny_reg <- function() {
  set.seed(42L)
  data.frame(
    x1 = stats::rnorm(30L),
    x2 = stats::rnorm(30L),
    y  = stats::rnorm(30L)
  )
}

# Binary classification (churn-like)
binary_df <- function(n = 100L, seed = 42L) {
  set.seed(seed)
  data.frame(
    age     = stats::rnorm(n, 40, 10),
    income  = abs(stats::rnorm(n, 50000, 20000)),
    churn   = sample(c("yes", "no"), n, replace = TRUE, prob = c(0.3, 0.7)),
    stringsAsFactors = FALSE
  )
}

# Multiclass 4-class dataset
multi_df <- function(n = 120L, seed = 42L) {
  set.seed(seed)
  data.frame(
    x1     = stats::rnorm(n),
    x2     = stats::rnorm(n),
    x3     = stats::rnorm(n),
    target = sample(c("A", "B", "C", "D"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}
