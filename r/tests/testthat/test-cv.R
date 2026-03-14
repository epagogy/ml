# ── ml_cv() ───────────────────────────────────────────────────────────────────

test_that("ml_cv creates correct number of folds", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L)
  expect_s3_class(c, "ml_cv_result")
  expect_length(c$folds, 5L)
})

test_that("ml_cv fold indices cover all dev rows", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L)
  n_dev <- nrow(s$dev)
  all_valid <- sort(unlist(lapply(c$folds, `[[`, "valid")))
  expect_equal(all_valid, seq_len(n_dev))
})

test_that("ml_cv folds are non-overlapping", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 3L, seed = 42L)
  valids <- lapply(c$folds, `[[`, "valid")
  for (i in seq_along(valids)) {
    for (j in seq_along(valids)) {
      if (i != j) expect_length(intersect(valids[[i]], valids[[j]]), 0L)
    }
  }
})

test_that("ml_cv result feeds into ml_fit", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 3L, seed = 42L)
  model <- ml_fit(c, "Species", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_false(is.null(model$scores_))
})

test_that("ml_cv seed reproducibility", {
  s <- ml_split(iris, "Species", seed = 42L)
  c1 <- ml_cv(s, "Species", folds = 5L, seed = 99L)
  c2 <- ml_cv(s, "Species", folds = 5L, seed = 99L)
  expect_identical(c1$folds[[1]]$valid, c2$folds[[1]]$valid)
})

test_that("ml_cv rejects non-split input", {
  expect_error(ml_cv(iris, "Species"), "ml_split_result")
})

test_that("ml_cv rejects folds < 2", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_error(ml_cv(s, "Species", folds = 1L, seed = 42L), "folds")
})

test_that("ml_cv rejects missing target", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_error(ml_cv(s, "nonexistent", folds = 3L, seed = 42L), "nonexistent")
})

test_that("ml_cv print method works", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L)
  out <- capture.output(print(c))
  expect_true(any(grepl("CV", out)))
  expect_true(any(grepl("5 folds", out)))
})

# ── ml_cv_temporal() ─────────────────────────────────────────────────────────

test_that("ml_cv_temporal creates expanding-window folds", {
  df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
  s <- ml_split_temporal(df, "y", time = "date")
  c <- ml_cv_temporal(s, "y", folds = 3L)
  expect_s3_class(c, "ml_cv_result")
  expect_length(c$folds, 3L)
  # Expanding: fold 2 train should be larger than fold 1 train
  expect_true(length(c$folds[[2]]$train) > length(c$folds[[1]]$train))
})

# ── ml_cv_group() ────────────────────────────────────────────────────────────

test_that("ml_cv_group no group leakage across folds", {
  df <- data.frame(
    pid = rep(1:20, each = 5),
    x = rnorm(100),
    y = sample(0:1, 100, TRUE)
  )
  s <- ml_split(df, "y", seed = 42L)
  c <- ml_cv_group(s, "y", groups = "pid", folds = 3L, seed = 42L)
  expect_s3_class(c, "ml_cv_result")
  # Check no group appears in both train and valid of any fold
  dev_data <- s$dev
  for (fold in c$folds) {
    train_groups <- unique(dev_data$pid[fold$train])
    valid_groups <- unique(dev_data$pid[fold$valid])
    expect_length(intersect(train_groups, valid_groups), 0L)
  }
})

test_that("ml_cv_group rejects missing groups column", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_error(ml_cv_group(s, "Species", groups = "nope", seed = 42L), "nope")
})

# ── ml_cv_temporal: embargo + window ─────────────────────────────────────────

test_that("ml_cv_temporal embargo shifts valid start by embargo rows", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
  s  <- ml_split_temporal(df, "target", time = "date")
  c0 <- ml_cv_temporal(s, "target", folds = 3L)
  ce <- ml_cv_temporal(s, "target", folds = 3L, embargo = 5L)
  # With embargo, valid starts later → fewer valid rows in first folds
  expect_gte(length(c0$folds[[1L]]$valid), length(ce$folds[[1L]]$valid))
})

test_that("ml_cv_temporal rejects negative embargo", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
  s <- ml_split_temporal(df, "target", time = "date")
  expect_error(
    ml_cv_temporal(s, "target", folds = 3L, embargo = -1L),
    regexp = "embargo.*>= 0|embargo.*negative"
  )
})

test_that("ml_cv_temporal sliding window requires window_size", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
  s <- ml_split_temporal(df, "target", time = "date")
  expect_error(
    ml_cv_temporal(s, "target", folds = 3L, window = "sliding"),
    regexp = "window_size"
  )
})

test_that("ml_cv_temporal sliding window creates folds", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
  s <- ml_split_temporal(df, "target", time = "date")
  c <- ml_cv_temporal(s, "target", folds = 3L, window = "sliding",
                       window_size = 50L)
  expect_s3_class(c, "ml_cv_result")
  expect_gte(length(c$folds), 1L)
})
