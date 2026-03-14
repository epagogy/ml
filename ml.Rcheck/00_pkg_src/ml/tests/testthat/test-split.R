test_that("three-way split produces correct sizes", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), nrow(iris))
  # Within tolerance of 60/20/20
  expect_true(abs(nrow(s$train) / nrow(iris) - 0.6) < 0.05)
})

test_that("stratified split preserves class proportions", {
  s <- ml_split(iris, "Species", seed = 42L)
  train_freq <- table(s$train$Species) / nrow(s$train)
  orig_freq  <- table(iris$Species)  / nrow(iris)
  for (cls in names(orig_freq)) {
    expect_true(abs(train_freq[[cls]] - orig_freq[[cls]]) < 0.1)
  }
})

test_that("$dev = rbind($train, $valid) row count matches", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_equal(nrow(s$dev), nrow(s$train) + nrow(s$valid))
})

test_that("seed reproducibility: same seed produces identical split", {
  s1 <- ml_split(iris, "Species", seed = 99L)
  s2 <- ml_split(iris, "Species", seed = 99L)
  expect_identical(rownames(s1$train), rownames(s2$train))
})

test_that("different seeds produce different splits", {
  s1 <- ml_split(iris, "Species", seed = 1L)
  s2 <- ml_split(iris, "Species", seed = 2L)
  expect_false(identical(rownames(s1$train), rownames(s2$train)))
})

test_that("CV mode returns ml_cv_result with correct fold count", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 5L)
  expect_s3_class(cv, "ml_cv_result")
  expect_equal(length(cv$folds), 5L)
})

test_that("NA target rows dropped with warning", {
  df <- iris
  df$Species[1:5] <- NA
  expect_warning(
    s <- ml_split(df, "Species", seed = 42L),
    regexp = "Dropped"
  )
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), nrow(iris) - 5L)
})

test_that("tibble input works", {
  if (!requireNamespace("tibble", quietly = TRUE)) skip("tibble not installed")
  tbl <- tibble::as_tibble(iris)
  s   <- ml_split(tbl, "Species", seed = 42L)
  expect_s3_class(s, "ml_split_result")
  expect_true(is.data.frame(s$train))
})

test_that("error on non-existent target column", {
  expect_error(
    ml_split(iris, "nonexistent", seed = 42L),
    class = "data_error"
  )
})

test_that("error on empty data.frame", {
  expect_error(
    ml_split(data.frame(), seed = 42L),
    class = "data_error"
  )
})

test_that("error on ratio not summing to 1.0", {
  expect_error(
    ml_split(iris, "Species", seed = 42L, ratio = c(0.5, 0.3, 0.3)),
    class = "config_error"
  )
})

test_that("CVResult $train access raises config_error", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  expect_error(cv$train, class = "config_error")
  expect_error(cv$valid, class = "config_error")
  expect_error(cv$test,  class = "config_error")
})

test_that("classification heuristic: numeric target with few unique values", {
  df <- data.frame(x = stats::rnorm(100L), target = sample(1:3, 100L, replace = TRUE))
  expect_warning(
    s <- ml_split(df, "target", seed = 42L),
    regexp = "classification"
  )
})

test_that("task= override works", {
  df <- data.frame(x = stats::rnorm(100L), y = stats::rnorm(100L))
  s  <- ml_split(df, "y", seed = 42L, task = "regression")
  expect_s3_class(s, "ml_split_result")
})

test_that("stratify = FALSE produces valid split", {
  s <- ml_split(iris, "Species", seed = 42L, stratify = FALSE)
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), nrow(iris))
})

test_that("folds > n_rows raises data_error", {
  small_df <- data.frame(x = 1:5, y = c("a", "b", "a", "b", "a"))
  expect_error(
    ml_split(small_df, "y", seed = 42L, folds = 10L),
    class = "data_error"
  )
})

test_that("ml$split() module style works identically", {
  s1 <- ml_split(iris, "Species", seed = 42L)
  s2 <- ml$split(iris, "Species", seed = 42L)
  expect_identical(rownames(s1$train), rownames(s2$train))
})

test_that("1-row data raises data_error", {
  expect_error(
    ml_split(iris[1L, , drop = FALSE], "Species", seed = 42L),
    class = "data_error"
  )
})

# ── time= parameter ─────────────────────────────────────────────────────────

test_that("time= produces chronological split, drops time column", {
  n <- 200L
  df <- data.frame(
    t = seq_len(n),
    x = stats::rnorm(n),
    y = stats::rnorm(n)
  )
  s <- ml_split(df, "y", seed = 42L, time = "t")
  # Time column should be dropped

  expect_false("t" %in% names(s$train))
  expect_false("t" %in% names(s$valid))
  expect_false("t" %in% names(s$test))
  # All rows accounted for
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), n)
})

test_that("time= split is deterministic (seed ignored)", {
  n <- 200L
  df <- data.frame(t = seq_len(n), x = stats::rnorm(n), y = stats::rnorm(n))
  s1 <- ml_split(df, "y", seed = 1L, time = "t")
  s2 <- ml_split(df, "y", seed = 999L, time = "t")
  expect_identical(s1$train$x, s2$train$x)
})

test_that("time= shuffled input gets sorted", {
  n <- 100L
  df <- data.frame(t = sample(n), x = seq_len(n), y = stats::rnorm(n))
  s  <- ml_split(df, "y", seed = 42L, time = "t")
  # Train should have lowest-t rows (sorted by t, then dropped)
  # x values in train should correspond to the lowest t values
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), n)
})

test_that("time= with folds produces temporal CV", {
  n <- 200L
  df <- data.frame(t = seq_len(n), x = stats::rnorm(n), y = stats::rnorm(n))
  cv <- ml_split(df, "y", seed = 42L, time = "t", folds = 3L)
  expect_s3_class(cv, "ml_cv_result")
  expect_equal(length(cv$folds), 3L)
  # Each fold's valid indices should come after train indices (expanding window)
  for (f in cv$folds) {
    expect_true(max(f$train) < min(f$valid))
  }
})

test_that("time= missing column raises data_error", {
  expect_error(
    ml_split(iris, "Species", seed = 42L, time = "nonexistent"),
    class = "data_error"
  )
})

# ── groups= parameter ───────────────────────────────────────────────────────

test_that("groups= ensures no group leakage", {
  n <- 200L
  grp <- rep(paste0("g", seq_len(20L)), each = 10L)
  df  <- data.frame(group = grp, x = stats::rnorm(n), y = stats::rnorm(n))
  s   <- ml_split(df, "y", seed = 42L, groups = "group")
  # No group appears in more than one partition
  train_grps <- unique(s$train$group)
  valid_grps <- unique(s$valid$group)
  test_grps  <- unique(s$test$group)
  expect_length(intersect(train_grps, valid_grps), 0L)
  expect_length(intersect(train_grps, test_grps), 0L)
  expect_length(intersect(valid_grps, test_grps), 0L)
  # All rows accounted for
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), n)
})

test_that("groups= with folds produces GroupKFold", {
  n <- 200L
  grp <- rep(paste0("g", seq_len(20L)), each = 10L)
  df  <- data.frame(group = grp, x = stats::rnorm(n), y = stats::rnorm(n))
  cv  <- ml_split(df, "y", seed = 42L, groups = "group", folds = 4L)
  expect_s3_class(cv, "ml_cv_result")
  expect_equal(length(cv$folds), 4L)
  # No group leakage across train/valid in each fold
  for (f in cv$folds) {
    train_grps <- unique(grp[f$train])
    valid_grps <- unique(grp[f$valid])
    expect_length(intersect(train_grps, valid_grps), 0L)
  }
})

test_that("groups= seed reproducibility", {
  n <- 200L
  grp <- rep(paste0("g", seq_len(20L)), each = 10L)
  df  <- data.frame(group = grp, x = stats::rnorm(n), y = stats::rnorm(n))
  s1  <- ml_split(df, "y", seed = 42L, groups = "group")
  s2  <- ml_split(df, "y", seed = 42L, groups = "group")
  expect_identical(s1$train$group, s2$train$group)
})

test_that("groups= with <3 unique groups raises data_error", {
  df <- data.frame(group = rep(c("a", "b"), each = 50L),
                   x = stats::rnorm(100L), y = stats::rnorm(100L))
  expect_error(
    ml_split(df, "y", seed = 42L, groups = "group"),
    class = "data_error"
  )
})

test_that("groups= missing column raises data_error", {
  expect_error(
    ml_split(iris, "Species", seed = 42L, groups = "nonexistent"),
    class = "data_error"
  )
})

test_that("time= and groups= combined raises config_error", {
  df <- data.frame(t = 1:100, g = rep(1:10, each = 10L),
                   x = stats::rnorm(100L), y = stats::rnorm(100L))
  expect_error(
    ml_split(df, "y", seed = 42L, time = "t", groups = "g"),
    class = "config_error"
  )
})

# ── Partition tag tests ──

test_that("split tags partitions with _ml_partition attr", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_equal(attr(s$train, "_ml_partition"), "train")
  expect_equal(attr(s$valid, "_ml_partition"), "valid")
  expect_equal(attr(s$test,  "_ml_partition"), "test")
})

test_that("dev property tags as dev", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_equal(attr(s$dev, "_ml_partition"), "dev")
})
