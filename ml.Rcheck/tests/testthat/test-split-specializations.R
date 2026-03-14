# Tests for ml_split_temporal() and ml_split_group() domain specializations

# ── Temporal holdout ─────────────────────────────────────────────────────────

test_that("ml_split_temporal produces ml_split_result", {
  df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
  s <- ml_split_temporal(df, "y", time = "date")
  expect_s3_class(s, "ml_split_result")
})

test_that("ml_split_temporal drops time column", {
  df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
  s <- ml_split_temporal(df, "y", time = "date")
  expect_false("date" %in% names(s$train))
  expect_false("date" %in% names(s$test))
})

test_that("ml_split_temporal tags partitions", {
  df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
  s <- ml_split_temporal(df, "y", time = "date")
  expect_equal(attr(s$train, "_ml_partition"), "train")
  expect_equal(attr(s$valid, "_ml_partition"), "valid")
  expect_equal(attr(s$test, "_ml_partition"), "test")
})

test_that("ml_split_temporal is deterministic", {
  df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
  s1 <- ml_split_temporal(df, "y", time = "date")
  s2 <- ml_split_temporal(df, "y", time = "date")
  expect_equal(s1$train, s2$train)
})

test_that("ml_split_temporal equivalent to ml_split(time=)", {
  df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
  s1 <- ml_split_temporal(df, "y", time = "date")
  s2 <- ml_split(df, "y", time = "date")
  expect_equal(nrow(s1$train), nrow(s2$train))
  expect_equal(nrow(s1$test), nrow(s2$test))
})

test_that("ml_split_temporal with folds produces ml_cv_result", {
  df <- data.frame(date = 1:200, x = rnorm(200), y = sample(0:1, 200, TRUE))
  cv <- ml_split_temporal(df, "y", time = "date", folds = 5)
  expect_s3_class(cv, "ml_cv_result")
  expect_equal(cv$k, 5L)
})

test_that("ml_split_temporal errors on missing time column", {
  df <- data.frame(x = rnorm(100), y = sample(0:1, 100, TRUE))
  expect_error(ml_split_temporal(df, "y", time = "nonexistent"), "not found")
})

test_that("ml_split_temporal dev property works", {
  df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
  s <- ml_split_temporal(df, "y", time = "date")
  expect_equal(nrow(s$dev), nrow(s$train) + nrow(s$valid))
})

# ── Group holdout ────────────────────────────────────────────────────────────

test_that("ml_split_group produces ml_split_result", {
  df <- data.frame(pid = rep(1:10, each = 5), x = rnorm(50), y = sample(0:1, 50, TRUE))
  s <- ml_split_group(df, "y", groups = "pid", seed = 42)
  expect_s3_class(s, "ml_split_result")
})

test_that("ml_split_group has no group overlap", {
  df <- data.frame(pid = rep(1:20, each = 5), x = rnorm(100), y = sample(0:1, 100, TRUE))
  s <- ml_split_group(df, "y", groups = "pid", seed = 42)
  train_g <- unique(s$train$pid)
  valid_g <- unique(s$valid$pid)
  test_g  <- unique(s$test$pid)
  expect_length(intersect(train_g, valid_g), 0)
  expect_length(intersect(train_g, test_g), 0)
  expect_length(intersect(valid_g, test_g), 0)
})

test_that("ml_split_group tags partitions", {
  df <- data.frame(pid = rep(1:10, each = 5), x = rnorm(50), y = sample(0:1, 50, TRUE))
  s <- ml_split_group(df, "y", groups = "pid", seed = 42)
  expect_equal(attr(s$train, "_ml_partition"), "train")
  expect_equal(attr(s$valid, "_ml_partition"), "valid")
  expect_equal(attr(s$test, "_ml_partition"), "test")
})

test_that("ml_split_group reproducible with same seed", {
  df <- data.frame(pid = rep(1:10, each = 5), x = rnorm(50), y = sample(0:1, 50, TRUE))
  s1 <- ml_split_group(df, "y", groups = "pid", seed = 42)
  s2 <- ml_split_group(df, "y", groups = "pid", seed = 42)
  expect_equal(sort(unique(s1$train$pid)), sort(unique(s2$train$pid)))
})

test_that("ml_split_group equivalent to ml_split(groups=)", {
  df <- data.frame(pid = rep(1:10, each = 5), x = rnorm(50), y = sample(0:1, 50, TRUE))
  s1 <- ml_split_group(df, "y", groups = "pid", seed = 42)
  s2 <- ml_split(df, "y", groups = "pid", seed = 42)
  expect_equal(sort(unique(s1$train$pid)), sort(unique(s2$train$pid)))
})

test_that("ml_split_group with folds produces ml_cv_result", {
  df <- data.frame(pid = rep(1:20, each = 5), x = rnorm(100), y = sample(0:1, 100, TRUE))
  cv <- ml_split_group(df, "y", groups = "pid", folds = 4, seed = 42)
  expect_s3_class(cv, "ml_cv_result")
  expect_equal(cv$k, 4L)
})

test_that("ml_split_group errors on missing groups column", {
  df <- data.frame(x = rnorm(50), y = sample(0:1, 50, TRUE))
  expect_error(ml_split_group(df, "y", groups = "nonexistent"), "not found")
})

# ── No-target usage ──────────────────────────────────────────────────────────

test_that("ml_split_temporal works without target", {
  df <- data.frame(date = 1:100, x = rnorm(100))
  s <- ml_split_temporal(df, time = "date")
  expect_s3_class(s, "ml_split_result")
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), 100L)
})

test_that("ml_split_group works without target", {
  df <- data.frame(pid = rep(1:10, each = 5), x = rnorm(50))
  s <- ml_split_group(df, groups = "pid", seed = 42)
  expect_s3_class(s, "ml_split_result")
})
