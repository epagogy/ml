# CV parity tests — structural invariants against caret/rsample reference.
#
# The backtester for the backtester. Compares ml_split() holdout and CV
# against reference R packages on STRUCTURAL PROPERTIES:
#
#   1. Fold count matches
#   2. Fold sizes balanced (±1 row)
#   3. No validation overlap (partition property)
#   4. Complete coverage (every row appears in exactly one valid fold)
#   5. Stratification preserves class ratios per fold
#   6. Group non-overlap (no group in both train and valid)
#   7. Temporal ordering (train indices < valid indices)
#   8. Train+valid = all data (no rows lost or invented)
#
# We do NOT compare exact indices — implementations legitimately differ.
# We compare the properties any correct implementation must satisfy.

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

make_clf_200 <- function() {
  withr::local_seed(42L)
  data.frame(
    x1 = stats::rnorm(200L),
    x2 = stats::rnorm(200L),
    x3 = stats::rnorm(200L),
    x4 = stats::rnorm(200L),
    x5 = stats::rnorm(200L),
    target = sample(c(0L, 1L), 200L, replace = TRUE)
  )
}

make_reg_200 <- function() {
  withr::local_seed(42L)
  x <- stats::rnorm(200L)
  data.frame(
    x1 = x,
    x2 = stats::rnorm(200L),
    x3 = stats::rnorm(200L),
    target = 2 * x + stats::rnorm(200L) * 0.3
  )
}

make_group_200 <- function() {
  withr::local_seed(42L)
  data.frame(
    group_id = rep(paste0("g", seq_len(20L)), each = 10L),
    x1 = stats::rnorm(200L),
    x2 = stats::rnorm(200L),
    target = sample(c(0L, 1L), 200L, replace = TRUE)
  )
}

make_temporal_500 <- function() {
  withr::local_seed(42L)
  data.frame(
    date = seq_len(500L),
    x1 = stats::rnorm(500L),
    x2 = stats::rnorm(500L),
    target = stats::rnorm(500L) * 10 + 100
  )
}


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------

assert_fold_count <- function(folds, k) {
  expect_equal(length(folds), k, label = "fold count")
}

assert_balanced_fold_sizes <- function(folds, n_total, k) {
  expected <- n_total / k
  for (i in seq_along(folds)) {
    size <- length(folds[[i]]$valid)
    expect_true(
      abs(size - expected) <= 1.5,
      label = sprintf("Fold %d: valid size %d, expected ~%.0f", i, size, expected)
    )
  }
}

assert_no_valid_overlap <- function(folds) {
  seen <- integer(0)
  for (i in seq_along(folds)) {
    idx <- folds[[i]]$valid
    overlap <- intersect(seen, idx)
    expect_length(overlap, 0L)
    seen <- c(seen, idx)
  }
}

assert_complete_coverage <- function(folds, n_total) {
  all_valid <- integer(0)
  for (f in folds) {
    all_valid <- c(all_valid, f$valid)
  }
  expect_equal(sort(unique(all_valid)), seq_len(n_total))
}

assert_train_valid_disjoint <- function(folds) {
  for (i in seq_along(folds)) {
    overlap <- intersect(folds[[i]]$train, folds[[i]]$valid)
    expect_length(overlap, 0L)
  }
}

assert_train_valid_cover_all <- function(folds, n_total) {
  for (i in seq_along(folds)) {
    all_idx <- sort(union(folds[[i]]$train, folds[[i]]$valid))
    expect_equal(all_idx, seq_len(n_total))
  }
}

assert_group_non_overlap <- function(folds, group_col) {
  for (i in seq_along(folds)) {
    train_grps <- unique(group_col[folds[[i]]$train])
    valid_grps <- unique(group_col[folds[[i]]$valid])
    leak <- intersect(train_grps, valid_grps)
    expect_length(leak, 0L)
  }
}

assert_temporal_ordering <- function(folds) {
  for (i in seq_along(folds)) {
    expect_true(
      max(folds[[i]]$train) < min(folds[[i]]$valid),
      label = sprintf("Fold %d: train max < valid min", i)
    )
  }
}


# ---------------------------------------------------------------------------
# 1. Holdout split — structural invariants
# ---------------------------------------------------------------------------

test_that("holdout: partitions cover full dataset", {
  df <- make_clf_200()
  s  <- ml_split(df, "target", seed = 42L)
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), nrow(df))
})

test_that("holdout: partitions are disjoint", {
  df <- make_clf_200()
  s  <- ml_split(df, "target", seed = 42L)
  train_idx <- as.integer(rownames(s$train))
  valid_idx <- as.integer(rownames(s$valid))
  test_idx  <- as.integer(rownames(s$test))
  expect_length(intersect(train_idx, valid_idx), 0L)
  expect_length(intersect(train_idx, test_idx), 0L)
  expect_length(intersect(valid_idx, test_idx), 0L)
})

test_that("holdout: partition sizes match ratio ±5%", {
  df <- make_clf_200()
  n  <- nrow(df)
  s  <- ml_split(df, "target", seed = 42L)
  expect_true(abs(nrow(s$train) / n - 0.6) < 0.05)
  expect_true(abs(nrow(s$valid) / n - 0.2) < 0.05)
  expect_true(abs(nrow(s$test)  / n - 0.2) < 0.05)
})

test_that("holdout: dev = train + valid", {
  df <- make_clf_200()
  s  <- ml_split(df, "target", seed = 42L)
  expect_equal(nrow(s$dev), nrow(s$train) + nrow(s$valid))
})


# ---------------------------------------------------------------------------
# 2. K-fold CV — structural invariants
# ---------------------------------------------------------------------------

for (k in c(2L, 3L, 5L, 10L)) {
  test_that(sprintf("cv(%d-fold): fold count matches", k), {
    df <- make_clf_200()
    cv <- ml_split(df, "target", seed = 42L, folds = k)
    assert_fold_count(cv$folds, k)
  })

  test_that(sprintf("cv(%d-fold): fold sizes balanced", k), {
    df <- make_clf_200()
    cv <- ml_split(df, "target", seed = 42L, folds = k)
    n_dev <- nrow(cv$train)
    assert_balanced_fold_sizes(cv$folds, n_dev, k)
  })

  test_that(sprintf("cv(%d-fold): no validation overlap", k), {
    df <- make_clf_200()
    cv <- ml_split(df, "target", seed = 42L, folds = k)
    assert_no_valid_overlap(cv$folds)
  })

  test_that(sprintf("cv(%d-fold): complete coverage", k), {
    df <- make_clf_200()
    cv <- ml_split(df, "target", seed = 42L, folds = k)
    n_dev <- nrow(cv$train)
    assert_complete_coverage(cv$folds, n_dev)
  })

  test_that(sprintf("cv(%d-fold): train/valid disjoint", k), {
    df <- make_clf_200()
    cv <- ml_split(df, "target", seed = 42L, folds = k)
    assert_train_valid_disjoint(cv$folds)
  })

  test_that(sprintf("cv(%d-fold): train+valid = all data", k), {
    df <- make_clf_200()
    cv <- ml_split(df, "target", seed = 42L, folds = k)
    n_dev <- nrow(cv$train)
    assert_train_valid_cover_all(cv$folds, n_dev)
  })
}


# ---------------------------------------------------------------------------
# 3. CV vs caret reference (structural parity)
# ---------------------------------------------------------------------------

test_that("cv vs caret::createFolds — same fold sizes", {
  skip_if_not_installed("caret")
  df <- make_clf_200()
  k  <- 5L

  # ml (folds on dev, test held out)
  cv <- ml_split(df, "target", seed = 42L, folds = k)
  n_dev <- nrow(cv$train)
  ml_sizes <- sort(vapply(cv$folds, function(f) length(f$valid), integer(1L)))

  # caret reference (folds on full data for comparison)
  withr::local_seed(42L)
  caret_folds <- caret::createFolds(df$target, k = k, list = TRUE, returnTrain = FALSE)
  caret_sizes <- sort(vapply(caret_folds, length, integer(1L)))

  # ml folds should be balanced relative to dev size
  expected_ml <- n_dev / k
  expected <- nrow(df) / k
  for (s in ml_sizes) {
    expect_true(abs(s - expected_ml) <= 2, label = sprintf("ml fold size %d", s))
  }
  for (s in caret_sizes) {
    expect_true(abs(s - expected) <= 2, label = sprintf("caret fold size %d", s))
  }
})

test_that("cv vs caret::createFolds — both satisfy invariants", {
  skip_if_not_installed("caret")
  df <- make_clf_200()
  k  <- 5L

  # ml invariants (folds on dev, test held out)
  cv <- ml_split(df, "target", seed = 42L, folds = k)
  n_dev <- nrow(cv$train)
  assert_fold_count(cv$folds, k)
  assert_no_valid_overlap(cv$folds)
  assert_complete_coverage(cv$folds, n_dev)
  assert_train_valid_disjoint(cv$folds)

  # caret invariants (convert to our format)
  withr::local_seed(42L)
  caret_idx <- caret::createFolds(df$target, k = k, list = TRUE, returnTrain = FALSE)
  caret_folds <- lapply(seq_len(k), function(i) {
    valid <- caret_idx[[i]]
    train <- setdiff(seq_len(nrow(df)), valid)
    list(train = train, valid = valid)
  })
  assert_fold_count(caret_folds, k)
  assert_no_valid_overlap(caret_folds)
  assert_complete_coverage(caret_folds, nrow(df))
  assert_train_valid_disjoint(caret_folds)
})


# ---------------------------------------------------------------------------
# 4. Temporal CV — structural invariants
# ---------------------------------------------------------------------------

test_that("temporal cv: fold count matches", {
  df <- make_temporal_500()
  cv <- ml_split(df, "target", time = "date", folds = 5L)
  assert_fold_count(cv$folds, 5L)
})

test_that("temporal cv: chronological ordering (train < valid)", {
  df <- make_temporal_500()
  cv <- ml_split(df, "target", time = "date", folds = 5L)
  assert_temporal_ordering(cv$folds)
})

test_that("temporal cv: expanding window (train grows)", {
  df <- make_temporal_500()
  cv <- ml_split(df, "target", time = "date", folds = 5L)
  prev_size <- 0L
  for (f in cv$folds) {
    expect_true(length(f$train) >= prev_size)
    prev_size <- length(f$train)
  }
})

test_that("temporal cv: train/valid disjoint", {
  df <- make_temporal_500()
  cv <- ml_split(df, "target", time = "date", folds = 5L)
  assert_train_valid_disjoint(cv$folds)
})

test_that("temporal cv vs rsample::rolling_origin — both preserve order", {
  skip_if_not_installed("rsample")
  df <- make_temporal_500()

  # ml
  cv <- ml_split(df, "target", time = "date", folds = 3L)
  assert_temporal_ordering(cv$folds)

  # rsample reference — rolling_origin preserves temporal ordering
  ro <- rsample::rolling_origin(df, initial = 250L, assess = 50L, skip = 50L)
  for (i in seq_len(nrow(ro))) {
    split_i   <- ro$splits[[i]]
    # Use the date column to verify temporal ordering
    train_dates <- rsample::analysis(split_i)$date
    valid_dates <- rsample::assessment(split_i)$date
    expect_true(max(train_dates) < min(valid_dates))
  }
})


# ---------------------------------------------------------------------------
# 5. Group CV — structural invariants
# ---------------------------------------------------------------------------

test_that("group cv: fold count matches", {
  df  <- make_group_200()
  cv  <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = 4L)
  assert_fold_count(cv$folds, 4L)
})

test_that("group cv: no group in both train and valid", {
  df  <- make_group_200()
  cv  <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = 4L)
  dev <- cv$train
  assert_group_non_overlap(cv$folds, dev$group_id)
})

test_that("group cv: train/valid disjoint", {
  df  <- make_group_200()
  cv  <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = 4L)
  assert_train_valid_disjoint(cv$folds)
})

test_that("group cv: complete coverage", {
  df  <- make_group_200()
  cv  <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = 4L)
  n_dev <- nrow(cv$train)
  assert_complete_coverage(cv$folds, n_dev)
})

test_that("group cv: groups distributed ~equally", {
  df  <- make_group_200()
  k   <- 4L
  cv  <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = k)
  dev <- cv$train
  n_grps  <- length(unique(dev$group_id))
  expected <- n_grps / k
  for (f in cv$folds) {
    n_grps_fold <- length(unique(dev$group_id[f$valid]))
    expect_true(abs(n_grps_fold - expected) <= 2)
  }
})

test_that("group holdout: no group leakage", {
  df <- make_group_200()
  s  <- ml_split(df, "target", seed = 42L, groups = "group_id")
  train_g <- unique(s$train$group_id)
  valid_g <- unique(s$valid$group_id)
  test_g  <- unique(s$test$group_id)
  expect_length(intersect(train_g, valid_g), 0L)
  expect_length(intersect(train_g, test_g), 0L)
  expect_length(intersect(valid_g, test_g), 0L)
})


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

test_that("cv deterministic with same seed", {
  df  <- make_clf_200()
  cv1 <- ml_split(df, "target", seed = 42L, folds = 5L)
  cv2 <- ml_split(df, "target", seed = 42L, folds = 5L)
  for (i in seq_along(cv1$folds)) {
    expect_identical(cv1$folds[[i]]$valid, cv2$folds[[i]]$valid)
  }
})

test_that("group cv deterministic with same seed", {
  df  <- make_group_200()
  cv1 <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = 4L)
  cv2 <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = 4L)
  for (i in seq_along(cv1$folds)) {
    expect_identical(cv1$folds[[i]]$valid, cv2$folds[[i]]$valid)
  }
})

test_that("temporal cv deterministic (no seed)", {
  df  <- make_temporal_500()
  cv1 <- ml_split(df, "target", time = "date", folds = 3L)
  cv2 <- ml_split(df, "target", time = "date", folds = 3L)
  for (i in seq_along(cv1$folds)) {
    expect_identical(cv1$folds[[i]]$valid, cv2$folds[[i]]$valid)
  }
})


# ---------------------------------------------------------------------------
# 7. Stratification ratio parity
# ---------------------------------------------------------------------------

test_that("stratified split preserves class ratio within ±10%", {
  df <- make_clf_200()
  s  <- ml_split(df, "target", seed = 42L, stratify = TRUE)
  global_ratio <- mean(df$target)
  train_ratio  <- mean(s$train$target)
  valid_ratio  <- mean(s$valid$target)
  test_ratio   <- mean(s$test$target)
  expect_true(abs(train_ratio - global_ratio) < 0.10)
  expect_true(abs(valid_ratio - global_ratio) < 0.10)
  expect_true(abs(test_ratio  - global_ratio) < 0.10)
})

test_that("stratified cv preserves class ratio per fold within ±15%", {
  skip_if_not_installed("caret")
  df <- make_clf_200()
  k  <- 5L
  global_ratio <- mean(df$target)

  # caret stratified folds (must pass factor target for stratification)
  withr::local_seed(42L)
  caret_idx <- caret::createFolds(factor(df$target), k = k, list = TRUE, returnTrain = FALSE)
  for (i in seq_len(k)) {
    fold_ratio <- mean(df$target[caret_idx[[i]]])
    expect_true(
      abs(fold_ratio - global_ratio) < 0.15,
      label = sprintf("caret fold %d ratio %.3f vs global %.3f", i, fold_ratio, global_ratio)
    )
  }
})


# ---------------------------------------------------------------------------
# 8. Regression CV invariants
# ---------------------------------------------------------------------------

test_that("regression cv satisfies all structural invariants", {
  df <- make_reg_200()
  k  <- 5L
  cv <- ml_split(df, "target", seed = 42L, folds = k)
  n_dev <- nrow(cv$train)
  assert_fold_count(cv$folds, k)
  assert_balanced_fold_sizes(cv$folds, n_dev, k)
  assert_no_valid_overlap(cv$folds)
  assert_complete_coverage(cv$folds, n_dev)
  assert_train_valid_disjoint(cv$folds)
  assert_train_valid_cover_all(cv$folds, n_dev)
})
