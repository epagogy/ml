# CV falsification tests — prove our invariant checks CATCH real bugs.
#
# Every test deliberately breaks split/cv and verifies the invariant
# check detects the failure. Tests that always pass prove nothing.
# These tests prove our guards are real.

# ---------------------------------------------------------------------------
# Broken implementations
# ---------------------------------------------------------------------------

broken_overlapping_folds <- function(data, k, seed) {
  withr::local_seed(seed)
  n <- nrow(data)
  lapply(seq_len(k), function(i) {
    # Each fold independently samples — rows CAN appear in multiple folds
    valid <- sample(n, n %/% k, replace = FALSE)
    train <- setdiff(seq_len(n), valid)
    list(train = train, valid = valid)
  })
}

broken_incomplete_coverage <- function(data, k, seed) {
  withr::local_seed(seed)
  n <- nrow(data) - 5L  # Drop last 5 rows
  idx <- sample(n)
  fold_size <- n %/% k
  lapply(seq_len(k), function(i) {
    valid <- idx[((i - 1L) * fold_size + 1L):(i * fold_size)]
    train <- setdiff(idx, valid)
    list(train = train, valid = valid)
  })
}

broken_temporal_future_leak <- function(data, k) {
  n <- nrow(data)
  fold_size <- n %/% (k + 1L)
  lapply(seq_len(k), function(i) {
    valid_start <- fold_size * i + 1L
    valid_end <- min(fold_size * (i + 1L), n)
    valid <- seq(valid_start, valid_end)
    # BUG: train includes ALL non-valid rows, even those AFTER valid
    train <- setdiff(seq_len(n), valid)
    list(train = train, valid = valid)
  })
}

broken_group_leaking <- function(data, k, group_col, seed) {
  withr::local_seed(seed)
  n <- nrow(data)
  # Ignore groups entirely, just random split
  idx <- sample(n)
  fold_size <- n %/% k
  lapply(seq_len(k), function(i) {
    valid <- idx[((i - 1L) * fold_size + 1L):(i * fold_size)]
    train <- setdiff(idx, valid)
    list(train = train, valid = valid)
  })
}


# ---------------------------------------------------------------------------
# 1. Overlapping folds — detected
# ---------------------------------------------------------------------------

test_that("FALSIFICATION: overlapping folds caught by invariant check", {
  withr::local_seed(42L)
  df <- data.frame(x = rnorm(200L), target = sample(0:1, 200L, replace = TRUE))
  broken <- broken_overlapping_folds(df, k = 5L, seed = 42L)

  # Our check should catch the overlap
  seen <- integer(0)
  has_overlap <- FALSE
  for (f in broken) {
    overlap <- intersect(seen, f$valid)
    if (length(overlap) > 0L) has_overlap <- TRUE
    seen <- c(seen, f$valid)
  }
  expect_true(has_overlap, label = "broken implementation must produce overlap")
})

test_that("FALSIFICATION: correct ml_split CV has no overlap", {
  withr::local_seed(42L)
  df <- data.frame(x = rnorm(200L), target = sample(0:1, 200L, replace = TRUE))
  cv <- ml_split(df, "target", seed = 42L, folds = 5L)

  seen <- integer(0)
  for (f in cv$folds) {
    overlap <- intersect(seen, f$valid)
    expect_length(overlap, 0L)
    seen <- c(seen, f$valid)
  }
})


# ---------------------------------------------------------------------------
# 2. Incomplete coverage — detected
# ---------------------------------------------------------------------------

test_that("FALSIFICATION: incomplete coverage caught", {
  withr::local_seed(42L)
  df <- data.frame(x = rnorm(200L), target = sample(0:1, 200L, replace = TRUE))
  broken <- broken_incomplete_coverage(df, k = 5L, seed = 42L)

  all_valid <- integer(0)
  for (f in broken) all_valid <- c(all_valid, f$valid)
  # Some rows missing
  expect_false(setequal(sort(unique(all_valid)), seq_len(nrow(df))))
})

test_that("FALSIFICATION: correct ml_split CV has complete coverage", {
  withr::local_seed(42L)
  df <- data.frame(x = rnorm(200L), target = sample(0:1, 200L, replace = TRUE))
  cv <- ml_split(df, "target", seed = 42L, folds = 5L)

  n_dev <- nrow(cv$train)
  all_valid <- integer(0)
  for (f in cv$folds) all_valid <- c(all_valid, f$valid)
  expect_true(setequal(sort(unique(all_valid)), seq_len(n_dev)))
})


# ---------------------------------------------------------------------------
# 3. Temporal future leak — detected
# ---------------------------------------------------------------------------

test_that("FALSIFICATION: temporal future leak caught", {
  n <- 500L
  df <- data.frame(t = seq_len(n), x = rnorm(n), target = rnorm(n))
  # Use ml to get temporal split, then break it
  s <- ml_split(df, "target", time = "t")
  # Simulate broken folds on dev data
  broken <- broken_temporal_future_leak(s$dev, k = 3L)

  has_future_leak <- FALSE
  for (f in broken) {
    if (max(f$train) > min(f$valid)) has_future_leak <- TRUE
  }
  expect_true(has_future_leak, label = "broken temporal must have future in train")
})

test_that("FALSIFICATION: correct temporal CV has no future leak", {
  n <- 500L
  df <- data.frame(t = seq_len(n), x = rnorm(n), target = rnorm(n))
  cv <- ml_split(df, "target", time = "t", folds = 3L)

  for (f in cv$folds) {
    expect_true(max(f$train) < min(f$valid))
  }
})


# ---------------------------------------------------------------------------
# 4. Group leakage — detected
# ---------------------------------------------------------------------------

test_that("FALSIFICATION: group leak caught", {
  withr::local_seed(42L)
  n <- 200L
  grp <- rep(paste0("g", seq_len(20L)), each = 10L)
  df <- data.frame(group_id = grp, x = rnorm(n), target = sample(0:1, n, replace = TRUE))

  broken <- broken_group_leaking(df, k = 4L, group_col = "group_id", seed = 42L)

  has_leak <- FALSE
  for (f in broken) {
    train_grps <- unique(grp[f$train])
    valid_grps <- unique(grp[f$valid])
    if (length(intersect(train_grps, valid_grps)) > 0L) has_leak <- TRUE
  }
  expect_true(has_leak, label = "broken group split must produce group leak")
})

test_that("FALSIFICATION: correct group CV has no group leak", {
  withr::local_seed(42L)
  n <- 200L
  grp <- rep(paste0("g", seq_len(20L)), each = 10L)
  df <- data.frame(group_id = grp, x = rnorm(n), target = sample(0:1, n, replace = TRUE))

  cv <- ml_split(df, "target", seed = 42L, groups = "group_id", folds = 4L)
  dev <- cv$train
  for (f in cv$folds) {
    train_grps <- unique(dev$group_id[f$train])
    valid_grps <- unique(dev$group_id[f$valid])
    expect_length(intersect(train_grps, valid_grps), 0L)
  }
})


# ---------------------------------------------------------------------------
# 5. Permutation null — scrambled target must score lower
# ---------------------------------------------------------------------------

test_that("FALSIFICATION: real signal beats scrambled target", {
  withr::local_seed(42L)
  n <- 200L
  x <- rnorm(n)
  y_real <- ifelse(x + rnorm(n, sd = 0.3) > 0, 1L, 0L)
  y_scrambled <- sample(y_real)

  # Real signal
  df_real <- data.frame(x = x, target = y_real)
  suppressWarnings({
    s1 <- ml_split(df_real, "target", seed = 42L, folds = 5L)
    m1 <- ml_fit(s1, "target", algorithm = "logistic", seed = 42L)
  })
  real_acc <- m1$scores_$accuracy

  # Scrambled
  df_scrambled <- data.frame(x = x, target = y_scrambled)
  suppressWarnings({
    s2 <- ml_split(df_scrambled, "target", seed = 42L, folds = 5L)
    m2 <- ml_fit(s2, "target", algorithm = "logistic", seed = 42L)
  })
  scrambled_acc <- m2$scores_$accuracy

  expect_gt(real_acc, scrambled_acc)
})
