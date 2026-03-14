# CV implementation audit — tests that SHOULD pass but currently FAIL.
#
# These expose real bugs found by auditing R against the Python reference.
# Each test documents the bug, the expected correct behavior, and the
# current broken behavior. Fix the implementation, then remove the skip.

# ---------------------------------------------------------------------------
# BUG 1: R CV does not stratify folds
# ---------------------------------------------------------------------------
# R split.R line 196: fold_ids <- sample(rep(seq_len(folds), length.out = n))
# This is plain random assignment. Python's cv() calls _stratified_kfold()
# when stratify=True, preserving class ratio per fold.
#
# Impact: With imbalanced data, some folds may have 0 minority class samples,
# causing models to fail or produce degenerate predictions.

test_that("BUG: CV folds should preserve class ratio (stratification)", {
  withr::local_seed(42L)
  n <- 300L
  # Imbalanced: 80% class 0, 20% class 1
  target <- c(rep(0L, 240L), rep(1L, 60L))
  df <- data.frame(x = rnorm(n), target = target)
  cv <- ml_split(df, "target", seed = 42L, folds = 5L)

  dev <- cv$train  # dev data (test held out)
  global_ratio <- mean(dev$target == 1L)
  for (i in seq_along(cv$folds)) {
    fold_target <- dev$target[cv$folds[[i]]$valid]
    fold_ratio <- mean(fold_target == 1L)
    # With stratification, each fold should be within ±0.08 of global
    # Without stratification (current bug), folds can deviate wildly
    expect_true(
      abs(fold_ratio - global_ratio) < 0.08,
      label = sprintf(
        "Fold %d: minority ratio %.3f vs global %.3f (diff=%.3f)",
        i, fold_ratio, global_ratio, abs(fold_ratio - global_ratio)
      )
    )
  }
})


# ---------------------------------------------------------------------------
# BUG 2: R temporal CV drops remainder rows
# ---------------------------------------------------------------------------
# R .temporal_cv() line 283: valid_end <- min(train_end + chunk_size, n)
# Python cv_temporal(): last fold sets valid_end = n.
# With n not divisible by (folds+1), R silently drops trailing rows.
#
# Impact: Some data points are never validated. Coverage < 100%.

test_that("BUG: temporal CV must cover all rows (no remainder drop)", {
  n <- 503L  # Not divisible by (3+1)=4
  df <- data.frame(t = seq_len(n), x = rnorm(n), target = rnorm(n))
  cv <- ml_split(df, "target", time = "t", folds = 3L)

  # Collect all rows that appear in ANY fold (train or valid)
  all_touched <- integer(0)
  for (f in cv$folds) {
    all_touched <- union(all_touched, c(f$train, f$valid))
  }

  # Every dev row should appear in at least one fold
  n_dev <- nrow(cv$train)
  expect_equal(
    length(all_touched), n_dev,
    label = sprintf("Expected %d rows covered, got %d", n_dev, length(all_touched))
  )
})

test_that("BUG: temporal CV last fold must extend to final row", {
  n <- 503L
  df <- data.frame(t = seq_len(n), x = rnorm(n), target = rnorm(n))
  cv <- ml_split(df, "target", time = "t", folds = 3L)

  last_fold <- cv$folds[[length(cv$folds)]]
  max_valid <- max(last_fold$valid)
  n_dev <- nrow(cv$train)
  expect_equal(
    max_valid, n_dev,
    label = sprintf("Last fold valid ends at %d, should be %d", max_valid, n_dev)
  )
})


# ---------------------------------------------------------------------------
# BUG 3: R CV has no test holdout
# ---------------------------------------------------------------------------
# Python: cv() takes SplitResult, operates on .dev, preserves .test.
# R: ml_split(folds=) creates CV on ALL data. No test partition.
#
# Impact: Users who do ml_split(folds=5) then ml_fit() have no held-out
# test set for ml_assess(). The entire assess() workflow breaks.
# This is the split/cv separation the Python package already enforces.

test_that("BUG: CV should preserve a test holdout for assess()", {
  df <- data.frame(
    x1 = rnorm(200L), x2 = rnorm(200L),
    target = sample(0:1, 200L, replace = TRUE)
  )

  # The correct workflow: split first (gets test), then CV on dev
  # R currently: ml_split(folds=5) creates CV on ALL data, no test
  cv <- ml_split(df, "target", seed = 42L, folds = 5L)

  # CV data should be dev (train+valid), not all data
  # With a 60/20/20 split on 200 rows, dev ≈ 160, test ≈ 40
  expect_true(
    nrow(cv$train) < nrow(df),
    label = sprintf(
      "CV dev has %d rows (should be < %d, test should be held out)",
      nrow(cv$train), nrow(df)
    )
  )
  # Test partition should exist and be non-empty
  expect_true(nrow(cv$test) > 0L)
})
