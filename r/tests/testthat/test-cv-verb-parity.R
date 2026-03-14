# ── ml_cv() verb parity with Python ml.cv() ──────────────────────────────────
#
# Tests structural invariants that MUST hold in both languages:
# 1. Fold count matches requested k
# 2. Fold sizes balanced (±1 row)
# 3. No validation overlap (partition property)
# 4. Complete coverage (every dev row in exactly one valid fold)
# 5. Stratification preserves class ratios per fold
# 6. Group non-overlap (ml_cv_group)
# 7. Temporal ordering (ml_cv_temporal: train indices < valid indices)
# 8. Train + valid = all dev rows (no rows lost or invented)
# 9. ml_fit(cv_result) produces scores_ (non-NULL)

# ── Structural: fold count ───────────────────────────────────────────────────

test_that("ml_cv: fold count matches k for k=2,3,5,10", {
  s <- ml_split(iris, "Species", seed = 42L)
  for (k in c(2L, 3L, 5L, 10L)) {
    c <- ml_cv(s, "Species", folds = k, seed = 42L)
    expect_length(c$folds, k)
  }
})

# ── Structural: fold size balance ────────────────────────────────────────────

test_that("ml_cv: fold sizes balanced within ±1", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L)
  sizes <- vapply(c$folds, function(f) length(f$valid), integer(1))
  expect_true(max(sizes) - min(sizes) <= 1L)
})

# ── Structural: no validation overlap ────────────────────────────────────────

test_that("ml_cv: no overlap between validation folds", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L)
  valids <- lapply(c$folds, `[[`, "valid")
  for (i in seq_along(valids)) {
    for (j in seq_along(valids)) {
      if (i < j) {
        expect_length(intersect(valids[[i]], valids[[j]]), 0L)
      }
    }
  }
})

# ── Structural: complete coverage ────────────────────────────────────────────

test_that("ml_cv: every dev row appears in exactly one valid fold", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L)
  all_valid <- sort(unlist(lapply(c$folds, `[[`, "valid")))
  expect_equal(all_valid, seq_len(nrow(s$dev)))
})

# ── Structural: train + valid = all dev rows ─────────────────────────────────

test_that("ml_cv: train + valid covers all dev rows per fold", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L)
  n_dev <- nrow(s$dev)
  for (fold in c$folds) {
    all_idx <- sort(c(fold$train, fold$valid))
    expect_equal(all_idx, seq_len(n_dev))
  }
})

# ── Structural: stratification preserves class ratios ────────────────────────

test_that("ml_cv: stratified folds preserve class proportions within 10%", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 5L, seed = 42L, stratify = TRUE)
  dev_data <- s$dev
  overall_freq <- table(dev_data$Species) / nrow(dev_data)
  for (fold in c$folds) {
    fold_y <- dev_data$Species[fold$valid]
    fold_freq <- table(fold_y) / length(fold_y)
    for (cls in names(overall_freq)) {
      expect_true(abs(fold_freq[[cls]] - overall_freq[[cls]]) < 0.10,
                  label = sprintf("class %s: fold=%.2f overall=%.2f",
                                  cls, fold_freq[[cls]], overall_freq[[cls]]))
    }
  }
})

# ── Structural: regression (no stratification) ───────────────────────────────

test_that("ml_cv: works with regression target", {
  df <- data.frame(x = rnorm(200), y = rnorm(200))
  s <- ml_split(df, "y", seed = 42L)
  c <- ml_cv(s, "y", folds = 5L, seed = 42L)
  expect_s3_class(c, "ml_cv_result")
  all_valid <- sort(unlist(lapply(c$folds, `[[`, "valid")))
  expect_equal(all_valid, seq_len(nrow(s$dev)))
})

# ── Integration: ml_fit(cv_result) produces scores_ ─────────────────────────

test_that("ml_fit on ml_cv result produces non-NULL scores_", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 3L, seed = 42L)
  model <- ml_fit(c, "Species", seed = 42L)
  expect_false(is.null(model$scores_))
  expect_true(is.list(model$scores_))
  expect_true(length(model$scores_) > 0L)
})

test_that("ml_fit on ml_cv: scores_ has expected metric names", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 3L, seed = 42L)
  model <- ml_fit(c, "Species", seed = 42L)
  expect_true("accuracy" %in% names(model$scores_))
})

# ── Integration: ml_assess after ml_cv fit ───────────────────────────────────

test_that("ml_assess works after ml_cv + ml_fit", {
  s <- ml_split(iris, "Species", seed = 42L)
  c <- ml_cv(s, "Species", folds = 3L, seed = 42L)
  model <- ml_fit(c, "Species", seed = 42L)
  evidence <- ml_assess(model, test = s$test)
  expect_s3_class(evidence, "ml_evidence")
})

# ── Temporal CV: train indices < valid indices ───────────────────────────────

test_that("ml_cv_temporal: train indices always before valid indices", {
  df <- data.frame(date = 1:200, x = rnorm(200), y = rnorm(200))
  s <- ml_split_temporal(df, "y", time = "date")
  c <- ml_cv_temporal(s, "y", folds = 5L)
  for (fold in c$folds) {
    expect_true(max(fold$train) < min(fold$valid))
  }
})

test_that("ml_cv_temporal: expanding window (later folds have more training)", {
  df <- data.frame(date = 1:200, x = rnorm(200), y = rnorm(200))
  s <- ml_split_temporal(df, "y", time = "date")
  c <- ml_cv_temporal(s, "y", folds = 5L)
  sizes <- vapply(c$folds, function(f) length(f$train), integer(1))
  # Each fold should have more training data than the previous
  expect_true(all(diff(sizes) > 0L))
})

# ── Group CV: no group leakage ───────────────────────────────────────────────

test_that("ml_cv_group: no group in both train and valid", {
  df <- data.frame(
    pid = rep(1:30, each = 5),
    x = rnorm(150),
    y = sample(c("a", "b"), 150, TRUE)
  )
  s <- ml_split(df, "y", seed = 42L)
  c <- ml_cv_group(s, "y", groups = "pid", folds = 5L, seed = 42L)
  dev_data <- s$dev
  for (fold in c$folds) {
    train_groups <- unique(dev_data$pid[fold$train])
    valid_groups <- unique(dev_data$pid[fold$valid])
    expect_length(intersect(train_groups, valid_groups), 0L)
  }
})

test_that("ml_cv_group: all groups covered across folds", {
  df <- data.frame(
    pid = rep(1:20, each = 5),
    x = rnorm(100),
    y = sample(c("a", "b"), 100, TRUE)
  )
  s <- ml_split(df, "y", seed = 42L)
  c <- ml_cv_group(s, "y", groups = "pid", folds = 4L, seed = 42L)
  dev_data <- s$dev
  all_valid_groups <- unique(unlist(lapply(c$folds, function(f) {
    unique(dev_data$pid[f$valid])
  })))
  expect_equal(sort(all_valid_groups), sort(unique(dev_data$pid)))
})

# ── Parity: ml_cv() vs ml_split(folds=) produce same structure ──────────────

test_that("ml_cv() and ml_split(folds=) both feed into ml_fit with scores_", {
  # Old path: ml_split(folds=)
  s_old <- ml_split(iris, "Species", seed = 42L, folds = 5L)
  m_old <- ml_fit(s_old, "Species", seed = 42L)

  # New path: ml_split() then ml_cv()
  s_new <- ml_split(iris, "Species", seed = 42L)
  c_new <- ml_cv(s_new, "Species", folds = 5L, seed = 42L)
  m_new <- ml_fit(c_new, "Species", seed = 42L)

  # Both should produce scores_
  expect_false(is.null(m_old$scores_))
  expect_false(is.null(m_new$scores_))

  # Both should have the same metric names
  expect_equal(sort(names(m_old$scores_)), sort(names(m_new$scores_)))
})
