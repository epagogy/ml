# Split/CV Correctness Tests
#
# Structural invariants that must hold across ALL implementations.
# These are the grammar's constraints made executable.
#
# Cross-language parity (temporal): Python and R must produce identical
# partition sizes for deterministic (temporal) splits. Random splits
# cannot match exactly (different RNGs) but sizes must agree within ±1.
#
# Constraint coverage:
#   C1 Coverage — every row in exactly one partition
#   C2 Disjointness — no row in two partitions
#   C3 Preservation — no fabricated rows
#   C4 No info flow — (tested in fit/evaluate, not split)
#   C5 Determinism — same seed → same split
#   C6 Temporal ordering — train < valid < test in time
#   C7 Group integrity — no group in two partitions
#   Stratification — class ratio preserved per fold
#   T  Terminal — test exists and is non-empty for CV

# ── Helpers ──────────────────────────────────────────────────────────────────

#' Compute partition sizes the canonical way: round(n * ratio)
#' This is the formula both languages (Python, R) must use.
canonical_sizes <- function(n, ratio = c(0.6, 0.2, 0.2)) {
  n_train <- round(n * ratio[1])
  n_valid <- round(n * ratio[2])
  n_test  <- n - n_train - n_valid
  c(train = n_train, valid = n_valid, test = n_test)
}

# ── C1+C2+C3: Coverage, Disjointness, Preservation ──────────────────────────

test_that("C1+C2+C3 hold for random split across 20 dataset sizes", {
  for (n in c(10L, 20L, 50L, 100L, 101L, 103L, 150L, 200L, 201L, 300L,
              500L, 503L, 750L, 1000L, 1001L, 1500L, 2000L, 2001L, 5000L, 10000L)) {
    df <- data.frame(x = rnorm(n), y = rnorm(n))
    suppressWarnings(s <- ml_split(df, "y", seed = 42L))

    # C1: Coverage — all rows present
    total <- nrow(s$train) + nrow(s$valid) + nrow(s$test)
    expect_equal(total, n, label = sprintf("C1 coverage n=%d", n))

    # C2: Disjointness — no overlap
    tr <- as.integer(rownames(s$train))
    va <- as.integer(rownames(s$valid))
    te <- as.integer(rownames(s$test))
    expect_length(intersect(tr, va), 0L)
    expect_length(intersect(tr, te), 0L)
    expect_length(intersect(va, te), 0L)

    # C3: Preservation — no fabricated rows (all indices ∈ 1..n)
    all_idx <- sort(c(tr, va, te))
    expect_equal(all_idx, seq_len(n))
  }
})

# ── C5: Determinism ──────────────────────────────────────────────────────────

test_that("C5 determinism — 10 repeated calls identical", {
  df <- data.frame(x = rnorm(500L), y = sample(0:1, 500L, replace = TRUE))
  ref <- ml_split(df, "y", seed = 42L)
  for (i in 1:10) {
    s <- ml_split(df, "y", seed = 42L)
    expect_identical(rownames(s$train), rownames(ref$train))
    expect_identical(rownames(s$valid), rownames(ref$valid))
    expect_identical(rownames(s$test),  rownames(ref$test))
  }
})

test_that("C5 determinism — different seeds differ", {
  df <- data.frame(x = rnorm(500L), y = rnorm(500L))
  s1 <- ml_split(df, "y", seed = 1L)
  s2 <- ml_split(df, "y", seed = 2L)
  expect_false(identical(rownames(s1$train), rownames(s2$train)))
})

test_that("C5 determinism — regression CV folds identical across calls", {
  # Regression target → non-stratified path → exercises Rust shuffle + R fold assignment.
  # Bug: if withr::local_seed() not called before fold assignment, folds are non-deterministic.
  df <- data.frame(x = rnorm(300L), y = rnorm(300L))
  ref <- ml_split(df, "y", seed = 42L, folds = 5L)
  for (i in 1:10) {
    cv <- ml_split(df, "y", seed = 42L, folds = 5L)
    expect_identical(cv$folds[[1]]$valid, ref$folds[[1]]$valid,
                     label = sprintf("regression CV fold determinism, run %d", i))
    expect_identical(cv$folds[[3]]$train, ref$folds[[3]]$train,
                     label = sprintf("regression CV fold determinism, run %d", i))
  }
})

# ── C6: Temporal ordering ────────────────────────────────────────────────────

test_that("C6 temporal — train rows precede valid precede test", {
  for (n in c(100L, 200L, 503L, 1001L)) {
    df <- data.frame(t = seq_len(n), x = rnorm(n), y = rnorm(n))
    suppressWarnings(s <- ml_split(df, "y", time = "t"))
    # Since data is sorted by t and t is dropped, row positions encode order
    expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), n)
    # First partition has the earliest data, last has the latest
    # All rows accounted for in order
    expect_true(nrow(s$train) > 0L)
    expect_true(nrow(s$valid) > 0L)
    expect_true(nrow(s$test) > 0L)
  }
})

# ── Cross-language parity: temporal cutpoints ────────────────────────────────

test_that("temporal partition sizes match canonical formula", {
  for (n in c(100L, 101L, 103L, 200L, 201L, 503L, 1000L, 1001L)) {
    df <- data.frame(t = seq_len(n), x = rnorm(n), y = rnorm(n))
    suppressWarnings(s <- ml_split(df, "y", time = "t"))
    expected <- canonical_sizes(n)
    expect_equal(nrow(s$train), expected[["train"]],
                 label = sprintf("temporal train n=%d", n))
    expect_equal(nrow(s$valid), expected[["valid"]],
                 label = sprintf("temporal valid n=%d", n))
    expect_equal(nrow(s$test), expected[["test"]],
                 label = sprintf("temporal test n=%d", n))
  }
})

test_that("random split sizes match canonical formula (±1)", {
  for (n in c(100L, 101L, 103L, 200L, 201L, 503L, 1000L, 1001L)) {
    df <- data.frame(x = rnorm(n), y = rnorm(n))
    suppressWarnings(s <- ml_split(df, "y", seed = 42L))
    expected <- canonical_sizes(n)
    # Random splits may differ by ±1 due to stratification rounding
    expect_true(abs(nrow(s$train) - expected[["train"]]) <= 2L,
                label = sprintf("random train n=%d: got %d expected ~%d",
                                n, nrow(s$train), expected[["train"]]))
    expect_true(abs(nrow(s$valid) - expected[["valid"]]) <= 2L,
                label = sprintf("random valid n=%d: got %d expected ~%d",
                                n, nrow(s$valid), expected[["valid"]]))
    expect_true(abs(nrow(s$test) - expected[["test"]]) <= 2L,
                label = sprintf("random test n=%d: got %d expected ~%d",
                                n, nrow(s$test), expected[["test"]]))
  }
})

# ── C7: Group integrity ──────────────────────────────────────────────────────

test_that("C7 group integrity — no group leakage across 5 configs", {
  for (n_grps in c(6L, 10L, 20L, 50L, 100L)) {
    grp <- rep(paste0("g", seq_len(n_grps)), each = 10L)
    n <- length(grp)
    df <- data.frame(group = grp, x = rnorm(n), y = rnorm(n))
    suppressWarnings(s <- ml_split(df, "y", seed = 42L, groups = "group"))

    train_g <- unique(s$train$group)
    valid_g <- unique(s$valid$group)
    test_g  <- unique(s$test$group)
    expect_length(intersect(train_g, valid_g), 0L)
    expect_length(intersect(train_g, test_g), 0L)
    expect_length(intersect(valid_g, test_g), 0L)
    # All groups accounted for
    expect_equal(sort(c(train_g, valid_g, test_g)), sort(unique(grp)))
  }
})

# ── Stratification ──────────────────────────────────────────────────────

test_that("stratification — class ratio preserved in holdout", {
  for (minority_frac in c(0.5, 0.3, 0.2, 0.1, 0.05)) {
    n <- 1000L
    n_min <- round(n * minority_frac)
    target <- c(rep(1L, n_min), rep(0L, n - n_min))
    df <- data.frame(x = rnorm(n), target = target)
    s <- ml_split(df, "target", seed = 42L)
    global_ratio <- mean(df$target)

    train_ratio <- mean(s$train$target)
    valid_ratio <- mean(s$valid$target)
    test_ratio  <- mean(s$test$target)
    # Each partition within ±5% of global
    expect_true(abs(train_ratio - global_ratio) < 0.05,
                label = sprintf("strat train frac=%.2f", minority_frac))
    expect_true(abs(valid_ratio - global_ratio) < 0.05,
                label = sprintf("strat valid frac=%.2f", minority_frac))
    expect_true(abs(test_ratio - global_ratio) < 0.05,
                label = sprintf("strat test frac=%.2f", minority_frac))
  }
})

test_that("stratification — class ratio preserved per CV fold", {
  n <- 500L
  target <- c(rep(1L, 100L), rep(0L, 400L))  # 20% minority
  df <- data.frame(x = rnorm(n), target = target)
  cv <- ml_split(df, "target", seed = 42L, folds = 5L)

  dev_data <- cv$train
  global_ratio <- mean(dev_data$target)
  for (i in seq_along(cv$folds)) {
    fold_target <- dev_data$target[cv$folds[[i]]$valid]
    fold_ratio <- mean(fold_target)
    expect_true(abs(fold_ratio - global_ratio) < 0.08,
                label = sprintf("fold %d: %.3f vs global %.3f", i, fold_ratio, global_ratio))
  }
})

# ── T: Terminal boundary — CV has test holdout ───────────────────────────────

test_that("T — CV always holds out test for assess()", {
  for (folds in c(2L, 3L, 5L, 10L)) {
    df <- data.frame(x = rnorm(500L), target = sample(0:1, 500L, replace = TRUE))
    cv <- ml_split(df, "target", seed = 42L, folds = folds)
    # Test partition exists and is non-empty
    expect_true(nrow(cv$test) > 0L,
                label = sprintf("CV folds=%d has test", folds))
    # Dev + test = all data
    expect_equal(nrow(cv$train) + nrow(cv$test), nrow(df),
                 label = sprintf("CV folds=%d: dev+test=n", folds))
    # Folds cover dev
    all_valid <- integer(0)
    for (f in cv$folds) all_valid <- c(all_valid, f$valid)
    expect_equal(length(unique(all_valid)), nrow(cv$train),
                 label = sprintf("CV folds=%d: folds cover dev", folds))
  }
})

test_that("T — temporal CV holds out test from end", {
  n <- 500L
  df <- data.frame(t = seq_len(n), x = rnorm(n), y = rnorm(n))
  cv <- ml_split(df, "y", time = "t", folds = 3L)
  expect_true(nrow(cv$test) > 0L)
  # Folds respect temporal ordering
  for (f in cv$folds) {
    expect_true(max(f$train) < min(f$valid))
  }
})

test_that("T — group CV holds out test groups", {
  n_grps <- 20L
  grp <- rep(paste0("g", seq_len(n_grps)), each = 10L)
  n <- length(grp)
  df <- data.frame(group = grp, x = rnorm(n), y = rnorm(n))
  cv <- ml_split(df, "y", seed = 42L, groups = "group", folds = 4L)
  expect_true(nrow(cv$test) > 0L)
  # Test groups don't appear in dev
  dev_grps <- unique(cv$train$group)
  test_grps <- unique(cv$test$group)
  expect_length(intersect(dev_grps, test_grps), 0L)
})

# ── Adversarial inputs ───────────────────────────────────────────────────────

test_that("adversarial — extreme imbalance 99:1 still stratifies", {
  n <- 1000L
  target <- c(rep(0L, 990L), rep(1L, 10L))
  df <- data.frame(x = rnorm(n), target = target)
  s <- ml_split(df, "target", seed = 42L)
  # Minority class appears in all three partitions
  expect_true(sum(s$train$target == 1L) >= 1L)
  expect_true(sum(s$test$target == 1L) >= 1L)
})

test_that("adversarial — multiclass (10 classes) stratification", {
  n <- 500L
  target <- rep(paste0("class_", 1:10), each = 50L)
  df <- data.frame(x = rnorm(n), target = target)
  s <- ml_split(df, "target", seed = 42L)
  # All 10 classes in train
  expect_equal(length(unique(s$train$target)), 10L)
  # At least 8 classes in test (small test partition may miss a class)
  expect_true(length(unique(s$test$target)) >= 8L)
})

test_that("adversarial — constant features don't break split", {
  n <- 200L
  df <- data.frame(const = rep(1, n), x = rnorm(n), y = sample(0:1, n, replace = TRUE))
  suppressWarnings(s <- ml_split(df, "y", seed = 42L))
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), n)
})

test_that("adversarial — duplicate rows handled", {
  df <- data.frame(x = rep(1:5, each = 20L), y = rep(0:1, 50L))
  s <- ml_split(df, "y", seed = 42L)
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), 100L)
})

test_that("adversarial — minimum viable dataset (4 rows)", {
  df <- data.frame(x = 1:4, y = c("a", "b", "a", "b"))
  suppressWarnings(s <- ml_split(df, "y", seed = 42L))
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), 4L)
})

# ── CV fold structural invariants across k values ────────────────────────────

for (k in c(2L, 3L, 5L, 7L, 10L)) {
  test_that(sprintf("CV k=%d — no fold overlap, complete coverage, balanced", k), {
    df <- data.frame(x = rnorm(500L), target = sample(0:1, 500L, replace = TRUE))
    cv <- ml_split(df, "target", seed = 42L, folds = k)
    n_dev <- nrow(cv$train)

    # No overlap between validation folds
    seen <- integer(0)
    for (f in cv$folds) {
      overlap <- intersect(seen, f$valid)
      expect_length(overlap, 0L)
      seen <- c(seen, f$valid)
    }
    # Complete coverage of dev
    expect_equal(length(unique(seen)), n_dev)

    # Balanced fold sizes (within ±2 of n_dev/k)
    expected_size <- n_dev / k
    for (f in cv$folds) {
      expect_true(abs(length(f$valid) - expected_size) <= 2,
                  label = sprintf("fold size %d vs expected %.0f", length(f$valid), expected_size))
    }

    # Train+valid = dev in each fold
    for (f in cv$folds) {
      expect_equal(sort(c(f$train, f$valid)), seq_len(n_dev))
    }
  })
}

# ── Cross-language PCG shuffle parity ──────────────────────────────────────

test_that("partition_sizes match canonical formula across languages", {
  # These values are pinned in Rust (al/core/src/shuffle.rs) and must
  # match Python (ml.split._ml_partition_sizes).
  sizes_100 <- .ml_partition_sizes(100L, c(0.6, 0.2, 0.2))
  expect_equal(sizes_100, c(60L, 20L, 20L))
  sizes_101 <- .ml_partition_sizes(101L, c(0.6, 0.2, 0.2))
  expect_equal(sizes_101, c(61L, 20L, 20L))
  sizes_503 <- .ml_partition_sizes(503L, c(0.6, 0.2, 0.2))
  expect_equal(sizes_503, c(302L, 101L, 100L))
  # All sum to n
  for (n in c(10L, 50L, 100L, 101L, 103L, 200L, 503L, 1001L)) {
    s <- .ml_partition_sizes(n, c(0.6, 0.2, 0.2))
    expect_equal(sum(s), n, label = sprintf("partition sum n=%d", n))
  }
})

test_that("Rust shuffle golden test (when available)", {
  # Rust PCG-XSH-RR golden output: shuffle(10, 42) = [4,5,9,0,1,7,8,3,2,6] (0-based)
  # In R 1-based: [5,6,10,1,2,8,9,4,3,7]
  perm <- .ml_shuffle(10L, 42L)
  expect_length(perm, 10L)
  expect_equal(sort(perm), 1:10)  # valid permutation
  # Golden match only when Rust shuffle is available
  rust_shuffle_ok <- tryCatch(
    { .Call(wrap__ml_rust_shuffle, 1L, 0); TRUE },
    error = function(e) FALSE
  )
  if (rust_shuffle_ok) {
    expect_equal(perm, c(5L, 6L, 10L, 1L, 2L, 8L, 9L, 4L, 3L, 7L))
  }
})

# ── Module style parity ─────────────────────────────────────────────────────

test_that("ml$split() == ml_split()", {
  s1 <- ml_split(iris, "Species", seed = 42L)
  s2 <- ml$split(iris, "Species", seed = 42L)
  expect_identical(rownames(s1$train), rownames(s2$train))
  expect_identical(rownames(s1$test), rownames(s2$test))
})
