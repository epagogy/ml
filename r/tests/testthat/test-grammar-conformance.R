# Grammar conformance tests — the paper is the spec.
# Tests the 8 conformance conditions (CC1-CC8) from the ML grammar paper.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
make_clf <- function(n = 300, p = 5, seed = 42) {
  set.seed(seed)
  X <- as.data.frame(matrix(rnorm(n * p), n, p))
  names(X) <- paste0("f", seq_len(p))
  X$target <- factor(sample(c("a", "b"), n, replace = TRUE, prob = c(0.4, 0.6)))
  X
}

make_reg <- function(n = 300, p = 5, seed = 42) {
  set.seed(seed)
  X <- as.data.frame(matrix(rnorm(n * p), n, p))
  names(X) <- paste0("f", seq_len(p))
  X$target <- X$f1 * 2 + X$f2 + rnorm(n, sd = 0.3)
  X
}

# ═══════════════════════════════════════════════════════════════════════════
# CC1: Partition integrity
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC1: split produces exhaustive partitions (no rows lost)", {
  d <- make_clf(n = 300, seed = 1)
  s <- ml_split(d, "target", seed = 1)
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), nrow(d))
})

test_that("CC1: split produces disjoint partitions", {
  d <- make_clf(n = 300, seed = 2)
  s <- ml_split(d, "target", seed = 2)
  all_rows <- c(rownames(s$train), rownames(s$valid), rownames(s$test))
  expect_equal(length(all_rows), length(unique(all_rows)))
})

test_that("CC1: dev equals train + valid", {
  d <- make_clf(n = 300, seed = 3)
  s <- ml_split(d, "target", seed = 3)
  expect_equal(nrow(s$dev), nrow(s$train) + nrow(s$valid))
})

test_that("CC1: temporal split preserves time ordering", {
  set.seed(4)
  n <- 300
  d <- data.frame(
    date = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1 = rnorm(n),
    target = rnorm(n)
  )
  s <- ml_split(d, "target", time = "date", seed = 4)
  expect_true(max(s$train$date) <= min(s$valid$date))
  expect_true(max(s$valid$date) <= min(s$test$date))
})

test_that("CC1: group split has no group leakage", {
  set.seed(5)
  n <- 300
  d <- data.frame(
    group = rep(1:30, each = 10),
    f1 = rnorm(n),
    target = factor(sample(c("a", "b"), n, replace = TRUE))
  )
  s <- ml_split_group(d, "target", groups = "group", seed = 5)
  train_g <- unique(s$train$group)
  test_g <- unique(s$test$group)
  expect_equal(length(intersect(train_g, test_g)), 0)
})

# ═══════════════════════════════════════════════════════════════════════════
# CC2: Provenance chain
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC2: model records algorithm name", {
  d <- make_clf(seed = 6)
  s <- ml_split(d, "target", seed = 6)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 6)
  expect_true(!is.null(m$algorithm))
  expect_equal(m$algorithm, "random_forest")
})

test_that("CC2: model records seed", {
  d <- make_clf(seed = 7)
  s <- ml_split(d, "target", seed = 7)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 7)
  expect_true(!is.null(m$seed))
})

test_that("CC2: verify returns provenance check", {
  d <- make_clf(seed = 8)
  s <- ml_split(d, "target", seed = 8)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 8)
  v <- ml_verify(m)
  expect_true(is.list(v))
  expect_true("status" %in% names(v))
})

# ═══════════════════════════════════════════════════════════════════════════
# CC3: Terminal assessment — assess is one-shot
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC3: assess returns evidence with accuracy", {
  d <- make_clf(seed = 9)
  s <- ml_split(d, "target", seed = 9)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 9)
  ev <- ml_assess(m, test = s$test)
  expect_true(is.numeric(ev[["accuracy"]]))
  expect_true(ev[["accuracy"]] >= 0 && ev[["accuracy"]] <= 1)
})

test_that("CC3: double assess is blocked", {
  d <- make_clf(seed = 10)
  s <- ml_split(d, "target", seed = 10)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 10)
  ml_assess(m, test = s$test)
  expect_error(ml_assess(m, test = s$test))
})

# ═══════════════════════════════════════════════════════════════════════════
# CC4: Evaluate/assess boundary
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC4: evaluate is repeatable (same metrics twice)", {
  d <- make_clf(seed = 11)
  s <- ml_split(d, "target", seed = 11)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 11)
  m1 <- ml_evaluate(m, s$valid)
  m2 <- ml_evaluate(m, s$valid)
  expect_equal(m1[["accuracy"]], m2[["accuracy"]])
})

test_that("CC4: evaluate returns ml_metrics, assess returns ml_evidence", {
  d <- make_clf(seed = 12)
  s <- ml_split(d, "target", seed = 12)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 12)
  metrics <- ml_evaluate(m, s$valid)
  evidence <- ml_assess(m, test = s$test)
  expect_true(inherits(metrics, "ml_metrics"))
  expect_true(inherits(evidence, "ml_evidence"))
})

# ═══════════════════════════════════════════════════════════════════════════
# CC5: Determinism
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC5: same seed produces identical splits", {
  d <- make_clf(seed = 13)
  s1 <- ml_split(d, "target", seed = 99)
  s2 <- ml_split(d, "target", seed = 99)
  expect_identical(s1$train, s2$train)
})

test_that("CC5: same seed produces identical predictions", {
  d <- make_clf(seed = 14)
  s <- ml_split(d, "target", seed = 14)
  m1 <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 14)
  m2 <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 14)
  p1 <- ml_predict(m1, s$valid)
  p2 <- ml_predict(m2, s$valid)
  expect_identical(p1, p2)
})

test_that("CC5: different seeds produce different splits", {
  d <- make_clf(seed = 15)
  s1 <- ml_split(d, "target", seed = 1)
  s2 <- ml_split(d, "target", seed = 2)
  expect_false(identical(s1$train, s2$train))
})

# ═══════════════════════════════════════════════════════════════════════════
# CC6: Type safety — wrong inputs rejected
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC6: split rejects missing target column", {
  d <- make_clf(seed = 16)
  expect_error(ml_split(d, "nonexistent", seed = 16))
})

test_that("CC6: fit rejects unknown algorithm", {
  d <- make_clf(seed = 17)
  s <- ml_split(d, "target", seed = 17)
  expect_error(ml_fit(s$train, "target", algorithm = "fake_algo", seed = 17))
})

test_that("CC6: logistic rejects regression target", {
  d <- make_reg(seed = 18)
  s <- ml_split(d, "target", seed = 18)
  expect_error(ml_fit(s$train, "target", algorithm = "logistic", seed = 18))
})

# ═══════════════════════════════════════════════════════════════════════════
# CC7: Cross-validation integrity
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC7: cv returns k folds", {
  d <- make_clf(seed = 19)
  s <- ml_split(d, "target", seed = 19)
  cv <- ml_cv(s, "target", folds = 5, seed = 19)
  expect_equal(cv$k, 5)
})

test_that("CC7: cv_temporal returns valid fold count", {
  set.seed(20)
  n <- 300
  d <- data.frame(
    date = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1 = rnorm(n),
    target = rnorm(n)
  )
  s <- ml_split(d, "target", time = "date", seed = 20)
  cv <- ml_cv_temporal(s, "target", folds = 3)
  expect_true(cv$k >= 2)
})

# ═══════════════════════════════════════════════════════════════════════════
# CC8: Composition closure — verb chains produce valid outputs
# ═══════════════════════════════════════════════════════════════════════════

test_that("CC8: full clf workflow chain completes", {
  d <- make_clf(seed = 21)
  s <- ml_split(d, "target", seed = 21)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 21)
  p <- ml_predict(m, s$valid)
  metrics <- ml_evaluate(m, s$valid)
  evidence <- ml_assess(m, test = s$test)
  expect_equal(length(p), nrow(s$valid))
  expect_true(metrics[["accuracy"]] > 0)
  expect_true(evidence[["accuracy"]] > 0)
})

test_that("CC8: full reg workflow chain completes", {
  d <- make_reg(seed = 22)
  s <- ml_split(d, "target", seed = 22)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 22)
  p <- ml_predict(m, s$valid)
  metrics <- ml_evaluate(m, s$valid)
  evidence <- ml_assess(m, test = s$test)
  expect_equal(length(p), nrow(s$valid))
  expect_true(metrics[["rmse"]] > 0)
  expect_true(is.finite(evidence[["r2"]]))
})

test_that("CC8: screen then fit best chain completes", {
  d <- make_clf(seed = 23)
  s <- ml_split(d, "target", seed = 23)
  lb <- ml_screen(s, "target", seed = 23)
  m <- ml_fit(s$train, "target", algorithm = lb$best, seed = 23)
  expect_true(ml_evaluate(m, s$valid)[["accuracy"]] > 0)
})

test_that("CC8: tune then fit chain completes", {
  d <- make_clf(seed = 24)
  s <- ml_split(d, "target", seed = 24)
  tr <- ml_tune(s$train, "target", algorithm = "random_forest", n_trials = 3, seed = 24)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 24)
  expect_true(ml_evaluate(m, s$valid)[["accuracy"]] > 0)
})

test_that("CC8: fit on dev then assess on test", {
  d <- make_clf(seed = 25)
  s <- ml_split(d, "target", seed = 25)
  m <- ml_fit(s$dev, "target", algorithm = "random_forest", seed = 25)
  evidence <- ml_assess(m, test = s$test)
  expect_true(evidence[["accuracy"]] > 0)
})

test_that("CC8: explain returns feature importances", {
  d <- make_clf(seed = 26)
  s <- ml_split(d, "target", seed = 26)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 26)
  exp <- ml_explain(m)
  expect_true(length(exp$features) > 0 || !is.null(exp$importance))
})

test_that("CC8: save/load roundtrip preserves predictions", {
  d <- make_clf(seed = 27)
  s <- ml_split(d, "target", seed = 27)
  m <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 27)
  p1 <- ml_predict(m, s$valid)
  path <- tempfile(fileext = ".rds")
  ml_save(m, path)
  m2 <- ml_load(path)
  p2 <- ml_predict(m2, s$valid)
  expect_identical(p1, p2)
  unlink(path)
})

# ═══════════════════════════════════════════════════════════════════════════
# Algorithm coverage — every algorithm runs the full DAG
# ═══════════════════════════════════════════════════════════════════════════

clf_algos <- c("random_forest", "decision_tree", "logistic", "knn", "naive_bayes")
reg_algos <- c("random_forest", "decision_tree", "ridge", "knn")

for (algo in clf_algos) {
  test_that(paste0("coverage: ", algo, " completes clf DAG"), {
    d <- make_clf(seed = 30)
    s <- ml_split(d, "target", seed = 30)
    m <- ml_fit(s$train, "target", algorithm = algo, seed = 30)
    p <- ml_predict(m, s$valid)
    metrics <- ml_evaluate(m, s$valid)
    expect_equal(length(p), nrow(s$valid))
    expect_true(metrics[["accuracy"]] >= 0 && metrics[["accuracy"]] <= 1)
  })
}

for (algo in reg_algos) {
  test_that(paste0("coverage: ", algo, " completes reg DAG"), {
    d <- make_reg(seed = 31)
    s <- ml_split(d, "target", seed = 31)
    m <- ml_fit(s$train, "target", algorithm = algo, seed = 31)
    p <- ml_predict(m, s$valid)
    metrics <- ml_evaluate(m, s$valid)
    expect_equal(length(p), nrow(s$valid))
    expect_true(is.finite(metrics[["rmse"]]))
  })
}
