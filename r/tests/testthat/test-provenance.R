# ── Content-Addressed Partition Provenance ──

test_that("fingerprint is deterministic", {
  df <- data.frame(x = 1:10, y = rnorm(10))
  fp1 <- ml:::.fingerprint(df)
  fp2 <- ml:::.fingerprint(df)
  expect_identical(fp1, fp2)
})

test_that("fingerprint differs for different data", {
  df1 <- data.frame(x = 1:10, y = rnorm(10))
  df2 <- data.frame(x = 1:10, y = rnorm(10))
  expect_false(ml:::.fingerprint(df1) == ml:::.fingerprint(df2))
})

test_that("registry stores and retrieves partitions", {
  ml:::.clear_registry()
  df <- data.frame(x = 1:10, y = rnorm(10))
  ml:::.register_partition(df, "train", "split_001")
  expect_equal(ml:::.identify_partition(df), "train")
  expect_equal(ml:::.registry_size(), 1L)
  ml:::.clear_registry()
})

test_that("split registers all three partitions in registry", {
  ml:::.clear_registry()
  s <- ml_split(iris, "Species", seed = 42L)

  expect_equal(ml:::.identify_partition(s$train), "train")
  expect_equal(ml:::.identify_partition(s$valid), "valid")
  expect_equal(ml:::.identify_partition(s$test), "test")

  ml:::.clear_registry()
})

test_that("dev partition is registered on access", {
  ml:::.clear_registry()
  s <- ml_split(iris, "Species", seed = 42L)
  dev <- s$dev
  expect_equal(ml:::.identify_partition(dev), "dev")
  ml:::.clear_registry()
})

test_that("fingerprint survives attr-stripping operations", {
  ml:::.clear_registry()
  s <- ml_split(iris, "Species", seed = 42L)

  # Simulate dplyr-like operation that strips attributes
  train_copy <- s$train
  attr(train_copy, "_ml_partition") <- NULL
  expect_null(attr(train_copy, "_ml_partition"))

  # Content-addressed lookup still works
  expect_equal(ml:::.resolve_partition(train_copy), "train")

  ml:::.clear_registry()
})

test_that("guard accepts data after attr stripped (content-addressed)", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  s <- ml_split(iris, "Species", seed = 42L)

  # Strip attr to simulate dplyr
  train_naked <- s$train
  attr(train_naked, "_ml_partition") <- NULL

  # fit() should still work via fingerprint
  model <- ml_fit(train_naked, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")

  ml:::.clear_registry()
})

test_that("guard rejects genuinely unsplit data", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  # Fresh data never passed through split
  raw <- iris
  expect_error(
    ml_fit(raw, "Species", algorithm = "logistic", seed = 42L),
    class = "partition_error"
  )

  ml:::.clear_registry()
})

test_that("assess guard works via content-address after attr strip", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  s <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)

  # Strip attr from test
  test_naked <- s$test
  attr(test_naked, "_ml_partition") <- NULL

  # assess should still work via fingerprint
  evidence <- ml_assess(model, test = test_naked)
  expect_s3_class(evidence, "ml_evidence")

  ml:::.clear_registry()
})

test_that("evaluate guard works via content-address after attr strip", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  s <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)

  # Strip attr from valid
  valid_naked <- s$valid
  attr(valid_naked, "_ml_partition") <- NULL

  # evaluate should still work via fingerprint
  metrics <- ml_evaluate(model, valid_naked)
  expect_s3_class(metrics, "ml_metrics")

  ml:::.clear_registry()
})

test_that("evaluate rejects test data via content-address", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  s <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)

  # Strip attr — fingerprint should still identify as "test"
  test_naked <- s$test
  attr(test_naked, "_ml_partition") <- NULL

  expect_error(
    ml_evaluate(model, test_naked),
    class = "partition_error"
  )

  ml:::.clear_registry()
})

test_that("fit rejects test data via content-address after attr strip", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  s <- ml_split(iris, "Species", seed = 42L)

  # Strip attr — fingerprint should still identify as "test"
  test_naked <- s$test
  attr(test_naked, "_ml_partition") <- NULL

  expect_error(
    ml_fit(test_naked, "Species", algorithm = "logistic", seed = 42L),
    class = "partition_error"
  )

  ml:::.clear_registry()
})

# ── Layer 2: Cross-Verb Provenance (Split-Shopping Detection) ──

test_that("assess catches split-shopping (different split lineage)", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  s1 <- ml_split(iris, "Species", seed = 42L)
  s2 <- ml_split(iris, "Species", seed = 99L)

  model <- ml_fit(s1$train, "Species", algorithm = "logistic", seed = 42L)

  # Same-split assess should work
  expect_s3_class(ml_assess(model, test = s1$test), "ml_evidence")

  # Different-split assess should fail (split-shopping)
  model2 <- ml_fit(s1$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_assess(model2, test = s2$test),
    class = "partition_error"
  )

  ml:::.clear_registry()
})

test_that("model stores provenance from fit", {
  ml:::.clear_registry()
  s <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)

  prov <- model[[".provenance"]]
  expect_true(!is.null(prov))
  expect_true(!is.null(prov$train_fingerprint))
  expect_true(!is.null(prov$split_id))
  expect_true(!is.null(prov$fit_timestamp))

  ml:::.clear_registry()
})

test_that("split-shopping works with attr-stripped data too", {
  ml:::.clear_registry()
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))

  s1 <- ml_split(iris, "Species", seed = 42L)
  s2 <- ml_split(iris, "Species", seed = 99L)

  model <- ml_fit(s1$train, "Species", algorithm = "logistic", seed = 42L)

  # Strip attr from s2$test
  test_naked <- s2$test
  attr(test_naked, "_ml_partition") <- NULL

  # Should still catch split-shopping via content-addressed lineage
  expect_error(
    ml_assess(model, test = test_naked),
    class = "partition_error"
  )

  ml:::.clear_registry()
})
