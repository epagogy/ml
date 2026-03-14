test_that("assess returns ml_evidence", {
  s      <- iris_split()
  model  <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_assess(model, test = s$test)
  expect_s3_class(result, "ml_evidence")
})

test_that("evaluate rejects test-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s      <- iris_split()
  model  <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_evaluate(model, s$test),
    class = "partition_error"
  )
})

test_that("assess errors on 2nd call (peeking detection)", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  ml_assess(model, test = s$test)  # first call: no error
  expect_error(
    ml_assess(model, test = s$test),
    regexp = "2 times"
  )
})

test_that("assess counter increments correctly", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_equal(.get_assess_count(model), 0L)
  ml_assess(model, test = s$test)
  expect_equal(.get_assess_count(model), 1L)
  # 2nd call errors, but counter was already incremented before the abort
  tryCatch(ml_assess(model, test = s$test), error = function(e) NULL)
  expect_equal(.get_assess_count(model), 2L)
})

test_that("assess auto-unwraps TuningResult", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  result <- ml_assess(tuned, test = s$test)
  expect_s3_class(result, "ml_evidence")
})

test_that("ml$assess() module style works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  result <- ml$assess(model, test = s$test)
  expect_s3_class(result, "ml_evidence")
})

test_that("assess accuracy is between 0 and 1", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_assess(model, test = s$test)
  expect_gte(m[["accuracy"]], 0)
  expect_lte(m[["accuracy"]], 1)
})

test_that("third call errors with count=3", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  ml_assess(model, test = s$test)
  tryCatch(ml_assess(model, test = s$test), error = function(e) NULL)
  expect_error(
    ml_assess(model, test = s$test),
    regexp = "3 times"
  )
})

# ── Partition guards ──

test_that("assess errors on train-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_assess(model, test = s$train),
    class = "partition_error"
  )
})

test_that("assess errors on valid-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_assess(model, test = s$valid),
    class = "partition_error"
  )
})

test_that("assess partition error does not burn counter", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  # Wrong partition -> error, counter should NOT increment

  tryCatch(ml_assess(model, test = s$train), error = function(e) NULL)
  expect_equal(.get_assess_count(model), 0L)
  # Correct partition -> should succeed
  result <- ml_assess(model, test = s$test)
  expect_s3_class(result, "ml_evidence")
})

test_that("assess rejects untagged data in strict mode", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  untagged <- data.frame(s$test)
  expect_null(attr(untagged, "_ml_partition"))
  expect_error(ml_assess(model, test = untagged), class = "partition_error")
})

test_that("assess allows untagged data with guards off", {
  ml_config(guards = "off")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  untagged <- data.frame(s$test)
  result <- ml_assess(model, test = untagged)
  expect_s3_class(result, "ml_evidence")
})
