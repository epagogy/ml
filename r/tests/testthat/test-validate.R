test_that("validate rules pass correctly", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.01"))
  expect_true(gate$passed)
})

test_that("validate rules fail correctly with informative message", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.9999"))
  if (!gate$passed) {
    expect_true(length(gate$failures) > 0L)
    expect_true(grepl("accuracy", gate$failures[[1]]))
  }
})

test_that("validate baseline comparison detects improvement", {
  s      <- iris_split()
  old_m  <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 1L)
  new_m  <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate   <- ml_validate(new_m, test = s$test, baseline = old_m)
  expect_s3_class(gate, "ml_validate_result")
})

test_that("validate baseline detects degradation", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  # Use same model as baseline — tolerance=0 exact match → no degradation
  gate  <- ml_validate(model, test = s$test, baseline = model, tolerance = 0)
  expect_true(gate$passed)
})

test_that("validate tolerance allows small degradation", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test, baseline = model, tolerance = 0.1)
  expect_true(gate$passed)
})

test_that("validate combined mode (rules + baseline)", {
  s      <- iris_split()
  model  <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate   <- ml_validate(model, test = s$test,
                         rules    = list(accuracy = ">0.01"),
                         baseline = model)
  expect_s3_class(gate, "ml_validate_result")
})

test_that("validate error: no rules and no baseline", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_validate(model, test = s$test),
    class = "config_error"
  )
})

test_that("validate lower-is-better metrics handled correctly (rmse)", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  gate  <- ml_validate(model, test = s$test,
                        rules = list(rmse = "<1000"))
  expect_true(gate$passed)
})

test_that("validate operators >, >=, <, <= all work", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate1 <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.0"))
  gate2 <- ml_validate(model, test = s$test, rules = list(accuracy = ">=0.0"))
  expect_true(gate1$passed)
  expect_true(gate2$passed)
})

test_that("validate error on invalid rule syntax", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_validate(model, test = s$test, rules = list(accuracy = "~0.85")),
    class = "config_error"
  )
})

test_that("validate custom print shows PASSED/FAILED header", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.01"))
  expect_output(print(gate), "PASSED|FAILED")
})

test_that("validate baseline target mismatch raises config_error", {
  s1 <- iris_split()
  s2 <- mtcars_split()
  m1 <- ml_fit(s1$train, "Species", algorithm = "logistic", seed = 42L)
  skip_if_not_installed("glmnet")
  m2 <- ml_fit(s2$train, "mpg", algorithm = "linear", seed = 42L)
  expect_error(
    ml_validate(m1, test = s1$test, baseline = m2),
    class = "config_error"
  )
})

test_that("validate auto-unwraps TuningResult", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  gate  <- ml_validate(tuned, test = s$test, rules = list(accuracy = ">0.0"))
  expect_s3_class(gate, "ml_validate_result")
})

test_that("ml$validate() module style works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml$validate(model, test = s$test, rules = list(accuracy = ">0.0"))
  expect_s3_class(gate, "ml_validate_result")
})

# ── Partition guard tests ───────────────────────────────────────────────────

test_that("validate rejects train-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_validate(model, test = s$train, rules = list(accuracy = ">0.0")),
    class = "partition_error"
  )
})

test_that("validate rejects valid-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_validate(model, test = s$valid, rules = list(accuracy = ">0.0")),
    class = "partition_error"
  )
})

test_that("validate accepts test-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.0"))
  expect_true(gate$passed)
})

test_that("validate rejects genuinely unsplit data in strict mode", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  # Use data that never passed through split
  unsplit <- iris[1:30, ]
  expect_error(
    ml_validate(model, test = unsplit, rules = list(accuracy = ">0.0")),
    class = "partition_error"
  )
})

test_that("validate allows unsplit data with guards off", {
  ml_config(guards = "off")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  unsplit <- iris[1:30, ]
  gate <- ml_validate(model, test = unsplit, rules = list(accuracy = ">0.0"))
  expect_s3_class(gate, "ml_validate_result")
})
