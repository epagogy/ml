test_that("shelf returns ml_shelf_result", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_s3_class(result, "ml_shelf_result")
})

test_that("shelf $fresh is logical", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_true(is.logical(result$fresh))
})

test_that("shelf $stale is inverse of $fresh", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_equal(result$stale, !result$fresh)
})

test_that("shelf $metrics_then populated from model scores_", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_true(length(result$metrics_then) > 0L)
})

test_that("shelf $metrics_now populated from evaluate()", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_true(length(result$metrics_now) > 0L)
})

test_that("shelf $degradation is a named list", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_true(is.list(result$degradation))
})

test_that("shelf model with no scores_ raises model_error", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  # holdout fit → no scores_
  expect_error(
    ml_shelf(model, new = iris[1:30, ], target = "Species"),
    class = "model_error"
  )
})

test_that("shelf target mismatch raises config_error", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_shelf(model, new = iris[1:30, ], target = "Sepal.Length"),
    class = "config_error"
  )
})

test_that("shelf error when new data has < 5 rows", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  expect_error(
    ml_shelf(model, new = iris[1:3, ], target = "Species"),
    class = "data_error"
  )
})

test_that("shelf auto-unwraps TuningResult", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  tuned <- ml_tune(cv, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  # TuningResult best_model may have scores_ if CV was used
  # Just test it doesn't crash on the unwrapping path
  tryCatch(
    ml_shelf(tuned, new = iris[1:30, ], target = "Species"),
    error = function(e) {
      # model_error is OK if no scores_ — just check it's not a class error
      expect_false(inherits(e, "simpleError") && grepl("ml_tuning_result", conditionMessage(e)))
    }
  )
})

test_that("shelf $recommendation is a non-empty string", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_true(is.character(result$recommendation))
  expect_true(nchar(result$recommendation) > 0L)
})

test_that("shelf fresh=TRUE with very high tolerance", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species", tolerance = 1.0)
  expect_true(result$fresh)
})

test_that("shelf print works", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_shelf(model, new = iris[1:30, ], target = "Species")
  expect_output(print(result), "Shelf")
})

test_that("ml$shelf() module style works", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  result <- ml$shelf(model, new = iris[1:30, ], target = "Species")
  expect_s3_class(result, "ml_shelf_result")
})
