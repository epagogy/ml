test_that("save/load roundtrip works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  path  <- file.path(tempdir(), "test_model.mlr")
  on.exit(unlink(path), add = TRUE)
  ml_save(model, path)
  loaded <- ml_load(path)
  expect_s3_class(loaded, "ml_model")
  expect_equal(loaded$algorithm, model$algorithm)
})

test_that("saved model predictions match original", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  path  <- file.path(tempdir(), "test_model2.mlr")
  on.exit(unlink(path), add = TRUE)
  ml_save(model, path)
  loaded <- ml_load(path)
  p1 <- predict(model, newdata = s$valid)
  p2 <- predict(loaded, newdata = s$valid)
  expect_identical(p1, p2)
})

test_that("version_error on major version mismatch", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  path  <- file.path(tempdir(), "test_version.mlr")
  on.exit(unlink(path), add = TRUE)
  wrapper <- list(
    version   = "99.0.0",
    type      = "ml_model",
    model     = model,
    timestamp = Sys.time()
  )
  saveRDS(wrapper, file = path)
  expect_error(ml_load(path), class = "version_error")
})

test_that("file not found raises error", {
  expect_error(ml_load("nonexistent_file.mlr"))
})

test_that("TuningResult save/load works", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  path  <- file.path(tempdir(), "test_tuned.mlr")
  on.exit(unlink(path), add = TRUE)
  ml_save(tuned, path)
  loaded <- ml_load(path)
  expect_s3_class(loaded, "ml_tuning_result")
})

test_that("ml$save() and ml$load() module style work", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  path  <- file.path(tempdir(), "test_module.mlr")
  on.exit(unlink(path), add = TRUE)
  ml$save(model, path)
  loaded <- ml$load(path)
  expect_s3_class(loaded, "ml_model")
})

test_that("save returns path invisibly", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  path  <- file.path(tempdir(), "test_ret.mlr")
  on.exit(unlink(path), add = TRUE)
  result <- ml_save(model, path)
  expect_true(file.exists(result))
})

test_that("same major version loads without error", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  path  <- file.path(tempdir(), "test_samever.mlr")
  on.exit(unlink(path), add = TRUE)
  ml_save(model, path)
  # Should succeed
  loaded <- ml_load(path)
  expect_s3_class(loaded, "ml_model")
})
