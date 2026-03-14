test_that("ml_report writes a file to the specified path", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  tmp   <- tempfile(fileext = ".html")
  on.exit(unlink(tmp))
  ml_report(model, data = s$valid, path = tmp)
  expect_true(file.exists(tmp))
})

test_that("ml_report output is HTML", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  tmp   <- tempfile(fileext = ".html")
  on.exit(unlink(tmp))
  ml_report(model, data = s$valid, path = tmp)
  content <- paste(readLines(tmp, warn = FALSE), collapse = "\n")
  expect_true(grepl("<!DOCTYPE html>", content, fixed = TRUE))
  expect_true(grepl("<html", content, fixed = TRUE))
})

test_that("ml_report returns path invisibly", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  tmp   <- tempfile(fileext = ".html")
  on.exit(unlink(tmp))
  result <- ml_report(model, data = s$valid, path = tmp)
  expect_true(is.character(result))
  expect_true(nchar(result) > 0L)
})
