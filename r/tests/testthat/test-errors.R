test_that("ml_error() is catchable as ml_error class", {
  expect_error(ml_error("test"), class = "ml_error")
})

test_that("config_error() is catchable as config_error and ml_error", {
  expect_error(config_error("bad config"), class = "config_error")
  expect_error(config_error("bad config"), class = "ml_error")
})

test_that("data_error() is catchable as data_error and ml_error", {
  expect_error(data_error("bad data"), class = "data_error")
  expect_error(data_error("bad data"), class = "ml_error")
})

test_that("model_error() is catchable as model_error and ml_error", {
  expect_error(model_error("bad model"), class = "model_error")
  expect_error(model_error("bad model"), class = "ml_error")
})

test_that("version_error() is catchable as version_error and ml_error", {
  expect_error(version_error("version mismatch"), class = "version_error")
  expect_error(version_error("version mismatch"), class = "ml_error")
})
