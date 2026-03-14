test_that("profile returns ml_profile_result with shape", {
  prof <- ml_profile(iris, "Species")
  expect_s3_class(prof, "ml_profile_result")
  expect_equal(prof$shape[[1]], nrow(iris))
  expect_equal(prof$shape[[2]], ncol(iris))
})

test_that("profile reports correct number of rows/columns", {
  prof <- ml_profile(mtcars, "mpg")
  expect_equal(prof$shape[[1]], nrow(mtcars))
  expect_equal(prof$shape[[2]], ncol(mtcars))
})

test_that("profile detects numeric stats (mean, sd, min, max, median)", {
  prof <- ml_profile(iris)
  sl_stats <- prof$columns[["Sepal.Length"]]
  expect_true(!is.null(sl_stats$mean))
  expect_true(!is.null(sl_stats$sd))
  expect_true(!is.null(sl_stats$min))
  expect_true(!is.null(sl_stats$max))
  expect_true(!is.null(sl_stats$median))
})

test_that("profile detects categorical stats (top, top_freq)", {
  prof <- ml_profile(iris, "Species")
  sp_stats <- prof$columns[["Species"]]
  expect_true(!is.null(sp_stats$top))
  expect_true(!is.null(sp_stats$top_freq))
})

test_that("profile detects missing values", {
  df <- iris
  df$Sepal.Length[1:10] <- NA
  prof <- ml_profile(df, "Species")
  expect_true(any(grepl("missing", prof$warnings, ignore.case = TRUE)))
})

test_that("profile detects constant columns", {
  df <- iris
  df$constant <- 1L
  prof <- ml_profile(df)
  expect_true(any(grepl("constant", prof$warnings, ignore.case = TRUE)))
})

test_that("profile detects imbalanced binary target", {
  df <- binary_df()
  df$churn <- ifelse(df$churn == "yes", "yes", "no")
  # Make it heavily imbalanced
  df$churn[1:85] <- "no"
  df$churn[86:100] <- "yes"
  prof <- ml_profile(df, "churn")
  expect_true(any(grepl("[Ii]mbalanced|minority", prof$warnings)))
})

test_that("profile detects classification task for factor target", {
  prof <- ml_profile(iris, "Species")
  expect_equal(prof$task, "classification")
})

test_that("profile detects regression task for numeric target", {
  prof <- ml_profile(mtcars, "mpg")
  expect_equal(prof$task, "regression")
})

test_that("profile error on empty data", {
  expect_error(ml_profile(data.frame()), class = "data_error")
})

test_that("profile error on bad target", {
  expect_error(ml_profile(iris, "nonexistent"), class = "data_error")
})

test_that("profile small dataset warning", {
  small_df <- iris[1:50, , drop = FALSE]
  prof     <- ml_profile(small_df, "Species")
  expect_true(any(grepl("[Ss]mall", prof$warnings)))
})

test_that("ml$profile() module style works", {
  prof <- ml$profile(iris, "Species")
  expect_s3_class(prof, "ml_profile_result")
})
