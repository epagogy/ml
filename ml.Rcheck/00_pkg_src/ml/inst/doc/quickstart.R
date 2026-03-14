## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 6,
  fig.height = 4,
  out.width = "100%"
)
has_ranger <- requireNamespace("ranger", quietly = TRUE)

## ----setup--------------------------------------------------------------------
library(ml)

## ----profile------------------------------------------------------------------
prof <- ml_profile(iris, "Species")
prof

## ----split--------------------------------------------------------------------
s <- ml_split(iris, "Species", seed = 42)
s

## ----screen, eval = has_ranger------------------------------------------------
lb <- ml_screen(s, "Species", seed = 42)
lb

## ----fit-evaluate-------------------------------------------------------------
model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42)
model

metrics <- ml_evaluate(model, s$valid)
metrics

## ----plot-importance, fig.height = 3.5----------------------------------------
ml_plot(model, kind = "importance")

## ----plot-confusion, fig.height = 4-------------------------------------------
ml_plot(model, data = s$valid, kind = "confusion")

## ----explain------------------------------------------------------------------
ml_explain(model)

## ----validate-----------------------------------------------------------------
gate <- ml_validate(model,
                    test  = s$test,
                    rules = list(accuracy = ">0.80"))
gate

## ----assess-------------------------------------------------------------------
verdict <- ml_assess(model, test = s$test)
verdict

## ----drift--------------------------------------------------------------------
# Simulate drift: shift one feature by 2 units
drifted <- s$test
drifted$Sepal.Length <- drifted$Sepal.Length + 2

result <- ml_drift(reference = s$train, new = drifted, target = "Species")
result

## ----io-----------------------------------------------------------------------
path <- file.path(tempdir(), "iris_model.mlr")
ml_save(model, path)
loaded <- ml_load(path)
identical(predict(model, s$valid), predict(loaded, s$valid))

## ----module-style-------------------------------------------------------------
m2 <- ml$fit(s$train, "Species", algorithm = "logistic", seed = 42)
identical(predict(model, s$valid), predict(m2, s$valid))

## ----regression, eval = has_ranger--------------------------------------------
s2    <- ml_split(mtcars, "mpg", seed = 42)
m_reg <- ml_fit(s2$train, "mpg", seed = 42)
ml_evaluate(m_reg, s2$valid)

## ----regression-plot, eval = has_ranger, fig.height = 4-----------------------
ml_plot(m_reg, data = s2$valid, kind = "residual")

## ----algorithms---------------------------------------------------------------
ml_algorithms()

