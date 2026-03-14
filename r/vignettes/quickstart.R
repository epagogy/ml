## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE  # examples shown but not run during check (require optional deps)
)


## ----setup--------------------------------------------------------------------
library(ml)


## ----profile------------------------------------------------------------------
prof <- ml_profile(iris, "Species")
prof


## ----split--------------------------------------------------------------------
s <- ml_split(iris, "Species", seed = 42)
s


## ----screen-------------------------------------------------------------------
lb <- ml_screen(s, "Species", seed = 42)
lb


## ----fit-evaluate-------------------------------------------------------------
model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42)
model

metrics <- ml_evaluate(model, s$valid)
metrics


## ----explain------------------------------------------------------------------
exp <- ml_explain(model)
exp


## ----validate-----------------------------------------------------------------
gate <- ml_validate(model,
                    test  = s$test,
                    rules = list(accuracy = ">0.70"))
gate


## ----assess-------------------------------------------------------------------
verdict <- ml_assess(model, test = s$test)
verdict


## ----io, eval = FALSE---------------------------------------------------------
# path <- file.path(tempdir(), "iris_model.mlr")
# ml_save(model, path)
# loaded <- ml_load(path)
# predict(loaded, s$valid)[1:5]


## ----module-style-------------------------------------------------------------
# Identical results — pick the style you prefer
m2 <- ml$fit(s$train, "Species", algorithm = "logistic", seed = 42)
identical(predict(model, s$valid), predict(m2, s$valid))


## ----regression---------------------------------------------------------------
s2   <- ml_split(mtcars, "mpg", seed = 42)
m_rf <- ml_fit(s2$train, "mpg", seed = 42)
ml_evaluate(m_rf, s2$valid)


## ----algorithms---------------------------------------------------------------
ml_algorithms()

