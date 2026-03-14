pkgname <- "ml"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
base::assign(".ExTimings", "ml-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('ml')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("ml")
### * ml

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml
### Title: The ml module - all verbs accessed via ml$verb()
### Aliases: ml
### Keywords: datasets

### ** Examples

s <- ml$split(iris, "Species", seed = 42)
model <- ml$fit(s$train, "Species", seed = 42)
ml$evaluate(model, s$valid)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_algorithms")
### * ml_algorithms

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_algorithms
### Title: List available ML algorithms
### Aliases: ml_algorithms

### ** Examples

ml_algorithms()
ml_algorithms(task = "classification")



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_algorithms", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_assess")
### * ml_assess

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_assess
### Title: Assess model on held-out test data (do once)
### Aliases: ml_assess

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
verdict <- ml_assess(model, test = s$test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_assess", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_best")
### * ml_best

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_best
### Title: Get the best model from a leaderboard
### Aliases: ml_best

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
lb <- ml_screen(s, "Species", seed = 42)
best <- ml_best(lb)
predict(best, s$valid)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_best", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_calibrate")
### * ml_calibrate

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_calibrate
### Title: Calibrate predicted probabilities
### Aliases: ml_calibrate

### ** Examples

## No test: 
s <- ml_split(ml_dataset("cancer"), "target", seed = 42)
model <- ml_fit(s$train, "target", algorithm = "xgboost", seed = 42)
cal <- ml_calibrate(model, data = s$valid)
ml_evaluate(cal, s$valid)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_calibrate", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_check")
### * ml_check

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_check
### Title: Verify bitwise reproducibility for a given dataset
### Aliases: ml_check

### ** Examples

## No test: 
result <- ml_check(iris, "Species", seed = 42)
result$passed
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_check", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_check_data")
### * ml_check_data

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_check_data
### Title: Pre-flight data quality checks
### Aliases: ml_check_data

### ** Examples

report <- ml_check_data(iris, "Species")
report$passed



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_check_data", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_compare")
### * ml_compare

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_compare
### Title: Compare pre-fitted models on the same data
### Aliases: ml_compare

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42)
m2 <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
ml_compare(list(m1, m2), s$valid)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_compare", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_config")
### * ml_config

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_config
### Title: Configure ml package settings
### Aliases: ml_config

### ** Examples

ml_config(guards = "off")    # disable guards
ml_config(guards = "strict") # re-enable (default)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_config", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_dataset")
### * ml_dataset

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_dataset
### Title: Load a built-in dataset
### Aliases: ml_dataset

### ** Examples

churn <- ml_dataset("churn")
head(churn)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_dataset", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_drift")
### * ml_drift

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_drift
### Title: Detect data drift between reference and new data
### Aliases: ml_drift

### ** Examples

s    <- ml_split(iris, "Species", seed = 42)
# Simulate drift by perturbing test data
new  <- s$test
new$Sepal.Length <- new$Sepal.Length + 2
result <- ml_drift(reference = s$train, new = new, target = "Species")
result$shifted
result$features_shifted



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_drift", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_embed")
### * ml_embed

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_embed
### Title: Embed texts into numeric features
### Aliases: ml_embed

### ** Examples

## No test: 
# Requires 'tm' package: install.packages("tm")
texts <- c("good product", "bad service", "great value", "poor quality")
emb <- ml_embed(texts, method = "tfidf", max_features = 20)
emb$vocab_size
nrow(emb$vectors)

# Transform new texts using the fitted vocabulary
new_texts <- c("excellent quality", "terrible service")
new_vecs <- emb$transform(new_texts)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_embed", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_enough")
### * ml_enough

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_enough
### Title: Check if sample size is sufficient before training
### Aliases: ml_enough

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
ml_enough(s, "Species")



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_enough", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_evaluate")
### * ml_evaluate

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_evaluate
### Title: Evaluate model on validation data (iterate freely)
### Aliases: ml_evaluate

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
metrics <- ml_evaluate(model, s$valid)
metrics[["accuracy"]]



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_evaluate", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_explain")
### * ml_explain

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_explain
### Title: Explain model via feature importance
### Aliases: ml_explain

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
ml_explain(model)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_explain", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_fit")
### * ml_fit

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_fit
### Title: Fit a machine learning model
### Aliases: ml_fit

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
model$algorithm



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_fit", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_leak")
### * ml_leak

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_leak
### Title: Detect potential data leakage
### Aliases: ml_leak

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
report <- ml_leak(s, "Species")
report$clean



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_leak", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_load")
### * ml_load

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_load
### Title: Load a model from disk
### Aliases: ml_load

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
path <- file.path(tempdir(), "iris_model.mlr")
ml_save(model, path)
loaded <- ml_load(path)
loaded$algorithm
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_load", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_plot")
### * ml_plot

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_plot
### Title: Visual diagnostics for a fitted model
### Aliases: ml_plot

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
ml_plot(model, kind = "importance")
ml_plot(model, data = s$valid, kind = "confusion")
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_plot", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_predict")
### * ml_predict

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_predict
### Title: Predict from a fitted model (ml_predict style)
### Aliases: ml_predict

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
preds <- ml_predict(model, s$valid)
head(preds)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_predict", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_predict_proba")
### * ml_predict_proba

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_predict_proba
### Title: Predict class probabilities
### Aliases: ml_predict_proba

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
probs <- ml_predict_proba(model, s$valid)
head(probs)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_predict_proba", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_prepare")
### * ml_prepare

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_prepare
### Title: Prepare data for ML: encode, impute, and scale
### Aliases: ml_prepare

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
p <- ml_prepare(s$train, "Species")
p$task       # "classification"
head(p$data) # encoded feature matrix
## End(No test)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_prepare", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_profile")
### * ml_profile

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_profile
### Title: Profile data before modeling
### Aliases: ml_profile

### ** Examples

ml_profile(iris, "Species")



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_profile", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_quick")
### * ml_quick

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_quick
### Title: One-call workflow: split + screen + fit + evaluate
### Aliases: ml_quick

### ** Examples

result <- ml_quick(iris, "Species", seed = 42)
result$model
result$metrics



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_quick", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_report")
### * ml_report

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_report
### Title: Generate an HTML training report
### Aliases: ml_report

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
tmp <- tempfile(fileext = ".html")
ml_report(model, data = s$valid, path = tmp)
unlink(tmp)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_report", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_save")
### * ml_save

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_save
### Title: Save a model to disk
### Aliases: ml_save

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
path <- file.path(tempdir(), "iris_model.mlr")
ml_save(model, path)
loaded <- ml_load(path)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_save", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_screen")
### * ml_screen

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_screen
### Title: Screen all algorithms on your data
### Aliases: ml_screen

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
lb <- ml_screen(s, "Species", seed = 42)
lb
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_screen", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_shelf")
### * ml_shelf

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_shelf
### Title: Check if a model is past its shelf life
### Aliases: ml_shelf

### ** Examples

## No test: 
cv    <- ml_split(iris, "Species", seed = 42, folds = 3)
model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42)
# Simulate a new labeled batch
new_batch <- iris[sample(nrow(iris), 30), ]
result <- ml_shelf(model, new = new_batch, target = "Species")
result$fresh
result$degradation
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_shelf", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_split")
### * ml_split

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_split
### Title: Split data into train/valid/test partitions or cross-validation
###   folds
### Aliases: ml_split

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
nrow(s$train)
nrow(s$dev)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_split", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_split_group")
### * ml_split_group

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_split_group
### Title: Split data with group non-overlap - no group leaks across
###   partitions
### Aliases: ml_split_group

### ** Examples

df <- data.frame(pid = rep(1:10, each = 5), x = rnorm(50), y = sample(0:1, 50, TRUE))
s <- ml_split_group(df, "y", groups = "pid", seed = 42)
nrow(s$train)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_split_group", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_split_temporal")
### * ml_split_temporal

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_split_temporal
### Title: Split data chronologically - no future leakage
### Aliases: ml_split_temporal

### ** Examples

df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
s <- ml_split_temporal(df, "y", time = "date")
nrow(s$train)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_split_temporal", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_stack")
### * ml_stack

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_stack
### Title: Ensemble stacking
### Aliases: ml_stack

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
stacked <- ml_stack(s$train, "Species", seed = 42)
predict(stacked, s$valid)
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_stack", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_tune")
### * ml_tune

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_tune
### Title: Tune hyperparameters via random or grid search
### Aliases: ml_tune

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
tuned <- ml_tune(s$train, "Species", algorithm = "xgboost", n_trials = 5, seed = 42)
tuned$best_params_
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_tune", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ml_validate")
### * ml_validate

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ml_validate
### Title: Validate model against rules and/or baseline
### Aliases: ml_validate

### ** Examples

## No test: 
s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
gate <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.80"))
gate$passed
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ml_validate", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("predict.ml_model")
### * predict.ml_model

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: predict.ml_model
### Title: Predict from a fitted model
### Aliases: predict.ml_model

### ** Examples

s <- ml_split(iris, "Species", seed = 42)
model <- ml_fit(s$train, "Species", seed = 42)
preds <- predict(model, newdata = s$valid)
head(preds)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("predict.ml_model", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
