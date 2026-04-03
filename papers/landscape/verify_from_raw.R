# Verify paper claims directly from raw JSONL. Base R only.
# Usage: Rscript verify_from_raw.R

library(jsonlite)

claims <- fromJSON("claims.json")

v1 <- stream_in(file("data/leakage_landscape_v1_final.jsonl"), verbose = FALSE)
v1_ok <- v1[v1$status == "ok", ]

v2 <- stream_in(file("data/leakage_landscape_v2.jsonl"), verbose = FALSE)

v3_an <- stream_in(file("data/v3_an.jsonl"), verbose = FALSE)
v3_an_ok <- v3_an[v3_an$v3_status == "ok", ]

dz <- function(x) {
  x <- x[!is.na(x)]
  mean(x) / sd(x)
}

check <- function(name, expected, got, tol = 0.002) {
  ok <- abs(expected - got) <= tol
  cat(sprintf("  %s: expected=%s, got=%s, %s\n", name, expected, round(got, 4),
              ifelse(ok, "PASS", "FAIL")))
  ok
}

passed <- 0L; total <- 0L

cat("=== Dataset counts ===\n")
total <- total + 1L; passed <- passed + check("n_datasets", claims$n_datasets, nrow(v1_ok), tol = 0)
total <- total + 1L; passed <- passed + check("corpus.median_n", claims$corpus$median_n,
                                                median(v1_ok$n_rows), tol = 0)

cat("\n=== Class I: Estimation ===\n")
d <- v1_ok$a_lr_gap_diff[!is.na(v1_ok$a_lr_gap_diff)]
total <- total + 1L; passed <- passed + check("norm_lr.dz", claims$norm_lr$dz, dz(d))

cat("\n=== Class II: Peeking ===\n")
d <- v1_ok$b_infl_k10[!is.na(v1_ok$b_infl_k10)]
total <- total + 1L; passed <- passed + check("peek.dz", claims$peek$dz, dz(d))
total <- total + 1L; passed <- passed + check("peek.auc", claims$peek$auc, mean(d), tol = 0.001)

cat("\n=== Class II: Seed ===\n")
d <- v2$ai_inflation[!is.na(v2$ai_inflation)]
total <- total + 1L; passed <- passed + check("seed.dz", claims$seed$dz, dz(d))

cat("\n=== Class II: Screen ===\n")
d <- v2$aq_k1_optimism[!is.na(v2$aq_k1_optimism)]
total <- total + 1L; passed <- passed + check("screen.dz", claims$screen$dz, dz(d))

cat("\n=== N-scaling ===\n")
total <- total + 1L; passed <- passed + check("nscale.n_main", claims$nscale$n_main,
                                                sum(v3_an_ok$an_n_full == 2000), tol = 0)
total <- total + 1L; passed <- passed + check("nscale.ext.n_datasets", claims$nscale$ext$n_datasets,
                                                sum(v3_an_ok$an_n_full == 10000), tol = 0)

cat(sprintf("\n%s\nRESULT: %d/%d checks passed\n", strrep("=", 40), passed, total))
if (passed == total) cat("ALL CLAIMS VERIFIED FROM RAW DATA.\n")
