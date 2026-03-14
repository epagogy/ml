test_that("ml_embed returns ml_embedder for tfidf", {
  skip_if_not_installed("tm")
  texts <- c("good product amazing", "bad service terrible", "great value excellent",
             "poor quality awful", "fantastic delivery fast", "slow response bad")
  emb <- ml_embed(texts, method = "tfidf", max_features = 20L)
  expect_s3_class(emb, "ml_embedder")
})

test_that("ml_embedder has correct structure", {
  skip_if_not_installed("tm")
  texts <- c("good product", "bad service", "great value", "poor quality",
             "fast delivery", "slow response")
  emb <- ml_embed(texts, max_features = 10L)
  expect_true(is.data.frame(emb$vectors))
  expect_equal(nrow(emb$vectors), length(texts))
  expect_equal(emb$method, "tfidf")
  expect_true(emb$vocab_size <= 10L)
})

test_that("vocab_size respects max_features", {
  skip_if_not_installed("tm")
  texts <- c("alpha beta gamma delta", "epsilon zeta eta theta",
             "iota kappa lambda mu", "nu xi omicron pi",
             "rho sigma tau upsilon", "phi chi psi omega")
  emb <- ml_embed(texts, max_features = 5L)
  expect_true(emb$vocab_size <= 5L)
  expect_equal(ncol(emb$vectors), emb$vocab_size)
})

test_that("transform() returns data.frame with same number of columns", {
  skip_if_not_installed("tm")
  texts     <- c("good product", "bad service", "great value", "poor quality",
                  "fast delivery", "slow response")
  emb       <- ml_embed(texts, max_features = 10L)
  new_texts <- c("excellent product", "terrible service")
  new_vecs  <- emb$transform(new_texts)
  expect_true(is.data.frame(new_vecs))
  expect_equal(nrow(new_vecs), 2L)
  expect_equal(ncol(new_vecs), emb$vocab_size)
})

test_that("embed error on empty texts", {
  skip_if_not_installed("tm")
  expect_error(ml_embed(character(0)), class = "data_error")
})

test_that("embed error on non-character input", {
  skip_if_not_installed("tm")
  expect_error(ml_embed(1:5), class = "data_error")
})

test_that("embed error on unknown method", {
  skip_if_not_installed("tm")
  texts <- c("text one", "text two", "text three", "text four", "text five")
  expect_error(ml_embed(texts, method = "sbert"), class = "config_error")
})

test_that("embed print works", {
  skip_if_not_installed("tm")
  texts <- c("good product", "bad service", "great value", "poor quality",
             "fast delivery", "slow response")
  emb <- ml_embed(texts, max_features = 5L)
  expect_output(print(emb), "Embedder")
})

test_that("ml$embed() module style works", {
  skip_if_not_installed("tm")
  texts <- c("good product", "bad service", "great value", "poor quality",
             "fast delivery", "slow response")
  emb <- ml$embed(texts, max_features = 5L)
  expect_s3_class(emb, "ml_embedder")
})

test_that("embed vectors column names are tfidf_N format", {
  skip_if_not_installed("tm")
  texts <- c("alpha beta gamma", "delta epsilon zeta", "eta theta iota",
             "kappa lambda mu", "nu xi omicron", "pi rho sigma")
  emb   <- ml_embed(texts, max_features = 5L)
  expect_true(all(grepl("^tfidf_", names(emb$vectors))))
})
