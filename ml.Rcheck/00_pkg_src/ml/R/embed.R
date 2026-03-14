#' Embed texts into numeric features
#'
#' Fits a text vectorizer on training texts and returns an embedder object
#' that stores the vocabulary for consistent transform at prediction time.
#'
#' Currently supports TF-IDF ('tm' package). SBERT and neural methods are
#' planned for future gates.
#'
#' @param texts A character vector of texts to embed
#' @param method Embedding method. Currently only "tfidf" is supported.
#' @param max_features Maximum vocabulary size (number of TF-IDF features).
#'   Default 100.
#' @returns An object of class `ml_embedder` with:
#'   - `$vectors`: data.frame of TF-IDF features (n_texts x max_features)
#'   - `$method`: the method used
#'   - `$vocab_size`: number of features generated
#'   - `$transform(new_texts)`: apply stored vocabulary to new texts
#' @export
#' @examples
#' \donttest{
#' # Requires 'tm' package: install.packages("tm")
#' texts <- c("good product", "bad service", "great value", "poor quality")
#' emb <- ml_embed(texts, method = "tfidf", max_features = 20)
#' emb$vocab_size
#' nrow(emb$vectors)
#'
#' # Transform new texts using the fitted vocabulary
#' new_texts <- c("excellent quality", "terrible service")
#' new_vecs <- emb$transform(new_texts)
#' }
ml_embed <- function(texts, method = "tfidf", max_features = 100L) {
  .embed_impl(texts = texts, method = method, max_features = max_features)
}

.embed_impl <- function(texts, method = "tfidf", max_features = 100L) {
  if (!requireNamespace("tm", quietly = TRUE)) {
    config_error("'tm' package required for embed(). Install with: install.packages('tm')")
  }

  # Validate inputs
  if (is.null(texts) || length(texts) == 0L) {
    data_error("Cannot embed empty texts (0 samples)")
  }
  if (!is.character(texts)) {
    data_error(paste0("Expected character vector, got ", class(texts)[[1]]))
  }
  if (!method %in% c("tfidf")) {
    config_error(paste0(
      "method='", method, "' not recognized. Choose from: tfidf"
    ))
  }

  max_features <- as.integer(max_features)

  if (method == "tfidf") {
    # Build corpus
    corpus <- tm::Corpus(tm::VectorSource(texts))
    corpus <- tm::tm_map(corpus, tm::content_transformer(tolower))
    corpus <- tm::tm_map(corpus, tm::removePunctuation)
    corpus <- tm::tm_map(corpus, tm::removeNumbers)
    corpus <- tm::tm_map(corpus, tm::stripWhitespace)

    # Build DTM with TF-IDF weighting
    dtm <- tm::DocumentTermMatrix(
      corpus,
      control = list(
        weighting    = tm::weightTfIdf,
        bounds       = list(global = c(1L, Inf)),
        wordLengths  = c(2L, Inf)
      )
    )

    # Convert to dense matrix (allows base R operations)
    mat_full   <- as.matrix(dtm)

    # Limit to max_features most frequent terms (by total TF-IDF weight)
    col_sums   <- colSums(mat_full)
    top_idx    <- order(col_sums, decreasing = TRUE)[seq_len(min(max_features, ncol(mat_full)))]
    top_terms  <- colnames(mat_full)[top_idx]
    mat        <- mat_full[, top_idx, drop = FALSE]

    vocab_size <- ncol(mat)
    colnames(mat) <- paste0("tfidf_", seq_len(vocab_size))
    vectors_df <- as.data.frame(mat)

    # Store vocabulary and IDF weights for transform()
    # doc_freq = number of documents containing each term
    doc_freq    <- colSums(mat_full[, top_idx, drop = FALSE] > 0)
    idf_weights <- log((nrow(mat) + 1L) / (doc_freq + 1L)) + 1L  # sklearn-style smooth IDF
    names(idf_weights) <- top_terms
    vocab       <- top_terms

    embedder <- new_ml_embedder(
      vectors    = vectors_df,
      method     = method,
      vocab_size = vocab_size,
      vocab      = vocab,
      idf        = idf_weights
    )
    return(embedder)
  }

  config_error(paste0("method='", method, "' not yet implemented."))
}

# ── S3 type ────────────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_embedder <- function(vectors, method, vocab_size, vocab, idf) {
  # transform closure captures vocab and idf
  transform_fn <- function(new_texts) {
    if (!requireNamespace("tm", quietly = TRUE)) {
      config_error("'tm' required. Install with: install.packages('tm')")
    }
    if (!is.character(new_texts) || length(new_texts) == 0L) {
      data_error("new_texts must be a non-empty character vector")
    }
    corpus <- tm::Corpus(tm::VectorSource(new_texts))
    corpus <- tm::tm_map(corpus, tm::content_transformer(tolower))
    corpus <- tm::tm_map(corpus, tm::removePunctuation)
    corpus <- tm::tm_map(corpus, tm::removeNumbers)
    corpus <- tm::tm_map(corpus, tm::stripWhitespace)

    dtm_new <- tm::DocumentTermMatrix(
      corpus,
      control = list(
        dictionary   = vocab,
        weighting    = tm::weightTf,
        wordLengths  = c(2L, Inf)
      )
    )

    # Apply stored IDF weights manually (TF × IDF)
    mat <- as.matrix(dtm_new)
    # Reorder/fill columns to match stored vocab
    aligned <- matrix(0, nrow = nrow(mat), ncol = length(vocab))
    colnames(aligned) <- vocab
    shared <- intersect(colnames(mat), vocab)
    if (length(shared) > 0L) aligned[, shared] <- mat[, shared, drop = FALSE]

    # Apply IDF
    for (j in seq_along(vocab)) {
      aligned[, j] <- aligned[, j] * idf[[vocab[[j]]]]
    }

    df_out <- as.data.frame(aligned)
    colnames(df_out) <- paste0("tfidf_", seq_len(ncol(df_out)))
    df_out
  }

  structure(
    list(
      vectors    = vectors,
      method     = method,
      vocab_size = vocab_size,
      vocab      = vocab,
      idf        = idf,
      transform  = transform_fn
    ),
    class = "ml_embedder"
  )
}

#' Print ml_embedder
#' @param x An ml_embedder object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_embedder <- function(x, ...) {
  cat(sprintf("-- Embedder [%s] --\n", x[["method"]]))
  cat(sprintf("  texts     : %d\n", nrow(x[["vectors"]])))
  cat(sprintf("  vocab_size: %d\n", x[["vocab_size"]]))
  cat(sprintf("  use $transform(new_texts) to embed new data\n"))
  cat("\n")
  invisible(x)
}
