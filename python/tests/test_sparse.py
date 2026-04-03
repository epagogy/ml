"""Tests for ml.sparse() — sparse TF-IDF pipeline."""

import numpy as np
import pandas as pd
import pytest  # noqa: F401

import ml


@pytest.fixture
def text_data():
    """Synthetic text classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    reviews = []
    labels = []
    pos_words = ["great", "excellent", "amazing", "wonderful", "fantastic", "love", "best"]
    neg_words = ["terrible", "awful", "horrible", "worst", "hate", "bad", "disappointing"]
    for _i in range(n):
        if rng.random() < 0.5:
            words = rng.choice(pos_words, size=rng.randint(3, 8), replace=True)
            labels.append(1)
        else:
            words = rng.choice(neg_words, size=rng.randint(3, 8), replace=True)
            labels.append(0)
        reviews.append(" ".join(words))
    return pd.DataFrame({"review": reviews, "score": rng.randn(n), "target": labels})


def test_sparse_basic(text_data):
    """sparse() returns SparseFrame with correct shape."""
    tok = ml.tokenize(text_data, columns=["review"], max_features=50)
    sf = ml.sparse(tok, text_data, target="target")

    assert isinstance(sf, ml.SparseFrame)
    assert len(sf) == len(text_data)
    assert sf.target == "target"
    # sparse features + numeric (score + target)
    assert sf.data.shape[1] <= 50  # max_features
    assert "score" in sf.numeric.columns
    assert "target" in sf.numeric.columns


def test_sparse_to_dense(text_data):
    """to_dense() round-trips correctly."""
    tok = ml.tokenize(text_data, columns=["review"], max_features=20)
    sf = ml.sparse(tok, text_data, target="target")
    dense = sf.to_dense()

    assert isinstance(dense, pd.DataFrame)
    assert len(dense) == len(text_data)
    assert "target" in dense.columns
    assert "score" in dense.columns
    # TF-IDF columns present
    tfidf_cols = [c for c in dense.columns if "tfidf" in c]
    assert len(tfidf_cols) > 0


def test_sparse_fit_predict(text_data):
    """Full sparse pipeline: tokenize → sparse → fit → predict."""
    s = ml.split(text_data, "target", seed=42)

    tok = ml.tokenize(s.train, columns=["review"], max_features=30)
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid)

    model = ml.fit(sf_train, "target", algorithm="logistic", seed=42)
    preds = ml.predict(model, sf_valid)

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset({0, 1})


def test_sparse_evaluate(text_data):
    """evaluate() accepts SparseFrame."""
    s = ml.split(text_data, "target", seed=42)
    tok = ml.tokenize(s.train, columns=["review"], max_features=30)
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid, target="target")

    model = ml.fit(sf_train, "target", algorithm="logistic", seed=42)
    metrics = ml.evaluate(model, sf_valid)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.5  # better than random on separable data


def test_sparse_tree_algorithm(text_data):
    """Tree-based algorithms auto-densify from SparseFrame."""
    s = ml.split(text_data, "target", seed=42)
    tok = ml.tokenize(s.train, columns=["review"], max_features=20)
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid)

    model = ml.fit(sf_train, "target", algorithm="random_forest", seed=42)
    preds = ml.predict(model, sf_valid)

    assert len(preds) == len(s.valid)


def test_sparse_repr(text_data):
    """SparseFrame has informative repr."""
    tok = ml.tokenize(text_data, columns=["review"], max_features=50)
    sf = ml.sparse(tok, text_data, target="target")
    r = repr(sf)
    assert "SparseFrame" in r
    assert "sparse" in r
    assert "dense" in r


def test_sparse_multi_column():
    """sparse() handles multiple text columns."""
    df = pd.DataFrame({
        "title": ["good book", "bad movie", "great show"] * 20,
        "body": ["loved it all", "hated everything", "fantastic work"] * 20,
        "target": [1, 0, 1] * 20,
    })
    tok = ml.tokenize(df, columns=["title", "body"], max_features=10)
    sf = ml.sparse(tok, df, target="target")

    # Feature names should be title_tfidf_* and body_tfidf_*
    title_feats = [f for f in sf.feature_names if f.startswith("title_")]
    body_feats = [f for f in sf.feature_names if f.startswith("body_")]
    assert len(title_feats) > 0
    assert len(body_feats) > 0


def test_sparse_no_scipy():
    """sparse() raises clear error when scipy missing."""
    import importlib
    sparse_mod = importlib.import_module("ml.sparse")
    old = sparse_mod.HAS_SCIPY
    sparse_mod.HAS_SCIPY = False
    try:
        tok = ml.tokenize(
            pd.DataFrame({"text": ["hello world"], "y": [1]}),
            columns=["text"],
        )
        with pytest.raises(ml.ConfigError, match="scipy"):
            ml.sparse(tok, pd.DataFrame({"text": ["test"], "y": [0]}))
    finally:
        sparse_mod.HAS_SCIPY = old


def test_sparse_missing_column():
    """sparse() raises DataError on missing text columns."""
    df = pd.DataFrame({"text": ["hello"], "y": [1]})
    tok = ml.tokenize(df, columns=["text"])
    with pytest.raises(ml.DataError, match="missing"):
        ml.sparse(tok, pd.DataFrame({"wrong": ["hello"], "y": [1]}))


def test_sparse_density():
    """Sparse matrix has expected sparsity for TF-IDF."""
    docs = [f"word_{i} word_{i+1}" for i in range(100)]
    df = pd.DataFrame({"text": docs, "y": range(100)})
    tok = ml.tokenize(df, columns=["text"], max_features=200)
    sf = ml.sparse(tok, df)

    # Each doc has ~2 unique words → density should be low
    density = sf.data.nnz / (sf.data.shape[0] * sf.data.shape[1])
    assert density < 0.1  # sparse


def test_sparse_fast_path_no_normalize(text_data):
    """Sparse fast-path for logistic skips normalize entirely."""
    s = ml.split(text_data, "target", seed=42)
    tok = ml.tokenize(s.train, columns=["review"], max_features=30)
    sf_train = ml.sparse(tok, s.train, target="target")

    model = ml.fit(sf_train, "target", algorithm="logistic", seed=42)

    # Model should be marked as sparse-trained
    assert getattr(model, "_sparse", False) is True
    # NormState should have no category maps, no scaler, no imputer
    ns = model._feature_encoder
    assert ns.category_maps == {}
    assert ns.scaler is None
    assert ns.imputer is None
    assert ns.onehot_encoder is None


def test_sparse_fast_path_predict(text_data):
    """Predict on SparseFrame uses fast path (no normalize)."""
    s = ml.split(text_data, "target", seed=42)
    tok = ml.tokenize(s.train, columns=["review"], max_features=30)
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid)

    model = ml.fit(sf_train, "target", algorithm="logistic", seed=42)

    # Sparse predict
    preds_sparse = ml.predict(model, sf_valid)

    # Dense predict (for comparison — same model, dense data)
    dense_valid = sf_valid.to_dense()
    preds_dense = ml.predict(model, dense_valid)

    # Results should match
    assert len(preds_sparse) == len(preds_dense)
    # Both predict the same classes (may differ slightly due to float precision
    # but should agree on most predictions)
    agreement = (preds_sparse.values == preds_dense.values).mean()
    assert agreement > 0.9


def test_sparse_regression(text_data):
    """Sparse path works for regression with linear algorithm."""
    text_data = text_data.copy()
    text_data["target"] = np.random.RandomState(42).randn(len(text_data))

    s = ml.split(text_data, "target", seed=42)
    tok = ml.tokenize(s.train, columns=["review"], max_features=20)
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid)

    model = ml.fit(sf_train, "target", algorithm="linear", seed=42)
    assert getattr(model, "_sparse", False) is True

    preds = ml.predict(model, sf_valid)
    assert len(preds) == len(s.valid)
    assert preds.dtype == np.float64


def test_sparse_svm(text_data):
    """SVM takes sparse fast-path."""
    s = ml.split(text_data, "target", seed=42)
    tok = ml.tokenize(s.train, columns=["review"], max_features=20)
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid)

    model = ml.fit(sf_train, "target", algorithm="svm", seed=42)
    assert getattr(model, "_sparse", False) is True

    preds = ml.predict(model, sf_valid)
    assert len(preds) == len(s.valid)


def test_sparse_naive_bayes(text_data):
    """NaiveBayes auto-densifies (GaussianNB needs dense)."""
    s = ml.split(text_data, "target", seed=42)
    tok = ml.tokenize(s.train, columns=["review"], max_features=20)
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid)

    model = ml.fit(sf_train, "target", algorithm="naive_bayes", seed=42)
    # NB goes through dense path, not sparse fast-path
    assert getattr(model, "_sparse", False) is not True

    preds = ml.predict(model, sf_valid)
    assert len(preds) == len(s.valid)


def test_sparse_elastic_net():
    """ElasticNet takes sparse fast-path for regression."""
    rng = np.random.RandomState(42)
    n = 100
    docs = [f"word_{rng.randint(0, 50)} word_{rng.randint(0, 50)}" for _ in range(n)]
    df = pd.DataFrame({"text": docs, "y": rng.randn(n)})

    s = ml.split(df, "y", seed=42)
    tok = ml.tokenize(s.train, columns=["text"], max_features=50)
    sf_train = ml.sparse(tok, s.train, target="y")
    sf_valid = ml.sparse(tok, s.valid)

    model = ml.fit(sf_train, "y", algorithm="elastic_net", seed=42)
    assert getattr(model, "_sparse", False) is True

    preds = ml.predict(model, sf_valid)
    assert len(preds) == len(s.valid)


def test_sparse_high_dimensional():
    """Sparse path handles high-dimensional TF-IDF efficiently."""
    import time

    rng = np.random.RandomState(42)
    n = 500
    # Generate docs with unique vocabulary → high-dimensional sparse matrix
    words = [f"word_{i}" for i in range(2000)]
    docs = []
    labels = []
    for _i in range(n):
        k = rng.randint(5, 15)
        doc_words = rng.choice(words, size=k, replace=False)
        docs.append(" ".join(doc_words))
        labels.append(rng.randint(0, 2))

    df = pd.DataFrame({"text": docs, "target": labels})
    s = ml.split(df, "target", seed=42)

    tok = ml.tokenize(s.train, columns=["text"], max_features=1000)

    # SPARSE PATH
    t0 = time.perf_counter()
    sf_train = ml.sparse(tok, s.train, target="target")
    sf_valid = ml.sparse(tok, s.valid)
    model_s = ml.fit(sf_train, "target", algorithm="logistic", seed=42)
    preds_s = ml.predict(model_s, sf_valid)
    t_sparse = time.perf_counter() - t0

    # DENSE PATH (tokenizer → dense DataFrame → fit → predict)
    t0 = time.perf_counter()
    train_dense = tok.transform(s.train)
    valid_dense = tok.transform(s.valid)
    model_d = ml.fit(train_dense, "target", algorithm="logistic", seed=42, engine="sklearn")
    preds_d = ml.predict(model_d, valid_dense)
    t_dense = time.perf_counter() - t0

    assert len(preds_s) == len(preds_d)
    # Both should produce valid predictions
    assert set(preds_s.unique()).issubset({0, 1})
    assert set(preds_d.unique()).issubset({0, 1})

    # Sparse should be faster (skip normalize + no DataFrame construction)
    # Not asserting ratio — just proving it works at scale
    print(f"\n  sparse={t_sparse:.3f}s  dense={t_dense:.3f}s  "
          f"speedup={t_dense/max(t_sparse,0.001):.1f}x")


def test_sparse_numerical_parity(text_data):
    """Sparse TF-IDF values match dense TF-IDF values."""
    tok = ml.tokenize(text_data, columns=["review"], max_features=30)

    # Dense path (existing tokenizer.transform)
    dense_df = tok.transform(text_data)
    tfidf_cols = [c for c in dense_df.columns if "tfidf" in c]
    dense_vals = dense_df[tfidf_cols].values

    # Sparse path
    sf = ml.sparse(tok, text_data)
    sparse_vals = sf.data.toarray()

    np.testing.assert_allclose(sparse_vals, dense_vals, atol=1e-12)
