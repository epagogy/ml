"""ml.help() — inline documentation.

Returns _HelpText object with __repr__ for clean display.
Does NOT print — returns object so callers can inspect or display.
"""

from __future__ import annotations

_VERB_DOCS = {
    'split': 'Split data into train/valid/test/dev sets.\n  s = ml.split(data, "target", seed=42)\n  s.train, s.valid, s.test, s.dev',
    'fit': 'Train a model.\n  model = ml.fit(s.train, "target", seed=42)\n  model = ml.fit(s.train, "target", algorithm="xgboost", seed=42)',
    'predict': 'Predict on new data.\n  preds = model.predict(s.valid)  # pd.Series\n  probs = model.predict_proba(s.valid)  # DataFrame',
    'evaluate': 'Compute metrics (use repeatedly during development).\n  metrics = ml.evaluate(model, s.valid)  # dict',
    'assess': 'Final evaluation — use ONCE on held-out test set.\n  verdict = ml.assess(model, s.test)',
    'explain': 'Feature importance.\n  exp = ml.explain(model)  # importance\n  exp = ml.explain(model, data=s.valid, method="shap", seed=42)',
    'screen': 'Quick algorithm comparison.\n  lb = ml.screen(s, "target", seed=42)',
    'tune': 'Hyperparameter search.\n  result = ml.tune(s.train, "target", algorithm="xgboost", seed=42)',
    'stack': 'Ensemble stacking.\n  model = ml.stack(s.train, "target", seed=42)',
    'compare': 'Compare pre-fitted models (no refitting).\n  lb = ml.compare([model1, model2], s.valid)',
    'save': 'Save model to disk.\n  ml.save(model, "model.ml")',
    'load': 'Load model from disk.\n  model = ml.load("model.ml")',
    'profile': 'Data quality report.\n  info = ml.profile(data, "target")',
    'validate': 'Assert model meets quality rules.\n  gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.85"}, baseline=baseline)',
    'drift': 'Detect distribution shift.\n  result = ml.drift(reference=s.train, new=new_data)',
    'calibrate': 'Calibrate class probabilities.\n  calibrated = ml.calibrate(model, data=s.valid)',
    'optimize': 'Find optimal classification threshold.\n  model = ml.optimize(model, data=s.valid, metric="f1")',
    'encode': 'Encode categorical/datetime features.\n  enc = ml.encode(data, columns=["cat1"], method="onehot")\n  enc = ml.encode(data, columns=["date"], method="datetime")',
    'plot': 'Visualize ML objects.\n  fig = ml.plot(model, kind="importance")\n  fig = ml.plot(model, data=s.valid, kind="roc")',
    'enough': 'Learning curve — is more data worth collecting?\n  result = ml.enough(model, data=s.valid, seed=42)',
    'interact': 'Detect feature interactions.\n  pairs = ml.interact(model, data=s.train, seed=42)',
    'shelf': 'Check model degradation over time.\n  result = ml.shelf(model, data=new_data)',
    'select': 'Select important features.\n  features = ml.select(model, data=s.valid, seed=42)',
    'quick': 'One-call workflow: split + screen + fit + evaluate.\n  model, metrics, s = ml.quick(data, "target", seed=42)',
}

_OVERVIEW = 'ml — The Linux of tabular ML\n=============================\n31 verbs. One import. Any dataset.\n\nQUICK START:\n  import ml\n  model, metrics, s = ml.quick(data, "target", seed=42)\n\nFULL WORKFLOW:\n  s = ml.split(data, "target", seed=42)         # split\n  lb = ml.screen(s, "target", seed=42)          # find best algorithm\n  model = ml.fit(s.train, "target", seed=42)    # train\n  metrics = ml.evaluate(model, s.valid)          # dev metrics\n  verdict = ml.assess(model, s.test)             # final exam\n  ml.save(model, "model.ml")                    # persist\n\nGET HELP ON A VERB:\n  ml.help("fit")\n  ml.help("split")\n\nALL VERBS:\n  split  fit  predict  evaluate  assess  explain  screen\n  tune   stack  compare  save  load  profile  validate\n  drift  calibrate  optimize  encode  plot  enough\n  interact  shelf  select  quick  check_data  report\n'


class _HelpText:
    """Container for help text with clean repr."""

    def __init__(self, text: str):
        self._text = text

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text


def help(verb = None):
    """Get help on ml verbs.

    Args:
        verb: Verb name (e.g. "fit", "split"). If None, shows overview.

    Returns:
        _HelpText with __repr__ for clean display.

    Example:
        >>> ml.help()           # overview
        >>> ml.help("fit")      # fit() docs
        >>> ml.help("tune")     # tune() docs
    """
    if verb is None:
        return _HelpText(_OVERVIEW)

    verb_lower = verb.lower().strip()
    if verb_lower in _VERB_DOCS:
        text = f"ml.{verb_lower}()" + chr(10) + chr(0x2500) * 40 + chr(10) + _VERB_DOCS[verb_lower]
        return _HelpText(text)

    import difflib

    matches = difflib.get_close_matches(verb_lower, _VERB_DOCS.keys(), n=3, cutoff=0.6)
    if matches:
        suggestions = ", ".join(f"'{m}'" for m in matches)
        text = f"Unknown verb '{verb}'. Did you mean: {suggestions}?" + chr(10) + "Run ml.help() for all verbs."
    else:
        text = f"Unknown verb '{verb}'. Run ml.help() for all verbs."
    return _HelpText(text)
