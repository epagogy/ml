"""query() — structured model interrogation.

Ask questions about a model's behavior: feature effects, decision
boundaries, counterfactuals, and what-if scenarios.

Usage:
    >>> answer = ml.query(model, "what drives churn?")
    >>> answer = ml.query(model, "what if tenure=24?", data=row)

Status: stub — not yet implemented. Planned for v2.
"""

from __future__ import annotations


def query(model, question, *, data=None):
    """Query a model about its behavior.

    Provides a natural-language interface to model interrogation:
    feature effects, counterfactuals, decision boundaries.

    Args:
        model: A trained ml.Model
        question: Natural-language question about model behavior
        data: Optional observation(s) for counterfactual queries

    Returns:
        QueryResult — structured answer with evidence

    Raises:
        NotImplementedError: Always (v2 feature)

    Example:
        >>> answer = ml.query(model, "what drives predictions?")
        >>> answer = ml.query(model, "what if age=30?", data=customer)
    """
    raise NotImplementedError(
        "ml.query() is planned for v2. "
        "For now, use ml.explain() for feature importance "
        "and ml.interact() for interaction effects."
    )
