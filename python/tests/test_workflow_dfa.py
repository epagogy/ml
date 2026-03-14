"""Test workflow DFA state transitions.

The ML workflow is a regular language recognized by a finite automaton:
CREATED(0) -> FITTED(1) -> EVALUATED(2) -> ASSESSED(3, terminal sink).

Idempotent verbs (explain, predict, calibrate, validate) don't change state.
assess() transitions to terminal ASSESSED — no further transitions allowed.
"""

import pytest

from ml._types import (
    WorkflowState,
    WorkflowStateError,
    check_workflow_transition,
)


class TestWorkflowStateValues:
    def test_state_values(self):
        assert WorkflowState.CREATED == 0
        assert WorkflowState.FITTED == 1
        assert WorkflowState.EVALUATED == 2
        assert WorkflowState.ASSESSED == 3

    def test_state_is_int(self):
        assert isinstance(WorkflowState.FITTED, int)
        assert WorkflowState.FITTED + 1 == 2

    def test_names_mapping(self):
        assert WorkflowState._NAMES[0] == "CREATED"
        assert WorkflowState._NAMES[3] == "ASSESSED"


class TestTransitionsFromFitted:
    """FITTED is the initial state after ml.fit()."""

    def test_evaluate_transitions_to_evaluated(self):
        new = check_workflow_transition(WorkflowState.FITTED, "evaluate")
        assert new == WorkflowState.EVALUATED

    def test_assess_transitions_to_assessed(self):
        new = check_workflow_transition(WorkflowState.FITTED, "assess")
        assert new == WorkflowState.ASSESSED

    def test_explain_stays_fitted(self):
        new = check_workflow_transition(WorkflowState.FITTED, "explain")
        assert new == WorkflowState.FITTED

    def test_predict_stays_fitted(self):
        new = check_workflow_transition(WorkflowState.FITTED, "predict")
        assert new == WorkflowState.FITTED

    def test_calibrate_stays_fitted(self):
        new = check_workflow_transition(WorkflowState.FITTED, "calibrate")
        assert new == WorkflowState.FITTED

    def test_validate_stays_fitted(self):
        new = check_workflow_transition(WorkflowState.FITTED, "validate")
        assert new == WorkflowState.FITTED


class TestTransitionsFromEvaluated:
    """EVALUATED: after at least one evaluate() call."""

    def test_evaluate_idempotent(self):
        new = check_workflow_transition(WorkflowState.EVALUATED, "evaluate")
        assert new == WorkflowState.EVALUATED

    def test_assess_transitions_to_assessed(self):
        new = check_workflow_transition(WorkflowState.EVALUATED, "assess")
        assert new == WorkflowState.ASSESSED

    def test_explain_stays_evaluated(self):
        new = check_workflow_transition(WorkflowState.EVALUATED, "explain")
        assert new == WorkflowState.EVALUATED

    def test_predict_stays_evaluated(self):
        new = check_workflow_transition(WorkflowState.EVALUATED, "predict")
        assert new == WorkflowState.EVALUATED


class TestAssessedTerminal:
    """ASSESSED is a terminal sink — no transitions out."""

    @pytest.mark.parametrize("verb", ["evaluate", "explain", "predict", "assess", "calibrate", "validate"])
    def test_all_verbs_rejected(self, verb):
        with pytest.raises(WorkflowStateError, match="assessed model"):
            check_workflow_transition(WorkflowState.ASSESSED, verb)

    def test_error_message_mentions_terminal(self):
        with pytest.raises(WorkflowStateError, match="terminal"):
            check_workflow_transition(WorkflowState.ASSESSED, "evaluate")


class TestCreatedState:
    """CREATED(0) — before fit. No valid transitions."""

    @pytest.mark.parametrize("verb", ["evaluate", "explain", "predict", "assess"])
    def test_verbs_rejected_before_fit(self, verb):
        with pytest.raises(WorkflowStateError):
            check_workflow_transition(WorkflowState.CREATED, verb)


class TestUnknownVerb:
    def test_unknown_verb_raises(self):
        with pytest.raises(WorkflowStateError):
            check_workflow_transition(WorkflowState.FITTED, "nonexistent_verb")


class TestIntegrationWithModel:
    """DFA wired into actual ml verbs via contextlib.suppress."""

    def test_model_starts_fitted(self):
        import pandas as pd

        import ml
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        model = ml.fit(df, "y", seed=42)
        assert model._workflow_state == WorkflowState.FITTED

    def test_evaluate_advances_to_evaluated(self):
        import pandas as pd

        import ml
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        model = ml.fit(df, "y", seed=42)
        ml.evaluate(model, df)
        assert model._workflow_state == WorkflowState.EVALUATED

    def test_assess_advances_to_assessed(self):
        import pandas as pd

        import ml
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        s = ml.split(df, "y", seed=42)
        model = ml.fit(s.dev, "y", seed=42)
        ml.assess(model, test=s.test)
        assert model._workflow_state == WorkflowState.ASSESSED

    def test_explain_preserves_state(self):
        import pandas as pd

        import ml
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        model = ml.fit(df, "y", seed=42)
        ml.explain(model)
        assert model._workflow_state == WorkflowState.FITTED

    def test_predict_preserves_state(self):
        import pandas as pd

        import ml
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        model = ml.fit(df, "y", seed=42)
        ml.predict(model, df.drop(columns=["y"]))
        assert model._workflow_state == WorkflowState.FITTED

    def test_full_workflow_sequence(self):
        """FITTED -> evaluate -> EVALUATED -> assess -> ASSESSED."""
        import pandas as pd

        import ml
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        s = ml.split(df, "y", seed=42)
        model = ml.fit(s.dev, "y", seed=42)
        assert model._workflow_state == WorkflowState.FITTED

        ml.evaluate(model, s.train)
        assert model._workflow_state == WorkflowState.EVALUATED

        ml.explain(model)
        assert model._workflow_state == WorkflowState.EVALUATED  # idempotent

        ml.assess(model, test=s.test)
        assert model._workflow_state == WorkflowState.ASSESSED

    def test_workflow_state_serializable(self):
        """WorkflowState is int — survives dataclass serialization."""
        import pandas as pd

        import ml
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        model = ml.fit(df, "y", seed=42)
        # int subclass — JSON-safe, pickle-safe, dataclass-safe
        assert int(model._workflow_state) == 1
        assert isinstance(model._workflow_state, int)
