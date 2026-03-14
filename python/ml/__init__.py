"""ml - ML that works. Simple, lovable, complete.

42 verbs + 3 data helpers:
    Core:     split, split_temporal, split_group, fit, predict, evaluate, assess, explain, save, load
    Screen:   screen, compare, tune, stack, validate, profile
    Monitor:  drift, shelf, calibrate
    Preproc:  tokenize, scale, encode, impute, pipe, discretize, null_flags
    Analysis: interact, enough, leak, blend, cluster_features, select, optimize,
              nested_cv, plot, report, check, check_data
    Data:     dataset, datasets, algorithms

Import and use:
    >>> import ml
    >>> data = ml.dataset("tips")          # bundled — no internet needed
    >>> s = ml.split(data, "tip", seed=42)
    >>> leaderboard = ml.screen(s, "tip", seed=42)   # quick filter
    >>> model = ml.fit(s.train, "tip", seed=42)
    >>> metrics = ml.evaluate(model, s.valid)         # practice exam, iterate freely
    >>> final = ml.fit(s.dev, "tip", seed=42)         # retrain on all dev data
    >>> gate = ml.validate(final, test=s.test)        # optional rules gate
    >>> verdict = ml.assess(final, test=s.test)       # one-time final exam
"""

__version__ = "1.1.1"

# Core workflow functions
from .split import split, split_temporal, split_group
from .prepare import prepare
from .fit import fit
from .evaluate import evaluate
from .assess import assess
from .explain import explain
from .io import save, load
from .utils import algorithms, dataset, datasets
from .profile import profile
from .screen import screen
from .predict import predict, predict_proba
from .compare import compare
from .tune import tune
from .stack import stack
from .validate import validate, ValidateResult
from .calibrate import calibrate
from .tokenize import tokenize, Tokenizer
from .scale import scale, Scaler
from .encode import encode, Encoder
from .impute import impute, Imputer
from .drift import drift, DriftResult
from .shelf import shelf, ShelfResult
from .pipeline import pipe, Pipeline
from .interact import interact, InteractResult
from .enough import enough, EnoughResult
from .optimize import optimize
from .cv import cv, cv_temporal, cv_group  # noqa: F401
from .nested import nested_cv, NestedCVResult
from .blend import blend
from .bin import discretize, Binner
from .null_flags import null_flags
from .cluster import cluster_features
from .select import select
from ._config import config
from ._help import help
from ._provenance import audit  # noqa: F401

# Held back for v1.1 — internal only
# embed/Embedder not exported: use ml._embed or import from ml.embed directly
from .leak import leak  # noqa: F401
from .check import check_data, check, CheckReport, CheckResult
from .report import report
from .plot import plot

# V2 stubs — wired but raise NotImplementedError
from .update import update
from .query import query

# Data structures
from ._types import (
    Model,
    SplitResult,
    CVResult,
    PreparedData,
    TuningResult,
    OptimizeResult,
    # Display types
    Evidence,  # noqa: F401
    Metrics,
    ProfileResult,
    Explanation,
    Leaderboard,
    LeakReport,  # noqa: F401
    # Errors
    MLError,
    ConfigError,
    DataError,
    ModelError,
    PartitionError,
    VersionError,
    WorkflowState,  # noqa: F401
    WorkflowStateError,  # noqa: F401
)


def quick(data, target, *, seed):
    """One-call workflow: split + screen + fit + evaluate.

    The fastest path from raw data to a trained, evaluated model.

    Args:
        data: DataFrame with features and target column
        target: Target column name
        seed: Random seed (keyword-only)

    Returns:
        tuple of (model, metrics, split_result)

    Example:
        >>> model, metrics, s = ml.quick(data, "target", seed=42)
        >>> print(metrics)
    """
    s = split(data, target, seed=seed)
    lb = screen(s, target, seed=seed, algorithms=["logistic", "random_forest", "xgboost"])
    best_algo = lb.best
    model = fit(s.train, target, algorithm=best_algo, seed=seed)
    metrics = evaluate(model, s.valid)
    return model, metrics, s


def quiet():
    """Suppress all ml warnings.

    Call ml.verbose() to restore warnings.

    Example:
        >>> ml.quiet()   # suppress warnings
        >>> model = ml.fit(data, "target", seed=42)
        >>> ml.verbose() # restore
    """
    import warnings
    warnings.filterwarnings("ignore", module="ml")


def verbose():
    """Restore ml warnings (after ml.quiet()).

    Example:
        >>> ml.quiet()   # suppress
        >>> ml.verbose() # restore
    """
    import warnings
    warnings.filterwarnings("default", module="ml")

# Public API — what shows up in docs and tab-completion
__all__ = [
    # Version
    "__version__",
    # Functions (40 verbs + 3 data helpers)
    "split",
    "split_temporal",
    "split_group",
    "prepare",
    "fit",
    "evaluate",
    "assess",
    "explain",
    "save",
    "load",
    "algorithms",
    "dataset",
    "datasets",
    "profile",
    "screen",
    "compare",
    "tune",
    "stack",
    "predict",
    "predict_proba",
    "validate",
    "calibrate",
    "tokenize",
    "scale",
    "encode",
    "impute",
    "drift",
    "shelf",
    "interact",
    "enough",
    "optimize",
    "nested_cv",
    "blend",
    "discretize",
    "null_flags",
    "cluster_features",
    "select",
    "config",
    "pipe",
    "leak",
    "plot",
    "help",
    "quick",
    "check_data",
    "check",
    "CheckResult",
    "quiet",
    "verbose",
    "report",
    "update",
    "query",
    # Data structures
    "Pipeline",
    "Tokenizer",
    "Scaler",
    "Encoder",
    "Imputer",
    "DriftResult",
    "ShelfResult",
    "InteractResult",
    "EnoughResult",
    "OptimizeResult",
    "Binner",
    "NestedCVResult",
    "CheckReport",
    "LeakReport",
    "Model",
    "SplitResult",
    "CVResult",
    "PreparedData",
    "TuningResult",
    "ValidateResult",
    # Display types
    "Metrics",
    "ProfileResult",
    "Explanation",
    "Leaderboard",
    # Errors
    "MLError",
    "ConfigError",
    "DataError",
    "ModelError",
    "PartitionError",
    "VersionError",
    "WorkflowState",
    "WorkflowStateError",
]
