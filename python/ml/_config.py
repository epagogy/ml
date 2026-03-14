"""Global mlw configuration."""
from __future__ import annotations

_CONFIG: dict = {"n_jobs": 1, "verbose": 0, "guards": "strict"}
_EXPLICITLY_SET: set = set()


def config(**kwargs) -> dict:
    """Get or set global mlw configuration.

    ml.config()                    # Returns current config dict
    ml.config(n_jobs=2)            # Limit parallelism
    ml.config(n_jobs=1)            # Force single-thread (8GB laptop safety)
    ml.config(verbose=1)           # Show progress
    ml.config(guards="strict")     # Partition guards: "strict" (default), "warn", "off"
    """
    from ._types import ConfigError

    if not kwargs:
        return dict(_CONFIG)
    for k, v in kwargs.items():
        if k not in _CONFIG:
            raise ConfigError(
                f"Unknown config key '{k}'. Valid keys: {sorted(_CONFIG.keys())}"
            )
        _CONFIG[k] = v
        _EXPLICITLY_SET.add(k)
    return dict(_CONFIG)
