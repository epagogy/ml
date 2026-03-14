"""Deprecation infrastructure for ml package.

Usage:
    from ._deprecation import deprecated

    @deprecated("Use ml.new_function() instead.", since="4.1", removal="5.0")
    def old_function():
        pass
"""

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def deprecated(message: str, *, since: str = "", removal: str = "") -> Callable[[F], F]:
    """Mark a function as deprecated.

    Emits DeprecationWarning when the decorated function is called.

    Args:
        message: Deprecation message shown to users.
        since: Version when deprecated (e.g. "4.1").
        removal: Version when it will be removed (e.g. "5.0").

    Returns:
        Decorator that wraps the function with a deprecation warning.

    Example:
        @deprecated("Use ml.new_func() instead.", since="4.1", removal="5.0")
        def old_func():
            pass
    """
    def decorator(func: F) -> F:
        since_str = f" (since {since})" if since else ""
        removal_str = f" Will be removed in {removal}." if removal else ""
        full_message = f"{func.__qualname__} is deprecated{since_str}. {message}{removal_str}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(full_message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = True
        wrapper.__deprecation_message__ = full_message
        return wrapper  # type: ignore[return-value]

    return decorator
