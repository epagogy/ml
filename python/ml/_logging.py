"""Structured debug logging for ml package.

Usage (for library users who want debug output):
    import logging
    logging.getLogger("ml").setLevel(logging.DEBUG)

By default, the ml logger uses a NullHandler (no output).
"""
import logging

logger = logging.getLogger("ml")
logger.addHandler(logging.NullHandler())
