"""Centralised logging setup for the rendering pipeline.

Replaces ad-hoc ``print(f'{x=}')`` debug statements scattered across the codebase.
Call :func:`setup_logging` once at the top of any entry script:

    from utils.logging_util import setup_logging
    setup_logging(debug=args.debug)

Module-level loggers should be obtained via ``logging.getLogger(__name__)``.
"""
from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(debug: bool = False, fmt: str | None = None) -> None:
    """Configure the root logger.

    Honours the ``MITSUBA_RENDER_LOG_LEVEL`` env var (DEBUG/INFO/WARNING/...)
    when ``debug`` is False, otherwise forces DEBUG.
    """
    if debug:
        level = logging.DEBUG
    else:
        level_name = os.environ.get("MITSUBA_RENDER_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt or _DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
