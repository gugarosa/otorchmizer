# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Standard logging without application-level configuration."""

import logging


def get_logger(logger_name: str) -> logging.Logger:
    """Return a standard logger without configuring handlers or levels.

    Args:
        logger_name: Name identifying the emitting module.

    Returns:
        The application's standard named logger.

    """

    return logging.getLogger(logger_name)
