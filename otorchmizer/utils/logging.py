# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Logging infrastructure for the Otorchmizer package."""

import logging
import sys
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "otorchmizer.log"
LOG_LEVEL = logging.DEBUG


class Logger(logging.Logger):
    """A customized Logger that supports file-only logging."""

    def to_file(self, msg: str, *args, **kwargs) -> None:
        """Log an info message while suppressing the first handler.

        Args:
            msg: Message or format string passed to the logger.
            *args: Positional message-formatting arguments.
            **kwargs: Keyword arguments forwarded to the info call.

        Notes:
            Package-created loggers attach the console handler first.
            Its level is temporarily set to CRITICAL and then reset to LOG_LEVEL.
            No message is emitted when the logger has no handlers.

        """

        if self.handlers:
            self.handlers[0].setLevel(logging.CRITICAL)
            self.info(msg, *args, **kwargs)
            self.handlers[0].setLevel(LOG_LEVEL)


def get_console_handler() -> StreamHandler:
    """Create a console handler for standard output.

    Returns:
        Stream handler using the package formatter.

    """

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Create a delayed file handler with midnight rotation.

    Returns:
        UTF-8 file handler using the package log path and formatter.

    """

    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight", encoding="utf-8")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Gets a named logger instance.

    Args:
        logger_name: The name of the logger.

    Returns:
        Named logger, with package handlers configured if it has no handlers.

    Notes:
        New loggers use the package's Logger subclass.
        Existing named loggers retain their class and any preconfigured handlers.

    """

    logging.setLoggerClass(Logger)

    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
        logger.propagate = False

    return logger
