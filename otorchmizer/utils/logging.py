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
        """Logs the message only to the logging file (suppresses console)."""

        if self.handlers:
            self.handlers[0].setLevel(logging.CRITICAL)
            self.info(msg, *args, **kwargs)
            self.handlers[0].setLevel(LOG_LEVEL)


def get_console_handler() -> StreamHandler:
    """Gets a console handler for logging to stdout."""

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Gets a timed rotating file handler for logging to files."""

    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Gets a named logger instance.

    Args:
        logger_name: The name of the logger.

    Returns:
        Logger instance.
    """

    logging.setLoggerClass(Logger)

    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
        logger.propagate = False

    return logger
