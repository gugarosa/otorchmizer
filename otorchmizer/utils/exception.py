"""Custom exceptions for the Otorchmizer package."""

from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """A generic Error class derived from Exception.

    Logs the error class and message to the logger.
    """

    def __init__(self, cls: str, msg: str) -> None:
        super().__init__(f"{cls}: {msg}")
        logger.error("%s: %s.", cls, msg)


class ArgumentError(Error):
    """Error for wrong number of provided arguments."""

    def __init__(self, error: str) -> None:
        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Error for classes not being built before use."""

    def __init__(self, error: str) -> None:
        super().__init__("BuildError", error)


class SizeError(Error):
    """Error for mismatched array/tensor sizes."""

    def __init__(self, error: str) -> None:
        super().__init__("SizeError", error)


class TypeError(Error):
    """Error for wrong variable types."""

    def __init__(self, error: str) -> None:
        super().__init__("TypeError", error)


class ValueError(Error):
    """Error for out-of-range values."""

    def __init__(self, error: str) -> None:
        super().__init__("ValueError", error)
