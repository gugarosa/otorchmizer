# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Custom exceptions for the Otorchmizer package."""

from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """A generic Error class derived from Exception.

    Notes:
        Logs the error class and message through the package logger.
        Specialized errors derive from this class rather than the similarly named built-in exceptions.

    """

    def __init__(self, cls: str, msg: str) -> None:
        """Initialize a labeled exception and log its diagnostic.

        Args:
            cls: Error category prefixed to the exception message.
            msg: Diagnostic describing the invalid argument or state.

        """

        super().__init__(f"{cls}: {msg}")
        logger.error("`%s`: %s.", cls, str(msg).rstrip("."))


class ArgumentError(Error):
    """Error for wrong number of provided arguments."""

    def __init__(self, error: str) -> None:
        """Initialize an argument-count diagnostic.

        Args:
            error: Message describing the incorrect arguments.

        """

        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Error for classes not being built before use."""

    def __init__(self, error: str) -> None:
        """Initialize an unbuilt-component diagnostic.

        Args:
            error: Message identifying the component that must be built.

        """

        super().__init__("BuildError", error)


class SizeError(Error):
    """Error for mismatched array/tensor sizes."""

    def __init__(self, error: str) -> None:
        """Initialize a shape or size mismatch diagnostic.

        Args:
            error: Message describing the incompatible sizes.

        """

        super().__init__("SizeError", error)


class TypeError(Error):
    """Error for wrong variable types."""

    def __init__(self, error: str) -> None:
        """Initialize an unsupported-type diagnostic.

        Args:
            error: Message identifying the value with the wrong type.

        """

        super().__init__("TypeError", error)


class ValueError(Error):
    """Error for out-of-range values."""

    def __init__(self, error: str) -> None:
        """Initialize an invalid-value diagnostic.

        Args:
            error: Message describing the invalid value or range.

        """

        super().__init__("ValueError", error)
