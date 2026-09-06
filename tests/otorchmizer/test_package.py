# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from importlib.metadata import distribution, version

import otorchmizer


def test_runtime_version_matches_package_metadata():
    assert otorchmizer.__version__ == version("otorchmizer")


def test_distribution_exposes_only_the_library_package():
    packages = distribution("otorchmizer").read_text("top_level.txt")

    assert packages is not None
    assert packages.splitlines() == ["otorchmizer"]
