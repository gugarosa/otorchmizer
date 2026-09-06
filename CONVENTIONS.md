# Otorchmizer conventions

Adapted from the cpmux/phitrain conventions discussed for Statys. These rules preserve
Otorchmizer's public APIs and numerical requirements rather than importing another
project's architecture.

## Contracts

- Keep one canonical implementation per public optimizer and update its consumers
  together when an explicitly versioned API change is required. Do not retain
  compatibility aliases or parallel old/new module layouts.
- Population positions have shape `(n_agents, n_variables, n_dimensions)`, and
  scalar fitness has shape `(n_agents,)`. Keep shape, device, and dtype metadata
  consistent with the stored tensors.
- Preserve the best evaluated valid candidate before selection or restart can discard
  it. Keeping a result archive does not require forcing elitist population selection.
- TreeSpace owns expression trees, terminal tensors, and their population phenotypes.
  Transfer the complete owned state rather than casting only phenotype outputs.
- Sample distributions directly on the requested device and dtype. An omitted dtype
  follows the PyTorch default rather than imposing a hidden float32 conversion.
- Unevaluated fitness uses positive infinity. `FLOAT_MAX` remains the public finite
  float32 limit, not a universal initialization sentinel.
- Evaluate each algorithm against its documented equations and invariants. A smoke
  test on sphere, an earlier approval, or a lower line count does not prove correctness.
- Preserve meaningful variation and shared responsibilities. Do not replace
  algorithm-specific behavior with generic random updates or a new framework.
- Hardware, objective purity, dtype, and PyTorch operator support constrain GPU,
  autocast, compilation, and CUDA Graph usage. Report untested capabilities explicitly.

## Code style

- Preserve Python 3.10+ support while using modern unions and builtin generics.
  ABCs come from `collections.abc`; import typing-specific constructs such as `Any`,
  `Literal`, and `TYPE_CHECKING` from `typing`. (R2)
- Imports are absolute and module-level, grouped stdlib, third-party, then local.
  Type-only imports may use a module-level `TYPE_CHECKING` guard to avoid cycles.
- Public APIs use Google-style docstrings with a single-sentence summary and concise,
  one-line `Args:`, `Returns:`, and `Raises:` entries. Put constructor arguments on
  `__init__`, and preserve useful detail in `Notes:` and examples. Do not add semicolons
  or `defaults to X` tails to entries. (R3, R13)
- Multiline docstrings have one blank line before the closing triple quote and one
  after it before code. Private helpers and framework-only callback overrides carry
  no docstring. User-facing optimizer methods remain documented.
- Data classes document each field in an `Attributes:` section, one line per field.
- Use specific builtin exceptions and explicit validation, not runtime assertions.
  Raised messages name a backticked offender and end with a period. Do not wrap
  builtin errors in parallel project-specific exception classes or log while raising.
  Dependency exceptions are not blindly translated. (R1)
- Reuse the standard named logger returned by `get_logger(__name__)` from
  `otorchmizer.utils.logging`. Do not configure handlers, change Python's logger
  class, create log files, or narrate routine construction from library modules.
  Warning/error diagnostics name a backticked offender and end with a period.
  Examples and command-line tools may print intentional results. (R14)
- Comments explain why, not what. Prefer none or one line, cap comment blocks at three
  lines, avoid banner separators and trailing periods, and preserve attribution.
  Copyright/license notices retain their prescribed punctuation. (R8)
- Separate real phases in function bodies of at least 12 lines with a blank line.
  Do not mechanically space every statement. (R11)
- Inline first. Extract a helper, constant, or parameter when another real use
  establishes shared responsibility. Do not remove public APIs based on local call counts. (R16)
- Use double-quoted string literals and a 120-column limit. Keep the existing Ruff
  tooling rather than adding a competing formatter or broad lint suppressions. (R9)
- Tests use clear behavior names, ordinary assertions without custom failure-message
  strings, and independently justified expectations. Do not weaken assertions to pass.

Every project Python file begins with:

```python
# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.
```

## Delivery

`pyproject.toml` owns dependencies, extras, package discovery, and tool configuration.
The source version and package metadata must agree. CPU CI uses uv's native PyTorch
backend selection without forcing CPU-only wheels on library consumers.

Build and inspect wheels and source distributions when changing packaging. Wheels
must contain the library, not the project's `tests` package.

Do not merge PRs or publish releases without authorization. Publishing runs the
interpreter, quality, documentation, and installed-wheel checks first, then requires
a matching `v<version>` tag and configured PyPI authentication.
