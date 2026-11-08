# Contributing

1. Create a small, focused branch; describe the problem and measurable outcome.
2. Install the editable package with the dev extra.
3. Add regression tests for bugs and numerical edge cases. Keep synthetic fixtures deterministic.
4. Run `python -m ruff check .`, `python -m ruff format --check .`, and `python -m pytest -q`.
5. Report dataset provenance, split unit, seeds, hardware, and failure cases with experimental changes.

Do not tune on a held-out test set or report synthetic results as real-world performance. Keep large datasets, secrets, and unrelated model binaries outside Git.
