# Contributing

Changes should preserve the separation between synthetic model development,
locked evaluation, and unlabelled real-data processing.

## Code changes

- Use Python 3.10+ syntax and PEP 8 formatting.
- Add concise docstrings for public functions and comments for non-obvious
  scientific or numerical choices. Avoid diary-style change notes in source
  files; record durable changes in the pull request or release notes.
- Keep physical units in names where ambiguity is possible (`*_hz`, `*_hz_s`,
  `*_mjd`). Preserve signed frequency increments.
- Validate external tables through the canonical schemas rather than accessing
  arbitrary columns in downstream modules.
- Add or update a regression test for every geometry, serialization, policy,
  or metric change.

Run before submitting:

```bash
python -m pip install -r requirements/dev.txt
isort stages tests
black stages tests
ruff check stages tests
bash scripts/check_repository.sh
```

## Scientific changes

A change to a checkpoint, preprocessing definition, candidate union,
association tolerance, evaluation population, threshold, or bootstrap unit is
a methodological change. Document it explicitly and do not overwrite a frozen
result. For a preregistered test, preserve the original output and report the
new run as a sensitivity analysis or a new version.

## Data and artifacts

Do not commit raw filterbanks, private candidate tables, extracted pair arrays,
or ordinary model binaries. Commit compact schemas, provenance, hashes,
machine-readable results, and curated figures. Confirm data-sharing permission
before publishing any candidate-level record.
