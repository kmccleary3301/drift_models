# PyPI Trusted Publishing Setup

This repository publishes with OpenID Connect (OIDC) via GitHub Actions:

- Production publish workflow: `.github/workflows/release.yml`
- TestPyPI publish workflow: `.github/workflows/release-testpypi.yml`

## One-time configuration on package indexes

Configure both publishers in package index settings:

- **Project**: `drift-models`
- **Owner**: `kmccleary3301`
- **Repository**: `drift_models`
- **Workflow filename (PyPI)**: `release.yml`
- **Workflow filename (TestPyPI)**: `release-testpypi.yml`
- **Environment (PyPI)**: _leave empty_ (or use `pypi` if you add it to workflow)
- **Environment (TestPyPI)**: `testpypi`

## Release flow

1. Ensure `main` CI is green and `pyproject.toml` version is finalized.
2. Run TestPyPI publish:
   - GitHub Actions → `Release (TestPyPI)` → `Run workflow`
3. Validate install from TestPyPI:
   - `pip install --index-url https://test.pypi.org/simple/ drift-models==<version>`
4. Tag production release:
   - `git tag v<version> && git push origin v<version>`
5. Confirm `Release` workflow uploads artifacts + publishes to PyPI.

## Safety notes

- Keep `id-token: write` on publish jobs.
- Do not add token-based upload fallbacks unless absolutely necessary.
- If a bad release is published, yank on PyPI immediately and cut a patch release.
