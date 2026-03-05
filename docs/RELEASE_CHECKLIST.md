# Release Checklist

## Versioning policy (SemVer)

- `MAJOR`: breaking API/CLI/docs contract changes.
- `MINOR`: backward-compatible feature additions and operational improvements.
- `PATCH`: bugfixes and documentation corrections without contract breaks.
- Pre-1.0 policy: still follow SemVer shape (`0.x.y`), but treat `0.MINOR` as potentially significant for research workflows.

## Preconditions

- [ ] CI checks are green on `main`
- [ ] `CHANGELOG.md` updated
- [ ] Claim-boundary docs audited (`docs/faithfulness_status.md`)
- [ ] Version updated in `pyproject.toml`
- [ ] Branch protection checks are configured per `docs/branch_protection.md`
- [ ] Deprecation matrix review completed:
  - [ ] `docs/deprecation_matrix.md` entries were audited
  - [ ] Any entry with `Remove no earlier than <= vX.Y.Z` was removed or explicitly deferred with rationale
- [ ] Table-8 config contract audit passes in CI:
  - [ ] CI artifact `faithful-toggle-audit-py*` is present
  - [ ] `faithful_toggle_audit.md` reports all configured rows as pass with no mismatches
- [ ] Script surface registry audit passes:
  - [ ] `python scripts/check_script_surface_registry.py`
- [ ] Packaging build passes:
  - [ ] `python -m build`
  - [ ] `twine check dist/*`

## Smoke validations

- [ ] Linux CPU quickstart passes
- [ ] Linux CUDA smoke run passes (if GPU runner available)
- [ ] Eval smoke path passes
- [ ] Install instructions validated from clean env

## Publish

- [ ] Create release tag `vX.Y.Z`
- [ ] Create GitHub Release notes
- [ ] Publish to TestPyPI (recommended preflight):
  - [ ] Run `.github/workflows/release-testpypi.yml` (Trusted Publishing)
  - [ ] Validate package install from TestPyPI
- [ ] Publish to PyPI
  - [ ] GitHub Trusted Publishing is configured for this repository
  - [ ] `.github/workflows/release.yml` has `id-token: write` and no API token fallback path
  - [ ] Trusted publisher settings match `docs/pypi_trusted_publishing.md`

## Post-release

- [ ] Update docs if publish behavior diverged from plan
- [ ] Update deprecation targets if no removals occurred this cycle
- [ ] Publish launch posts (Reddit + X)
- [ ] Start KPI tracking cadence (`docs_tmp/socials/PUBLICITY_KPI_DASHBOARD_V1.md`)

## Rollback and hotfix

- [ ] If publish is broken, yank affected PyPI release immediately.
- [ ] Cut hotfix branch from release tag.
- [ ] Apply minimal fix, run full CI, and publish `PATCH` increment.
- [ ] Add explicit hotfix notes in `CHANGELOG.md` and GitHub release notes.
