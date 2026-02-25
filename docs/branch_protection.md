# Branch Protection Policy (`main`)

Required status checks before merge:

- `lint`
- `tests (3.10)`
- `tests (3.12)`
- `build`
- `runtime-preflight (ubuntu-latest, 3.12)`
- `runtime-preflight (macos-latest, 3.12)`
- `runtime-preflight (windows-latest, 3.12)`
- `runtime-preflight-summary`

Non-blocking informational checks:

- `runtime-preflight-experimental (mps)`
- `runtime-preflight-experimental (rocm)`

Recommended branch protection toggles:

- Require pull request before merging
- Require approvals (>=1)
- Dismiss stale approvals on new commits
- Require status checks to pass before merging
- Require conversation resolution before merging
- Block force pushes and branch deletions on `main`
