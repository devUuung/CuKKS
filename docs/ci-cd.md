# CI/CD Overview

This document explains how CuKKS validates changes, publishes packages, and creates releases.

## Workflow map

### 1. CI validation: `.github/workflows/build-wheels.yml`

Triggered on:

- pull requests
- pushes to `main`
- manual `workflow_dispatch`

What it does:

- builds CUDA Docker images for 11.8, 12.1, 12.4, and 12.8
- builds GPU wheels for each CUDA package
- validates wheel contents across Python 3.10-3.13
- builds the main `cukks` package

What it does not do:

- it does not publish to PyPI
- it does not create releases

### 2. Package publishing: `.github/workflows/publish-packages.yml`

Triggered by:

- `workflow_call` from the release workflow
- manual `workflow_dispatch` for controlled recovery only

Safety checks:

- validates package versions against the requested release version
- checks whether the target version already exists on PyPI
- uses GitHub OIDC trusted publishing with `environment: pypi`

Packages published:

- `cukks`
- `cukks-cu118`
- `cukks-cu121`
- `cukks-cu124`
- `cukks-cu128`

### 3. Milestone release orchestration: `.github/workflows/release-milestone.yml`

Triggered by:

- manual `workflow_dispatch`

Inputs:

- `milestone_number`
- `version`
- `target_ref`
- `draft`
- `publish_pypi`

What it does:

- validates that the milestone exists and is closed
- checks that milestone PRs are merged into `main`
- verifies package versions match the requested release version
- generates release notes from merged PRs and closed issues in the milestone
- creates an annotated tag
- creates a GitHub release
- calls `publish-packages.yml`

## Contributor view

External contributors usually interact with CI only through PRs.

- open an issue first
- submit a PR linked to the issue
- check the `Build Wheels` workflow status on the PR
- maintainers decide milestone placement and release timing

## Maintainer view

Maintainers own release preparation.

Before releasing:

- make sure the milestone is closed
- make sure all milestone PRs are merged
- bump all package versions consistently
- verify PyPI Trusted Publishing still matches:
  - owner: `devUuung`
  - repo: `CuKKS`
  - workflow: `.github/workflows/publish-packages.yml`
  - environment: `pypi`

## Operational notes

- normal GitHub release creation in the UI is not the publish path
- historical release restoration should not be used as a publish mechanism
- release publication is intentionally separated from PR/main CI
