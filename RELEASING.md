# Releasing CuKKS

This repository uses an issue-driven release flow:

1. create or refine an issue
2. merge a PR tied to the target milestone
3. close the milestone
4. run the milestone release workflow

## Required GitHub setup

- GitHub Issues must remain enabled for the repository.
- PyPI Trusted Publishing must be configured for:
  - `cukks`
  - `cukks-cu118`
  - `cukks-cu121`
  - `cukks-cu124`
  - `cukks-cu128`
- The `pypi` environment should exist if you want approval gates before publish jobs run.

## Normal development flow

### 1. Track work with issues and milestones

- create an issue for the work
- assign the issue to the target milestone
- open a PR for the issue
- assign the PR to the same milestone

External contributors should follow `CONTRIBUTING.md` for the issue and PR workflow. Milestone planning and release execution remain maintainer responsibilities.

### 2. Merge PRs into `main`

- all PR validation happens in `.github/workflows/build-wheels.yml`
- this workflow builds docker images, GPU wheels, tests wheel structure, and builds the main package
- this workflow does not publish packages

### 3. Prepare a release

Before cutting a release:

- confirm the target milestone is closed
- confirm all milestone PRs are merged into `main`
- use the milestone title itself as the release version (for example, `0.1.4`)
- you do not need to manually edit package version files before running the release workflow

## Cut a release from a milestone

Run `.github/workflows/release-milestone.yml` with:

- `milestone_name`: the closed milestone title, used directly as the release version
- `target_ref`: normally `main`
- `draft`: keep the GitHub release as draft after build/publish completes
- `publish_pypi`: whether to publish packages to PyPI

The workflow will:

1. validate milestone state and package versions
2. verify that milestone PRs are merged into `main`
3. sync tracked version files to the milestone title and create a release prep commit on `main` when needed
4. generate release notes from merged PRs and closed issues in the milestone
5. create an annotated git tag from the release prep commit
6. create a draft GitHub release
7. call `.github/workflows/publish-packages.yml` to build and publish packages from that tag
8. optionally publish the GitHub release if `draft` is false

See `docs/ci-cd.md` for the full automation layout and how this workflow connects to CI and package publishing.

## Recovery and manual publish

Use `.github/workflows/publish-packages.yml` only for controlled recovery.

- it accepts an explicit `ref`
- it derives the release version from the checked-out source
- it fails if the target version already exists on PyPI

## Safety rules

- do not publish from `release.published`
- do not publish by creating a GitHub release manually in the UI
- do not reuse old tags for new package contents
- do not use a milestone title that is not a valid package version string
- do not release a milestone while it still contains unmerged PRs
