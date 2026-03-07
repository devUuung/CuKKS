# Contributing to CuKKS

Thanks for contributing to CuKKS.

This project uses an issue-first workflow:

1. open or refine an issue
2. maintainers triage and assign labels/milestones
3. contributors open a PR linked to the issue
4. CI validates the PR
5. maintainers review, merge, and include the work in a milestone release

## Before you start

- read `README.md` for installation and project overview
- check existing issues before opening a new one
- use the issue templates in `.github/ISSUE_TEMPLATE/`
- for feature work, prefer discussing scope in an issue before sending a large PR

## Contribution flow

### 1. Open an issue

- use `Bug Report` for defects and regressions
- use `Feature Request` for enhancements or new capabilities
- include a minimal reproduction or concrete proposal when possible

Maintainers own triage and milestone assignment. Outside contributors do not need to create milestones.

### 2. Work on a branch

```bash
git clone https://github.com/devUuung/CuKKS.git
cd CuKKS
git checkout -b your-branch-name
```

Follow existing code patterns and keep changes focused.

### 3. Run local checks before opening a PR

Typical Python checks:

```bash
pytest -q
python -m build
```

If your change touches GPU packaging or release logic, mention what you verified in the PR.

### 4. Open a pull request

- use the PR template in `.github/pull_request_template.md`
- summarize the change clearly
- list concrete testing evidence
- link the issue with `Closes #123` when appropriate
- if maintainers gave the PR a milestone, keep the PR aligned with that release target

### 5. Review and merge

- maintainers review code, release fit, and milestone scope
- merged PRs targeting a closed milestone are included in milestone-based releases

## CI expectations for contributors

PRs and pushes to `main` run `.github/workflows/build-wheels.yml`.

That workflow:

- builds CUDA Docker images used by wheel builds
- builds GPU wheel artifacts for CUDA 11.8, 12.1, 12.4, and 12.8
- checks installed wheel structure across Python 3.10-3.13
- builds the main `cukks` package

CI validates changes but does not publish packages from a normal PR.

For the full automation model, see `docs/ci-cd.md`.

## Release model

CuKKS releases are milestone-based.

- issues and PRs are grouped under a milestone
- maintainers close the milestone when the release scope is ready
- maintainers run the manual milestone release workflow

For maintainer release operations, see `RELEASING.md`.
