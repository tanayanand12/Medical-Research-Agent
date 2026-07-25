# GitHub Migration and CI/CD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current GitHub repository root with the present medical research agent, preserve the old implementation, and publish tested versioned containers to GHCR.

**Architecture:** Preserve remote `main` through an immutable tag and frozen branch, then perform the replacement on a migration branch and merge through a CI-gated pull request. GitHub Actions separates offline Python validation from tag-triggered container publication.

**Tech Stack:** Git, GitHub CLI, GitHub Actions, Python 3.12, pytest, Docker Buildx, FastAPI/Uvicorn, GitHub Container Registry.

## Global Constraints

- Remote repository: `https://github.com/tanayanand12/Medical-Research-Agent.git`.
- Source directory: `F:\GitHub uploads\Medical-Research-Agent-Paper\medical-research-agent`.
- Migration clone: `F:\GitHub uploads\Medical-Research-Agent-migration`.
- Preserve pre-migration `main` as `legacy/v1` and annotated tag `v1.0.0-legacy`.
- Never force-push or rewrite remote history.
- The source directory contents become repository-root contents.
- Do not commit credentials, `.env`, virtual environments, logs, indexes, databases, caches, or local runtime state.
- Publish `ghcr.io/tanayanand12/medical-research-agent` only from `v*` tags or explicit manual workflow dispatch.
- Initial current-system release tag: `v2.0.0`.

---

### Task 1: Preserve the legacy repository state

**Files:**
- No repository files changed.

**Interfaces:**
- Consumes: current remote `main`.
- Produces: remote branch `legacy/v1` and tag `v1.0.0-legacy` at the same commit.

- [ ] **Step 1: Verify GitHub authentication and remote state**

Run:

```powershell
gh auth status
gh repo view tanayanand12/Medical-Research-Agent --json defaultBranchRef,url,isPrivate
gh api repos/tanayanand12/Medical-Research-Agent/git/ref/heads/main
```

Expected: authenticated account can write to the repository and the default branch is `main`.

- [ ] **Step 2: Clone the repository into an isolated migration directory**

Run:

```powershell
git clone https://github.com/tanayanand12/Medical-Research-Agent.git "F:\GitHub uploads\Medical-Research-Agent-migration"
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" status --short
```

Expected: clone succeeds and status output is empty.

- [ ] **Step 3: Confirm preservation names do not already point elsewhere**

Run:

```powershell
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" ls-remote --heads origin legacy/v1
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" ls-remote --tags origin v1.0.0-legacy
```

Expected: both outputs are empty. If either exists, compare its object commit to `origin/main`; stop rather than overwrite a different object.

- [ ] **Step 4: Create and push the preservation branch and tag**

Run:

```powershell
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" branch legacy/v1 origin/main
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" tag -a v1.0.0-legacy origin/main -m "Preserve legacy implementation before precision evidence migration"
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" push origin legacy/v1
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" push origin v1.0.0-legacy
```

Expected: both refs are created without force.

- [ ] **Step 5: Verify preservation refs**

Run:

```powershell
$main = git -C "F:\GitHub uploads\Medical-Research-Agent-migration" rev-parse origin/main
$legacy = git -C "F:\GitHub uploads\Medical-Research-Agent-migration" rev-parse origin/legacy/v1
$legacyTag = git -C "F:\GitHub uploads\Medical-Research-Agent-migration" rev-list -n 1 v1.0.0-legacy
if ($main -ne $legacy -or $main -ne $legacyTag) { throw "Legacy preservation refs do not match pre-migration main" }
```

Expected: command exits successfully.

### Task 2: Assemble the sanitized migration branch

**Files:**
- Replace: all tracked project files at repository root.
- Preserve through import: `docs/superpowers/specs/2026-07-25-github-migration-ci-cd-design.md`
- Preserve through import: `docs/superpowers/plans/2026-07-25-github-migration-ci-cd.md`

**Interfaces:**
- Consumes: local source directory and remote `main`.
- Produces: branch `migration/precision-evidence-v2` with sanitized current-system files.

- [ ] **Step 1: Create the migration branch from fresh remote main**

Run:

```powershell
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" fetch origin
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" switch -c migration/precision-evidence-v2 origin/main
```

Expected: current branch is `migration/precision-evidence-v2`.

- [ ] **Step 2: Remove legacy tracked files on the migration branch**

Run:

```powershell
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" rm -r -- .
```

Expected: legacy files are staged for deletion while `.git` remains intact.

- [ ] **Step 3: Copy current source while excluding local artifacts**

Run:

```powershell
robocopy "F:\GitHub uploads\Medical-Research-Agent-Paper\medical-research-agent" "F:\GitHub uploads\Medical-Research-Agent-migration" /E /XD .git venv .venv __pycache__ .pytest_cache .mypy_cache .ruff_cache logs data\indexes /XF .env *.pyc *.pyo *.index *.faiss *.db
if ($LASTEXITCODE -gt 7) { throw "robocopy failed with exit code $LASTEXITCODE" }
```

Expected: robocopy exit code 0–7 and no nested `medical-research-agent` directory.

- [ ] **Step 4: Audit copied paths for secrets and excluded artifacts**

Run:

```powershell
Get-ChildItem "F:\GitHub uploads\Medical-Research-Agent-migration" -Force -Recurse |
  Where-Object {
    $_.FullName -notlike "*\.git\*" -and
    ($_.Name -eq ".env" -or $_.Name -match "\.(db|index|faiss|pyc|pyo)$" -or $_.Name -in @("venv", ".venv", "__pycache__", "logs"))
  } |
  Select-Object -ExpandProperty FullName
```

Expected: no output.

- [ ] **Step 5: Stage imported source and selected reproducibility results**

Run:

```powershell
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" add -A -- . ':!results'
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" add -f -- results/paper_eval_summary.json results/retrieval_grounding_eval.json results/routing_holdout_eval_final.json results/benchmark_all_agents_20260724_214522_rejudged_audited.json
git -C "F:\GitHub uploads\Medical-Research-Agent-migration" status --short
```

Expected: project source, paper, design, plan, and four canonical result artifacts are staged; no excluded artifact appears.

### Task 3: Add container packaging

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: `requirements.txt`, `research_agent_api_v2:app`.
- Produces: OCI image exposing port 8000 and running as a non-root user.

- [ ] **Step 1: Add a failing packaging assertion**

Run before creating the files:

```powershell
$required = @("Dockerfile", ".dockerignore")
$missing = $required | Where-Object { -not (Test-Path "F:\GitHub uploads\Medical-Research-Agent-migration\$_") }
if ($missing.Count -eq 0) { throw "Expected packaging files to be absent before implementation" }
```

Expected: command exits successfully because at least one packaging file is absent.

- [ ] **Step 2: Create `Dockerfile`**

Create:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/logs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "research_agent_api_v2:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `.dockerignore`**

Create:

```text
.git
.github
.env
.venv
venv
__pycache__
*.py[cod]
.pytest_cache
.mypy_cache
.ruff_cache
logs
data/indexes
*.index
*.faiss
*.db
results
agentic-pipeline-clinical
```

- [ ] **Step 4: Build the image**

Run:

```powershell
docker build --tag medical-research-agent:ci "F:\GitHub uploads\Medical-Research-Agent-migration"
```

Expected: Docker exits 0 and creates `medical-research-agent:ci`.

- [ ] **Step 5: Verify container startup and health**

Run:

```powershell
docker run --detach --rm --name mra-ci -p 18000:8000 medical-research-agent:ci
Start-Sleep -Seconds 10
$health = Invoke-RestMethod http://127.0.0.1:18000/health
if (-not $health) { throw "Health endpoint returned no response" }
docker stop mra-ci
```

Expected: `/health` returns JSON and the container stops cleanly.

### Task 4: Add GitHub Actions CI and GHCR delivery

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release-container.yml`

**Interfaces:**
- Consumes: repository source, `Dockerfile`, `requirements.txt`.
- Produces: required test/build checks and tag-triggered GHCR images.

- [ ] **Step 1: Verify workflows are absent**

Run:

```powershell
if (Test-Path "F:\GitHub uploads\Medical-Research-Agent-migration\.github\workflows\ci.yml") { throw "ci.yml unexpectedly exists" }
if (Test-Path "F:\GitHub uploads\Medical-Research-Agent-migration\.github\workflows\release-container.yml") { throw "release-container.yml unexpectedly exists" }
```

Expected: command exits successfully.

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

Create:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -r requirements.txt
      - run: python -m compileall -q .
      - name: Run deterministic offline suite
        run: >-
          python -m pytest
          test_phase4_integration.py
          test_reliability.py
          eval/test_metrics.py
          eval/test_finalize_results.py
          eval/test_retrieval_metrics.py
          eval/test_routing_eval.py
          test_skill_router_precision.py
          -q

  container:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          tags: medical-research-agent:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 3: Create `.github/workflows/release-container.yml`**

Create:

```yaml
name: Release container

on:
  push:
    tags: ["v*"]
  workflow_dispatch:
    inputs:
      image_tag:
        description: Container tag to publish
        required: true
        default: manual

permissions:
  contents: read
  packages: write

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/tanayanand12/medical-research-agent
          tags: |
            type=ref,event=tag
            type=raw,value=${{ inputs.image_tag }},enable=${{ github.event_name == 'workflow_dispatch' }}
            type=raw,value=latest,enable=${{ startsWith(github.ref, 'refs/tags/v') }}
          labels: |
            org.opencontainers.image.source=https://github.com/tanayanand12/Medical-Research-Agent
            org.opencontainers.image.revision=${{ github.sha }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 4: Validate workflow syntax and Docker references**

Run:

```powershell
python -c "import yaml, pathlib; [yaml.safe_load(p.read_text(encoding='utf-8')) for p in pathlib.Path(r'F:\GitHub uploads\Medical-Research-Agent-migration\.github\workflows').glob('*.yml')]"
Select-String -Path "F:\GitHub uploads\Medical-Research-Agent-migration\.github\workflows\*.yml" -Pattern "ghcr.io/tanayanand12/medical-research-agent"
```

Expected: YAML parsing exits 0 and the release workflow contains the lowercase GHCR image path.

### Task 5: Document migration and container usage

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: workflow and container behavior from Tasks 3–4.
- Produces: public migration, CI, image, and runtime documentation.

- [ ] **Step 1: Add legacy migration notice after the README introduction**

Add:

```markdown
> **Version history:** The pre-LangGraph implementation is frozen on
> [`legacy/v1`](../../tree/legacy/v1) and tagged as `v1.0.0-legacy`.
> The `main` branch contains the current precision evidence orchestration
> system evaluated in the accompanying paper.
```

- [ ] **Step 2: Add container usage**

Add after the local run instructions:

```markdown
### Container

Versioned images are published to GitHub Container Registry:

```bash
docker pull ghcr.io/tanayanand12/medical-research-agent:v2.0.0
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/tanayanand12/medical-research-agent:v2.0.0
```

Provider credentials and model configuration are supplied at runtime and are
never included in the image.
```

- [ ] **Step 3: Add CI/CD documentation**

Add:

```markdown
## Continuous integration and delivery

Pull requests and pushes to `main` run the deterministic offline test suite and
build the container. Tags matching `v*` publish versioned and `latest` images
to `ghcr.io/tanayanand12/medical-research-agent`.

Live-provider evaluations are intentionally excluded from CI because they need
credentials and query mutable external evidence services.
```

- [ ] **Step 4: Check documentation links and repository-root paths**

Run:

```powershell
Select-String -Path "F:\GitHub uploads\Medical-Research-Agent-migration\README.md" -Pattern "legacy/v1","ghcr.io/tanayanand12/medical-research-agent","research_agent_api_v2:app"
```

Expected: all three concepts are present.

### Task 6: Validate, commit, and open the migration PR

**Files:**
- Commit all staged migration files, packaging, workflows, and documentation.

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: reviewed migration pull request into `main`.

- [ ] **Step 1: Run the complete offline validation suite**

Run:

```powershell
Set-Location "F:\GitHub uploads\Medical-Research-Agent-migration"
python -m pytest test_phase4_integration.py test_reliability.py eval/test_metrics.py eval/test_finalize_results.py eval/test_retrieval_metrics.py eval/test_routing_eval.py test_skill_router_precision.py -q
```

Expected: 46 tests pass with no failures.

- [ ] **Step 2: Inspect staged changes and secret indicators**

Run:

```powershell
git status --short
git diff --cached --stat
git diff --cached -- . ':!results/*.json' | Select-String -Pattern "sk-[A-Za-z0-9]","BEGIN PRIVATE KEY","api[_-]?key\s*[:=]\s*[^\s$]"
```

Expected: no credential-like value appears; only intended files are staged.

- [ ] **Step 3: Commit the migration**

Run:

```powershell
git add Dockerfile .dockerignore .github/workflows/ci.yml .github/workflows/release-container.yml README.md docs/superpowers/specs/2026-07-25-github-migration-ci-cd-design.md docs/superpowers/plans/2026-07-25-github-migration-ci-cd.md
git commit -m "feat: migrate precision evidence platform with CI/CD"
```

Expected: commit succeeds without bypassing hooks.

- [ ] **Step 4: Push the migration branch**

Run:

```powershell
git branch --show-current
git push -u origin HEAD
```

Expected: current branch is `migration/precision-evidence-v2` and push succeeds.

- [ ] **Step 5: Create the pull request**

Run:

```powershell
gh pr create --repo tanayanand12/Medical-Research-Agent --base main --head migration/precision-evidence-v2 --title "Migrate precision evidence platform and add CI/CD" --body "## Summary`n- preserve the previous implementation on legacy/v1 and v1.0.0-legacy`n- replace the repository root with the current LangGraph medical research agent and paper artifacts`n- add offline Python CI, container build verification, and GHCR release automation`n`n## Validation`n- deterministic offline pytest suite`n- production Docker image build and health check`n- workflow YAML validation`n`n## Release`nAfter merge, v2.0.0 publishes ghcr.io/tanayanand12/medical-research-agent:v2.0.0."
```

Expected: command returns the new pull request URL.

### Task 7: Merge and publish v2.0.0

**Files:**
- No additional source files changed.

**Interfaces:**
- Consumes: passing migration PR.
- Produces: migrated `main`, release tag `v2.0.0`, and GHCR image.

- [ ] **Step 1: Watch CI to completion**

Run:

```powershell
gh pr checks --repo tanayanand12/Medical-Research-Agent --watch
```

Expected: Python tests and container build both pass.

- [ ] **Step 2: Merge without rewriting history**

Run:

```powershell
gh pr merge --repo tanayanand12/Medical-Research-Agent --merge --delete-branch
```

Expected: pull request merges into `main`.

- [ ] **Step 3: Tag the merged main commit**

Run:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git tag -a v2.0.0 -m "Release precision evidence orchestration platform v2.0.0"
git push origin v2.0.0
```

Expected: tag push triggers `Release container`.

- [ ] **Step 4: Verify release workflow and package**

Run:

```powershell
gh run list --repo tanayanand12/Medical-Research-Agent --workflow "Release container" --limit 1
gh run watch --repo tanayanand12/Medical-Research-Agent
gh api /users/tanayanand12/packages/container/medical-research-agent/versions --jq '.[0].metadata.container.tags'
```

Expected: workflow succeeds and returned tags include `v2.0.0` and `latest`.

- [ ] **Step 5: Verify final branch and legacy invariants**

Run:

```powershell
$main = git rev-parse origin/main
$legacy = git rev-parse origin/legacy/v1
$legacyTag = git rev-list -n 1 v1.0.0-legacy
if ($main -eq $legacy) { throw "Main was not migrated" }
if ($legacy -ne $legacyTag) { throw "Legacy branch and tag diverged" }
gh repo view tanayanand12/Medical-Research-Agent --json defaultBranchRef,url
```

Expected: default branch is `main`, migrated `main` differs from legacy, and the legacy branch/tag remain identical.
