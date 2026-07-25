# GitHub Migration and CI/CD Design

## Objective

Make the contents of `medical-research-agent/` the repository root of
`tanayanand12/Medical-Research-Agent`, while preserving the repository's
existing implementation and adding reproducible continuous integration and
container delivery.

## Migration strategy

1. Preserve the current remote default-branch commit as both:
   - branch `legacy/v1`;
   - annotated tag `v1.0.0-legacy`.
2. Create a migration branch from the current remote `main`.
3. Replace the tracked project files on that branch with the sanitized
   contents of the local `medical-research-agent/` directory.
4. Open a pull request into `main`; do not rewrite or force-push history.
5. Merge only after CI succeeds.
6. Create tag `v2.0.0` from the merged `main` commit to publish the first
   current-system container image.

The imported folder becomes the repository root. It is not nested inside an
additional `medical-research-agent/` directory.

## Import exclusions

The migration excludes local-only or sensitive files, including:

- `.env` and credentials;
- virtual environments and Python caches;
- logs, checkpoints, SQLite databases, and local runtime state;
- downloaded model files and FAISS indexes;
- IDE state and temporary test artifacts.

Versioned benchmark definitions, manuscript files, audited evaluation results,
and reproducibility metadata remain included unless they contain sensitive or
licensed material.

## Continuous integration

GitHub Actions runs on pull requests and pushes to `main`:

1. check out the repository;
2. install the supported Python version and cached dependencies;
3. compile Python modules;
4. run the deterministic offline validation suite documented by the paper;
5. build the Docker image without publishing it.

Live API and paid-provider evaluations are not CI gates because they require
credentials and mutable external services.

## Continuous delivery

A separate GitHub Actions workflow runs for version tags matching `v*` and can
also be dispatched manually. It:

1. authenticates to GitHub Container Registry using `GITHUB_TOKEN`;
2. builds the production image with Buildx;
3. publishes immutable version tags and a `latest` tag for stable releases;
4. records standard OCI source and revision labels.

The initial release is `ghcr.io/tanayanand12/medical-research-agent:v2.0.0`.
No external cloud deployment is included.

## Container runtime

The image starts the FastAPI application via:

`uvicorn research_agent_api_v2:app --host 0.0.0.0 --port 8000`

Runtime model keys, provider URLs, and optional observability configuration are
injected through environment variables and never baked into the image.

## Failure handling and rollback

- A failed CI run blocks merge.
- A failed image publication leaves `main` intact and can be retried.
- The old implementation remains accessible through `legacy/v1` and
  `v1.0.0-legacy`.
- Rollback of a deployment uses a previously published immutable image tag.

## Acceptance criteria

- Remote `legacy/v1` and `v1.0.0-legacy` point to the pre-migration `main`.
- Remote `main` contains the current project at repository root.
- No secret or local runtime artifact is committed.
- CI tests and the container build succeed.
- Tag `v2.0.0` publishes a pullable GHCR image.
- README documentation identifies the legacy branch and current architecture.
