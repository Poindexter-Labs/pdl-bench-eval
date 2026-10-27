# PDL-Bench evaluation environments

Downloadable Docker evaluation packages are provided alongside the datasets.
Build and run them on your own machine with your own provider credentials; no
Poindexter account, private repository, Firebase, or hosted demo is required.

| Benchmark | Tasks | Package and instructions | Execution |
| --- | ---: | --- | --- |
| STEM | 44 | [STEM](stem/README.md) | Native provider tools, per-model settings |
| Business Operations | 10 | [Business Operations](business-operations/README.md) | Native tools or E2B, per model |
| Life Administration | 8 | [Life Administration](life-administration/README.md) | Native tools or E2B, per model |
| GDPVal Localisation | 24 | [GDPVal](gdpval/README.md) | Stirrup + E2B + Brave |
| Software Delivery | 20 | [Software Delivery](software-delivery/README.md) | Reference agent loop in Docker sandboxes; hidden-test and judge-panel grading |
| Cyber Operations | 10 | [Cyber Operations](cyber-operations/README.md) | Reference agent loop in Docker sandboxes; hidden-test grading |

Each package has `START_HERE.md`, a Dockerfile, pinned dependencies, source code,
model presets, `.env.example`, local tasks/inputs, and checksums. The four
single-turn packages share a common runner with benchmark-specific
configurations; the two repository-task packages share the reference agent
loop and the reference evaluator (`pdl-bench-eval`). Read `DATASET-NOTES.md`
for the grading protocol: in particular, the Business Operations runner retains
the corrected PDS038 rubric, which differs from the older published parquet,
and the repository-task packages carry the two operator overrides the
published report used (the 2M-token run profile and judge policy v0.3).

Example: download and validate GDPVal without making any API calls:

```sh
python -m pip install -U huggingface_hub
hf download Poindexter-Labs/PDL-Bench --repo-type dataset \
  --include "environments/gdpval/**" --local-dir pdl-bench
cd pdl-bench/environments/gdpval
docker build -t poindexter/mini-bench:gdpval-corrected .
docker run --rm --network none -v "$PWD/data:/data:ro" \
  poindexter/mini-bench:gdpval-corrected plan
```

Add `--revision COMMIT_SHA` to pin downloads to an environment release. For real
runs, copy `.env.example` to `.env`, set your keys, and follow the package README.
E2B configurations require a template in your own E2B account; the build command
is included. APIs, E2B, and Brave can incur charges. Runtime keys are not included
or baked into the image. Docker builds and offline validation start no benchmark.

The repository-task packages drive your Docker daemon through its socket (one
sandbox per episode, one evaluator container per grading step) instead of a
provider sandbox; an attempt there is a full agent episode with a 2,000,000-token
budget, so read the cost section of their README before a full run. They also
ship a free offline calibration: the reference patch scores 1.0 and the empty
submission 0.0 on every task.

The Software Delivery/Cyber Operations Dockerfiles below are retained unchanged.

---

# Software Delivery and Cyber Operations environments

Two images cover all 30 repository tasks. Every task fixture is **standard-library
only** — `pytest` is the sole installed dependency, and the three package-looking
imports (`evalscore`, `relaygrid`, `sealock`) are vendored inside their own
`repository.zip`.

| Image | Tasks | Used by | Graded by |
| --- | --- | --- | --- |
| `pdl-bench/swe:1.0.0` | 20 | `software_delivery` (executable), `cyber_operations` | held-out pytest suite |
| `pdl-bench/sweqna:1.0.0` | 10 | `software_delivery` (`pdl_qna_*`, `chronicle_*`) | rubric over `answers/answer.md` |

Each task's `manifest.yaml` names its image in `environment.prebuilt_image` and
points at the Dockerfile that builds it in `environment.image_source`.

## Build

```bash
docker build -t pdl-bench/swe:1.0.0     environments/swe
docker build -t pdl-bench/sweqna:1.0.0  environments/sweqna
```

These task environments are distributed here as buildable Dockerfiles. The base
layer is pinned by digest, so a build today and a build next year produce the same
interpreter.

## Running a task

```bash
unzip tasks/<task_id>/repository.zip   -d work/
unzip tasks/<task_id>/hidden_tests.zip -d hidden/      # swe only
docker run --rm -v "$PWD/work/repository:/workspace" \
                -v "$PWD/hidden:/hidden-tests:ro" \
                pdl-bench/swe:1.0.0 \
                bash -lc 'python -m pytest tests/public -q && python -m pytest /hidden-tests -q'
```

`manifest.yaml` carries the authoritative `public_command`, `build_command`,
`hidden_command`, and the `patch_policy` that says which paths an agent may edit.

## Why Python 3.12

A floor, not a preference. One public suite
(`swe_rl_cybersecurity_task_01/tests/public/test_crypto_parsing.py`) uses a PEP 701
f-string with a backslash in the expression, which does not parse on 3.11. All 675
task `.py` files parse cleanly on 3.12 and 3.14.

## Integrity

`repository.sha256` in each manifest is the SHA-256 of that task's
`repository.zip`. `base_commit` is `null`: these fixtures are distributed as
archives and have no upstream git history, so the archive hash is the integrity
anchor.

## Why `python -m pytest`, not `pytest`

The hidden suites import the task package as `src`. Bare `pytest` does not put the
working directory on `sys.path`, so with a rootdir of `/hidden-tests` the import fails,
the suite collects zero tests, and the run reports no failures — which reads as a pass.
`python -m pytest` inserts the CWD and resolves it. Four tasks ship their own
`conftest.py` inside the hidden bundle and work either way; the other sixteen do not.

All 20 executable tasks have been replayed in this image: every baseline fails its
hidden suite, every gold patch turns it green, and every public suite is green in both
states.
