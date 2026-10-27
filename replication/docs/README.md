# Poindexter @@BENCH@@ — Docker replication package

Run the PDL-Bench **@@TITLE@@** mini-benchmark (@@TASKS@@ repository tasks) the way
Poindexter's published numbers were produced, without a Poindexter account,
the private RL repository, Kubernetes, GCS or BigQuery. The Docker image is the
experiment controller: it runs the reference agent loop against each task in
its own Docker sandbox, freezes the agent's work into a patch, and grades that
patch with the reference evaluator.

These are downloadable evaluation packages. Credentials and results are not
baked into the image. No benchmark starts during build, `plan` or `doctor`.

## What a run reproduces

| Family | Tasks | Agent prompt | Run profile | Scoring | Pass |
| --- | ---: | --- | --- | --- | --- |
| executable repository tasks | @@EXECUTABLE@@ | `agent_prompt_v0.3` | `python_standard_2m_v0.1` | `scoring_v0.2`: hidden-test fraction × (0.80 + 0.15 regression + 0.05 build) | reward > 0.95 |
@@QNA_SECTION@@
Default models: @@MODELS@@, eight attempts each, exactly the configurations behind
the published report (preset `selection_evidence` names the console jobs:
executable @@EXEC_JOBS@@; question answering @@QNA_JOBS@@). The JSON preset is the
authority; `plan` prints the resolved budgets (2,000,000 tokens, 200 turns,
1,280 tool calls, 640 shell commands, 150 min for executable tasks; 800 tool
calls, 400 shell commands, 180 min for question answering), the tool
vocabulary and the required keys. `settings-evidence/` lists every
configuration ever recorded on these tasks, with counts, so that earlier
scoring policies and image digests are visible and never pooled.

Two operator overrides the published runs made are carried in the preset and
stated in `DATASET-NOTES.md`: the executable tasks' manifests pin a 1M-token
profile and the published report ran the 2M sibling; the question-answering
manifests pin judge policy v0.2 and the report graded under v0.3.

## Download and build

```sh
python -m pip install -U huggingface_hub
hf download Poindexter-Labs/PDL-Bench --repo-type dataset \
  --include "environments/@@BENCH@@/**" --local-dir pdl-bench
cd pdl-bench/environments/@@BENCH@@
cp .env.example .env
# Fill the keys `plan` lists for the models you will run.
docker build --target @@BENCH@@ -t poindexter/mini-bench:@@BENCH@@ .
mkdir -p results
```

Add `--revision COMMIT_SHA` to the download to pin the package release. The
controller needs the host's Docker daemon: every command below mounts
`/var/run/docker.sock`. That socket is root-equivalent on the host, which is
why the image does not switch to an unprivileged user. On Linux, pass
`--user "$(id -u):$(id -g)" --group-add "$(stat -c %g /var/run/docker.sock)"`
to keep the results directory owned by you.

```sh
# Offline: validate every bundle hash, print the resolved configuration and required keys.
docker run --rm --network none -v "$PWD/data:/data:ro" \
  poindexter/mini-bench:@@BENCH@@ plan

# Check keys and imports, reach Docker, build the sandbox and evaluator images. No provider calls.
docker run --rm --env-file .env -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/data:/data:ro" poindexter/mini-bench:@@BENCH@@ doctor

# Free calibration on the executable tasks: the reference patch scores 1.0 and the empty
# submission 0.0 (no keys needed; the question-answering baselines need the judge keys).
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD/data:/data:ro" \
  -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ run --agent gold --family executable --limit 1
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD/data:/data:ro" \
  -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ run --agent noop --family executable --limit 1

# Paid smoke run: one task, one attempt, one model.
docker run --rm --env-file .env -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/data:/data:ro" -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ \
  run --model gpt-6-astra --limit 1 --attempts 1

# Full run/resume: all preset models, eight attempts each.
docker run --rm --env-file .env -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/data:/data:ro" -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ run --parallel 4
```

The sandboxes and evaluator containers are started by the host daemon, so
the controller translates its own bind-mounted paths (`/data`, `/results`) to
the host paths behind them before naming them in a mount; `doctor` prints the
translation. Keep both directories bind-mounted from a location the daemon
can share (on Docker Desktop, inside a shared file-sharing path); a results
directory the daemon cannot see is refused up front rather than graded empty.

Provider access is account-specific. The reference routes are Azure OpenAI
for `gpt-6-astra`, the Anthropic API for the Claude models and Vertex AI
(global endpoint) for the Gemini models; the preset records the route and the
results record what actually served each episode. There is no silent model
substitution. A different route or model is a different configuration and is
reported separately.

## Cost and time

An attempt is a complete agent episode with a 2,000,000-token budget, not one
API call. Episodes typically take 5 to 15 minutes and cost dollars, not
cents; the full default run is @@TASKS@@ tasks × 8 attempts × 5 models. `plan`
prints the attempt count. `--parallel N` runs N episodes at once, each in its
own sandbox (4 vCPU / 16 GiB each for executable tasks); the default is
sequential.

## Saved episodes, retries and reporting

Every attempt writes `results/attempts/<key>.json` and
`results/episodes/<key>/` (`trajectory.jsonl` with every model call, tool
call and result; `patch.diff`; `logs.txt`; `episode_record.json` in the
reference runner's record schema). A finished episode is saved before grading;
`grade` regrades saved episodes without touching a model; `--retry-failed`
re-solves attempts that failed for technical reasons. An attempt's identity is
the task, the configuration, the attempt number and the code that produced the
episode (the vendored loop and runner); the grading code and judge panel are
part of its evaluation identity, so a grader fix regrades saved episodes
rather than discarding them, and the report still never pools across either. An attempt marked
`SOLVING` was interrupted; check the saved state before retrying it.

```sh
docker run --rm --env-file .env -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/data:/data:ro" -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ grade
docker run --rm -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ report
```

`report.json` separates technical failures (ungraded, never counted against a
model) from graded attempts, and reports two statistics under their own
names: `pass_rate` = passes / valid attempts, where an attempt that the
evaluator voided is excluded and an attempt the agent invalidated (patch
outside the allowed paths, planted interpreter hook, stray background
process) counts as a failure — this is the published PDL-Bench statistic; and
`observed_pass_at_k`, the other packages' statistic. Different
configurations, code versions or grader settings are never pooled.

## Data contract and provenance

`data/tasks.jsonl` indexes the bundles under `data/tasks/<id>/`
(`manifest.yaml`, `repository.zip`, `hidden_tests.zip`, `gold.diff`) with the
SHA-256 of each file; `plan` refuses a bundle whose hash differs. Pinned to
dataset revision `@@REVISION@@` (snapshot @@SNAPSHOT_DATE@@). The solver never
sees the hidden suite, the rubric or the reference patch; the gold baseline is
a scripted agent that applies the reference patch through the same
`write_patch` tool a model would use.

`vendor/pdl_bench_eval` is the reference evaluator (validated against the
recorded results: 39 of 40 recorded episodes re-graded to the identical
reward) and, under `_vendor/`, the agent loop exported file-for-file from the
PD RL Environment repository: the prompts, tool schema, run profiles,
contract, provider adapters, toolbox and Docker sandbox the reference runs
used. `vendor/SOURCE_MANIFEST.json` names both source commits and every file
hash; the adaptations are listed there (credentials from the environment
instead of Secret Manager; Docker as the only sandbox; no Kubernetes, cloud
publishing, pull-request or DynamicSWE code).

Known differences from the reference runs, stated rather than hidden: the
reference episodes ran on Poindexter's `python-standard:0.4` image under
gVisor on Kubernetes; this package rebuilds the same pinned base for both the
agent sandbox and the evaluator, and the re-grade above is the evidence that
the image does not move results. Model behaviour is stochastic and provider
model versions change; exact scores are not guaranteed across runs. Record
the built image digests with any published result.

## Offline verification

```sh
docker run --rm --network none -v "$PWD/data:/data:ro" \
  --entrypoint python poindexter/mini-bench:@@BENCH@@ -m unittest discover -s tests -v
```

Synthetic-provider tests of the preset, the data contract, the attempt
records and the report; the sandbox tests run only where Docker is reachable
and spend nothing. `VALIDATION.md` records the dated live checks and their limits.
