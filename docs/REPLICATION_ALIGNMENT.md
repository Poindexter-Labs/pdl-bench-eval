# Aligning the SWE reproducibility work with the non-SWE replication packages

Scoping note, 2026-10-27. **Status:** decisions 1 to 6 below were taken the same day (vendor the
loop; two folders; Docker socket by default; adopt the attempt-record format; allow `--parallel`;
upload the Harbor tree) and the packages were built: see `docs/REPLICATION_PACKAGES.md`. Compares the Docker replication packages a colleague
published for STEM, Business Operations, Life Administration and corrected
GDPVal (`environments/<benchmark>/` in `Poindexter-Labs/PDL-Bench`, commit
0db2764f) with the SWE-side work in this repository (`pdl-bench-eval` graders
and the Harbor export), and proposes how the 30 repository tasks
(`software_delivery`, `cyber_operations`) get a package of the same shape.

## 1. What the colleague built

One folder per benchmark, generated from the pointdexter repository
(`replication/` plus `tools/replication/{vendor,export,bundle}.py`). The four
folders carry identical code and differ only in `data/` and the default
preset. The Docker image is the experiment controller; the model runs on the
provider's native tools or in an E2B sandbox; the judge runs from the
controller.

| Element | What it is | What it guarantees |
| --- | --- | --- |
| `bench.py` | one CLI: `plan`, `doctor`, `run`, `grade`, `report`, `template` | build, `plan` and `doctor` never call a provider; `plan` validates every input hash and prints the resolved configuration and the required keys |
| `presets/<benchmark>.json` | per-model configurations (provider, model, reasoning, attempts, environment, code, web, azure, solver prompt, timeout) plus one `evaluation` block | the preset is the authority; no silent model substitution; route overrides are explicit and recorded |
| `selection_evidence` (in the preset) and `settings-evidence/` | which recorded run each default was taken from, and counts of every historical configuration observed | stated basis ("latest observed successful configuration") and its limits (no per-attempt timestamps) |
| `data/tasks.jsonl`, `data/attachments/`, `data/manifest.json` | the tasks with input metadata (path, mime, kind, sha256), pinned to a dataset revision | input bytes are checked before any spend; `tasks_sha256` and `source_revision` tie the package to the public dataset |
| `vendor/` + `SOURCE_MANIFEST.json` | dependency-closed export of the production solver and judge code, with the source commit, per-file hashes and the list of adaptations | researchers run the production code paths without the application repository |
| `replication_{harness,clients,storage}.py` | thin adapters: lazy credential-bearing clients, local durable storage instead of GCS/Firebase, sequential execution instead of Redis leases | a Claude-only run never needs an OpenAI key; artifacts are saved atomically before grading |
| `results/attempts/<key>.json`, `report.json` | one record per attempt keyed by hash(task, config, attempt, implementation); statuses `SOLVING`, `ANSWER_SAVED`, `SOLVER_FAILED`, `GRADED`, `GRADING_FAILED` | resumable; grade-only re-runs; technical failures stay ungraded and are never counted as model failures; the report never pools different configurations or grader versions; observed pass@k only over tasks with all first k attempts graded |
| `tests/test_workflow.py` | 12 offline integration tests with synthetic providers | routing, schemas, sandbox cleanup, evidence integrity, retries verified without spend |
| `README.md`, `START_HERE.md`, `DATASET-NOTES.md`, `VALIDATION.md`, `PUBLICATION-VALIDATION.json`, `THIRD_PARTY.md`, `CHECKSUMS.json`, `licenses/`, `requirements.lock`, `.env.example` | the document set | pinned base image digest, pinned dependencies, dated validation record, attribution, package hashes |

Conventions worth copying verbatim: secrets only via `.env` and censored
out of error text; the base image pinned by digest; an unprivileged container
user; the results directory as the only writable state; a `--limit 1
--attempts 1` paid smoke run as the first live step.

## 2. What exists on the SWE side

| Piece | Where | Status |
| --- | --- | --- |
| Grading of a patch (`scoring_v0.2`) and of a QnA answer (`scoring_llm_v0.3`) | `pdl-bench-eval` `grade-repo`, `grade-sweqna`; `calibrate`, `summarize` | validated: gold 20/20, 39/40 recorded episodes identical, 2/2 SWEQnA identical |
| Task data | dataset `tasks/<id>/` (repository.zip, hidden_tests.zip, additional_task_files.zip, manifest.yaml, gold.diff) | published; `repository.sha256` in each manifest |
| Task images | dataset `environments/swe/Dockerfile`, `environments/sweqna/Dockerfile` | published; base pinned by digest |
| External-harness entry point | `harbor-export` (30 Harbor tasks, verifier = our grader) | validated with the oracle agent; not yet uploaded |
| Provenance of the published numbers | `docs/REPRODUCTION.md` | names jobs, profiles, prompts, policies, routes |
| The agent half: tool loop, provider adapters, prompts, budgets | private RL repository only: `pd_rl/providers/base.py` (`ModelAgent`), `providers/{openai,anthropic,google}_provider.py`, `runner.py` (`ToolBox`, episode loop), `sandbox.py` (`DockerSandbox`, `LocalSandbox`), `prompts/agent_prompt_v0.3.yaml`, `prompts/agent_prompt_qa_v0.1.yaml`, `contracts/tools_v0.1.yaml`, `profiles/python_standard_2m_v0.1.yaml`, `profiles/python_qa_2m_v0.1.yaml` | not public; about 6,700 lines across the agent-side modules before pruning |
| Per-episode settings record | BigQuery `pdl_rl.episodes`: prompt_version, tool_schema_version, scoring_version, evaluator_version, model_parameters, environment_image_digest, sandbox_backend, termination_reason, tokens, cost, judge panel | complete for every published job |

So the grading half already meets the colleague's bar. The two missing
halves are the solver (`run`) and the package envelope around it.

## 3. Gap table

| Colleague's element | SWE today | Needed for alignment |
| --- | --- | --- |
| `plan` / `doctor` / `run` / `grade` / `report` | `grade-*` on one submission at a time; no `run`; `summarize` over records | the same five verbs over the whole benchmark, with the same offline guarantees |
| Presets with per-model configs | run profiles and prompts in the private repo; routes listed in REPRODUCTION.md | `presets/software-delivery.json`, `presets/cyber-operations.json`: see section 5 |
| Selection evidence: latest observed successful configuration | exact console job ids per published figure | stronger basis: `selection_evidence` = the job ids and dates behind the report; no ordering caveat needed |
| Settings evidence from the sheet audit | BigQuery episodes | `settings-evidence/*.json` generated from BigQuery (section 6) |
| Solver environment: native API or E2B | Kubernetes (gVisor) internally; `DockerSandbox` exists in the runner | Docker sandbox built from `environments/swe` and `environments/sweqna` (default); E2B template from the same recipe as an option |
| `data/tasks.jsonl` + attachments + manifest | `tasks/<id>/` bundles | `data/tasks.jsonl` index (id, benchmark, family, issue text, bundle files with sha256, calibrated hidden total, scoring policy) plus the bundles copied under `data/tasks/`; `manifest.json` with `source_revision` |
| Vendored production judge | ported graders, validated against recorded results | a `SOURCE_MANIFEST.json`-style record naming the RL commit each port was taken from, plus the validation JSON already in `docs/validation/` |
| Attempt records and `report.json` | the runner's episode `record.json` | wrap the episode record in the colleague's attempt envelope; report both statistics (section 8) |
| 12 offline integration tests | 11 unit tests, Harbor oracle runs | offline tests of the agent loop with a scripted provider (gold and noop agents already exist) |
| Document set | README, REPRODUCTION, HARBOR, validation JSON | the full set: START_HERE, DATASET-NOTES, VALIDATION, PUBLICATION-VALIDATION, THIRD_PARTY, CHECKSUMS, licenses, requirements.lock, .env.example |
| None | Harbor tree | keep as the second entry point for external harnesses; the package reproduces the reference agent, Harbor reproduces the grader under any agent |

## 4. Proposed shape

Two folders in the dataset, mirroring the colleague's one-folder-per-benchmark
rule and sharing all code:

```
environments/software-delivery/        20 tasks: 10 executable + 10 QnA over a repository
environments/cyber-operations/         10 executable tasks
  Dockerfile                           controller image; targets per preset
  bench.py                             plan | doctor | run | grade | report
  presets/software-delivery.json, cyber-operations.json
  settings-evidence/*.json             generated from BigQuery
  data/tasks.jsonl, data/tasks/<id>/*, data/manifest.json
  sandbox/swe.Dockerfile, sandbox/sweqna.Dockerfile   the task images (copies of environments/swe|sweqna)
  tests/                               offline tests, scripted agents
  README.md, START_HERE.md, DATASET-NOTES.md, VALIDATION.md,
  PUBLICATION-VALIDATION.json, THIRD_PARTY.md, CHECKSUMS.json, licenses/,
  requirements.lock, .env.example, .dockerignore, .gitignore
```

`bench.py` is a thin entry over `pdl_bench_eval.replication`, installed in
the image from a pinned commit of this repository (or vendored, see section
7). `pdl-bench-eval` stays the library: single-submission grading, Harbor
export, calibration. The colleague intends to publish their shared runner
separately later; adopting their attempt-record and report format now makes
that convergence a merge of two storage modules rather than a rewrite.

## 5. Presets for an agentic benchmark

The colleague's model entry describes one API call. Ours describes a
multi-turn agent, so the entry needs the agent-side settings that the episode
record already stores:

```json
{
  "id": "gpt-6-astra",
  "provider": "openai-azure",
  "model": "gpt-6-astra",
  "route": "openai-azure:swedencentral",
  "model_parameters": {"api_mode": "responses", "sampling": "provider-default"},
  "attempts": 8,
  "agent": {"prompt": "agent_prompt_v0.3", "tools": "tools_v0.1"},
  "profile": "python_standard_2m_v0.1",
  "sandbox": {"image": "pdl-bench/swe:1.0.0", "network": "none"}
}
```

The profile expands to the published budgets: 2,000,000 tokens, 200 turns,
1,280 tool calls, 640 shell commands, 150 min agent timeout, 15 min evaluator
timeout, 60 s per test, 4 vCPU / 16 GiB (executable); 800 tool calls, 400
shell commands, 180 min, 2 vCPU / 8 GiB (QnA). These are copied into the
preset so a researcher never needs the private profile files.

`software-delivery` has two task families with different prompts, profiles
and scoring, so its preset carries two `evaluation` blocks keyed by family:

| Family | Prompt | Profile | Scoring | Pass |
| --- | --- | --- | --- | --- |
| executable (10 + the 10 cyber tasks) | `agent_prompt_v0.3` | `python_standard_2m_v0.1` | `scoring_v0.2` | reward > 0.95 |
| QnA over a repository (10) | `agent_prompt_qa_v0.1` | `python_qa_2m_v0.1` | `scoring_llm_v0.3`, panel gemini-3.1-pro-preview + claude-opus-5 + gpt-5.6-sol, per-item median | reward > 0.95 |

Default model list = the five behind the published report: `gpt-6-astra`
(Azure, Sweden Central), `claude-fable-5-1` and `claude-opus-5-5` (Anthropic
API), `gemini-3.8-flash` and `gemini-4-argon` (Vertex, global). Model
parameters come from the episode records: Anthropic `max_tokens 8192` with
ephemeral prompt cache; Gemini `temperature 1`; Azure Responses API with
provider-default sampling.

## 6. Evidence from BigQuery

The sheet audit lost per-attempt timestamps; the episode table has
everything. One query grouped by profile, prompt, tool schema, scoring,
evaluator, provider, model, parameters, image digest and sandbox backend
gives the settings evidence directly. The 2026-10-27 run of that query shows,
for example, 160 Astra episodes under `python_standard_2m_v0.1`,
`agent_prompt_v0.3`, `scoring_v0.2`, sandbox k8s, image digest `e0cb07…`
(156 valid, 4 void), and 80 Astra QnA episodes under `scoring_llm_v0.3`.

Two protocol facts the evidence makes explicit, for `DATASET-NOTES.md`:

- QnA: the published numbers use `scoring_llm_v0.3` (jobs of 2026-10-04,
  2026-10-05, 2026-10-14). Earlier `scoring_llm_v0.2` runs (August to
  September) are evidence only and must not be pooled.
- Executable: the published numbers use `scoring_v0.2` on image digest
  `e0cb07…` from 2026-09-03 onward. The 2026-08-16 probes used `scoring_v0.1`,
  a different digest and a local sandbox; excluded.

The query and its output should ship as `settings-evidence/<benchmark>.json`
plus the SQL text, and `selection_evidence` lists the console job ids named
in `docs/REPRODUCTION.md`.

## 7. The agent half: three options

| Option | What | Fidelity | Cost |
| --- | --- | --- | --- |
| A. Vendor the RL runner's agent loop | dependency-closed export (the colleague's `vendor.py` technique applied to the RL repo): `ModelAgent`, the three provider adapters, `ToolBox`, `DockerSandbox`, prompts, profiles, contracts; strip Kubernetes, GCS, BigQuery, Secret Manager | byte-identical prompts, tool schemas and budget accounting | about 3 to 4 days; requires a decision to publish those parts of the private repository (Apache-2.0, as this package) |
| B. Port the agent loop into `pdl-bench-eval` | the same approach as the graders: a port validated against recorded trajectories (prompt bytes, tool schemas, termination reasons) | high, but a port can drift in details the graders did not have (retry policy, truncation, tool-result formatting) | about 5 to 6 days plus validation runs |
| C. Harbor only | no reference agent; researchers bring their own | cannot reproduce the published numbers (different agent) | nothing new |

Recommendation: A, with C kept as the complementary entry point. The vendored
code gets a `SOURCE_MANIFEST.json` naming the RL commit and per-file hashes,
exactly as the colleague did for pointdexter.

Credentials: the adapters read keys from Secret Manager today. In the package
they read `.env` (`AZURE_OPENAI_KEY` + `AZURE_OPENAI_BASE_URL`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` or Vertex ADC, judge keys for the QnA
panel). Same variable names as the colleague's `.env.example` where they
overlap.

## 8. Differences to keep explicit

- **Cost and time.** A non-SWE attempt is one API call; an SWE attempt is a
  2M-token agent episode (about 5 to 15 min, dollars not cents). A full run
  (30 tasks × 8 attempts × 5 models) sequentially would take days. Propose an
  explicit `--parallel N` with a cost estimate printed by `plan`; the
  colleague's sequential default stays the default.
- **Sandbox privilege.** The controller must start sandbox containers:
  mount the Docker socket (as Harbor does), or run the controller on the host
  via `pip`, or use an E2B template built from the same Dockerfile (the
  colleague's `template` verb). Docker socket as default, E2B as a later
  option; the README states the privilege implication.
- **Statistics.** The colleague reports mean score and observed pass@k over
  graded attempts. The published SWE figure is pass rate = passes / valid
  attempts, where `void_*` is excluded and `invalid_*` (policy violation,
  planted hook, non-applying patch) counts as a failure. `report.json` should
  carry both, under their own names, never one relabelled as the other.
- **Judges.** The QnA panel is three members on three routes; a researcher
  may substitute a panel they can reach (`--judges`), and the record names
  what was used.
- **Image provenance.** Internal episodes ran on `python-standard:0.4`
  (digest `e0cb07…`); the public image is rebuilt from the same pinned base.
  The 39/40 re-grade is the evidence that this does not move results; say so
  rather than claiming identity.
- **Task versions.** `cyber_07` was revised after 2026-09-03 and `cyber_03`'s
  suite grew from 60 to 106 tests; the package pins the dataset revision and
  states that the report should be read against those versions.

## 9. Validation plan (same shape as the colleague's)

Offline: `plan` checks every bundle hash against `manifest.yaml`
`repository.sha256` and the calibrated hidden totals; `doctor` checks Docker,
images and keys without provider calls; the test suite drives the agent loop
with the scripted gold and noop agents (gold = 1.0 and noop = 0.0 on all 20
executable tasks is the existing `calibrate` replay; SWEQnA gold answers judged
by the panel). Live: one task, one attempt, one model per route (Azure,
Anthropic, Vertex), then one QnA task through the panel. Fidelity: grading is
already proven; for the agent, re-run one model on a handful of tasks and
compare termination reasons and reward spread with the recorded episodes
(stochastic, so a distribution check, not equality). Record all of it in
`VALIDATION.md` and `PUBLICATION-VALIDATION.json`.

## 10. Work plan

| Phase | Content | Effort |
| --- | --- | --- |
| 0 | decisions below | |
| 1 | `pdl_bench_eval.replication`: data index generator from the Hub bundles, preset schema, settings-evidence generator from BigQuery, `plan` / `doctor` / `report`, attempt records in the colleague's format | 2 days |
| 2 | `run`: vendored agent loop, Docker sandbox from the task images, `.env` credentials, per-attempt resume, `--parallel` | 3 to 4 days |
| 3 | packaging: Dockerfile, requirements.lock, offline tests, document set, CHECKSUMS, bundle generator mirroring `tools/replication/bundle.py`, validation runs | 2 days |
| 4 | upload `environments/software-delivery/` and `environments/cyber-operations/`; update `environments/README.md` and the dataset card; upload the Harbor tree | half a day |
| 5 | with the colleague, when their shared runner is published: one storage and report module, one preset schema | later |

## 11. Decisions needed

1. Publish the agent loop by vendoring from the RL repository (recommended) or re-port it.
2. Two folders mirroring the colleague (recommended) or one `swe` folder with two presets.
3. Sandbox launch: Docker socket by default, E2B template as an option, or E2B only.
4. Adopt the colleague's attempt-record and report format now, with `pdl-bench-eval` as the SWE engine, so the later runner merge is mechanical.
5. Allow `--parallel` for the SWE package, given episode cost and duration.
6. Upload the Harbor tree under `harbor/` as the external-harness entry point alongside the packages.
