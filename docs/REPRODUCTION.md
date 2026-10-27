# How the published PDL-Bench numbers were produced

This page names everything behind the figures in *Poindexter Mini-Benchmark Samples* (dataset
`docs/`), so that a reader can tell exactly what was measured and re-run it with this package: the grading
half with the graders below, and the agent half with `pdl-bench-replicate`, which runs the reference
agent loop (exported from the PD RL Environment repository) under the prompts, tool surface and
budgets named here (`docs/REPLICATION_PACKAGES.md`).

## Repository tasks: Software Delivery (executable) and Cyber Operations

| Item | Value |
| --- | --- |
| Tasks | 20 (`grading_type = tests`): `swe_rl_task_02`, `pdl_swe_edit_merge_019`, `contributor_task_01/03/04/06/07/08/09/10`, `swe_rl_cybersecurity_task_01..10` |
| Run profile | `python_standard_2m_v0.1`: 2,000,000 total tokens, 200 turns, agent timeout 150 min, evaluator timeout 15 min, per-test limit 60 s, 4 vCPU / 16 GiB / 30 GiB |
| Agent prompt | `agent_prompt_v0.3` (tool vocabulary `tools_v0.1`: read, search, write/patch, shell) |
| Scoring policy | `scoring_v0.2` (progress-scaled; this package's `policies/scoring_v0.2.yaml`) |
| Repeats | 8 attempts per model per task |
| Pass | reward > 0.95 |
| Console jobs | Astra/Fable/Gemini 3.8: `rlj-20260911-182105-d16163` (SWE), `rlj-20260911-181246-68ed7c` (Cyber); Opus 5.5: `rlj-20260925-204349-0090bb`, `rlj-20260925-210731-44bc8a`; Gemini 4 Argon: `rlj-20261013-232626-6fc6bd` |
| Models and routes | `gpt-6-astra` (Azure OpenAI, Sweden Central), `claude-fable-5-1` and `claude-opus-5-5` (Anthropic API), `gemini-3.8-flash` and `gemini-4-argon` (Vertex AI, global) |
| Sandbox | Kubernetes (gVisor) for the agent; evaluation in a fresh container with the network disabled |

Grading steps, in order (each a port in `pdl_bench_eval/repo_grader.py`):

1. Every path the patch touches must match `patch_policy.allowed_paths` and none of
   `forbidden_paths` (manifest); otherwise `invalid_agent_fault`, reward 0.
2. Bytecode caches are purged from the graded tree; a hook-named file the fixture did not ship
   (`conftest.py`, `sitecustomize.py`, `*.pth`, `pytest.ini`, ...) is `invalid_agent_fault`.
3. In fresh containers, each prefixed by the task's `setup_command`: `public_command`
   (regression suite), `build_command`, `hidden_command`; counts are the sums of every
   `N passed` / `N failed` / `N error` token in pytest's output.
4. Zero hidden tests collected is an evaluator failure (void). A collected total that differs from
   the calibrated total voids too, unless the build step failed on the graded tree: then the
   graded code does not import and every uncollected test counts as failed.
5. Reward under `scoring_v0.2`; a published pass is reward > 0.95.

Calibrated hidden totals come from replaying every gold patch in the public image
(`pdl_bench_eval/calibration.json`). Two tasks have a non-zero baseline under this policy
(`swe_rl_cybersecurity_task_03`: 13/106 hidden tests pass before any change, reward 0.1226;
`swe_rl_cybersecurity_task_07`: 8/60, reward 0.1333).

## SWEQnA tasks (question answering over a repository)

| Item | Value |
| --- | --- |
| Tasks | 10 (`grading_type = hybrid`, ids `pdl_qna_*` and `chronicle_versioned_escalation_001`) |
| Run profile | `python_qa_2m_v0.1`: 2,000,000 tokens, 200 turns, agent timeout 180 min, 2 vCPU / 8 GiB |
| Agent prompt | `agent_prompt_qa_v0.1` |
| Scoring policy | `scoring_llm_v0.3`: rubric criteria → binary checkpoints; three-member panel, one call each; item score = median of votes; reward = weighted mean |
| Judge panel | `gemini-3.1-pro-preview` (Vertex AI), `claude-opus-5` (Anthropic API), `gpt-5.6-sol` (Azure OpenAI) |
| Judge prompt | `judge_prompt_v0.2` (verbatim in `policies/scoring_llm_v0.3.yaml`) |
| Console jobs | `rlj-20261004-175352-1f4a04` (Astra/Fable/Gemini 3.8), `rlj-20261005-005743-894c05` (Opus 5.5), `rlj-20261014-113935-1586f8` (Argon) |
| Pass | reward > 0.95 |

The hidden suite of a SWEQnA task is a property gate (file exists, size bounds, required headings);
it must collect, and it is recorded, but it never contributes to the reward.

## Single-turn Q&A: STEM, Business Operations, Life Administration

| Item | Value |
| --- | --- |
| Tasks | 62 (`objective`, `rubric` or `hybrid`) |
| Attempts | 8 responses per model per task, no tools beyond the model's own, attachments supplied inline |
| Answer match | deterministic normalised match, else the comparator prompt `qa_answer_match_v1` with `gpt-5.4` (Azure OpenAI), reasoning `medium` |
| Rubric grade | `sheet-rubric-v1` with `gpt-5.6` (Azure deployment `gpt-5.6-sol`), reasoning `medium`; HIT/MISS per criterion, weight 1 unless stated; score = 10 × earned / total |
| Pass | objective: match; rubric and hybrid: score > 8.5 |
| Denominator | every attempt, including provider failures recorded as attempts |

## How this package was validated

* `calibrate` replays every executable task's gold patch and the pristine fixture in the public image:
  20/20 gold = 1.0 (one gold patch needs GNU `patch` because `git apply` rejects it), 18/20 baseline = 0.0,
  2 with the small non-zero baselines above.
* 40 recorded console episodes (patches from the artifacts bucket; 9 models, all 20 tasks, rewards
  spanning 0 to 1) re-graded here: 39/40 rewards identical to the recorded `reward_total`. The
  exception (`ep-20260903-213051-19cc`, `swe_rl_cybersecurity_task_07`) was recorded before that task's
  fixture was revised; its patch no longer applies. The published report should be read against
  the task versions in the dataset.
* Two recorded SWEQnA episodes re-judged with the reference panel reproduce their rewards exactly
  (0.75, 0.964286). Six stored single-turn responses re-graded with the reference rubric judge: three
  identical, three differ by exactly one criterion verdict (a single-call judge is not deterministic).

## Known differences from the internal runner

* The internal runs used Poindexter's own image (`python-standard:0.4`); the public image is rebuilt
  from the same pinned base. Every fixture is standard-library only, and the 39/40 re-grade shows no
  divergence attributable to the image.
* The internal runner enforces the per-test time limit through a pytest plugin staged beside the
  hidden suite; this package ships the same plugin and applies the same 60 s limit.
* Judges are called at temperature 1.0 (checkpoints) or with reasoning `medium` (Q&A), as in the
  reference pipeline; the Claude member is streamed with a 32,768-token output budget, a member that
  refuses is dropped down to a two-member floor and recorded, and a member that cannot deliver after
  its retry voids the grade (never a 0), all as in the reference; a judge is non-deterministic, so single-call rubric scores can move by one
  criterion between runs. The checkpoint panel's per-item median is far more stable.
