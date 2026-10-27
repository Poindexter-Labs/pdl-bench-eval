# pdl-bench-eval

The reference evaluator for [PDL-Bench](https://huggingface.co/datasets/Poindexter-Labs/PDL-Bench),
Poindexter Labs' open benchmark. It grades submissions **exactly as the published results were
graded**: the same scoring policies, the same test-count rules, the same judge prompts and
panel, the same pass threshold. Every rule is a port of the code that produced the report.

| Mini-benchmark | Tasks | Grader | What it needs |
| --- | --- | --- | --- |
| Software Delivery (executable), Cyber Operations | 20 | `grade-repo`: patch → hidden suite in the pinned image → `scoring_v0.2` | Docker |
| Software Delivery (QnA over a repository) | 10 | `grade-sweqna`: property gate + LLM judge panel over rubric checkpoints → `scoring_llm_v0.3` | Docker + judge API keys |
| STEM, Business Operations, Life Administration | 62 | `grade-qa`: final-answer match and/or rubric judge | judge API key |

The 30 repository tasks are also published as a **Harbor dataset** inside the HF repository
(`harbor/`), so any Harbor-compatible agent harness can run them: see [docs/HARBOR.md](docs/HARBOR.md).

## Install

```bash
pip install "pdl-bench-eval[judges] @ git+https://github.com/Poindexter-Labs/pdl-bench-eval"
pdl-bench-eval fetch            # downloads the task bundles, images and parquet files from the Hub
```

The `judges` extra pins the provider SDKs to the versions the validation runs used. Docker is required for the repository graders (the evaluator builds `pdl-bench/swe:1.0.0` and
`pdl-bench/sweqna:1.0.0` from the Dockerfiles shipped in the dataset; the base layer is pinned by digest).

## Grade a submission

```bash
# a patch (unified diff against repository.zip) for an executable task
pdl-bench-eval grade-repo --task swe_rl_task_02 --patch my_patch.diff --out record.json

# a written answer for a QnA-over-repository task, judged by the reference panel
export GOOGLE_CLOUD_PROJECT=... ANTHROPIC_API_KEY=... OPENAI_API_KEY=... OPENAI_BASE_URL=...
pdl-bench-eval grade-sweqna --task pdl_qna_close_provenance_010 --answer answer.md
#   or any panel you can reach: --judges "google:gemini-3.1-pro-preview,anthropic:claude-opus-5,openai:gpt-5.6-sol@https://host/openai/v1"

# a response to a single-turn task
pdl-bench-eval grade-qa --task PDS030 --response response.txt
#   judges: --match-judge openai:gpt-5.4  --rubric-judge openai:gpt-5.6-sol@https://host/openai/v1

# the published statistic over a directory of records: passes / valid attempts
pdl-bench-eval summarize records/
```

Every command prints a JSON record with the outcome, the reward, the pass flag, every
sub-score and count, and notes explaining any void or invalid outcome.

## What a published number means

* **Repository tasks**: reward = hidden-test fraction × (0.80·1 + 0.15·regression + 0.05·build),
  weights renormalised when a bucket is absent (`scoring_v0.2`). A no-op scores 0.
* **SWEQnA tasks**: reward = weighted mean over rubric checkpoints of the per-item median of the
  panel's binary votes (`scoring_llm_v0.3`); the property gate must collect but is never scored.
* **Single-turn Q&A**: objective tasks pass on a final-answer match; rubric and hybrid tasks pass
  on a rubric score strictly above 8.5/10.
* **Pass** on repository and SWEQnA tasks is a reward **strictly greater than 0.95**.
* **Pass rate** = passes / valid attempts; `void_*` outcomes (evaluator or setup faults) are excluded,
  `invalid_*` outcomes (patch-policy violations, planted hooks, non-applying patches) count as failures.

[docs/REPRODUCTION.md](docs/REPRODUCTION.md) names the exact jobs, run profile, agent prompt,
scoring policy and model routes behind each published figure, and reports how this package was
validated against the recorded results.

## Validation (2026-10-21)

* Gold patches score 1.0 and the pristine fixture scores 0.0 on all 20 executable tasks
  (`docs/validation/calibration_replay_2026-10-21.json`).
* 40 recorded console episodes (9 models, all 20 tasks, rewards from 0 to 1) re-graded from their
  recorded patches: **39/40 identical rewards**. The remaining episode predates a fixture revision
  of its task and its patch no longer applies (`docs/validation/regrade_console_episodes_2026-10-21.json`).
* Two recorded SWEQnA episodes re-judged with the reference panel: **identical rewards** (0.75 and
  0.964286; the per-item median absorbs single-judge noise).
* Six stored single-turn responses re-graded with the reference rubric judge: 3 identical, 3 within
  one criterion flip (single-call judge variance); three answer-match verdicts: 3/3 identical.

## Replicate the published runs

`pdl-bench-replicate` runs the **reference agent loop** (exported file-for-file from the PD RL
Environment repository, see `pdl_bench_eval/_vendor/SOURCE_MANIFEST.json`) on the repository tasks in
Docker sandboxes, under the published prompts, tool schema and budgets, and grades each episode with
the graders above. It is packaged for researchers as two self-contained folders in the dataset,
`environments/software-delivery` and `environments/cyber-operations`, with the same `plan` / `doctor` /
`run` / `grade` / `report` verbs, presets, settings evidence and attempt records as the other PDL-Bench
replication packages (`docs/REPLICATION_PACKAGES.md`, `docs/REPLICATION_ALIGNMENT.md`).

```bash
pdl-bench-replicate build-data --preset software-delivery --data /tmp/sd      # tasks.jsonl + bundles from the Hub
pdl-bench-replicate plan --preset software-delivery --data /tmp/sd            # offline: hashes, budgets, required keys
pdl-bench-replicate run  --preset software-delivery --data /tmp/sd --results /tmp/sd-results --agent gold --limit 1
pdl-bench-replicate run  --preset software-delivery --data /tmp/sd --results /tmp/sd-results --model gpt-6-astra --parallel 4
```

## Licence

Apache-2.0. The dataset itself is CC BY 4.0, with third-party input files reproduced for
evaluation only (see the dataset card).
