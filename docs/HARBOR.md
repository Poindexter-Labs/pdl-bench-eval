# Running the repository tasks with Harbor

`pdl-bench-eval harbor-export --out-dir harbor` writes the 30 repository tasks as Harbor packages
(https://harborframework.com). The dataset repository on the Hub carries this tree under
`harbor/`, so any Harbor-compatible agent harness can run them without a download step:

```bash
harbor run --repo https://huggingface.co/datasets/Poindexter-Labs/PDL-Bench -p harbor -a <agent> -m <model> -k 8
harbor run -p harbor/tasks/swe_rl_task_02 -a oracle      # the gold patch: reward 1.0
```

## Package layout

| File | Content |
| --- | --- |
| `instruction.md` | the task's issue text plus the working rules the reference agents had (allowed paths, public tests, how grading works) |
| `task.toml` | schema 1.3; metadata (bench, grading type, scoring policy, calibrated hidden total, canary); timeouts and resources from the published run profiles |
| `environment/Dockerfile`, `environment/app/` | the public image recipe with the fixture at `/app` |
| `tests/test.sh`, `tests/grade.py`, `tests/grade.json` | a self-contained (stdlib) port of the grader: patch policy against `tests/fixture_index.json`, bytecode purge, planted-hook check, public/build/hidden runs, calibrated-total checks, `scoring_v0.2` |
| `tests/hidden/` | the held-out suite, or the property gate and rubric for SWEQnA tasks |
| `tests/_pdl_eval/` | the 60 s per-test timeout plugin |
| `solution/solve.sh`, `solution/gold.diff` | the reference solution (`git apply`, falling back to `patch -p1`) |

The verifier writes `/logs/verifier/reward.json` (`reward` plus sub-scores and counts) and
`reward.txt`; `grade.json` beside them carries the outcome and notes. An unscorable trial (setup
fault, zero or partial collection without a build failure) writes an **empty** reward file, so Harbor
records an invalid trial rather than a silent zero.

## Network policy and the SWEQnA judge

The reference agents had no network access, so every task sets the agent sandbox to
`no-network`. Executable tasks verify inside that same sandbox, offline.

SWEQnA tasks need an LLM judge panel at verify time, so they declare a **dedicated verifier
sandbox** (`[verifier] environment_mode = "separate"`, built from `tests/Dockerfile`, network on)
and `artifacts = ["/app"]`, which makes Harbor copy the agent's tree into that sandbox before
`tests/test.sh` runs. The verifier runs the property gate, installs `pdl-bench-eval` and calls the
checkpoint judge. It needs, in the environment `harbor run` is launched from:

```bash
export PDL_JUDGES="google:gemini-3.1-pro-preview,anthropic:claude-opus-5,openai:gpt-5.6-sol@https://<azure-resource>/openai/v1"
export GOOGLE_CLOUD_PROJECT=... ANTHROPIC_API_KEY=... OPENAI_API_KEY=...     # or GEMINI_API_KEY / OPENAI_BASE_URL
```

`PDL_JUDGES` is any comma-separated list of `provider:model[@base_url]`; the line above is the
reference panel. Without `PDL_JUDGES` the verifier records the property gate in `gate.json` and
writes an empty reward (an invalid, unjudged trial), never a 0. A judge that cannot be reached
also yields an empty reward, with the reason in `test-stdout.txt`.

Credentials inside the verifier container: the Google member needs `GEMINI_API_KEY` (Application
Default Credentials from `gcloud` are not available inside the sandbox, so `GOOGLE_CLOUD_PROJECT`
alone will fail there); Anthropic and OpenAI-compatible members take their keys from the variables
above. The verifier installs `pdl-bench-eval[judges]`, whose SDK versions are pinned to the ones
the validation runs used.

Resources and timeouts in `task.toml` are the published run profiles: 4 vCPU / 16 GiB / 150 min
agent timeout for executable tasks, 2 vCPU / 8 GiB / 180 min for SWEQnA, 15 min for the verifier.
