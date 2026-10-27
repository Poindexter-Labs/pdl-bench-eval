# Validation — 27 October 2026

What was checked before this package was published, and what was not.

## Provenance checks

- The agent loop was exported from the PD RL Environment repository at commit
  `716e67fc4059c14cde9466aee9f1e14908b609e0` with no local modifications to any
  of the 18 source files (`vendor/pdl_bench_eval/_vendor/SOURCE_MANIFEST.json`).
  Every exported module imports; the only functional adaptations are the ones
  listed in that manifest.
- The evaluator is `pdl-bench-eval` 0.1.0, whose graders were validated on
  2026-10-21 against the recorded results: gold patches 20/20 at reward 1.0,
  39 of 40 recorded console episodes re-graded to the identical reward, two
  recorded SWEQnA episodes re-judged to the identical reward.
- Settings evidence was generated from the episode table in BigQuery (every
  recorded episode carries its prompt version, tool schema, scoring policy,
  model parameters, image digest and sandbox backend); counts are exact.

## Offline checks

- `plan` validates every bundle hash against `data/tasks.jsonl` and the
  manifests' `repository.sha256`, inside the controller image with networking
  disabled.
- The controller image builds from the pinned base; the package's offline
  test suite (presets, data contract, tamper refusal, attempt records and
  report arithmetic) passes inside it; the evaluator's own unit tests pass.
- The published layout was exercised the way a researcher runs it: the
  package was downloaded from the Hub, its controller image built, and
  `doctor`, a gold attempt and a noop attempt were run through the mounted
  Docker socket (`swe_rl_cybersecurity_task_01`: 1.0 and 0.0). This caught and
  fixed a socket-mode defect before publication: evaluator bind mounts issued
  from inside the controller are now translated to host paths, and a daemon or
  mount fault voids the grade instead of scoring it.
- Calibration through the full path (sandbox episode → frozen patch →
  grader) on every task, with the two scripted baselines the reference runner
  calibrates with. The question-answering baselines are judged by the
  reference panel. Tables below are generated from the attempt records
  (`tools/validation_record.py`), never typed.

<!-- validation-tables:begin -->
| Benchmark | Baseline | Tasks | Result |
| --- | --- | ---: | --- |
| software-delivery | gold | 20 | 17 at reward 1.0; other: pdl_qna_access_path_planning_017 0.961538; pdl_qna_signed_package_resolution_014 0.96; pdl_qna_incremental_build_promotion_015 0.96 |
| software-delivery | noop | 20 | 20 at reward 0.0 |
| cyber-operations | gold | 10 | 10 at reward 1.0 |
| cyber-operations | noop | 10 | 8 at reward 0.0; documented non-zero baselines: swe_rl_cybersecurity_task_07 0.133333; swe_rl_cybersecurity_task_03 0.122642 |

| Model | Route | Task | Termination | Outcome | Reward | Tool calls | Tokens in / out | Minutes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gpt-6-astra | openai-azure:swedencentral | swe_rl_task_02 | agent_submits | valid_scored | 1.0 | 17 | 166,602 / 11,427 | 5.0 |
| gemini-3.8-flash | google-vertex:global | swe_rl_task_02 | token_budget_reached | valid_scored | 1.0 | 28 | 1,982,496 / 130,449 | 16.9 |
| claude-fable-5-1 | anthropic:api | pdl_qna_close_provenance_010 | agent_submits | valid_scored | 1.0 | 13 | 451,026 / 24,082 | 6.3 |
<!-- validation-tables:end -->

The question-answering property gate reports 0 of 2 assertions on these
baselines, as it did on the recorded reference episodes of the same tasks; the
gate is collected and recorded but never scored, so this matches the
reference behaviour rather than contradicting it. Two executable tasks have
documented non-zero noop baselines under `scoring_v0.2`
(`swe_rl_cybersecurity_task_03`: 13 of 106 hidden tests pass before any
change; `swe_rl_cybersecurity_task_07`: 8 of 60); the table shows them.

## Live checks

One paid episode per reference route (Azure OpenAI, Vertex AI, Anthropic API)
was run to exercise the vendored adapters end to end; the second table above
and `PUBLICATION-VALIDATION.json` record them. These are smoke tests of the
plumbing, not fresh benchmark results, and they are not pooled with anything.
One of them ended on `token_budget_reached` and still graded 1.0: the frozen
workspace is evaluated whatever ended the loop (the contract's
`budget_exhaustion_scored: true`), and the termination reason is recorded so
the two cases stay distinguishable in the report.

## Judge voids

A judge member that cannot deliver a verdict set within its retry budget (a
malformed or empty reply twice in a row) voids that grading: the attempt stays
`GRADING_FAILED`, is never scored 0, and `grade` retries it. The recorded
reference episodes show the same behaviour on the same tasks (10 of 105 gold
and noop baseline episodes under the question-answering profile voided this
way). A member that *refuses* is dropped and recorded in `judge_refusals`,
down to a two-member floor, exactly as in the reference. The Claude member is
called the way the reference calls it: streamed, 32,768 output tokens,
temperature 1.0.

## Limits

- Model behaviour is stochastic and hosted model versions change; a re-run
  reproduces the protocol, not the exact numbers.
- The reference episodes ran under gVisor on Kubernetes on Poindexter's
  `python-standard:0.4` image; this package rebuilds the same pinned base for
  the agent sandbox and the evaluator on a plain Docker daemon. The 39/40
  re-grade is the evidence that grading does not move; agent-side differences
  (host CPU speed against the per-command timeouts, kernel) are possible and
  would show up as different termination reasons, which the report lists.
- Access to the reference routes (Azure deployment names, Vertex global
  endpoint, Anthropic model ids) is account-specific; `doctor` checks that
  the keys are present, not that the account can reach every model.
