# Dataset and grading protocol

This package contains the @@TASKS@@ tasks of the `@@BENCH@@` mini-benchmark of
PDL-Bench, pinned to dataset revision `@@REVISION@@` (snapshot @@SNAPSHOT_DATE@@).
Bundle bytes match that release; `data/tasks.jsonl` carries every file's
SHA-256 and `plan` checks them before anything runs. Public task ids are used
in results. The solver never receives the hidden suite, the rubric or the
reference patch.

Grading is the reference evaluator (`vendor/pdl_bench_eval`), a port of the
code that produced the published numbers, validated by re-grading recorded
episodes (see its `docs/REPRODUCTION.md`).

Two settings differ from what the task manifests pin, because the published
report was produced with operator overrides and this package reproduces the
report, not the manifest defaults:

- Executable tasks pin `python_standard_1m_v0.1`; the published runs used
  `python_standard_2m_v0.1` (same image digest, twice the token budget). The
  preset names the 2M profile and every episode record says so.
- Question-answering tasks pin `scoring_llm_v0.2`; the published runs were
  graded under `scoring_llm_v0.3` (itemised checkpoints, per-item panel
  median). The preset names v0.3.

Earlier episodes on these tasks under other settings (`scoring_v0.1`, an
earlier image digest, a local sandbox, `scoring_llm_v0.2`) are listed in
`settings-evidence/@@BENCH@@.json` with counts and dates. They are evidence of
what was run, not part of the published result set, and `report` never pools
them with anything.

One reference patch was repaired in the dataset revision this package pins:
`contributor_task_08/gold.diff` had lost its final context line (a
whitespace-only line stripped at export), so `git apply` rejected it while GNU
`patch` tolerated it. The missing line was restored; the patch is otherwise
byte-identical, applies cleanly, and scores 93 of 93 hidden tests. Recorded
agent episodes were never affected (agents write their own patches); only the
scripted gold baseline through the `write_patch` tool was.

Two tasks changed after some recorded episodes (`swe_rl_cybersecurity_task_07`
had its fixture revised; `swe_rl_cybersecurity_task_03`'s hidden suite grew
from 60 to 106 tests). This package carries the versions in the pinned
dataset revision; the published report should be read against those versions.
