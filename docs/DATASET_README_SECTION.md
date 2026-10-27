
## Evaluating against PDL-Bench

The reference evaluator, [`pdl-bench-eval`](https://github.com/Poindexter-Labs/pdl-bench-eval),
grades submissions exactly as the published results were graded (same scoring policies, test-count
rules, judge prompts and panel, pass threshold) and documents the jobs, profiles and prompts behind
every figure. The 30 repository tasks are also packaged under [`harbor/`](./harbor) in Harbor's
task format, so any Harbor-compatible harness can run them:

```bash
pip install "pdl-bench-eval[judges] @ git+https://github.com/Poindexter-Labs/pdl-bench-eval"
pdl-bench-eval grade-repo --task swe_rl_task_02 --patch my_patch.diff
harbor run --repo https://huggingface.co/datasets/Poindexter-Labs/PDL-Bench -p harbor -a <agent> -m <model> -k 8
```
