# Run @@BENCH@@

@@TASKS@@ tasks. Start in this directory with Docker installed and running; the
controller drives your Docker daemon through its socket to start one sandbox
per episode. The complete download command is in README.md. Supply only your
own credentials. @@QNA_START@@

```sh
cp .env.example .env
# Fill the keys required by the models and judges you will run (`plan` lists them).
docker build -t poindexter/mini-bench:@@BENCH@@ .
mkdir -p results
# Offline validation, no provider calls:
docker run --rm --network none -v "$PWD/data:/data:ro" poindexter/mini-bench:@@BENCH@@ plan
# Keys, imports, Docker and the sandbox/evaluator images, still no provider calls:
docker run --rm --env-file .env -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/data:/data:ro" poindexter/mini-bench:@@BENCH@@ doctor
# Free calibration: the reference patch on the first executable task must score 1.0:
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD/data:/data:ro" \
  -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ run --agent gold --family executable --limit 1
# Paid run: one task, one attempt, one model:
docker run --rm --env-file .env -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/data:/data:ro" -v "$PWD/results:/results" poindexter/mini-bench:@@BENCH@@ \
  run --model gpt-6-astra --limit 1 --attempts 1
```

Omit `--limit 1 --attempts 1` for all tasks and eight attempts for the
selected model; omit `--model` too for all preset models. Add `--parallel N`
to run N episodes at once. README.md covers cost, Linux file ownership,
regrading, retries, result reports and the judge panel. DATASET-NOTES.md
records the exact task and grading versions. Keep `.env` private; never
commit it.
