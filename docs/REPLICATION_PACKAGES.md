# The SWE replication packages (maintainer guide)

`environments/software-delivery/` and `environments/cyber-operations/` in the
dataset are generated from this repository. They follow the conventions of the
non-SWE packages (see `docs/REPLICATION_ALIGNMENT.md`): one self-contained
folder per benchmark, a controller image with `plan` / `doctor` / `run` /
`grade` / `report`, presets as the authority, settings evidence, attempt
records keyed by task, configuration, attempt and implementation, and a
report that never pools.

## Pieces

| Piece | Where | Regenerate with |
| --- | --- | --- |
| the agent loop exported from the RL repository | `pdl_bench_eval/_vendor/` | `python tools/vendor_rl.py --rl-root "/path/to/PD RL Environment"` |
| the runner (`plan`/`doctor`/`run`/`grade`/`report`/`build-data`) | `pdl_bench_eval/replication/` | edit; `pdl-bench-replicate` is the console script |
| presets | `pdl_bench_eval/replication/presets/*.json` | edit by hand; `selection_evidence` names the console jobs behind the report |
| agent sandbox images | `pdl_bench_eval/replication/sandbox/*/Dockerfile` | built on demand by `doctor`/`run` (`pdl-bench/swe-agent:1.0.0`, `pdl-bench/sweqna-agent:1.0.0`) |
| settings evidence | `replication/settings-evidence/*.json` | `python tools/settings_evidence.py --dataset-dir .dataset --out replication/settings-evidence` (needs `bq`) |
| package scaffolding and documents | `replication/` (`Dockerfile`, `bench.py`, `.env.example`, `requirements.lock`, `docs/`, `tests/`) | edit; `requirements.lock` is `pip freeze` of a clean venv with `.[judges]` |
| the packages | `dist/environments/<benchmark>/` | `python tools/bundle_swe.py --dataset-dir .dataset --out dist/environments --validation replication/validation.json` |

The vendored RL commit, the per-file hashes and the adaptations are in
`pdl_bench_eval/_vendor/SOURCE_MANIFEST.json`; the bundle tool copies this
package under `vendor/pdl_bench_eval` and writes `vendor/SOURCE_MANIFEST.json`
with both provenances.

## Running without Docker-in-Docker

The controller can also run on the host: `pip install -e .[judges]`, then

```sh
pdl-bench-replicate build-data --preset software-delivery --dataset-dir .dataset --data /tmp/sd
pdl-bench-replicate plan   --preset software-delivery --data /tmp/sd
pdl-bench-replicate run    --preset software-delivery --data /tmp/sd --results /tmp/sd-results --agent gold --limit 1
```

## Validation before publishing

1. `python -m unittest discover -s replication/tests -v` with `PDL_DATA` pointing at a built data directory (sandbox test needs Docker).
2. Gold and noop through the full path on one executable task (1.0 / 0.0) and one QnA task with the reference panel (1.0 / 0.0).
3. Build the controller image from `dist/environments/<benchmark>` and run `plan` with `--network none` and the test suite inside it.
4. One paid episode per route (one task, one attempt) with real keys.
5. Record all of it in `replication/docs/VALIDATION.md` and `replication/validation.json`, then rebuild the packages.

## Publishing

Upload `dist/environments/software-delivery` and `dist/environments/cyber-operations`
to the dataset under `environments/`, add the two rows to `environments/README.md`
(the rendered version is `replication/docs/environments-README.md`), and pin
`source_revision` in each `data/manifest.json` to the dataset commit the data
came from. Pushes to the Hub happen only after explicit approval.
