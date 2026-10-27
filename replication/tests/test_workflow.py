"""Offline tests: presets, the data contract, attempt records and the report; sandbox tests only where Docker is reachable."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "vendor"))          # inside a package folder
sys.path.insert(0, str(ROOT.parent))               # inside the pdl-bench-eval repository

from pdl_bench_eval.replication import data, presets, records  # noqa: E402

DATA = Path(os.environ.get("PDL_DATA") or (ROOT / "data"))
HAVE_DATA = (DATA / "tasks.jsonl").is_file()
HAVE_DOCKER = shutil.which("docker") is not None and subprocess.run(
    ["docker", "version"], capture_output=True).returncode == 0


class PresetTests(unittest.TestCase):
    def test_every_preset_validates_and_names_vendored_assets(self):
        for name in presets.available():
            preset = presets.load(name)
            for fam in preset["families"].values():
                self.assertIn(fam["kind"], ("executable", "qna"))
                self.assertGreater(presets.budgets(fam)["agent_budget"]["max_tokens_total"], 0)

    def test_published_budgets_are_the_2m_profiles(self):
        preset = presets.load("software-delivery")
        exe = presets.family_for(preset, "executable")
        self.assertEqual(exe["profile"], "python_standard_2m_v0.1")
        self.assertEqual(presets.budgets(exe)["agent_budget"]["max_tokens_total"], 2_000_000)
        self.assertEqual(presets.budgets(exe)["timing"]["per_test_timeout_seconds"], 60)
        qna = presets.family_for(preset, "qna")
        self.assertEqual((qna["profile"], qna["scoring"]), ("python_qa_2m_v0.1", "scoring_llm_v0.3"))

    def test_required_keys_follow_the_route(self):
        preset = presets.load("software-delivery")
        cfgs = presets.resolve_models(preset, ["gpt-6-astra", "claude-fable-5-1", "gemini-4-argon"])
        keys = presets.required_keys(cfgs, {})
        self.assertEqual(keys, sorted({"AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                                       "GOOGLE_CLOUD_PROJECT"}))
        self.assertEqual(presets.required_keys(presets.resolve_models(preset, agent="gold"), {}), [])

    def test_env_aliases_from_the_other_packages(self):
        saved = {k: os.environ.pop(k, None) for k in ("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY")}
        try:
            os.environ["AZURE_OPENAI_KEY"] = "alias-value"
            presets.apply_env_aliases()
            self.assertEqual(os.environ["AZURE_OPENAI_API_KEY"], "alias-value")
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_unknown_model_or_attempts_rejected(self):
        preset = presets.load("cyber-operations")
        with self.assertRaises(ValueError):
            presets.resolve_models(preset, ["not-a-model"])
        with self.assertRaises(ValueError):
            presets.validate_config({"id": "x", "provider": "openai", "endpoint": None, "model": "m", "attempts": 9})


@unittest.skipUnless(HAVE_DATA, "no data directory (set PDL_DATA)")
class DataContractTests(unittest.TestCase):
    def test_index_matches_bundles(self):
        bench = json.loads((DATA / "manifest.json").read_text())["benchmark"]
        rows = data.load(DATA, bench)
        self.assertEqual(len(rows), json.loads((DATA / "manifest.json").read_text())["tasks"])
        self.assertTrue(all(r["task_hash"] for r in rows))

    def test_tampered_bundle_is_refused_before_spend(self):
        bench = json.loads((DATA / "manifest.json").read_text())["benchmark"]
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copytree(DATA, tmp / "data")
            row = json.loads((tmp / "data" / "tasks.jsonl").read_text().split("\n")[0])
            victim = tmp / "data" / "tasks" / row["id"] / "repository.zip"
            victim.write_bytes(victim.read_bytes() + b"\0")
            with self.assertRaises(ValueError):
                data.load(tmp / "data", bench)
            with self.assertRaises(ValueError):
                data.load(tmp / "data", "some-other-benchmark")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RecordTests(unittest.TestCase):
    def _record(self, store, task, cfg_id, attempt, status, outcome=None, reward=None, passed=False, term="agent_submits"):
        key = records.attempt_key(task, {"id": cfg_id, "provider": "x"}, attempt, "impl")
        rec = {"id": key, "task_id": task, "task_hash": task, "config": {"id": cfg_id, "provider": "x"},
               "attempt": attempt, "implementation": "impl", "status": status, "evaluation_hash": "e",
               "episode": {"termination_reason": term}}
        if outcome:
            rec["grade"] = {"outcome": outcome, "reward": reward, "passed": passed}
        store.save(rec)

    def test_report_separates_void_invalid_and_technical_failures(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            store = records.AttemptStore(tmp)
            self._record(store, "t1", "m", 1, "GRADED", "valid_scored", 1.0, True)
            self._record(store, "t1", "m", 2, "GRADED", "valid_scored", 0.5, False)
            self._record(store, "t2", "m", 1, "GRADED", "invalid_agent_fault", 0.0, False)
            self._record(store, "t2", "m", 2, "GRADED", "void_evaluator_failure", None, False)
            self._record(store, "t3", "m", 1, "SOLVER_FAILED")
            self._record(store, "t1", "other", 1, "GRADED", "valid_scored", 1.0, True)
            out = records.report(tmp)
            groups = {g["model"]: g for g in out["groups"]}
            m = groups["m"]
            self.assertEqual((m["attempts_recorded"], m["graded"], m["ungraded"]), (5, 4, 1))
            self.assertEqual((m["void"], m["valid_attempts"], m["invalid_attempts"], m["passes"]), (1, 3, 1, 1))
            self.assertAlmostEqual(m["pass_rate"], 1 / 3)
            self.assertAlmostEqual(m["mean_reward_valid_scored"], 0.75)
            self.assertEqual(m["observed_pass_at_k"]["1"]["complete_tasks"], 2)
            self.assertEqual(groups["other"]["passes"], 1)      # never pooled with m
            self.assertTrue((tmp / "report.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_attempt_key_changes_with_configuration_and_implementation(self):
        a = records.attempt_key("t", {"id": "m"}, 1, "i1")
        self.assertNotEqual(a, records.attempt_key("t", {"id": "m"}, 2, "i1"))
        self.assertNotEqual(a, records.attempt_key("t", {"id": "m2"}, 1, "i1"))
        self.assertNotEqual(a, records.attempt_key("t", {"id": "m"}, 1, "i2"))


@unittest.skipUnless(HAVE_DATA and HAVE_DOCKER, "needs the data directory and Docker")
class SandboxTests(unittest.TestCase):
    def test_noop_episode_freezes_an_empty_patch_and_records_schema_valid_episode(self):
        from pdl_bench_eval.replication.episode import run_episode
        from pdl_bench_eval.replication.grading import validate_record

        bench = json.loads((DATA / "manifest.json").read_text())["benchmark"]
        preset = presets.load(bench)
        row = next(r for r in data.load(DATA, bench) if r["family"] == "executable")
        fam = presets.family_for(preset, "executable")
        tmp = Path(tempfile.mkdtemp())
        try:
            episode = run_episode(row["bundle"], config={"id": "noop", "provider": "noop", "attempts": 1}, family=fam,
                                  out_dir=tmp / "ep", work_dir=tmp / "work", log=lambda m: None)
            self.assertEqual(episode["patch_text"], "")
            self.assertEqual(episode["termination_reason"], "agent_submits")
            self.assertIsNone(episode["outcome"])          # pending: graded afterwards
            with self.assertRaises(RuntimeError):
                validate_record(episode)                    # a pending record is never persisted as final
            clean = validate_record({**episode, "outcome": "valid_scored"})
            self.assertEqual(clean["benchmark_profile_id"], fam["profile"])
            self.assertTrue((tmp / "ep" / "trajectory.jsonl").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
