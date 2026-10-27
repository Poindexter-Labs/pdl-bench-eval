import json
import zipfile
from pathlib import Path

import pytest

from pdl_bench_eval import checkpoints, patches, pytest_counts, rubric, scoring
from pdl_bench_eval.qa_grader import direct_answers_match, normalize_direct_answer, score_criteria

DATASET = Path(__file__).resolve().parent.parent / ".dataset"   # optional local snapshot for data tests


def test_scoring_v02_progress_scaled():
    p = scoring.load_policy("scoring_v0.2")
    assert scoring.score(p, {"hidden_test_fraction": 1.0, "regression_test_fraction": 1.0, "build_quality": 1.0}) == 1.0
    assert scoring.score(p, {"hidden_test_fraction": 0.0, "regression_test_fraction": 1.0, "build_quality": 1.0}) == 0.0
    assert scoring.score(p, {"hidden_test_fraction": 0.5, "regression_test_fraction": 0.0, "build_quality": 1.0}) == pytest.approx(0.5 * (0.8 + 0.05))
    assert scoring.score(p, {"hidden_test_fraction": 1.0, "regression_test_fraction": 1.0, "build_quality": None}) == 1.0
    assert scoring.score(p, {"hidden_test_fraction": 13 / 106, "regression_test_fraction": 1.0, "build_quality": 1.0}) == 0.122642


def test_scoring_v01_plain_weighted_sum():
    p = scoring.load_policy("scoring_v0.1")
    assert scoring.score(p, {"hidden_test_fraction": 0.0, "regression_test_fraction": 1.0, "build_quality": 1.0}) == 0.2


def test_pass_threshold_is_strict():
    assert scoring.passed(0.950001) and not scoring.passed(0.95) and not scoring.passed(None)


def test_pytest_counts_sums_every_token():
    assert pytest_counts.pytest_counts("60 passed in 0.5s") == {"passed": 60, "failed": 0}
    assert pytest_counts.pytest_counts("3 failed, 57 passed, 1 error in 2s") == {"passed": 57, "failed": 4}
    assert pytest_counts.pytest_counts("Interrupted: 2 errors during collection\n2 error in 0.1s") == {"passed": 0, "failed": 4}


def test_patch_paths_and_policy():
    diff = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n--- /dev/null\n+++ b/tests/t.py\n@@ -0,0 +1 @@\n+x\n"
    assert patches.patch_paths(diff) == {"src/x.py", "tests/t.py"}
    policy = {"allowed_paths": ["src/**"], "forbidden_paths": ["tests/**"]}
    assert patches.policy_violations(patches.patch_paths(diff), policy) == ["tests/t.py"]
    with pytest.raises(ValueError):
        patches.patch_paths("@@ -1 +1 @@\n-a\n+b\n")


def test_rubric_numbered_and_rid():
    r = rubric.parse_rubric("1. First criterion\n   Weight: 2\n2. Second (3 points)\n")
    assert [c.id for c in r.criteria] == ["1", "2"] and r.criteria[0].weight == 2.0
    r2 = rubric.parse_rubric("## Criteria\n\nR1 - Verdict\n\nMET when: x.\n\nR2 (2 points) - Parents\n\n## Notes\nignored\n")
    assert [c.id for c in r2.criteria] == ["R1", "R2"] and r2.criteria[1].weight == 2.0
    items = checkpoints.derive_from_rubric("1. a (25%)\n2. b\n")
    assert [i.weight for i in items] == [25.0, 1.0]


def test_weighted_score_median_semantics():
    items = checkpoints.derive_from_rubric("1. a\n2. b\n3. c\n")
    score, gate = checkpoints.weighted_score(items, {"1": 1.0, "2": 0.5, "3": 0.0})
    assert score == 0.5 and gate == []


def test_qa_direct_match_and_rubric_score():
    assert direct_answers_match("$\\frac{1}{2}$", " \\( \\frac{1}{2} \\) ")
    assert not direct_answers_match("", "")
    assert normalize_direct_answer('"42"') == "42"
    g = score_criteria({"criteria": [{"criterion": "a", "weight": 2, "verdict": "HIT", "evidence": "", "reason": ""},
                                     {"criterion": "b", "weight": 1, "verdict": "MISS", "evidence": "", "reason": ""}]})
    assert g["score"] == pytest.approx(10 * 2 / 3)


@pytest.mark.skipif(not DATASET.exists(), reason="no local dataset snapshot at .dataset")
def test_public_rubrics_parse_to_declared_counts():
    for task_dir in sorted((DATASET / "tasks").iterdir()):
        with zipfile.ZipFile(task_dir / "hidden_tests.zip") as z:
            if "hidden_tests/rubric.md" not in z.namelist():
                continue
            text = z.read("hidden_tests/rubric.md").decode()
        r = rubric.parse_rubric(text)
        assert r.criteria, task_dir.name
        assert not [p for p in r.problems if "mixed numbering" in p], (task_dir.name, r.problems)
        if r.declared_total is not None:
            assert r.declared_total == len(r.criteria), task_dir.name


def test_calibration_file_covers_every_executable_task():
    from importlib import resources
    cal = json.loads(resources.files("pdl_bench_eval").joinpath("calibration.json").read_text())["tasks"]
    assert len(cal) == 20 and all(v["hidden_total"] > 0 for v in cal.values())


@pytest.mark.skipif(not DATASET.exists(), reason="no local dataset snapshot at .dataset")
def test_harbor_export_layout(tmp_path):
    from pdl_bench_eval.harbor_export import export
    info = export(DATASET, tmp_path / "harbor")
    assert info["tasks"] == 30
    t = tmp_path / "harbor" / "tasks" / "swe_rl_task_02"
    for rel in ("instruction.md", "task.toml", "environment/Dockerfile", "environment/app/src", "tests/test.sh",
                "tests/grade.py", "tests/grade.json", "tests/hidden", "tests/fixture_index.json", "solution/solve.sh", "solution/gold.diff"):
        assert (t / rel).exists(), rel
    cfg = json.loads((t / "tests" / "grade.json").read_text())
    assert cfg["hidden_total"] == 16 and cfg["weights"] == {"h": 0.8, "r": 0.15, "b": 0.05}
    assert (tmp_path / "harbor" / "dataset.toml").read_text().count("[[tasks]]") == 30
    import tomllib
    q = tmp_path / "harbor" / "tasks" / "pdl_qna_close_provenance_010"
    cfg = tomllib.loads((q / "task.toml").read_text())
    assert cfg["artifacts"] == ["/app"] and cfg["verifier"]["environment_mode"] == "separate"
    assert (q / "tests" / "Dockerfile").exists() and (q / "tests" / "judge.py").exists()
    assert "artifacts" not in cfg.get("metadata", {})


def test_judge_panel_drops_a_refusing_member_but_voids_on_failure(monkeypatch):
    from pdl_bench_eval import judges
    from pdl_bench_eval.checkpoints import derive_from_rubric
    from pdl_bench_eval.scoring import load_policy

    policy = load_policy("scoring_llm_v0.3")
    items = derive_from_rubric("1. Names the module. (1 point)\n2. States the value. (1 point)")
    ids = [it.id for it in items]
    ok = '{"results": [' + ",".join(f'{{"id": "{i}", "pass": true, "evidence": "x"}}' for i in ids) + "]}"

    def fake(member, system, user, schema=None, reasoning=None):
        if member.model == "refuser":
            raise judges.JudgeRefusal("declined")
        if member.model == "broken":
            raise judges.JudgeError("garbage")
        return ok, member.model + "-v1"

    monkeypatch.setattr(judges, "complete", fake)
    monkeypatch.setattr(judges, "JUDGE_RETRY_BACKOFF_S", 0)
    members = [judges.JudgeMember("openai", "refuser"), judges.JudgeMember("openai", "a"), judges.JudgeMember("openai", "b")]
    out = judges.judge_checkpoints(policy, members, "brief", items, "deliverable")
    assert [m["model"] for m in out["judge_panel"]] == ["a", "b"]
    assert out["judge_refusals"][0]["model"] == "refuser" and out["aggregated"] == 1.0
    with pytest.raises(judges.JudgeError):     # two refusals leave one member: below the floor
        judges.judge_checkpoints(policy, [judges.JudgeMember("openai", "refuser"), judges.JudgeMember("openai", "refuser"),
                                          judges.JudgeMember("openai", "a")], "brief", items, "deliverable")
    with pytest.raises(judges.JudgeError):     # a member that cannot deliver voids, even with others present
        judges.judge_checkpoints(policy, [judges.JudgeMember("openai", "broken"), judges.JudgeMember("openai", "a"),
                                          judges.JudgeMember("openai", "b")], "brief", items, "deliverable")


def test_graded_episode_record_matches_reference_schema():
    """A judged grade maps onto the reference record schema (panel entries, scores, refusals)."""
    from pdl_bench_eval.replication import grading

    record = {"episode_id": "ep-x", "task_id": "t", "task_version": 1, "model_provider": "scripted", "model_name": "gold",
              "model_version": "v", "model_parameters": {}, "prompt_version": "scripted_v0", "tool_schema_version": "tools_v0.1",
              "sandbox_backend": "docker", "environment_image_digest": "sha256:" + "0" * 64, "evaluator_version": "x",
              "evaluator_image_digest": "sha256:" + "0" * 64, "scoring_version": "scoring_llm_v0.3",
              "benchmark_profile_id": "python_qa_2m_v0.1", "random_seed": 1, "start_time": "2026-10-27T00:00:00+00:00",
              "end_time": "2026-10-27T00:01:00+00:00", "outcome": None, "termination_reason": "agent_submits",
              "reward_total": None, "hidden_tests_passed": None, "hidden_tests_total": None, "property_gate_passed": None,
              "property_gate_total": None, "regression_score": None, "build_quality_score": None, "tokens_in": 0,
              "tokens_out": 0, "tool_calls": 2, "model_cost": 0, "compute_cost": None, "judge_model": None,
              "judge_model_version": None, "judge_prompt_version": None, "judge_scores": None, "judge_panel": None,
              "judge_refusals": None, "pr_state": None, "condition": None, "success": None, "upstream_state": None,
              "trajectory_uri": "t.jsonl", "patch_uri": "p.diff", "logs_uri": "l.txt", "patch_text": ""}
    grade = {"outcome": "valid_scored", "reward": 0.9, "image_id": "sha256:" + "1" * 64, "property_gate_passed": 0,
             "property_gate_total": 2, "judge_model": "panel:a,b",
             "judge": {"judge_prompt_version": "judge_prompt_v0.2", "aggregated": 0.9,
                       "judge_panel": [{"model": "a", "provider": "google", "model_version": "a-1", "score": 1.0},
                                       {"model": "b", "provider": "openai", "model_version": None, "score": 0.8}],
                       "judge_refusals": [{"model": "c", "provider": "anthropic", "reason": "declined"}]}}
    # exercise the mapping without Docker: apply the same field updates grade_episode makes
    judge, panel = grade["judge"], grade["judge"]["judge_panel"]
    record.update({"outcome": grade["outcome"], "reward_total": grade["reward"], "evaluator_image_digest": grade["image_id"],
                   "property_gate_passed": 0, "property_gate_total": 2, "judge_model": grade["judge_model"],
                   "judge_prompt_version": judge["judge_prompt_version"],
                   "judge_panel": [grading._panel_entry(m) for m in panel], "judge_refusals": judge["judge_refusals"],
                   "judge_scores": [m["score"] for m in panel]})
    clean = grading.validate_record(record)
    assert clean["judge_panel"][0] == {"model": "a", "provider": "google", "endpoint": None, "model_version": "a-1",
                                       "scores": [1.0], "median": 1.0}
    assert clean["judge_scores"] == [1.0, 0.8] and "patch_text" not in clean


def test_docker_cli_failure_is_an_evaluator_fault(monkeypatch):
    import subprocess
    from pdl_bench_eval import docker_env

    class P:
        returncode, stdout, stderr = 125, "", "docker: Error response from daemon: denied: path is not shared from the host"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: P())
    monkeypatch.setattr(docker_env, "host_path", lambda p: str(p))
    with pytest.raises(docker_env.EvaluatorError):
        docker_env.DockerEnv("img").evaluate("true", Path("/tmp/r"), Path("/tmp/h"), {}, 10)
