"""One replication episode: the reference agent loop in a Docker sandbox, then the validated grader.

A port of the reference runner's ``run_episode`` restricted to what the
published repository tasks use: stage the fixture, run the task's setup
command, take a tamper-proof git baseline, run the agent through the vendored
toolbox under the profile's budgets, freeze the workspace into a patch with the
same invariants, then grade that patch with ``pdl_bench_eval``'s graders. The
pull-request, DynamicSWE and Kubernetes paths of the reference runner are not
needed by these tasks and are not here.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


from .. import EVALUATOR_ID
from .._vendor.pd_rl import images as images_mod
from .._vendor.pd_rl.sandbox import SandboxError, make_sandbox
from .._vendor.pd_rl.toolbox import ToolBox, Trajectory, _TERMINATION_REASONS, _changed_files
from ..bundles import TaskBundle, unpack
from . import SANDBOX_ROOT, assets
from .agents import _AgentFault, _Submit, make_agent

CONTRACT_SUPPORTED = ("evaluation_contract_v0.2",)
RECORD_OUTCOMES = ("valid_scored", "invalid_agent_fault", "void_setup_failure", "void_infra_failure",
                   "void_evaluator_failure")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def effective_limits(profile: dict, manifest: dict) -> dict:
    """Profile sections shadowed by any manifest overrides, as the reference runner merges them."""
    limits = {}
    overrides = manifest.get("overrides") or {}
    for section in ("agent_budget", "timing", "resources", "terminal_limits"):
        merged = dict(profile.get(section) or {})
        merged.update(overrides.get(section) or {})
        limits[section] = merged
    return limits


def load_task_images(fixture_dir: Path, manifest: dict) -> list:
    rels = images_mod.task_image_paths(manifest)
    if len(rels) > images_mod.MAX_TASK_IMAGES:
        raise images_mod.ImageError(f"issue.images lists {len(rels)} images; at most {images_mod.MAX_TASK_IMAGES}")
    out = []
    for rel in rels:
        path = (Path(fixture_dir) / rel).resolve()
        if Path(fixture_dir).resolve() not in path.parents or not path.is_file():
            raise images_mod.ImageError(f"issue.images entry {rel!r} is not a file in the fixture")
        out.append(images_mod.from_bytes(rel, path.read_bytes()))
    return out


def specs_for(task: TaskBundle, family: dict) -> dict:
    manifest = task.manifest
    contract_id = (manifest.get("contract") or {}).get("evaluation_contract")
    if contract_id not in CONTRACT_SUPPORTED:
        raise ValueError(f"{task.task_id}: contract {contract_id!r} is not one this runner carries")
    contract = assets.contract(contract_id)
    profile = assets.profile(family["profile"])
    tools = assets.tools(contract["agent_interface"]["tool_schema_reference"])
    prompt_spec = assets.prompt(family["prompt"])
    eco = profile.get("ecosystem") or {}
    env_family = eco.get("task_family") or "swe"
    if env_family not in (prompt_spec.get("environments") or ["swe"]):
        raise ValueError(f"prompt {family['prompt']} declares environments {prompt_spec.get('environments')}, "
                         f"but profile {family['profile']} is the {env_family} environment")
    return {"manifest": manifest, "contract": contract, "profile": profile, "tools": tools,
            "scoring": {"scoring_policy_id": family["scoring"]}, "prompt": prompt_spec}


def run_episode(task: TaskBundle, *, config: dict, family: dict, out_dir: Path, work_dir: Path,
                sandbox_root: Path = SANDBOX_ROOT, log=print) -> dict:
    """Run the agent; return the episode record (schema-valid) with ``patch_text`` beside it.

    The record's ``outcome`` is one of the contract's classes once known at this
    stage (``invalid_agent_fault``, ``void_setup_failure``, ``void_infra_failure``);
    an episode that reached a frozen patch has outcome ``None`` and is graded by
    ``grade_episode`` afterwards.
    """
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs_for(task, family)
    manifest, profile, contract = specs["manifest"], specs["profile"], specs["contract"]
    limits = effective_limits(profile, manifest)
    repo_dir, _hidden = unpack(task, work_dir)
    task_images = load_task_images(repo_dir, manifest)
    agent = make_agent(config, family["prompt"], gold_path=task.task_dir / "gold.diff")
    if task_images and hasattr(agent, "adapter") and not getattr(agent, "accepts_images", False):
        raise ValueError(f"{task.task_id} shows the model {len(task_images)} image(s), but "
                         f"{agent.name}:{agent.model_name} cannot view images")
    episode_id = f"ep-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{random.randrange(16**4):04x}"
    random_seed = random.randrange(2**31)
    start_time = _now()
    logs = []

    def note(msg):
        logs.append(msg)
        log(msg)

    trajectory = Trajectory(out_dir / "trajectory.jsonl", episode_id, assets.trajectory_schema())
    sandbox = make_sandbox("docker", resources=limits["resources"], image=family["sandbox_image"])
    outcome = termination_reason = None
    reward = None
    patch_text = ""
    toolbox = None
    trajectory.emit("episode_start", {
        "task_id": manifest["task_id"], "task_version": manifest["task_version"],
        "benchmark_profile_id": profile["profile_id"], "model_provider": agent.provider,
        "model_name": agent.model_name, "random_seed": random_seed,
        **({"issue_images": [img.describe() for img in task_images]} if task_images else {}),
    })
    try:
        # --- stage + setup (failure here = void_setup_failure) ---------------
        sandbox.start(repo_dir, root=sandbox_root)
        setup_timeout = limits["timing"].get("setup_timeout_minutes", 15) * 60
        setup_env = ((profile.get("ecosystem") or {}).get("dependency_install") or {}).get("setup_env") or {}
        rc, out = sandbox.run(manifest["environment"]["setup_command"], timeout=setup_timeout, env=setup_env)
        if rc != 0:
            note(f"[setup] FAILED (rc={rc}):\n{out[-2000:]}")
            outcome, termination_reason = "void_setup_failure", None
            trajectory.emit("error", {"message": f"setup failed rc={rc}", "recoverable": False})
            raise _Finish()
        rc, out = sandbox.git("init -q")
        if rc != 0:
            raise SandboxError(f"git init failed: {out}")
        ignore = (profile.get("ecosystem") or {}).get("workspace_ignore") or ["__pycache__/", "*.pyc", ".pytest_cache/"]
        sandbox.write_git_exclude("\n".join(ignore) + "\n")
        for args in ("-c user.email=runner@pdl -c user.name=pdl-runner add -A",
                     "-c user.email=runner@pdl -c user.name=pdl-runner commit -qm baseline"):
            rc, out = sandbox.git(args)
            if rc != 0:
                raise SandboxError(f"baseline commit failed: {out}")
        note(f"[setup] ok (sandbox={sandbox.kind}, agent={agent.name})")

        # --- agent loop ------------------------------------------------------
        agent_deadline = time.monotonic() + limits["timing"].get("agent_timeout_minutes", 45) * 60
        toolbox = ToolBox(sandbox, specs, limits, trajectory, agent_deadline=agent_deadline)
        toolbox.agent_accepts_images = getattr(agent, "accepts_images", True)
        ctx = {"task_dir": task.task_dir, "issue_text": manifest["issue"]["text"], "root": assets.ASSETS,
               "condition": None, "controller": None, "manifest": manifest, "contract": contract,
               "limits": limits, "specs": specs, "images": task_images}
        try:
            toolbox.turn = 1
            agent.run(ctx, toolbox)
            raise _AgentFault("agent returned without submitting")
        except _Submit as stop:
            reason = str(stop)
            termination_reason = reason if reason in _TERMINATION_REASONS else "max_turns_reached"
        note(f"[agent] finished: {termination_reason} ({toolbox.tool_calls} tool calls, {toolbox.shell_commands} shell)")

        # --- freeze + validity ----------------------------------------------
        rc, out = sandbox.git("add -A", timeout=120)
        if rc != 0:
            raise SandboxError(f"staging for patch extraction failed: {out}")
        rc, patch_text = sandbox.git("diff --cached --binary HEAD", timeout=120)
        if rc != 0:
            raise SandboxError(f"patch extraction failed: {patch_text}")
        rc, staged = sandbox.git("diff --cached --name-only HEAD", timeout=60)
        if rc != 0:
            raise SandboxError(f"patch verification failed: {staged}")
        staged_paths = {p.strip() for p in staged.splitlines() if p.strip()}
        changed_paths = set(_changed_files(sandbox))
        if staged_paths != changed_paths:
            raise SandboxError("recorded patch does not cover the workspace — "
                               f"missing from patch: {sorted(changed_paths - staged_paths)}; "
                               f"in patch but unreported: {sorted(staged_paths - changed_paths)}")
        stray = sandbox.reap_stray_processes()
        if stray:
            note(f"[validity] background processes survived the episode: {stray}")
            outcome, reward = "invalid_agent_fault", 0
            trajectory.emit("error", {"message": f"background processes survived the episode: {stray}",
                                      "recoverable": False})
            raise _Finish()
        patch_cap_mib = limits["terminal_limits"].get("max_patch_mib", 5)
        if len(patch_text.encode()) > patch_cap_mib * 1024 * 1024:
            note(f"[validity] extracted patch exceeds max_patch_mib={patch_cap_mib}")
            outcome, reward = "invalid_agent_fault", 0
            trajectory.emit("error", {"message": f"patch exceeds max_patch_mib={patch_cap_mib}", "recoverable": False})
            raise _Finish()
        note(f"[freeze] patch of {len(patch_text):,} characters over {len(changed_paths)} path(s)")
    except _Finish:
        pass
    except _AgentFault as exc:
        note(f"[agent] unrecoverable error: {exc}")
        outcome, termination_reason, reward = "invalid_agent_fault", "unrecoverable_agent_error", 0
        trajectory.emit("error", {"message": str(exc), "recoverable": False})
    except SandboxError as exc:
        note(f"[infra] failure: {exc}")
        outcome, termination_reason = "void_infra_failure", "infrastructure_failure"
        trajectory.emit("error", {"message": str(exc), "recoverable": False})
    except Exception as exc:  # noqa: BLE001 - a platform or provider fault records a void episode
        note(f"[infra] unexpected error: {type(exc).__name__}: {exc}")
        outcome, termination_reason = "void_infra_failure", "infrastructure_failure"
        trajectory.emit("error", {"message": f"unexpected {type(exc).__name__}: {exc}", "recoverable": False})
    finally:
        env_digest = sandbox.env_digest()
        sandbox.close()

    trajectory.emit("episode_end", {"termination_reason": termination_reason, "outcome": outcome or "pending_grade"})
    trajectory.close()
    (out_dir / "patch.diff").write_text(patch_text or "")
    (out_dir / "logs.txt").write_text("\n".join(logs) + "\n")
    record = {
        "episode_id": episode_id,
        "task_id": manifest["task_id"],
        "task_version": manifest["task_version"],
        "model_provider": agent.provider,
        "model_name": agent.model_name,
        "model_version": agent.model_version,
        "model_parameters": agent.model_parameters,
        "prompt_version": agent.prompt_version,
        "tool_schema_version": contract["agent_interface"]["tool_schema_reference"],
        "sandbox_backend": sandbox.kind,
        "environment_image_digest": env_digest,
        "evaluator_version": EVALUATOR_ID,
        "evaluator_image_digest": env_digest,       # replaced by the evaluator image when graded
        "scoring_version": specs["scoring"]["scoring_policy_id"],
        "benchmark_profile_id": profile["profile_id"],
        "random_seed": random_seed,
        "start_time": start_time,
        "end_time": _now(),
        "outcome": outcome,
        "termination_reason": termination_reason,
        "reward_total": reward,
        "hidden_tests_passed": None, "hidden_tests_total": None,
        "property_gate_passed": None, "property_gate_total": None,
        "regression_score": None, "build_quality_score": None,
        "tokens_in": agent.usage["tokens_in"],
        "tokens_out": agent.usage["tokens_out"],
        "tool_calls": toolbox.tool_calls if toolbox else 0,
        "model_cost": 0 if agent.provider == "scripted" else None,
        "compute_cost": None,
        "judge_model": None, "judge_model_version": None, "judge_prompt_version": None,
        "judge_scores": None, "judge_panel": None, "judge_refusals": None,
        "pr_state": None, "condition": None, "success": None, "upstream_state": None,
        "trajectory_uri": str(out_dir / "trajectory.jsonl"),
        "patch_uri": str(out_dir / "patch.diff"),
        "logs_uri": str(out_dir / "logs.txt"),
    }
    record["patch_text"] = patch_text   # stripped before schema validation / persistence of the record proper
    return record


class _Finish(Exception):
    """Internal: leave the staged try-block with the outcome already decided."""
