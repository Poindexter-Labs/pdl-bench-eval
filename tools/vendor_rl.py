#!/usr/bin/env python3
"""Export the agent loop of the PD RL Environment repository into pdl_bench_eval/_vendor.

Dependency-closed and explicit, after the pattern of pointdexter's
tools/replication/vendor.py: named functions and classes are extracted with
their transitive module-level dependencies, imports are trimmed to what the
extracted code references, and SOURCE_MANIFEST.json records the source commit,
every source file's hash, every emitted file's hash and the adaptations.

What is exported: the provider-independent agent loop (ModelAgent), the three
provider adapters plus the Vertex MaaS adapter, the toolbox that executes
tools_v0.1 calls against a sandbox, the Docker sandbox backend, the trajectory
writer, the scripted-agent base, the image helper, and the versioned assets
(prompts, tool schema, contract, run profiles, record schemas).

What is NOT exported: the Kubernetes sandbox, GCS/BigQuery publishing, Secret
Manager access, the console watcher, pull-request and DynamicSWE machinery, the
judge (pdl_bench_eval ships its own validated port), state/document graders.

Usage: python tools/vendor_rl.py --rl-root "/path/to/PD RL Environment"
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import symtable
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "pdl_bench_eval"
DEST = PKG / "_vendor"
HEADER = "# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {a.asname or a.name.split(".")[0] for a in node.names}
    return set()


def dependencies(node):
    def visit(table):
        found = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
        for child in table.get_children():
            found |= visit(child)
        return found

    return visit(symtable.symtable(ast.unparse(node), "export", "exec"))


def subset(text: str, seeds, overrides=None) -> str:
    """Source for `seeds` and everything they reference at module level."""
    tree = ast.parse(text)
    lookup = {name: node for node in tree.body for name in names(node)}
    overrides = overrides or {}
    wanted = set(seeds)
    pending = list(seeds)
    while pending:
        name = pending.pop()
        if name in overrides or name not in lookup:
            continue
        for dep in dependencies(lookup[name]):
            if dep not in wanted:
                wanted.add(dep)
                pending.append(dep)
    missing = [s for s in seeds if s not in lookup and s not in overrides]
    if missing:
        raise SystemExit(f"seeds not found: {missing}")
    future, sections = [], []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future.append(ast.get_source_segment(text, node))
            continue
        selected = (names(node) & wanted) - overrides.keys()
        if not selected:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            node.names = [a for a in node.names if (a.asname or a.name.split(".")[0]) in selected]
            sections.append(ast.unparse(node))
        else:
            sections.append(ast.get_source_segment(text, node))
    sections = [v for k, v in overrides.items() if k in wanted] + sections
    return HEADER + "\n".join(future) + ("\n\n" if future else "") + "\n\n".join(dict.fromkeys(sections)) + "\n"


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(text, node)
            return text.replace(segment, replacement.rstrip("\n"))
    raise SystemExit(f"function {name} not found")


def rewrite_imports(code: str) -> str:
    code = re.sub(r"^(\s*)from pd_rl(\.|\s+import)", r"\1from pdl_bench_eval._vendor.pd_rl\2", code, flags=re.M)
    code = re.sub(r"^(\s*)import pd_rl\b", r"\1import pdl_bench_eval._vendor.pd_rl as pd_rl", code, flags=re.M)
    return code


GCP_SECRET_STUB = '''def gcp_secret(name: str) -> str:
    """Replication build: credentials come from environment variables only.

    The reference runner falls back to Google Secret Manager; a researcher's
    copy never does, so a missing key is reported by the caller as missing.
    """
    return ""
'''

SANDBOX_OVERRIDES = {
    "IMAGES": '''# Replication build: the agent runs in images derived from the public task
# images (pdl-bench/swe:1.0.0, pdl-bench/sweqna:1.0.0) with the `agent` user and
# keeper process the reference image provides. Built from sandbox/<kind>/.
IMAGES = {
    "python-standard": ("pdl-bench/swe-agent:1.0.0", "sandbox/swe"),
    "python-qa": ("pdl-bench/sweqna-agent:1.0.0", "sandbox/sweqna"),
}''',
    "DEFAULT_IMAGE": 'DEFAULT_IMAGE = "python-standard"',
    "make_sandbox": '''def make_sandbox(kind: str, resources=None, image_digest=None, image=None):
    """Replication build: the Docker backend is the only sandbox."""
    if kind != "docker":
        raise ValueError(f"unsupported sandbox kind for replication: {kind!r} (docker only)")
    return DockerSandbox(resources, image=image)''',
}

ASSETS = [
    "prompts/agent_prompt_v0.3.yaml",
    "prompts/agent_prompt_qa_v0.1.yaml",
    "contracts/tools_v0.1.yaml",
    "contracts/evaluation_contract_v0.2.yaml",
    "profiles/python_standard_2m_v0.1.yaml",
    "profiles/python_qa_2m_v0.1.yaml",
    "schemas/trajectory_event.schema.json",
    "schemas/episode_record.schema.json",
]
COPIED = [
    "pd_rl/images.py",
    "pd_rl/providers/__init__.py",
    "pd_rl/providers/openai_provider.py",
    "pd_rl/providers/anthropic_provider.py",
    "pd_rl/providers/vertex_maas.py",
]
ADAPTATIONS = [
    "Imports rewritten from `pd_rl` to `pdl_bench_eval._vendor.pd_rl`; nothing else in the copied adapters changes.",
    "providers/base.py: gcp_secret() returns '' so credentials come from environment variables only (no Secret Manager).",
    "providers/google.py: the Vertex project default is empty; GOOGLE_CLOUD_PROJECT must be set explicitly.",
    "sandbox.py: only the Docker backend is exported; the image table points at the replication agent images built from the public task images.",
    "toolbox.py: the Trajectory writer, ToolBox, patch-format check and changed-files helper are extracted from runner.py; the episode orchestration is reimplemented in pdl_bench_eval.replication.episode and the evaluator is pdl_bench_eval's validated port.",
    "agents.py: only the exceptions, the scripted-agent base and the noop agent are exported; the replication gold agent (write_patch of gold.diff, then submit) is defined in pdl_bench_eval.replication.agents.",
    "No pull-request, DynamicSWE, Kubernetes, GCS, BigQuery, console or judge code is exported.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rl-root", required=True)
    args = ap.parse_args()
    root = Path(args.rl_root).expanduser().resolve()
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "pd_rl" / "providers").mkdir(parents=True)
    (DEST / "pd_rl_assets").mkdir()
    (DEST / "__init__.py").write_text(HEADER + '"""Code exported from the PD RL Environment repository (see SOURCE_MANIFEST.json)."""\n')
    (DEST / "pd_rl" / "__init__.py").write_text(HEADER + '"""The reference agent loop, exported."""\n')
    sources, outputs = {}, {}

    def src(rel: str) -> str:
        data = (root / rel).read_bytes()
        sources[rel] = digest(data)
        return data.decode("utf-8-sig")

    def emit(rel: str, code: str):
        path = DEST / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        outputs[rel] = digest(code.encode())

    # whole-module copies (import paths rewritten)
    for rel in COPIED:
        emit(rel, HEADER + rewrite_imports(src(rel)))
    # providers/base.py: whole module, Secret Manager stubbed out
    base = replace_function(src("pd_rl/providers/base.py"), "gcp_secret", GCP_SECRET_STUB)
    emit("pd_rl/providers/base.py", HEADER + rewrite_imports(base))
    # providers/google.py: no implicit project
    google = src("pd_rl/providers/google.py")
    needle = 'PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "poindexter-labs-rl-environment")'
    if needle not in google:
        raise SystemExit("google.py: PROJECT line changed; update the exporter")
    google = google.replace(needle, 'PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")')
    emit("pd_rl/providers/google.py", HEADER + rewrite_imports(google))
    # subsets
    emit("pd_rl/agents.py", rewrite_imports(subset(src("pd_rl/agents.py"),
         ["_Submit", "_AgentFault", "ScriptedAgent", "NoopAgent"])))
    emit("pd_rl/sandbox.py", rewrite_imports(subset(src("pd_rl/sandbox.py"),
         ["SandboxError", "DockerSandbox", "make_sandbox", "WORKDIR", "HIDDEN_MOUNT", "GIT_DIR",
          "ORIGIN_PATH", "MERGE_PATH"], SANDBOX_OVERRIDES)))
    emit("pd_rl/toolbox.py", rewrite_imports(subset(src("pd_rl/runner.py"),
         ["_now", "Trajectory", "_patch_format_error", "_TERMINATION_REASONS", "ToolBox",
          "_looks_binary", "_changed_files", "SETUP_FAILED_RC"])))
    # assets
    for rel in ASSETS:
        data = (root / rel).read_bytes()
        sources[rel] = digest(data)
        target = DEST / "pd_rl_assets" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        outputs["pd_rl_assets/" + rel] = digest(data)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *sources], cwd=root, text=True)
    manifest = {
        "source_repository": "Poindexter-Labs/PD-RL-Environment",
        "source_commit": commit,
        "source_files_with_local_modifications": [line[3:] for line in dirty.splitlines() if line.strip()],
        "sources": dict(sorted(sources.items())),
        "files": dict(sorted(outputs.items())),
        "adaptations": ADAPTATIONS,
    }
    (DEST / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"source_commit": commit, "sources": len(sources), "files": len(outputs),
                      "dirty": manifest["source_files_with_local_modifications"]}, indent=2))


if __name__ == "__main__":
    main()
