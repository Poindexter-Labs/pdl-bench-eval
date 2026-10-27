"""Fresh-container evaluation, mirroring the reference runner's docker backend.

Every evaluator command runs in its own ``docker run --rm`` container with the
network disabled, the graded tree mounted at /workspace and the hidden bundle
mounted read-only at /hidden-tests. A task's ``setup_command`` is prefixed to
every command (containers are fresh, so nothing installed for one command
survives to the next); a failing setup is an infrastructure fault that voids the
evaluation rather than scoring it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

WORKDIR = "/workspace"
HIDDEN_MOUNT = "/hidden-tests"
SETUP_FAILED_RC = 97
IMAGE_TAGS = {"swe": "pdl-bench/swe:1.0.0", "sweqna": "pdl-bench/sweqna:1.0.0"}


class SetupFailed(RuntimeError):
    """The task's setup_command failed inside the evaluator container."""


class EvaluatorError(RuntimeError):
    """The evaluator container could not run at all (daemon, mount or image fault): void, never a 0."""


DOCKER_CLI_FAILURE_RCS = (125, 126, 127)    # docker run failed / command not invocable / not found


_MOUNTS = None


def container_mounts() -> list:
    """(destination, source) pairs of this process's own container, if it runs in one with the Docker socket.

    When the evaluator runs inside a controller container and talks to the host daemon, every
    ``-v`` source it names is resolved on the HOST. A path under one of our own bind mounts is
    translated to its host source; any other path would arrive empty, so it is refused.
    """
    global _MOUNTS
    if _MOUNTS is not None:
        return _MOUNTS
    _MOUNTS = []
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        import json
        import socket

        p = subprocess.run(["docker", "inspect", "--format", "{{json .Mounts}}", socket.gethostname()],
                           capture_output=True, text=True)
        if p.returncode == 0:
            for m in json.loads(p.stdout or "[]"):
                if m.get("Type") == "bind" and m.get("Source") and m.get("Destination"):
                    _MOUNTS.append((Path(m["Destination"]), Path(m["Source"])))
        _MOUNTS.sort(key=lambda d: -len(str(d[0])))
    return _MOUNTS


def host_path(path: Path) -> str:
    """The path the host daemon must be given for ``path`` as seen by this process."""
    path = Path(path).resolve()
    mounts = container_mounts()
    if not mounts and not (Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()):
        return str(path)
    for dest, src in mounts:
        if path == dest or dest in path.parents:
            return str(src / path.relative_to(dest))
    raise RuntimeError(f"{path} is not under a bind mount of this container; the host daemon could not see it. "
                       "Keep the results directory bind-mounted (-v \"$PWD/results:/results\").")


class DockerEnv:
    def __init__(self, image: str):
        self.image = image

    @staticmethod
    def build(context_dir: Path, tag: str) -> str:
        p = subprocess.run(["docker", "build", "-q", "-t", tag, str(context_dir)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"docker build failed for {tag}: {p.stderr[-800:]}")
        return p.stdout.strip()

    @staticmethod
    def exists(tag: str) -> bool:
        return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0

    @classmethod
    def ensure(cls, dataset_root: Path, kind: str) -> "DockerEnv":
        tag = IMAGE_TAGS[kind]
        if not cls.exists(tag):
            cls.build(Path(dataset_root) / "environments" / kind, tag)
        return cls(tag)

    def image_id(self) -> str | None:
        p = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
                           capture_output=True, text=True)
        return p.stdout.strip() or None

    def evaluate(self, cmd: str, repo_dir: Path, hidden_dir: Path, env: dict | None, timeout: int) -> tuple:
        args = ["docker", "run", "--rm", "--network", "none",
                "-v", f"{host_path(repo_dir)}:{WORKDIR}",
                "-v", f"{host_path(hidden_dir)}:{HIDDEN_MOUNT}:ro",
                "-w", WORKDIR]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += [self.image, "timeout", "-k", "5", str(timeout), "sh", "-c", cmd]
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 60)
        except subprocess.TimeoutExpired as exc:
            return 124, (exc.stdout or "") + (exc.stderr or "") + "\n[evaluator] docker run exceeded its timeout"
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode in DOCKER_CLI_FAILURE_RCS and ("docker:" in out or "denied:" in out or "Error response from daemon" in out
                                                       or "not shared from the host" in out):
            raise EvaluatorError(f"evaluator container could not run (rc={p.returncode}): {out.strip()[-1500:]}")
        return p.returncode, out

    def evaluate_with_setup(self, setup_cmd: str | None, cmd: str, repo_dir: Path, hidden_dir: Path,
                            env: dict | None, timeout: int) -> tuple:
        if not setup_cmd:
            return self.evaluate(cmd, repo_dir, hidden_dir, env, timeout)
        full = (f"( {setup_cmd} ) > /tmp/pdl-setup.log 2>&1 || "
                f"{{ echo '--- setup_command failed ---'; tail -c 4000 /tmp/pdl-setup.log; exit {SETUP_FAILED_RC}; }}\n"
                f"{cmd}")
        rc, out = self.evaluate(full, repo_dir, hidden_dir, env, timeout)
        if rc == SETUP_FAILED_RC:
            raise SetupFailed(f"setup_command failed ({setup_cmd!r}):\n" + out.strip()[-2000:])
        return rc, out
