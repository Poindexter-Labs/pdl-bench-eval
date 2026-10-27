# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
from __future__ import annotations

# Replication build: the agent runs in images derived from the public task
# images (pdl-bench/swe:1.0.0, pdl-bench/sweqna:1.0.0) with the `agent` user and
# keeper process the reference image provides. Built from sandbox/<kind>/.
IMAGES = {
    "python-standard": ("pdl-bench/swe-agent:1.0.0", "sandbox/swe"),
    "python-qa": ("pdl-bench/sweqna-agent:1.0.0", "sandbox/sweqna"),
}

DEFAULT_IMAGE = "python-standard"

def make_sandbox(kind: str, resources=None, image_digest=None, image=None):
    """Replication build: the Docker backend is the only sandbox."""
    if kind != "docker":
        raise ValueError(f"unsupported sandbox kind for replication: {kind!r} (docker only)")
    return DockerSandbox(resources, image=image)

import subprocess

import uuid

from pathlib import Path

WORKDIR = "/workspace/repo"

HIDDEN_MOUNT = "/hidden-tests"

GIT_DIR = "/var/lib/pdl-git"

ORIGIN_PATH = "/workspace/origin.git"

MERGE_PATH = "/var/lib/pdl-merge"

class SandboxError(Exception):
    """Infrastructure-level sandbox failure (maps to void_infra_failure)."""

class DockerSandbox:
    kind = "docker"

    def __init__(self, resources=None, image=None):
        self.resources = resources or {}
        image = image or DEFAULT_IMAGE
        if image not in IMAGES:
            raise SandboxError(f"unknown episode image '{image}' (profile ecosystem.base_image)")
        self.image = image
        self.image_tag, self.dockerfile_dir = IMAGES[image]
        self.workdir = WORKDIR
        self._cid = None
        self._image_id = None
        self._keeper_pid = None

    # -- docker plumbing -----------------------------------------------------
    @staticmethod
    def _docker(*args, input_text=None, timeout=600):
        try:
            proc = subprocess.run(
                ["docker", *args],
                input=input_text,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise SandboxError("docker CLI not found — is Docker installed and running?") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"docker {' '.join(args[:2])} timed out") from exc
        return proc

    def ensure_image(self, root: Path) -> None:
        """Build the episode image locally if absent (also used standalone by
        verify-task --sandbox docker, which never starts an agent container)."""
        if self._docker("image", "inspect", self.image_tag).returncode != 0:
            print(f"[sandbox] building {self.image_tag} from {self.dockerfile_dir}/ ...")
            proc = self._docker("build", "-t", self.image_tag, str(root / self.dockerfile_dir))
            if proc.returncode != 0:
                raise SandboxError(f"image build failed:\n{proc.stderr[-2000:]}")
        proc = self._docker("image", "inspect", "--format", "{{.Id}}", self.image_tag)
        self._image_id = proc.stdout.strip()

    # -- lifecycle -----------------------------------------------------------
    def start(self, fixture_dir: Path, root: Path = None) -> None:
        self.ensure_image(root or Path.cwd())
        name = f"pdl-ep-{uuid.uuid4().hex[:12]}"
        args = [
            "run", "-d", "--name", name,
            "--init",  # PID 1 reaps zombies so defunct entries don't accumulate
            "--network", "none",
        ]
        if self.resources.get("cpu"):
            args += ["--cpus", str(self.resources["cpu"])]
        if self.resources.get("memory_gib"):
            args += ["--memory", f"{self.resources['memory_gib']}g"]
        if self.resources.get("max_processes"):
            args += ["--pids-limit", str(self.resources["max_processes"])]
        args += [self.image_tag]
        proc = self._docker(*args)
        if proc.returncode != 0:
            raise SandboxError(f"container start failed: {proc.stderr.strip()}")
        self._cid = proc.stdout.strip()
        # stage the fixture and hand it to the non-root agent user
        proc = self._docker("cp", f"{fixture_dir}/.", f"{self._cid}:{WORKDIR}")
        if proc.returncode != 0:
            raise SandboxError(f"fixture copy failed: {proc.stderr.strip()}")
        self._docker("exec", "-u", "root", self._cid, "chown", "-R", "agent", "/workspace")
        # macOS Finder junk must not reach the agent's workspace
        self._docker(
            "exec", "-u", "root", self._cid, "sh", "-c",
            f"find {WORKDIR} -name .DS_Store -delete",
        )
        # root-owned external git dir, unreadable by the agent user (audit #2)
        self._docker(
            "exec", "-u", "root", self._cid, "sh", "-c",
            f"mkdir -p {GIT_DIR} && chmod 700 {GIT_DIR}",
        )
        # pin the keeper process pid so reap_stray_processes can't be evaded
        # by an agent daemonizing a look-alike command
        self._keeper_pid = None
        proc = self._docker("exec", "-u", "root", self._cid, "ps", "-eo", "pid=,args=")
        for line in proc.stdout.splitlines():
            pid, _, cmd_args = line.strip().partition(" ")
            if cmd_args.strip() == "sleep infinity":
                self._keeper_pid = pid
                break

    def run(self, cmd, timeout, env=None, input_text=None):
        args = ["exec", "-i", "-u", "agent", "-w", WORKDIR]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        # in-container `timeout` so the process tree dies on expiry even though
        # killing `docker exec` host-side would not kill it (kill_process_tree
        # semantics from the contract's terminal section)
        args += [self._cid, "timeout", "-k", "5", str(timeout), "sh", "-c", cmd]
        proc = self._docker(*args, input_text=input_text, timeout=timeout + 30)
        out = proc.stdout + proc.stderr
        if proc.returncode == 124:
            out += f"\ncommand timed out after {timeout}s"
        return proc.returncode, out

    def git(self, args, timeout=60, input_text=None):
        """Run a git command as root against the external git dir (audit #2)."""
        proc = self._docker(
            "exec", "-i", "-u", "root", "-w", WORKDIR,
            "-e", f"GIT_DIR={GIT_DIR}", "-e", f"GIT_WORK_TREE={WORKDIR}",
            self._cid, "sh", "-c", f"git {args}",
            input_text=input_text, timeout=timeout + 30,
        )
        return proc.returncode, proc.stdout + proc.stderr

    def write_git_exclude(self, text: str) -> None:
        proc = self._docker(
            "exec", "-i", "-u", "root", self._cid, "sh", "-c",
            f"mkdir -p {GIT_DIR}/info && cat > {GIT_DIR}/info/exclude",
            input_text=text,
        )
        if proc.returncode != 0:
            raise SandboxError(f"writing git exclude failed: {proc.stderr.strip()}")

    def admin(self, cmd, timeout=180, input_text=None):
        """Privileged shell for PR staging and merge construction (see the
        LocalSandbox docstring). Deliberately does NOT set GIT_DIR/GIT_WORK_TREE:
        these commands address the agent's own repository and the bare origin,
        not the runner's tamper-proof baseline. `input_text` is how the
        DynamicSWE series reaches the root-only store without a file copy."""
        proc = self._docker(
            "exec", "-i", "-u", "root", self._cid, "sh", "-c", cmd,
            input_text=input_text, timeout=timeout + 30,
        )
        return proc.returncode, proc.stdout + proc.stderr

    def disk_usage_kib(self):
        proc = self._docker("exec", "-u", "root", self._cid, "du", "-sk", WORKDIR)
        try:
            return int(proc.stdout.split()[0])
        except (ValueError, IndexError):
            return None  # watchdog malfunction is not an episode fault

    def reap_stray_processes(self):
        """List and kill processes surviving the agent loop (audit #7).
        Legitimate: pid 1 (docker-init), the pinned keeper, this ps itself.
        Zombies (<defunct>) are unreaped corpses, not background processes."""
        proc = self._docker("exec", "-u", "root", self._cid, "ps", "-eo", "pid=,args=")
        stray = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            pid, _, args = line.partition(" ")
            if pid in ("1", self._keeper_pid) or "ps -eo" in args or "<defunct>" in args:
                continue
            stray.append(line)
            self._docker("exec", "-u", "root", self._cid, "kill", "-9", pid)
        return stray

    def export_workspace(self, dest: Path, source: str = WORKDIR) -> None:
        proc = self._docker("cp", f"{self._cid}:{source}", str(dest))
        if proc.returncode != 0:
            raise SandboxError(f"workspace export failed: {proc.stderr.strip()}")

    def evaluate(self, cmd, frozen_dir: Path, hidden_dir: Path, env, timeout):
        """Fresh evaluator container: frozen workspace + read-only hidden mount.
        The agent container is never given the hidden bundle."""
        args = [
            "run", "--rm", "--network", "none",
            "-v", f"{frozen_dir}:{WORKDIR}",
            "-v", f"{hidden_dir}:{HIDDEN_MOUNT}:ro",
            "-w", WORKDIR,
        ]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += [self.image_tag, "timeout", "-k", "5", str(timeout), "sh", "-c", cmd]
        proc = self._docker(*args, timeout=timeout + 60)
        return proc.returncode, proc.stdout + proc.stderr

    def env_digest(self) -> str:
        return self._image_id or "sha256:" + "0" * 64

    def close(self) -> None:
        if self._cid:
            self._docker("rm", "-f", self._cid)
