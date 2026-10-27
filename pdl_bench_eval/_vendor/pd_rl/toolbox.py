# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
from __future__ import annotations

import json

import re

import shlex

import time

from datetime import datetime, timezone

from pathlib import Path

from jsonschema import Draft202012Validator

from pdl_bench_eval._vendor.pd_rl import images as images_mod

from pdl_bench_eval._vendor.pd_rl.agents import _AgentFault, _Submit

from pdl_bench_eval._vendor.pd_rl.sandbox import SandboxError

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

_SECTION_PREFIXES = (
    "--- ", "+++ ", "@@", "diff ", "index ", "new ", "deleted ", "old ",
    "similarity ", "rename ", "copy ", "Binary ",
)

def _patch_format_error(text: str):
    """Return a description of a count-inconsistent unified diff, or None.

    git-apply parses each hunk body by the counts declared in its @@ header
    and treats surplus body lines as patch-terminating garbage — the patch
    half-applies and exits 0 (DEC-034, found by the task-5 probe: the model's
    correct fix lost its final line silently). We refuse such patches outright
    rather than repair the header, so the malformed call stays the model's
    cost.
    """
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        header = _HUNK_HEADER_RE.match(lines[i])
        if header is None:
            i += 1  # file/git headers and inter-file noise: git validates those
            continue
        hunk_at = i + 1
        old_left = int(header.group(1)) if header.group(1) is not None else 1
        new_left = int(header.group(2)) if header.group(2) is not None else 1
        i += 1
        while old_left > 0 or new_left > 0:
            if i >= n:
                return (f"hunk at line {hunk_at} ends before its @@ header's "
                        "declared line counts are satisfied")
            body = lines[i]
            if body.startswith("\\"):  # "\ No newline at end of file"
                i += 1
                continue
            if body.startswith("+"):
                new_left -= 1
            elif body.startswith("-"):
                old_left -= 1
            elif body.startswith(" ") or body == "":
                old_left -= 1
                new_left -= 1
            else:
                return f"hunk at line {hunk_at}: invalid body line {i + 1}"
            if old_left < 0 or new_left < 0:
                return (f"hunk at line {hunk_at}: body is longer than its @@ "
                        "header declares")
            i += 1
        while i < n and lines[i].startswith("\\"):
            i += 1
        if i < n and lines[i].startswith(("+", "-", " ")) \
                and not lines[i].startswith(_SECTION_PREFIXES):
            return (f"hunk at line {hunk_at}: body continues past its @@ "
                    f"header's declared counts (surplus at line {i + 1})")
    return None

_TERMINATION_REASONS = {
    "agent_submits",
    "max_turns_reached",
    "token_budget_reached",
    "wall_clock_timeout_reached",
}

SETUP_FAILED_RC = 97

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

class Trajectory:
    """Append-only JSONL event log, schema-validated at write time (DEC-018)."""

    def __init__(self, path: Path, episode_id: str, schema: dict):
        self._fh = path.open("w")
        self._episode_id = episode_id
        self._validator = Draft202012Validator(schema)
        self._seq = 0

    def emit(self, event_type: str, payload: dict) -> None:
        event = {
            "episode_id": self._episode_id,
            "seq": self._seq,
            "timestamp": _now(),
            "type": event_type,
            "payload": payload,
        }
        errors = list(self._validator.iter_errors(event))
        if errors:
            raise RuntimeError(f"invalid trajectory event {event_type}: {errors[0].message}")
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()
        self._seq += 1

    def close(self) -> None:
        self._fh.close()

class ToolBox:
    """Executes validated tool calls against the sandbox, enforcing budgets
    and logging every call/result to the trajectory."""

    def __init__(self, sandbox, specs, limits, trajectory, agent_deadline=None):
        self._sandbox = sandbox
        self._limits = limits
        self._trajectory = trajectory
        self._tool_specs = specs["tools"]["tools"]
        self._validators = {
            name: Draft202012Validator(tool["parameters"])
            for name, tool in self._tool_specs.items()
        }
        self._agent_deadline = agent_deadline  # monotonic; audit #1
        self.turn = 0
        self.tool_calls = 0
        self.shell_commands = 0
        self._call_seq = 0
        # DynamicSWE (P1-038): set by ModelAgent.run; scripted agents leave
        # `turn` at 1, so turn-kind triggers read the tool-call count instead.
        self.turn_driven = False
        self.controller = None  # upstream.Controller for a swe_dyn episode
        # read_file returns images only on a tool surface that declares it
        # (`media: [text, image]`, tools_noexec_v0.2 — P1-044), and only to an
        # agent that can view them; the runner sets agent_accepts_images.
        read_spec = self._tool_specs.get("read_file") or {}
        self.read_file_images = "image" in (read_spec.get("media") or [])
        self.agent_accepts_images = True
        self._pending_images = []

    def _check_wall_clock(self):
        if self._agent_deadline is not None and time.monotonic() > self._agent_deadline:
            raise _Submit("wall_clock_timeout_reached")

    def emit_model_call(self, payload: dict) -> None:
        """Model agents log their API calls through the toolbox (DEC-018)."""
        self._check_wall_clock()
        self._trajectory.emit("model_call", payload)

    def emit_budget_warning(self, payload: dict) -> None:
        """Record that the agent was warned its budget was running out.

        On the trajectory rather than only in the log, so a low-scoring episode
        can be read as "was warned and still did not write" versus "was never
        warned" without re-deriving it from token counts.
        """
        self._trajectory.emit("budget_warning", payload)

    def take_images(self) -> list:
        """Images the last tool call returned, for the adapter to attach (P1-044)."""
        shown, self._pending_images = self._pending_images, []
        return shown

    def _read_image(self, path: str, timeout: int):
        """(rc, text) for read_file on an image; the image itself is queued for take_images.

        Fetched as base64 so the bytes survive the sandbox's text decode on
        every backend; `base64 < file` behaves the same under GNU and BSD
        coreutils. Size is checked before the fetch, so an oversized file never
        crosses the wire.
        """
        if not self.agent_accepts_images:
            return 1, (f"error: {path} is an image, and the model on this episode cannot view "
                       "images. Work from the problem text.")
        quoted = shlex.quote(path)
        rc, size = self._sandbox.run(f"wc -c < {quoted}", timeout=timeout)
        if rc != 0:
            return rc, size
        try:
            nbytes = int(size.strip())
        except ValueError:
            return 1, f"error: could not read {path}"
        if nbytes > images_mod.MAX_IMAGE_BYTES:
            return 1, (f"error: {path} is {nbytes:,} bytes; images larger than "
                       f"{images_mod.MAX_IMAGE_BYTES:,} bytes cannot be shown")
        rc, encoded = self._sandbox.run(f"base64 < {quoted}", timeout=timeout)
        if rc != 0:
            return rc, encoded
        import base64
        import binascii

        try:
            img = images_mod.from_bytes(path, base64.b64decode("".join(encoded.split()), validate=True))
        except (binascii.Error, images_mod.ImageError) as exc:
            return 1, f"error: {exc}"
        self._pending_images.append(img)
        return 0, f"[image {path}: {img.mime}, {len(img.data):,} bytes — shown to you with this result]"

    def _truncate(self, output: str, cap_kib=None):
        if cap_kib is None:
            cap_kib = self._limits["terminal_limits"].get("max_command_output_to_agent_kib", 64)
        cap = cap_kib * 1024
        if len(output) > cap:
            return output[:cap] + "\n[output truncated]", True
        return output, False

    def call(self, name: str, params: dict) -> str:
        self._check_wall_clock()
        budget = self._limits["agent_budget"]
        if self.tool_calls >= budget.get("max_tool_calls", 10**9):
            raise _Submit("max_tool_calls_reached")
        if name not in self._tool_specs:
            raise _AgentFault(f"unknown tool: {name}")
        errors = list(self._validators[name].iter_errors(params or {}))
        if errors:
            raise _AgentFault(f"invalid params for {name}: {errors[0].message}")

        self.tool_calls += 1
        self._call_seq += 1
        call_id = f"c{self._call_seq}"
        self._trajectory.emit(
            "tool_call",
            {"turn": max(self.turn, 1), "call_id": call_id, "tool": name, "params": params or {}},
        )
        # DynamicSWE (DS-5): the synchronous controller sees every call before
        # and after it runs. It returns at once outside the dynamic condition,
        # and `submit` is never a trigger source (S-4).
        ctl_turn = self.turn if self.turn_driven else self.tool_calls
        if self.controller is not None:
            self.controller.before(name, params or {}, ctl_turn, self.tool_calls)

        if name == "submit":
            self._trajectory.emit(
                "tool_result", {"call_id": call_id, "exit_code": 0, "output": "", "truncated": False}
            )
            raise _Submit("agent_submits")

        timeout = self._limits["terminal_limits"].get("default_command_timeout_seconds", 120)
        read_cap_kib = None
        shown = None
        if name == "read_file" and self.read_file_images and \
                Path(params["path"]).suffix.lower() in images_mod.IMAGE_SUFFIXES:
            rc, out = self._read_image(params["path"], timeout)
            shown = [img.describe() for img in self._pending_images]
        elif name == "read_file":
            path = params["path"]
            # shlex.quote, NOT json.dumps (P1-040). Both produce a quoted
            # string, but json.dumps produces a DOUBLE-quoted one, and `sh`
            # expands $(...), `...` and $VAR inside double quotes. A path of
            # `$(cmd)` therefore ran `cmd` — arbitrary execution through a tool
            # that is not `shell`, without incrementing shell_commands or
            # appearing in the trajectory as a shell call. Single quotes do not
            # expand anything.
            rc, out = self._sandbox.run(f"cat -- {shlex.quote(path)}", timeout=timeout)
            read_cap_kib = self._limits["terminal_limits"].get("max_file_read_kib", 256)  # audit #4
            if rc == 0 and _looks_binary(out):
                # read_file is `cat`: on a .docx/.xlsx/.pptx/.pdf it returns the
                # compressed container. The sandbox decodes tolerantly so this
                # can no longer crash the episode, but handing the model tens of
                # thousands of replacement characters is worse than useless — it
                # burns the token budget and teaches nothing. Refuse with the
                # route that works instead. Exit code 1 makes it an ordinary
                # recoverable tool error, not a fault.
                rc, out = 1, (
                    f"error: {path} is a binary file, and this tool reads text. "
                    "Word/Excel/PowerPoint/PDF files are compressed containers. "
                    "To inspect one, run a script with the shell tool that opens "
                    "it and prints what you need — for example python-docx for "
                    ".docx, openpyxl for .xlsx, python-pptx for .pptx, pypdf for "
                    ".pdf. The same applies to writing one: write a generator "
                    "script and run it."
                )
        elif name == "search":
            # Same double-quote expansion defect as read_file above (P1-040).
            rc, out = self._sandbox.run(
                f"grep -rn --include='*' -e {shlex.quote(params['query'])} . | head -200",
                timeout=timeout,
            )
        elif name == "write_patch":
            patch_cap_mib = self._limits["terminal_limits"].get("max_patch_mib", 5)
            if len(params["patch"].encode()) > patch_cap_mib * 1024 * 1024:  # audit #5
                raise _AgentFault(f"patch exceeds max_patch_mib={patch_cap_mib}")
            fmt_error = _patch_format_error(params["patch"])
            if fmt_error:
                rc, out = 1, f"error: malformed patch: {fmt_error}"
            else:
                rc, out = self._sandbox.run(
                    "git apply --whitespace=nowarn", timeout=timeout, input_text=params["patch"]
                )
        elif name == "shell":
            if self.shell_commands >= budget.get("max_shell_commands", 10**9):
                raise _Submit("max_tool_calls_reached")
            self.shell_commands += 1
            timeout = min(
                params.get("timeout_seconds", timeout),
                self._limits["terminal_limits"].get("long_command_timeout_seconds", 600),
            )
            rc, out = self._sandbox.run(params["command"], timeout=timeout)
        else:  # pragma: no cover - tools schema and this dispatch must agree
            raise _AgentFault(f"tool {name} has no implementation")

        out, truncated = self._truncate(out, cap_kib=read_cap_kib)
        self._trajectory.emit(
            "tool_result", {"call_id": call_id, "exit_code": rc, "output": out, "truncated": truncated,
                            **({"images": shown} if shown else {})}
        )
        if self.controller is not None:
            self.controller.after(name, params or {}, ctl_turn, self.tool_calls, output=out)
        if name in ("write_patch", "shell"):  # audit #3: disk watchdog after mutating calls
            used_kib = self._sandbox.disk_usage_kib()
            cap_kib = self._limits["resources"]["writable_disk_gib"] * 1024 * 1024
            if used_kib is not None and used_kib > cap_kib:
                raise _AgentFault(
                    f"disk budget exceeded: {used_kib} KiB used > {cap_kib} KiB limit"
                )
        return out

BINARY_REPLACEMENT_RATIO = 0.15

BINARY_REPLACEMENT_MIN = 8

def _looks_binary(text: str) -> bool:
    """Whether a tool's output is a binary file rather than readable text.

    Decided AFTER the read, on the decoded string, so it behaves identically on
    every sandbox backend and costs no extra round trip. A NUL byte is decisive
    on its own; otherwise it is the density of replacement characters.
    """
    if not text:
        return False
    if "\x00" in text:
        return True
    sample = text[:8192]
    bad = sample.count("\ufffd")
    return bad >= BINARY_REPLACEMENT_MIN and bad / len(sample) > BINARY_REPLACEMENT_RATIO

def _changed_files(sandbox, timeout=60):
    rc, out = sandbox.git("status --porcelain", timeout=timeout)
    if rc != 0:
        raise SandboxError(f"git status failed: {out}")
    files = []
    for line in out.splitlines():
        if line.strip():
            files.append(line[3:].split(" -> ")[-1].strip())
    return files
