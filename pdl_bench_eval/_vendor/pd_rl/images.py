# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
"""Images a model agent can see (P1-044).

Two routes put an image in front of a model, and both carry this type:

* **the task's own images** — `issue.images` in the manifest, workspace-relative
  paths into the fixture. They are loaded host-side from the task directory and
  attached to the opening message, beside the problem text, so every model sees
  them from turn one. That is how the problems were posed where they came from
  (the sheet's image_input column went to the model with the prompt).
* **read_file on an image** — only on a tool surface whose read_file declares
  `media: [text, image]` (tools_noexec_v0.2). The runner fetches the file from
  the sandbox as base64 and the adapter hands the image back with the tool
  result.

The format is decided from the bytes, never the file name: a `.png` that is a
text file is not an image, and a provider rejects a mislabelled one outright.

The size cap is the tightest of the providers' per-image limits: Anthropic's
5 MB applies to the base64 payload, so the raw file must stay under 3.75 MB.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_IMAGE_BYTES = 3_750_000
MAX_TASK_IMAGES = 8
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


class ImageError(ValueError):
    """Not an image a model can be shown."""


def sniff(data: bytes):
    """The image's media type from its magic bytes, or None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass(frozen=True)
class Image:
    name: str      # workspace-relative path, as the model is told it
    mime: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.b64()}"

    def describe(self) -> dict:
        """What the trajectory records: enough to identify the image, not the bytes."""
        return {"name": self.name, "mime": self.mime, "bytes": len(self.data), "sha256": self.sha256}


def from_bytes(name: str, data: bytes) -> Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"{name} is {len(data):,} bytes; images are capped at {MAX_IMAGE_BYTES:,}")
    mime = sniff(data)
    if mime is None:
        raise ImageError(f"{name} is not a PNG, JPEG, GIF or WebP image")
    return Image(name=name, mime=mime, data=data)


def _safe_rel(rel: str) -> bool:
    p = PurePosixPath(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts


def task_image_paths(manifest: dict) -> list:
    return list((manifest.get("issue") or {}).get("images") or [])


def load_task_images(task_dir: Path, manifest: dict) -> list:
    """The manifest's issue.images, read from the task's fixture. Raises ImageError.

    Read on the HOST from the task directory, not from the sandbox: they are
    task content, fixed at authoring time, and must reach the model whatever
    the agent later does to its workspace.
    """
    rels = task_image_paths(manifest)
    if len(rels) > MAX_TASK_IMAGES:
        raise ImageError(f"issue.images lists {len(rels)} images; at most {MAX_TASK_IMAGES}")
    fixture = (Path(task_dir) / (manifest.get("repository") or {}).get("path", "fixtures/repo")).resolve()
    out = []
    for rel in rels:
        if not _safe_rel(rel):
            raise ImageError(f"issue.images entry {rel!r} must be a workspace-relative path")
        path = (fixture / rel).resolve()
        if fixture not in path.parents or not path.is_file():
            raise ImageError(f"issue.images entry {rel!r} is not a file in the fixture")
        out.append(from_bytes(rel, path.read_bytes()))
    return out
