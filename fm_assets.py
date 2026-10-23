"""Asset parsing and inventory for Finetune Manager.

This module deliberately has no Wan2GP or Gradio imports.  It is the single
place that answers "what files does this recipe need, and where do they live?"
so the UI, the validator and the downloader cannot disagree.

The rules mirror Wan2GP's ``shared/utils/files_locator.py`` and the launcher's
recipe asset resolver:

* a remote source is ``https://host/file?query|subfolder``;
* ``|%lora_dir`` means the architecture's LoRA collection;
* LoRA entries default to the LoRA collection, everything else to ckpts/;
* a single bare string in ``URLs``/``URLs2``/``loras`` is a model reference,
  not a missing file;
* ``|subfolder`` is untrusted input and must never escape its base folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".gguf",
    ".sft",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
)

# (role key, model field, human label, lora collection?)
ROLE_SPECS = (
    ("model", "URLs", "model", False),
    ("second", "URLs2", "second phase", False),
    ("text_encoder", "text_encoder_URLs", "text encoder", False),
    ("vae", "VAE_URLs", "VAE", False),
    ("preload", "preload_URLs", "preload", False),
    ("lora", "loras", "LoRA", True),
    ("custom1", "custom_url_1", "custom URL 1", False),
    ("custom2", "custom_url_2", "custom URL 2", False),
    ("custom3", "custom_url_3", "custom URL 3", False),
)

ROLE_BY_KEY = {spec[0]: spec for spec in ROLE_SPECS}
ROLE_BY_FIELD = {spec[1]: spec for spec in ROLE_SPECS}


@dataclass(frozen=True)
class AssetSource:
    role: str
    field: str
    label: str
    index: int
    source: str
    is_lora: bool
    is_remote: bool
    url: str | None
    subfolder: str | None
    filename: str
    reference: bool = False


@dataclass(frozen=True)
class Asset:
    id: str
    role: str
    field: str
    label: str
    index: int
    source: str
    url: str | None
    subfolder: str | None
    filename: str
    destination: str | None
    present: bool
    bytes: int
    downloadable: bool
    needs_token: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "field": self.field,
            "label": self.label,
            "index": self.index,
            "source": self.source,
            "url": self.url,
            "subfolder": self.subfolder,
            "filename": self.filename,
            "destination": self.destination,
            "present": self.present,
            "bytes": self.bytes,
            "downloadable": self.downloadable,
            "needsToken": self.needs_token,
            "error": self.error,
        }


def is_pack_id(fin_id: str) -> bool:
    return str(fin_id or "").startswith("adultpack_")


def pack_display_id(fin_id: str) -> str:
    """The id a user sees for a pack recipe: ``adultpack_foo`` -> ``foo``."""
    value = str(fin_id or "")
    return value[len("adultpack_") :] if is_pack_id(value) else value


def finetune_candidates(fin_id: str) -> list[str]:
    """File stems to try for an id, matching the launcher's resolver.

    ``foo`` and ``adultpack_foo`` are the same recipe.  The exact id wins, so
    editing an existing file never silently switches to the other spelling.
    """
    value = str(fin_id or "").strip()
    if not value:
        return []
    out = [value]
    if is_pack_id(value):
        bare = value[len("adultpack_") :]
        if bare and bare not in out:
            out.append(bare)
    else:
        packed = f"adultpack_{value}"
        if packed not in out:
            out.append(packed)
    return out


def source_strings(value: Any) -> list[str]:
    """Normalise a FINETUNES field that may be a string or a list."""
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return []


def usable_weight_source(source: str) -> bool:
    """True when a value looks like a file/URL rather than a reuse id."""
    value = str(source or "").strip()
    if not value:
        return False
    lower = value.casefold()
    if lower.startswith(("http://", "https://", "/", "\\\\")):
        return True
    if ":/" in value or ":\\" in value or "/" in value or "\\" in value:
        return True
    return lower.endswith(WEIGHT_EXTENSIONS)


def is_reference(model: dict, role: str, source: str) -> bool:
    """A bare value in a single-value field is FINETUNES' reuse form."""
    value = str(source or "").strip()
    if value.startswith("="):
        return True
    spec = ROLE_BY_KEY.get(role)
    if not spec:
        return False
    field_value = model.get(spec[1])
    if not isinstance(field_value, str):
        return False
    return not usable_weight_source(value)


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def url_file_name(url: str) -> str:
    """The name a download will be saved under, without query/fragment.

    A ``#/name.ext`` fragment is an explicit filename override: Wan2GP's
    ``files_locator`` takes the basename of the whole URL, and fragments are
    never sent to the server. CivitAI recipes use it because their download
    URLs end in a numeric version id, not a filename.
    """
    raw = str(url or "").split("|", 1)[0]
    _, marker, override = raw.partition("#/")
    if marker and _basename(override):
        return _basename(override)
    parsed = urlsplit(raw)
    name = _basename(parsed.path or raw)
    return name


_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _reserved_stem(name: str) -> bool:
    stem = name.split(".", 1)[0].rstrip(" .").upper()
    return stem in _WINDOWS_RESERVED


def safe_file_name(name: str) -> str | None:
    """Reject names that would escape a folder or are invalid on Windows."""
    if not name or len(name) > 180:
        return None
    if name in (".", "..") or ".." in name or "/" in name or "\\" in name:
        return None
    bad = set('<>:"|?*')
    if any(ch in bad or ord(ch) < 0x20 for ch in name):
        return None
    if name.endswith((".", " ")):
        return None
    if _reserved_stem(name):
        return None
    return name


def safe_subdir(subfolder: str) -> Path:
    """Validate a ``|subfolder`` suffix; reject absolute paths and traversal."""
    raw = str(subfolder or "").strip().replace("\\", "/")
    if not raw:
        return Path()
    if raw.startswith("/") or raw.startswith("//"):
        raise ValueError(f"unsafe absolute subfolder: {subfolder}")
    if len(raw) > 1 and raw[1] == ":":
        raise ValueError(f"unsafe absolute subfolder: {subfolder}")
    parts: list[str] = []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == ".." or ":" in part or "\x00" in part:
            raise ValueError(f"unsafe subfolder: {subfolder}")
        if safe_file_name(part) is None:
            raise ValueError(f"unsafe subfolder segment: {part}")
        parts.append(part)
    return Path(*parts) if parts else Path()


def parse_source(raw: str) -> tuple[bool, str | None, str | None, str]:
    """Return (is_remote, clean_url, subfolder, filename).

    Local paths keep their value as the filename and have no subfolder.
    """
    value = str(raw or "").strip()
    if value.startswith(("http://", "https://")):
        url, _, sub = value.partition("|")
        return True, url, (sub or None), url_file_name(url)
    return False, None, None, _basename(value)


def asset_path(
    source: str,
    role: str,
    ckpts_dir: str | Path,
    lora_dir: str | Path | None,
) -> Path:
    """Where one recipe entry lands, mirroring files_locator."""
    spec = ROLE_BY_KEY.get(role)
    is_lora = bool(spec and spec[3])
    ckpts = Path(ckpts_dir)
    loras = Path(lora_dir) if lora_dir else None
    raw = str(source or "").strip()
    if not raw:
        raise ValueError("empty source")

    if raw.startswith(("http://", "https://")):
        url, _, sub = raw.partition("|")
        name = safe_file_name(url_file_name(url))
        if not name:
            raise ValueError(f"unsafe file name: {raw}")
        base = loras if is_lora else ckpts
        if sub == "%lora_dir":
            if loras is None:
                raise ValueError("source uses %lora_dir but no LoRA folder is available")
            return loras / name
        if sub:
            return base / safe_subdir(sub) / name
        if base is None:
            raise ValueError("source is a LoRA but no LoRA folder is available")
        return base / name

    path = Path(raw)
    if path.is_absolute():
        return path
    safe = safe_subdir(raw)
    base = loras if is_lora else ckpts
    if base is None:
        raise ValueError("source is a LoRA but no LoRA folder is available")
    return base / safe


def iter_asset_sources(model: dict) -> Iterator[AssetSource]:
    """Yield every file source in the order Wan2GP reads it."""
    if not isinstance(model, dict):
        return
    for role, field, label, is_lora in ROLE_SPECS:
        value = model.get(field)
        for index, source in enumerate(source_strings(value)):
            is_remote, url, subfolder, filename = parse_source(source)
            reference = is_reference(model, role, source)
            # preload_URLs may name a model type rather than a file.
            if role == "preload" and not usable_weight_source(source):
                reference = True
            yield AssetSource(
                role=role,
                field=field,
                label=label,
                index=index,
                source=source,
                is_lora=is_lora,
                is_remote=is_remote,
                url=url,
                subfolder=subfolder,
                filename=filename,
                reference=reference,
            )


def is_civitai_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        host = (urlsplit(url).hostname or "").casefold()
    except ValueError:
        return False
    return host == "civitai.com" or host.endswith(".civitai.com") or host == "civitai.red" or host.endswith(".civitai.red")


def build_inventory(
    model: dict,
    ckpts_dir: str | Path,
    lora_dir: str | Path | None,
    exists: Callable[[Path], bool] | None = None,
    size_of: Callable[[Path], int] | None = None,
) -> list[Asset]:
    """Build one asset row per source, including present/missing state."""
    def _exists(path: Path) -> bool:
        if exists is not None:
            return bool(exists(path))
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _size(path: Path) -> int:
        if size_of is not None:
            return int(size_of(path))
        try:
            return int(path.stat().st_size) if path.is_file() else 0
        except OSError:
            return 0

    assets: list[Asset] = []
    for spec in iter_asset_sources(model):
        if spec.reference:
            continue
        destination: Path | None = None
        error: str | None = None
        try:
            destination = asset_path(spec.source, spec.role, ckpts_dir, lora_dir)
        except (ValueError, OSError) as exc:
            error = str(exc)
        present = bool(destination and error is None and _exists(destination))
        size = _size(destination) if destination and error is None else 0
        assets.append(
            Asset(
                id=f"{spec.role}::{spec.index}",
                role=spec.role,
                field=spec.field,
                label=spec.label,
                index=spec.index,
                source=spec.source,
                url=spec.url,
                subfolder=spec.subfolder,
                filename=spec.filename,
                destination=str(destination) if destination else None,
                present=present,
                bytes=size,
                downloadable=bool(spec.is_remote and destination and error is None),
                needs_token=is_civitai_url(spec.url),
                error=error,
            )
        )
    return assets


def inventory_summary(assets: Iterable[Asset]) -> dict[str, int]:
    rows = list(assets)
    return {
        "total": len(rows),
        "present": sum(1 for row in rows if row.present),
        "missing": sum(1 for row in rows if not row.present),
        "downloadable": sum(
            1 for row in rows if not row.present and row.downloadable
        ),
        "errors": sum(1 for row in rows if row.error),
    }


def format_inventory(assets: Iterable[Asset], show_destination: bool = True) -> str:
    """Human-readable status used by the editor's download panel."""
    rows = list(assets)
    if not rows:
        return "No file assets found."
    lines: list[str] = []
    for row in rows:
        if row.present:
            mark = "on disk"
        elif row.error:
            mark = f"error: {row.error}"
        elif row.downloadable:
            mark = "needs download"
        else:
            mark = "local only"
        line = f"{row.label}: {row.filename or row.source} - {mark}"
        if show_destination and row.destination:
            line += f"\n    -> {row.destination}"
        lines.append(line)
    summary = inventory_summary(rows)
    lines.append(
        f"{summary['present']}/{summary['total']} present, "
        f"{summary['downloadable']} downloadable missing"
    )
    return "\n".join(lines)
