"""Finetune-manager download and CivitAI recipe helpers.

This module deliberately has no Gradio or Wan2GP imports.  The plugin supplies
the destination paths, so the plugin UI, tests and tools all use the same
rules.  The v4.0 slim-down removed the adult-pack runtime (catalog, bulk
downloads, inventory, updates): the curated pack lives in the
``wan2gp-finetunes-adult`` HF Space as plain finetune JSONs and is refreshed
with ``tools/build_adult_registry.py`` / ``tools/publish_adult_space.py``.

What remains:

- ``download_verified`` — resumable, SHA-256-verified file download used by
  the plugin's asset prefetch path.
- The CivitAI recipe builders used by Create/Edit "Fill from CivitAI":
  weight-file filtering, ``#/filename`` URL fragments, plain-text blurbs and
  the ``civitai_recipe`` payload builder.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlsplit

try:
    from . import fm_assets, fm_security
except ImportError:  # pragma: no cover - direct script/import fallback
    import fm_assets
    import fm_security

CIVITAI_SITES = ("civitai.com", "civitai.red")
USER_AGENT = "Wan2GP-FinetuneManager/3.7"
# Downloads from a finetune recipe deliberately allow any public HTTPS host;
# that is the product's job. Catalog traffic and its redirects stay inside the
# CivitAI/HF families (WEIGHT_HOSTS).
WEIGHT_HOSTS = ("huggingface.co", "hf.co", "civitai.com", "civitai.red")


def _basename(value: str) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def _safe_name(value: str) -> str:
    name = fm_assets.safe_file_name(_basename(value))
    if not name:
        raise ValueError(f"unsafe file name: {value}")
    return name


def detect_pair(name: str) -> str | None:
    n = _basename(name).lower()
    if "high" in n or "_hn" in n or n.endswith("_h.safetensors"):
        return "high"
    if "low" in n or "_ln" in n or n.endswith("_l.safetensors"):
        return "low"
    return None


def split_halves(names: list[str], paths: list[str]) -> tuple[list[str], list[str]]:
    high: list[str] = []
    low: list[str] = []
    for name, path in zip(names, paths):
        if detect_pair(name) == "low":
            low.append(path)
        else:
            high.append(path)
    if not high and low:
        high, low = low, []
    return high, low


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _civitai_host(url: str) -> bool:
    try:
        host = urlsplit(str(url)).hostname or ""
    except Exception:
        return False
    return any(host == site or host.endswith("." + site) for site in CIVITAI_SITES)


def with_civitai_token(url: str, token: str | None) -> str:
    value = str(url or "")
    key = str(token or "").strip()
    if not key or not value or not _civitai_host(value):
        return value
    parts = urlsplit(value)
    if "token" in _parse_qs(parts.query):
        return value
    query = parts.query + "&" if parts.query else ""
    query += "token=" + quote(key, safe="")
    return parts._replace(query=query).geturl()


def _parse_qs(query: str) -> dict[str, list[str]]:
    from urllib.parse import parse_qs

    return parse_qs(query)


def download_verified(
    url: str,
    destination: str | Path,
    *,
    sha256: str | None = None,
    expected_bytes: int = 0,
    token: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    label: str = "",
    allowed_hosts: tuple[str, ...] | list[str] | None = WEIGHT_HOSTS,
) -> dict:
    """Download one file with resume and SHA-256 verification.

    ``allowed_hosts=None`` accepts any public HTTPS host (recipe downloads),
    while the default restricts CivitAI/HF catalog traffic and its redirects
    to the expected CDN families. Returns
    ``{"status": "exists"|"downloaded", "bytes": n, "path": ...}``.
    """
    dest = Path(destination)
    expected = int(expected_bytes or 0)
    digest = str(sha256 or "").strip().lower()
    cap = expected or fm_security.MAX_DOWNLOAD_BYTES
    if dest.is_file():
        size = dest.stat().st_size
        if (not expected or size == expected) and (not digest or sha256_file(dest) == digest):
            if progress:
                progress(expected or size, expected or size, label or dest.name)
            return {"status": "exists", "bytes": size, "path": str(dest)}
    problem = fm_security.url_problem(str(url), allowed_hosts=allowed_hosts)
    if problem:
        raise fm_security.UnsafeUrl(problem)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    if expected and offset > expected:
        part.unlink(missing_ok=True)
        offset = 0
    if expected and offset == expected:
        if not digest or sha256_file(part) == digest:
            os.replace(part, dest)
            if progress:
                progress(expected, expected, label or dest.name)
            return {"status": "downloaded", "bytes": expected, "path": str(dest)}
        part.unlink(missing_ok=True)
        offset = 0
    if offset and digest and not expected and sha256_file(part) == digest:
        size = part.stat().st_size
        os.replace(part, dest)
        if progress:
            progress(size, size, label or dest.name)
        return {"status": "downloaded", "bytes": size, "path": str(dest)}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/octet-stream,*/*"}
    if token and _civitai_host(url):
        headers["Authorization"] = f"Bearer {str(token).strip()}"
    target = with_civitai_token(url, token)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        with fm_security.open_url(
            target,
            headers=headers,
            timeout=180,
            allowed_hosts=allowed_hosts,
        ) as resp:
            status = int(resp.status)
            if offset and status == 206:
                mode = "ab"
                content_length = int(resp.headers.get("Content-Length") or 0)
                total = expected or (offset + content_length)
            else:
                if offset and status == 200:
                    part.unlink(missing_ok=True)
                    offset = 0
                mode = "wb"
                content_length = int(resp.headers.get("Content-Length") or 0)
                total = expected or content_length
            if content_length and content_length > cap:
                raise ValueError(
                    f"response is {content_length} bytes, above the {cap} byte limit"
                )
            problem = fm_security.content_problem(resp.headers.get("Content-Type", ""))
            if problem:
                raise ValueError(problem)
            disposition = str(resp.headers.get("Content-Disposition") or "")
            if "filename=" in disposition.lower() and any(
                bad in disposition for bad in ("../", "..\\", "/", "\\", "\x00")
            ):
                raise ValueError("unsafe filename in Content-Disposition")
            written = offset
            with part.open(mode) as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > cap:
                        raise ValueError(f"download exceeds the {cap} byte limit")
                    fh.write(chunk)
                    if progress:
                        progress(written, total or cap, label or dest.name)
        size = part.stat().st_size
        if expected and size != expected:
            raise ValueError(f"size mismatch: expected {expected}, got {size}")
        if digest and sha256_file(part) != digest:
            raise ValueError("SHA-256 mismatch")
        os.replace(part, dest)
        return {"status": "downloaded", "bytes": size, "path": str(dest)}
    except ValueError:
        # Never keep a body that failed size/type/hash checks.
        part.unlink(missing_ok=True)
        raise
    except Exception:
        # Network errors leave a valid partial file for the next resume.
        if part.is_file() and expected and part.stat().st_size > expected:
            part.unlink(missing_ok=True)
        raise


CIVITAI_WEIGHT_TYPES = frozenset(
    {
        "",
        "model",
        "diffusion model",
        "pruned model",
        "unet",
        "checkpoint",
        "lora",
        "locon",
        "dora",
    }
)


def civitai_weight_file(file: dict) -> bool:
    """True when a CivitAI version file is model weight, not an accessory.

    ``Model`` is the generic value, but checkpoints converted from other
    ecosystems are published as ``Diffusion Model`` or ``UNet``. Accepting
    those is required for MiniMax H3 and several ComfyUI-origin checkpoints;
    training data, previews, VAEs, configs and text encoders stay excluded.
    """
    return str(file.get("type") or "").strip().lower() in CIVITAI_WEIGHT_TYPES


def _civitai_file_name(file: dict, download_url: str) -> str:
    name = file.get("name")
    if name:
        return _safe_name(str(name))
    return _safe_name(str(download_url).split("?", 1)[0])


def _civitai_sha(file: dict) -> str | None:
    hashes = file.get("hashes")
    if isinstance(hashes, dict):
        value = str(hashes.get("SHA256") or "").strip().lower()
        if len(value) == 64:
            return value
    return None


CIVITAI_LORA_TYPES = ("LORA", "LOCON", "DORA")


def civitai_kind(model_type: str | None) -> str | None:
    """Map a CivitAI model ``type`` to the Wan2GP asset kind.

    Returns ``"lora"`` or ``"checkpoint"``; ``None`` for entries that are not
    model weights (workflows, archives, training data, embeddings, ...).
    """
    value = str(model_type or "").strip().upper()
    if value in CIVITAI_LORA_TYPES:
        return "lora"
    if value == "CHECKPOINT":
        return "checkpoint"
    return None


def civitai_file_url(file: dict, site: str = "civitai.com") -> str:
    """Absolute download URL for one CivitAI file entry."""
    value = str(file.get("downloadUrl") or "").strip()
    if value.startswith("/"):
        host = site if site in CIVITAI_SITES else CIVITAI_SITES[0]
        value = f"https://{host}{value}"
    return value


def civitai_recipe_url(file: dict, site: str = "civitai.com") -> str:
    """Download URL that Wan2GP saves under the file's real name.

    CivitAI's download endpoint ends in the version id, so Wan2GP would save
    ``2466604`` instead of ``name.gguf``. A URL fragment is never sent to the
    server and ``files_locator.extract_alternate_path`` takes the basename of
    the whole string, so ``...#/name.gguf`` downloads normally while landing
    as ``name.gguf``.
    """
    url = civitai_file_url(file, site)
    name = _civitai_file_name(file, url)
    base = _basename(url.split("|", 1)[0].split("#", 1)[0])
    if base == name:
        return url
    return f"{url.split('#', 1)[0]}#/{name}"


def civitai_weight_files(
    files: Iterable[dict], *, primary_only: bool = False
) -> list[dict]:
    """Loadable weight files of one CivitAI version.

    CivitAI puts every format of a release in ``files``. ``primary_only``
    keeps just the files CivitAI marks primary (or the first file when the
    release does not mark one), which is what a recipe should reference.
    """
    weights = [
        file
        for file in (files or [])
        if isinstance(file, dict)
        and civitai_weight_file(file)
        and str(file.get("downloadUrl") or "").strip()
    ]
    if primary_only:
        primary = [file for file in weights if file.get("primary")]
        return primary or weights[:1]
    return weights


def civitai_download_urls(
    version: dict,
    *,
    site: str = "civitai.com",
    primary_only: bool = False,
) -> list[str]:
    files = version.get("files") if isinstance(version, dict) else []
    return [
        civitai_file_url(file, site)
        for file in civitai_weight_files(files, primary_only=primary_only)
    ]


def civitai_plain_text(value: Any, limit: int = 1200) -> str:
    """Strip CivitAI's HTML descriptions into a compact plain-text blurb."""
    import html as _html

    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def civitai_recipe(
    model: dict,
    version: dict,
    *,
    architecture: str = "",
    primary_only: bool = False,
    site: str = "civitai.com",
) -> dict:
    """Build a Wan2GP finetune recipe for one CivitAI version.

    The recipe references the public CivitAI download URLs, so Wan2GP can
    fetch the weights later. Each URL carries a ``#/filename`` fragment so
    Wan2GP saves it under the real name instead of the numeric version id.
    API tokens are never embedded: the plugin's own Download button supplies
    the token at request time.
    """
    model = model if isinstance(model, dict) else {}
    version = version if isinstance(version, dict) else {}
    kind = civitai_kind(model.get("type"))
    if kind is None:
        raise ValueError(
            f"CivitAI type '{model.get('type') or 'unknown'}' is not a "
            "checkpoint or LoRA"
        )
    files = civitai_weight_files(
        version.get("files") or [], primary_only=primary_only
    )
    if not files:
        raise ValueError("this CivitAI release has no downloadable weight file")
    urls = [civitai_recipe_url(file, site) for file in files]
    model_name = str(model.get("name") or "CivitAI model").strip()
    version_name = str(version.get("name") or "").strip()
    name = f"{model_name} \u2014 {version_name}" if version_name else model_name

    notes: list[str] = []
    base = str(version.get("baseModel") or "").strip()
    if base:
        notes.append(f"Base model: {base}")
    model_id = model.get("id")
    version_id = version.get("id")
    if model_id and version_id:
        host = site if site in CIVITAI_SITES else CIVITAI_SITES[0]
        notes.append(
            f"Source: https://{host}/models/{model_id}"
            f"?modelVersionId={version_id}"
        )
    description = civitai_plain_text(model.get("description"))
    if description:
        notes.append(description)

    entry: dict[str, Any] = {"name": name}
    if architecture:
        entry["architecture"] = str(architecture)
    if notes:
        entry["description"] = "\n\n".join(notes)
    if kind == "lora":
        entry["loras"] = urls
        entry["loras_multipliers"] = [1] * len(urls)
    else:
        names = [
            _civitai_file_name(file, url) for file, url in zip(files, urls)
        ]
        high, low = split_halves(names, urls)
        entry["URLs"] = high
        if low:
            entry["URLs2"] = low
    tags = [
        str(tag).strip()
        for tag in (model.get("tags") or [])
        if isinstance(tag, str) and str(tag).strip()
    ]
    tags = list(dict.fromkeys(["civitai"] + tags))[:12]
    if tags:
        entry["tags"] = tags
    return {"model": entry}
