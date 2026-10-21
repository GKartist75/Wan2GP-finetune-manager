"""Adult pack catalog, downloads and recipe generation.

This module deliberately has no Gradio or Wan2GP imports.  The caller supplies
the checkpoint root, the LoRA-directory resolver and the finetunes directory, so
the plugin UI, tests and future tools all use the same rules.

The catalog is the launcher's curated manifest (bundled as
``adult_pack_manifest.json``).  Checkpoints are recipes written to
``finetunes/adultpack_<id>.json``; LoRAs are loaded from the collection Wan2GP
already scans.  Weights always land under an ``adult-pack`` subfolder so the
curated set stays separate from the user's own files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import quote, unquote, urlencode, urlsplit

try:
    from . import fm_assets
except ImportError:  # pragma: no cover - direct script/import fallback
    import fm_assets

PACK_DIR = "adult-pack"
MANIFEST_NAME = "adult_pack_manifest.json"
STATE_NAME = "pack_state.json"
CUSTOM_NAME = "adult-pack-custom.json"
WEIGHT_EXTENSIONS = (".safetensors", ".gguf", ".sft", ".ckpt", ".pt", ".pth", ".bin")
CIVITAI_SITES = ("civitai.com", "civitai.red")
HF_API = "https://huggingface.co/api"
CIVITAI_API = "https://civitai.red/api/v1"
USER_AGENT = "Wan2GP-FinetuneManager/3.7"
HF_SORTS = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "likes": "likes",
    "newest": "lastModified",
}
CIVITAI_SORTS = {
    "trending": "Trending",
    "downloads": "Most Downloaded",
    "likes": "Most Liked",
    "newest": "Newest",
}
CIVITAI_PERIODS = {"week": "Week", "month": "Month", "year": "Year"}


# ---------------------------------------------------------------------------
# Manifest / state
# ---------------------------------------------------------------------------


def manifest_path() -> Path:
    return Path(__file__).parent / MANIFEST_NAME


def load_manifest(path: str | Path | None = None) -> dict:
    p = Path(path) if path else manifest_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("adult pack manifest is not a JSON object")
    for key in ("loras", "finetunes", "civitai", "groups", "archMap"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def default_state(manifest: dict) -> dict:
    return {
        "enabled": False,
        "ageConfirmed": False,
        "civitaiEnabled": False,
        "selections": {
            "loras": [str(x.get("id")) for x in manifest.get("loras", []) if x.get("defaultSelected")],
            "finetunes": [
                str(x.get("id")) for x in manifest.get("finetunes", []) if x.get("defaultSelected")
            ],
            "civitai": [
                str(x.get("id")) for x in manifest.get("civitai", []) if x.get("defaultSelected")
            ],
        },
        "civitaiPinned": {},
        "civitaiInstalled": {},
        "pins": {},
        "updates": {},
    }


def load_state(path: str | Path | None = None, manifest: dict | None = None) -> dict:
    base = default_state(manifest or load_manifest())
    p = Path(path) if path else manifest_path().with_name(STATE_NAME)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                base.update(data)
        except Exception:
            pass
    for key in ("selections", "civitaiPinned", "civitaiInstalled", "pins", "updates"):
        if not isinstance(base.get(key), dict):
            base[key] = {}
    for key in ("loras", "finetunes", "civitai"):
        vals = base["selections"].get(key, [])
        base["selections"][key] = [str(v) for v in vals] if isinstance(vals, list) else []
    return base


def save_state(state: dict, path: str | Path | None = None) -> Path:
    p = Path(path) if path else manifest_path().with_name(STATE_NAME)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------------------
# User additions
# ---------------------------------------------------------------------------


def custom_path() -> Path:
    return Path(__file__).parent / CUSTOM_NAME


def load_custom(path: str | Path | None = None) -> dict:
    p = Path(path) if path else custom_path()
    data: dict = {}
    if p.is_file():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = raw
        except Exception:
            data = {}
    for key in ("loras", "finetunes"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def save_custom(custom: dict, path: str | Path | None = None) -> Path:
    p = Path(path) if path else custom_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(custom, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


def drop_custom(ident: str, path: str | Path | None = None) -> dict:
    """Forget a user-added model so the catalog stops offering a deleted row."""
    custom = load_custom(path)
    for bucket in ("loras", "finetunes"):
        custom[bucket] = [x for x in custom[bucket] if str(x.get("id")) != str(ident)]
    save_custom(custom, path)
    return custom


def merged_catalog(manifest: dict, custom: dict | None = None) -> dict:
    """Curated manifest plus the user's own additions.

    Custom ids never collide with curated ones (``custom_*`` prefix), and the
    curated copy wins if one ever does so a manifest refresh cannot hide it.
    """
    out = {key: list(manifest.get(key) or []) for key in ("loras", "finetunes", "civitai")}
    out["groups"] = list(manifest.get("groups") or [])
    out["archMap"] = list(manifest.get("archMap") or [])
    for key in ("loras", "finetunes"):
        seen = {str(x.get("id")) for x in out[key]}
        for entry in (custom or {}).get(key) or []:
            if isinstance(entry, dict) and str(entry.get("id")) not in seen:
                out[key].append(entry)
    return out


def pinned_entry(entry: dict, pins: dict | None = None) -> dict:
    """Apply an update pin to one entry without mutating the manifest."""
    pin = (pins or {}).get(str(entry.get("id")))
    if not isinstance(pin, dict):
        return entry
    out = dict(entry)
    sha = str(pin.get("sha256") or "").strip().lower()
    size = int(pin.get("bytes") or 0)
    if sha:
        out["sha256"] = sha
    if size:
        out["bytes"] = size
    return out


def pinned_finetune(entry: dict, pins: dict | None = None) -> dict:
    out = pinned_entry(entry, pins)
    low = (pins or {}).get(f"{entry.get('id')}#low")
    if isinstance(low, dict):
        out = dict(out)
        if low.get("sha256"):
            out["sha256Low"] = str(low["sha256"]).strip().lower()
        if low.get("bytes"):
            out["bytesLow"] = int(low["bytes"])
    return out



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def safe_id(value: str) -> str:
    text = str(value or "").strip()
    return text if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", text) else ""


def pack_weight_url(url: str) -> str:
    """Point a remote model URL at the ``adult-pack`` checkpoint subfolder."""
    value = str(url or "").strip()
    if value.startswith(("http://", "https://")) and "|" not in value:
        return f"{value}|{PACK_DIR}"
    return value


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


def human_bytes(value: int | float) -> str:
    n = float(value or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def lora_arch(entry: dict) -> str:
    ids = entry.get("wan2gpIds") or []
    if isinstance(ids, list) and ids:
        return str(ids[0])
    family = str(entry.get("family") or "").strip()
    return {
        "wan": "i2v_2_2",
        "h3": "minimax_h3_ref2va_pruned",
        "ltx": "ltx2_22B",
        "image": "qwen_image_20B",
    }.get(family, "i2v_2_2")


def lora_destination(entry: dict, lora_dir_for: Callable[[str], str | Path]) -> Path:
    base = Path(lora_dir_for(lora_arch(entry)))
    return base / PACK_DIR / _safe_name(entry.get("file") or entry.get("url") or "")


def checkpoint_assets(entry: dict, ckpts_dir: str | Path) -> list[dict]:
    """Return ``{role, source, destination}`` for a finetune's weight URLs."""
    out: list[dict] = []
    for key, role in (("urls", "model"), ("urls2", "second")):
        values = entry.get(key) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            source = pack_weight_url(value)
            try:
                dest = Path(fm_assets.asset_path(source, role, ckpts_dir, None))
            except Exception as exc:
                out.append(
                    {"role": role, "source": source, "destination": None, "error": str(exc)}
                )
                continue
            out.append({"role": role, "source": source, "destination": dest})
    return out


def recipe_path(entry: dict, finetunes_dir: str | Path) -> Path | None:
    ident = safe_id(str(entry.get("id") or ""))
    return Path(finetunes_dir) / f"adultpack_{ident}.json" if ident else None


def build_recipe(entry: dict, urls: Iterable[str] | None = None) -> dict:
    """Build the Wan2GP finetune JSON for a curated checkpoint entry."""
    values = list(urls if urls is not None else (entry.get("urls") or []))
    urls_out = [pack_weight_url(str(u)) for u in values if str(u or "").strip()]
    model: dict[str, Any] = {
        "name": str(entry.get("name") or ""),
        "architecture": str(entry.get("architecture") or ""),
        "description": str(entry.get("description") or ""),
        "URLs": urls_out,
    }
    urls2 = entry.get("urls2") or []
    if isinstance(urls2, str):
        urls2 = [urls2]
    urls2_out = [pack_weight_url(str(u)) for u in urls2 if str(u or "").strip()]
    if urls2_out:
        model["URLs2"] = urls2_out
    preload = entry.get("preloadLoras") or []
    if isinstance(preload, str):
        preload = [preload]
    preload = [str(x) for x in preload if str(x or "").strip()]
    if preload:
        model["loras"] = preload
    multipliers = entry.get("loraMultipliers")
    if multipliers is None:
        multipliers = entry.get("lorasMultipliers")
    if multipliers is not None:
        model["loras_multipliers"] = multipliers
    note = str(entry.get("licenseNote") or "").strip()
    if note:
        model["infos"] = f"{note}\n\nFetched by the Finetune Manager adult pack."

    root: dict[str, Any] = {"model": model}
    recommended = entry.get("recommended")
    if isinstance(recommended, dict):
        if recommended.get("steps") is not None:
            try:
                root["steps"] = int(recommended["steps"])
            except (TypeError, ValueError):
                pass
        if recommended.get("shift") is not None:
            try:
                root["flow_shift"] = float(recommended["shift"])
            except (TypeError, ValueError):
                pass
        if recommended.get("sampler"):
            root["sampler"] = str(recommended["sampler"])
    return root


def write_recipe(entry: dict, finetunes_dir: str | Path) -> Path:
    p = recipe_path(entry, finetunes_dir)
    if p is None:
        raise ValueError(f"unsafe finetune id: {entry.get('id')}")
    doc = build_recipe(entry)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _file_state(path: Path | None, expected: int = 0) -> dict:
    if path is None:
        return {"path": None, "present": False, "bytes": 0, "expected": expected, "stale": False}
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    present = size > 0
    stale = bool(present and expected and size != expected)
    return {
        "path": str(path),
        "present": present,
        "bytes": size,
        "expected": expected,
        "stale": stale,
    }


def lora_status(
    entry: dict, lora_dir_for: Callable[[str], str | Path], pins: dict | None = None
) -> dict:
    entry = pinned_entry(entry, pins)
    try:
        dest = lora_destination(entry, lora_dir_for)
    except Exception as exc:
        return {"id": entry.get("id"), "error": str(exc), "present": False}
    state = _file_state(dest, int(entry.get("bytes") or 0))
    state["id"] = entry.get("id")
    state["name"] = entry.get("name")
    state["group"] = entry.get("group")
    state["pair"] = entry.get("pair")
    state["custom"] = bool(entry.get("custom"))
    state["family"] = entry.get("family")
    return state


def finetune_status(
    entry: dict,
    ckpts_dir: str | Path,
    finetunes_dir: str | Path,
    pins: dict | None = None,
) -> dict:
    entry = pinned_finetune(entry, pins)
    assets = checkpoint_assets(entry, ckpts_dir)
    rows = []
    for asset in assets:
        dest = asset.get("destination")
        rows.append(
            {
                "role": asset.get("role"),
                "source": asset.get("source"),
                **_file_state(Path(dest) if dest else None),
                "error": asset.get("error"),
            }
        )
    rp = recipe_path(entry, finetunes_dir)
    return {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "architecture": entry.get("architecture"),
        "family": entry.get("family"),
        "custom": bool(entry.get("custom")),
        "recipe": str(rp) if rp else None,
        "recipeWritten": bool(rp and rp.is_file()),
        "assets": rows,
        "weightsPresent": bool(rows) and all(r.get("present") for r in rows),
    }


def catalog_status(
    manifest: dict,
    ckpts_dir: str | Path,
    lora_dir_for: Callable[[str], str | Path],
    finetunes_dir: str | Path,
    state: dict | None = None,
    *,
    custom: dict | None = None,
    pins: dict | None = None,
) -> dict:
    catalog = merged_catalog(manifest, custom)
    pins = pins if pins is not None else (state or {}).get("pins")
    loras = [lora_status(x, lora_dir_for, pins) for x in catalog.get("loras", [])]
    finetunes = [
        finetune_status(x, ckpts_dir, finetunes_dir, pins)
        for x in catalog.get("finetunes", [])
    ]
    civitai = []
    installed = (state or {}).get("civitaiInstalled", {})
    for entry in catalog.get("civitai", []):
        rec = installed.get(str(entry.get("id")), {})
        files = rec.get("files", []) if isinstance(rec, dict) else []
        present = [p for p in files if Path(p).is_file()]
        civitai.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "installed": bool(files) and len(present) == len(files),
                "files": files,
                "present": len(present),
                "expected": len(files),
            }
        )
    return {"loras": loras, "finetunes": finetunes, "civitai": civitai}


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


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
) -> dict:
    """Download one file with resume and SHA-256 verification.

    Returns ``{"status": "exists"|"downloaded", "bytes": n, "path": ...}``.
    """
    import requests

    dest = Path(destination)
    expected = int(expected_bytes or 0)
    digest = str(sha256 or "").strip().lower()
    if dest.is_file():
        size = dest.stat().st_size
        if (not expected or size == expected) and (not digest or sha256_file(dest) == digest):
            if progress:
                progress(expected or size, expected or size, label or dest.name)
            return {"status": "exists", "bytes": size, "path": str(dest)}
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
    headers = {"User-Agent": "Wan2GP-FinetuneManager/3.6"}
    if token and _civitai_host(url):
        headers["Authorization"] = f"Bearer {str(token).strip()}"
    target = with_civitai_token(url, token)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        resp = requests.get(target, headers=headers, stream=True, timeout=(30, 180))
        if offset and resp.status_code == 206:
            mode = "ab"
            total = expected or (offset + int(resp.headers.get("Content-Length") or 0))
        else:
            if offset and resp.status_code == 200:
                part.unlink(missing_ok=True)
                offset = 0
            mode = "wb"
            total = expected or int(resp.headers.get("Content-Length") or 0)
        resp.raise_for_status()
        written = offset
        with part.open(mode) as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if progress:
                    progress(written, total or written, label or dest.name)
        size = part.stat().st_size
        if expected and size != expected:
            raise ValueError(f"size mismatch: expected {expected}, got {size}")
        if digest and sha256_file(part) != digest:
            raise ValueError("SHA-256 mismatch")
        os.replace(part, dest)
        return {"status": "downloaded", "bytes": size, "path": str(dest)}
    except Exception:
        # Keep a valid partial file for resume, but never leave a corrupt one
        # that will pass the next size check.
        if part.is_file() and expected and part.stat().st_size > expected:
            part.unlink(missing_ok=True)
        raise


def fetch_lora(
    entry: dict,
    lora_dir_for: Callable[[str], str | Path],
    *,
    token: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    pins: dict | None = None,
) -> dict:
    entry = pinned_entry(entry, pins)
    dest = lora_destination(entry, lora_dir_for)
    return download_verified(
        str(entry.get("url") or ""),
        dest,
        sha256=entry.get("sha256"),
        expected_bytes=int(entry.get("bytes") or 0),
        token=token,
        progress=progress,
        label=str(entry.get("name") or dest.name),
    )


def fetch_finetune(
    entry: dict,
    ckpts_dir: str | Path,
    *,
    token: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    pins: dict | None = None,
) -> list[dict]:
    entry = pinned_finetune(entry, pins)
    out = []
    assets = checkpoint_assets(entry, ckpts_dir)
    for asset in assets:
        if asset.get("error") or not asset.get("destination"):
            out.append({"ok": False, "error": asset.get("error") or "no destination"})
            continue
        source = str(asset.get("source") or "")
        url = source.split("|", 1)[0]
        is_low = asset.get("role") == "second"
        expected = int(entry.get("bytesLow") or 0) if is_low else int(entry.get("bytes") or 0)
        digest = entry.get("sha256Low") if is_low else entry.get("sha256")
        try:
            out.append(
                download_verified(
                    url,
                    asset["destination"],
                    sha256=digest,
                    expected_bytes=expected,
                    token=token,
                    progress=progress,
                    label=str(entry.get("name") or url),
                )
            )
        except Exception as exc:
            out.append({"ok": False, "error": str(exc), "path": str(asset["destination"])})
    return out


def pair_base(entry: dict) -> str:
    """Normalise a HIGH/LOW LoRA id so both halves share one key.

    ``pairId``/``sourceId`` identify a whole source collection, not one
    high/low pair, so the id itself is the only reliable key.
    """
    ident = str(entry.get("id") or "")
    return re.sub(r"-(?:high|low)(?:-v\d+)?$", "", ident, flags=re.IGNORECASE)


def selected_ids_with_pairs(manifest: dict, ids: Iterable[str]) -> list[str]:
    chosen = {str(x) for x in ids if str(x or "").strip()}
    bases = {
        pair_base(entry)
        for entry in manifest.get("loras", [])
        if str(entry.get("id")) in chosen and entry.get("pair")
    }
    if bases:
        for entry in manifest.get("loras", []):
            if entry.get("pair") and pair_base(entry) in bases:
                chosen.add(str(entry.get("id")))
    return sorted(chosen)


def selected_total_bytes(
    manifest: dict, lora_ids: Iterable[str], finetune_ids: Iterable[str]
) -> int:
    loras = {str(x.get("id")): x for x in manifest.get("loras", [])}
    fins = {str(x.get("id")): x for x in manifest.get("finetunes", [])}
    total = 0
    for ident in selected_ids_with_pairs(manifest, lora_ids):
        total += int((loras.get(ident) or {}).get("bytes") or 0)
    for ident in finetune_ids:
        entry = fins.get(str(ident)) or {}
        total += int(entry.get("bytes") or 0) + int(entry.get("bytesLow") or 0)
    return total


# ---------------------------------------------------------------------------
# CivitAI
# ---------------------------------------------------------------------------


def _civitai_api(
    site: str, path: str, token: str | None, params: dict | None = None
) -> dict:
    import requests

    base = f"https://{site}/api/v1/"
    headers = {"User-Agent": "Wan2GP-FinetuneManager/3.6"}
    key = str(token or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = requests.get(
        base + path.lstrip("/"), headers=headers, params=params or {}, timeout=60
    )
    if resp.status_code in (401, 403):
        if key:
            raise ValueError("CivitAI rejected the stored token (HTTP 401/403)")
        raise ValueError("this CivitAI release needs an API token")
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def resolve_civitai_version(
    spec: dict,
    *,
    site: str = "civitai.red",
    token: str | None = None,
    pinned: int | str | None = None,
) -> int:
    pinned = pinned or spec.get("civitaiVersionId")
    if pinned:
        return int(pinned)
    query = str(spec.get("query") or "").strip()
    if not query:
        raise ValueError("CivitAI entry has no query or pinned version")
    data = _civitai_api(site, "models", token, {"query": query, "limit": 20})
    items = data.get("items") or []
    filter_text = str(spec.get("baseModelFilter") or "").strip().lower()
    for model in items:
        if not isinstance(model, dict):
            continue
        base = str(model.get("baseModel") or "").lower()
        if filter_text and filter_text not in base:
            continue
        versions = model.get("modelVersions") or []
        if versions and isinstance(versions[0], dict) and versions[0].get("id"):
            return int(versions[0]["id"])
    raise ValueError(f'no CivitAI release found for "{query}"')


def civitai_weight_file(file: dict) -> bool:
    ftype = str(file.get("type") or "")
    return not ftype or ftype == "Model"


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


def _civitai_kind(spec: dict, meta: dict) -> str:
    declared = str(spec.get("kind") or "").strip().lower()
    if declared in ("lora", "locon", "dora"):
        return "lora"
    if declared:
        return "checkpoint"
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    api_kind = str(model.get("type") or "").strip().lower()
    return "lora" if api_kind in ("lora", "locon", "dora") else "checkpoint"


def fetch_civitai(
    spec: dict,
    *,
    ckpts_dir: str | Path,
    lora_dir_for: Callable[[str], str | Path],
    finetunes_dir: str | Path,
    site: str = "civitai.red",
    token: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    pinned: int | str | None = None,
) -> dict:
    ident = safe_id(str(spec.get("id") or ""))
    if not ident:
        raise ValueError("unsafe CivitAI id")
    version_id = resolve_civitai_version(spec, site=site, token=token, pinned=pinned)
    meta = _civitai_api(site, f"model-versions/{version_id}", token)
    files = meta.get("files") or []
    if not isinstance(files, list) or not files:
        raise ValueError("this CivitAI release has no downloadable files")
    kind = _civitai_kind(spec, meta)
    if kind == "lora":
        dest_dir = Path(lora_dir_for(str(spec.get("architecture") or lora_arch(spec)))) / PACK_DIR
    else:
        dest_dir = Path(ckpts_dir) / PACK_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    paths: list[str] = []
    total = 0
    for file in files:
        if not isinstance(file, dict) or not civitai_weight_file(file):
            continue
        dl = str(file.get("downloadUrl") or "").strip()
        if not dl:
            continue
        if dl.startswith("/"):
            dl = f"https://{site}{dl}"
        name = _civitai_file_name(file, dl)
        sha = _civitai_sha(file)
        expected = int(round(float(file.get("sizeKB") or 0) * 1024))
        dest = dest_dir / name
        result = download_verified(
            dl,
            dest,
            sha256=sha,
            expected_bytes=0 if sha else expected,
            token=token,
            progress=progress,
            label=str(spec.get("name") or name),
        )
        total += int(result.get("bytes") or 0)
        saved.append(name)
        paths.append(str(dest))
    if not saved:
        raise ValueError("nothing downloadable in this CivitAI release")
    recipe = None
    if kind != "lora":
        high, low = split_halves(saved, paths)
        entry = dict(spec)
        entry["urls"] = high
        entry["urls2"] = low
        recipe = str(write_recipe(entry, finetunes_dir))
    return {
        "ok": True,
        "id": ident,
        "kind": kind,
        "versionId": version_id,
        "files": saved,
        "paths": paths,
        "bytes": total,
        "recipe": recipe,
        "dir": str(dest_dir),
    }


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def lora_inventory(root: str | Path, known_paths: Iterable[str] | None = None) -> list[dict]:
    base = Path(root)
    known = {str(Path(p).resolve()).casefold() for p in (known_paths or [])}
    out: list[dict] = []
    if not base.exists():
        return out
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in WEIGHT_EXTENSIONS:
            continue
        try:
            rel = str(p.relative_to(base)).replace("\\", "/")
            size = p.stat().st_size
        except OSError:
            continue
        out.append(
            {
                "rel": rel,
                "name": p.name,
                "bytes": size,
                "known": str(p.resolve()).casefold() in known,
            }
        )
    out.sort(key=lambda x: x["rel"].lower())
    return out


# ---------------------------------------------------------------------------
# Deletion and update checks
# ---------------------------------------------------------------------------


def delete_lora(entry: dict, lora_dir_for: Callable[[str], str | Path]) -> dict:
    dest = lora_destination(entry, lora_dir_for)
    removed = False
    if dest.is_file():
        dest.unlink()
        removed = True
    try:
        dest.parent.rmdir()
    except OSError:
        pass
    return {"id": entry.get("id"), "removed": removed, "path": str(dest)}


def delete_finetune(
    entry: dict, ckpts_dir: str | Path, finetunes_dir: str | Path
) -> dict:
    rp = recipe_path(entry, finetunes_dir)
    removed_recipe = False
    if rp is not None and rp.is_file():
        rp.unlink()
        removed_recipe = True
    removed_weights: list[str] = []
    for asset in checkpoint_assets(entry, ckpts_dir):
        dest = asset.get("destination")
        if not dest:
            continue
        p = Path(dest)
        if p.is_file():
            p.unlink()
            removed_weights.append(str(p))
            try:
                p.parent.rmdir()
            except OSError:
                pass
    return {
        "id": entry.get("id"),
        "recipe": str(rp) if rp else None,
        "removedRecipe": removed_recipe,
        "removedWeights": removed_weights,
    }


def delete_civitai(
    spec: dict,
    state: dict,
    ckpts_dir: str | Path,
    lora_dir_for: Callable[[str], str | Path],
    finetunes_dir: str | Path,
) -> dict:
    ident = safe_id(str(spec.get("id") or ""))
    installed = state.setdefault("civitaiInstalled", {})
    rec = installed.get(ident, {})
    files = rec.get("files", []) if isinstance(rec, dict) else []
    removed: list[str] = []
    for value in files:
        p = Path(str(value))
        if p.is_file():
            p.unlink()
            removed.append(str(p))
            try:
                p.parent.rmdir()
            except OSError:
                pass
    rp = Path(finetunes_dir) / f"adultpack_{ident}.json"
    removed_recipe = False
    if rp.is_file():
        rp.unlink()
        removed_recipe = True
    installed.pop(ident, None)
    state.setdefault("civitaiPinned", {}).pop(ident, None)
    return {
        "id": ident,
        "removedFiles": removed,
        "removedRecipe": removed_recipe,
    }


# ---------------------------------------------------------------------------
# Find, inspect and add models
# ---------------------------------------------------------------------------


def _http_json(url: str, *, params=None, headers: dict | None = None, timeout: int = 60):
    import requests

    hdrs = {"User-Agent": USER_AGENT}
    hdrs.update(headers or {})
    resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
    if resp.status_code in (401, 403):
        raise ValueError(f"HTTP {resp.status_code}: authentication required")
    resp.raise_for_status()
    data = resp.json()
    return data


def repo_id_ok(repo: str) -> bool:
    parts = str(repo or "").split("/")
    if len(parts) != 2:
        return False
    ok = lambda s: bool(s) and len(s) <= 100 and re.fullmatch(r"[A-Za-z0-9._-]+", s)
    return bool(ok(parts[0]) and ok(parts[1]))


def encode_repo_path(path: str) -> str:
    out = []
    for seg in str(path or "").split("/"):
        part = ""
        for byte in seg.encode("utf-8"):
            char = chr(byte)
            if char == "%":
                part += "%25"
            elif char.isascii() and (char.isalnum() or char in "-_.!~*'()"):
                part += char
            else:
                part += f"%{byte:02X}"
        out.append(part)
    return "/".join(out)


def parse_source_url(value: str) -> tuple[str, str] | None:
    """Resolve a pasted repo, model URL or bare id into ``(source, id)``."""
    text = str(value or "").strip()
    if not text:
        return None
    low = text.lower()
    hf_prefix = "https://huggingface.co/"
    if low.startswith(hf_prefix):
        rest = text[len(hf_prefix) :]
        parts = re.split(r"[/?#]", rest)
        ident = "/".join(parts[:2])
        return ("hf", ident) if repo_id_ok(ident) else None
    for host in ("https://civitai.red/models/", "https://civitai.com/models/"):
        if low.startswith(host):
            digits = ""
            for char in text[len(host) :]:
                if not char.isdigit():
                    break
                digits += char
            return ("civitai", digits) if digits else None
    if "://" not in text:
        if repo_id_ok(text):
            return ("hf", text)
        if text.isdigit():
            return ("civitai", text)
    return None


def arch_entry(manifest: dict, architecture: str) -> dict | None:
    for entry in manifest.get("archMap", []):
        if str(entry.get("architecture")) == str(architecture):
            return entry
    return None


def family_for_arch(manifest: dict, architecture: str) -> str:
    entry = arch_entry(manifest, architecture)
    return str(entry.get("id") or "") if entry else ""


def match_arch(manifest: dict, haystack: str) -> dict | None:
    """Pick the most specific archMap entry whose keyword appears in the text."""
    lower = str(haystack or "").lower()
    best: tuple[int, dict] | None = None
    for entry in manifest.get("archMap", []):
        for needle in entry.get("match") or []:
            text = str(needle or "").strip().lower()
            if text and text in lower and (best is None or len(text) > best[0]):
                best = (len(text), entry)
    return best[1] if best else None


def classify_file(path: str, bytes: int) -> str:
    n = str(path or "").lower()
    if any(k in n for k in ("lora", "adapter", "lightx2v", "joyfox", "dmd")):
        return "lora"
    if bytes > 3 * 1024**3:
        return "checkpoint"
    if bytes < 2 * 1024**3:
        return "lora"
    return "checkpoint"


def precision_of(file: str) -> str:
    n = str(file or "").lower()
    if n.endswith(".gguf"):
        for level in (
            "q4_k_m", "q4_k_s", "q4_k", "q4_1", "q4_0", "q5_k_m", "q5_k",
            "q6_k", "q8_0", "iq3_s", "iq4_xs", "iq4_nl", "q2_k",
        ):
            if level in n:
                return f"gguf_{level}"
        return "gguf"
    if any(k in n for k in ("int8", "convrot", "w4a8", "nf4")):
        return "quanto_bf16_int8"
    if "fp8" in n:
        return "fp8"
    if "fp16" in n or "fp32" in n:
        return "fp16"
    return "bf16"


def slugify(value: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return out[:48]


def map_hf_hit(manifest: dict, item: dict) -> dict | None:
    ident = str(item.get("id") or "")
    if not repo_id_ok(ident):
        return None
    tags = [str(t) for t in (item.get("tags") or []) if isinstance(t, str)]
    arch = match_arch(manifest, f"{ident} {' '.join(tags)}")
    if not arch:
        return None
    card = item.get("cardData") or {}
    widget = (card.get("widget") or [{}])[0] if isinstance(card.get("widget"), list) else {}
    preview = (
        card.get("image_url")
        or card.get("cover_image_url")
        or ((widget or {}).get("output") or {}).get("url")
        or ""
    )
    return {
        "source": "hf",
        "id": ident,
        "title": ident,
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
        "updated": item.get("lastModified") or "",
        "releasedAt": item.get("createdAt") or "",
        "preview": preview,
        "archLabel": str(arch.get("label") or ""),
        "family": str(arch.get("id") or ""),
        "architecture": str(arch.get("architecture") or ""),
        "wan2gpIds": list(arch.get("wan2gpIds") or []),
        "repoPage": f"https://huggingface.co/{ident}",
        "versionId": None,
    }


def map_civitai_hit(manifest: dict, item: dict) -> dict | None:
    title = str(item.get("name") or "")
    base = str(item.get("baseModel") or "")
    arch = match_arch(manifest, f"{title} {base}")
    if not arch:
        return None
    version = (item.get("modelVersions") or [{}])[0] if item.get("modelVersions") else {}
    examples = []
    for image in (version.get("images") or [])[:8]:
        if not isinstance(image, dict) or not image.get("url"):
            continue
        examples.append(
            {
                "url": str(image["url"]),
                "kind": "video" if image.get("type") == "video" else "image",
                "nsfw": image.get("nsfw", 1),
            }
        )
    mid = item.get("id") or 0
    return {
        "source": "civitai",
        "id": mid,
        "title": title,
        "baseModel": base,
        "downloads": int((item.get("stats") or {}).get("monthlyDownloadCount") or 0),
        "likes": int((item.get("stats") or {}).get("favoriteCount") or 0),
        "versionId": version.get("id") or 0,
        "versionName": version.get("name") or "",
        "releasedAt": version.get("createdAt") or "",
        "modelCreatedAt": item.get("createdAt") or "",
        "examples": examples,
        "preview": examples[0]["url"] if examples else "",
        "page": f"https://civitai.red/models/{mid}",
        "archLabel": str(arch.get("label") or ""),
        "family": str(arch.get("id") or ""),
        "architecture": str(arch.get("architecture") or ""),
        "wan2gpIds": list(arch.get("wan2gpIds") or []),
        "repoPage": f"https://civitai.red/models/{mid}",
        "needsToken": True,
    }


def hf_search(manifest: dict, query: str, kind: str = "any", limit: int = 24) -> list[dict]:
    q = str(query or "").strip()
    if len(q) < 2:
        raise ValueError("Type at least two characters")
    params: list[tuple[str, str]] = [
        ("search", q),
        ("limit", str(max(1, min(int(limit or 24), 50)))),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("expand[]", "cardData"),
        ("expand[]", "createdAt"),
    ]
    if kind == "lora":
        params.append(("filter", "lora"))
    elif kind == "gguf":
        params.append(("filter", "gguf"))
    data = _http_json(f"{HF_API}/models", params=params)
    hits = data if isinstance(data, list) else []
    out = [map_hf_hit(manifest, x) for x in hits if isinstance(x, dict)]
    return [x for x in out if x]


def hf_browse(manifest: dict, sort: str = "trending", kind: str = "any", limit: int = 24) -> list[dict]:
    params: list[tuple[str, str]] = [
        ("limit", str(max(1, min(int(limit or 24), 50)))),
        ("sort", HF_SORTS.get(sort, "downloads")),
        ("direction", "-1"),
        ("expand[]", "cardData"),
        ("expand[]", "createdAt"),
    ]
    if kind == "lora":
        params.append(("filter", "lora"))
    elif kind == "gguf":
        params.append(("filter", "gguf"))
    data = _http_json(f"{HF_API}/models", params=params)
    hits = data if isinstance(data, list) else []
    out = [map_hf_hit(manifest, x) for x in hits if isinstance(x, dict)]
    return [x for x in out if x]


def _civitai_types_params(kind: str) -> list[tuple[str, str]]:
    if kind == "lora":
        return [("types", "LORA"), ("types", "DoRA"), ("types", "LoCon")]
    if kind == "checkpoint":
        return [("types", "Checkpoint")]
    return [
        ("types", "Checkpoint"),
        ("types", "LORA"),
        ("types", "DoRA"),
        ("types", "LoCon"),
    ]


def civitai_search(
    manifest: dict,
    query: str,
    kind: str = "any",
    limit: int = 24,
    *,
    token: str | None = None,
    site: str = "civitai.red",
) -> list[dict]:
    q = str(query or "").strip()
    if len(q) < 2:
        raise ValueError("Type at least two characters")
    params = [("query", q), ("limit", str(max(1, min(int(limit or 24), 50)))), ("nsfw", "1")]
    params += _civitai_types_params(kind)
    data = _civitai_api(site, "models", token, params)
    items = data.get("items") or []
    out = [map_civitai_hit(manifest, x) for x in items if isinstance(x, dict)]
    return [x for x in out if x]


def civitai_browse(
    manifest: dict,
    sort: str = "trending",
    period: str = "week",
    kind: str = "any",
    limit: int = 24,
    *,
    token: str | None = None,
    site: str = "civitai.red",
) -> list[dict]:
    params = [
        ("sort", CIVITAI_SORTS.get(sort, "Most Downloaded")),
        ("period", CIVITAI_PERIODS.get(period, "Week")),
        ("limit", str(max(1, min(int(limit or 24), 50)))),
        ("nsfw", "1"),
    ]
    params += _civitai_types_params(kind)
    data = _civitai_api(site, "models", token, params)
    items = data.get("items") or []
    out = [map_civitai_hit(manifest, x) for x in items if isinstance(x, dict)]
    return [x for x in out if x]


def _is_image(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"))


def _is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".webm", ".mov", ".mkv"))


def media_files(files: list[dict]) -> tuple[list[str], list[str]]:
    images: list[tuple[str, int]] = []
    videos: list[tuple[str, int]] = []
    for item in files:
        path = str(item.get("path") or "")
        if not path or path.startswith(".") or "undefined" in path.lower():
            continue
        size = int(item.get("size") or 0)
        if _is_image(path) and size < 6 * 1024**2:
            images.append((path, size))
        elif _is_video(path) and size < 120 * 1024**2:
            videos.append((path, size))
    images.sort(key=lambda x: x[1])
    videos.sort(key=lambda x: x[1], reverse=True)
    return (
        [p for p, _ in images[:12]],
        [p for p, _ in videos[:6]],
    )


def hf_tree(repo: str, rev: str = "main") -> list[dict]:
    if not repo_id_ok(repo):
        raise ValueError("Invalid HuggingFace repository id")
    data = _http_json(
        f"{HF_API}/models/{repo}/tree/{rev}", params={"recursive": "true"}, timeout=90
    )
    return [x for x in (data if isinstance(data, list) else []) if isinstance(x, dict)]


def inspect_hf(repo: str, architecture: str = "") -> dict:
    if not repo_id_ok(repo):
        raise ValueError("Invalid HuggingFace repository id")
    tree = hf_tree(repo)
    files = []
    for item in tree:
        if item.get("type") != "file":
            continue
        path = str(item.get("path") or "")
        size = int(item.get("size") or 0)
        if not path.lower().endswith((".safetensors", ".gguf")) or size < 1024**2:
            continue
        files.append(
            {
                "file": path,
                "bytes": size,
                "kind": classify_file(path, size),
                "pair": detect_pair(path),
                "sha256": ((item.get("lfs") or {}).get("oid") or None),
                "url": f"https://huggingface.co/{repo}/resolve/main/{encode_repo_path(path)}",
                "needsToken": False,
            }
        )
    files.sort(key=lambda x: (str(x.get("kind")), int(x.get("bytes") or 0)))
    images, videos = media_files(tree)
    base = f"https://huggingface.co/{repo}/resolve/main/"
    samples = [
        {"url": base + encode_repo_path(p), "name": _basename(p)} for p in videos
    ]
    preview = base + encode_repo_path(images[0]) if images else ""
    return {
        "ok": True,
        "source": "hf",
        "id": repo,
        "title": repo,
        "repoPage": f"https://huggingface.co/{repo}",
        "preview": preview,
        "architecture": architecture,
        "architectureKnown": True,
        "versions": [],
        "versionId": None,
        "versionName": "",
        "files": files,
        "samples": samples,
    }


def inspect_civitai(
    model_id: str | int,
    *,
    version: str | int | None = None,
    token: str | None = None,
    site: str = "civitai.red",
) -> dict:
    ident = str(model_id or "").strip()
    if not ident.isdigit():
        raise ValueError("Invalid CivitAI model id")
    data = _civitai_api(site, f"models/{ident}", token)
    versions = [x for x in (data.get("modelVersions") or []) if isinstance(x, dict)]
    wanted = str(version or "").strip()
    chosen = next(
        (x for x in versions if str(x.get("id")) == wanted),
        versions[0] if versions else {},
    )
    files = []
    for item in chosen.get("files") or []:
        if not isinstance(item, dict) or not civitai_weight_file(item):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        size = int(round(float(item.get("sizeKB") or 0) * 1024))
        download = str(item.get("downloadUrl") or "")
        if download.startswith("/"):
            download = f"https://{site}{download}"
        files.append(
            {
                "file": name,
                "bytes": size,
                "kind": classify_file(name, size),
                "pair": detect_pair(name),
                "sha256": _civitai_sha(item),
                "url": download,
                "needsToken": bool(download),
            }
        )
    files.sort(key=lambda x: (str(x.get("kind")), int(x.get("bytes") or 0)))
    samples = []
    preview = ""
    for image in (chosen.get("images") or [])[:8]:
        if not isinstance(image, dict) or not image.get("url"):
            continue
        samples.append(
            {
                "url": str(image["url"]),
                "kind": "video" if image.get("type") == "video" else "image",
                "name": _basename(str(image["url"])),
            }
        )
    if samples:
        preview = samples[0]["url"]
    return {
        "ok": True,
        "source": "civitai",
        "id": ident,
        "title": str(data.get("name") or "CivitAI model"),
        "repoPage": f"https://civitai.red/models/{ident}",
        "preview": preview,
        "architecture": "",
        "versions": [
            {
                "id": v.get("id"),
                "name": v.get("name") or "",
                "baseModel": v.get("baseModel") or "",
                "fileCount": len(v.get("files") or []),
            }
            for v in versions
        ],
        "versionId": chosen.get("id"),
        "versionName": chosen.get("name") or "",
        "files": files,
        "samples": samples,
    }


def inspect_source(
    manifest: dict,
    source: str,
    ident: str,
    *,
    architecture: str = "",
    version: str | int | None = None,
    token: str | None = None,
    site: str = "civitai.red",
) -> dict:
    source = str(source or "").strip().lower()
    if source == "hf":
        info = inspect_hf(ident, architecture)
    elif source == "civitai":
        info = inspect_civitai(ident, version=version, token=token, site=site)
    else:
        raise ValueError("Unknown source")
    if architecture:
        info["architecture"] = architecture
    info["architectureKnown"] = bool(arch_entry(manifest, architecture)) if architecture else True
    return info


def hf_gallery(repo: str, rev: str = "main") -> dict:
    tree = hf_tree(repo, rev)
    images, videos = media_files(tree)
    base = f"https://huggingface.co/{repo}/resolve/{rev}/"
    return {
        "ok": True,
        "repo": repo,
        "rev": rev,
        "images": [{"url": base + encode_repo_path(p), "name": _basename(p)} for p in images],
        "videos": [{"url": base + encode_repo_path(p), "name": _basename(p)} for p in videos],
    }


def civitai_gallery(
    spec: dict, *, token: str | None = None, site: str = "civitai.red"
) -> dict:
    model_id = spec.get("civitaiModelId")
    if not model_id:
        version_id = resolve_civitai_version(spec, site=site, token=token)
        meta = _civitai_api(site, f"model-versions/{version_id}", token)
        images = []
        for image in (meta.get("images") or [])[:12]:
            if isinstance(image, dict) and image.get("url"):
                images.append(
                    {
                        "url": str(image["url"]),
                        "nsfw": bool(image.get("nsfw")),
                        "width": image.get("width") or 0,
                        "height": image.get("height") or 0,
                    }
                )
        return {"ok": True, "name": meta.get("name") or "", "images": images}
    data = _civitai_api(site, f"models/{model_id}", token)
    version = (data.get("modelVersions") or [{}])[0]
    images = [
        {
            "url": str(image["url"]),
            "nsfw": bool(image.get("nsfw")),
            "width": image.get("width") or 0,
            "height": image.get("height") or 0,
        }
        for image in (version.get("images") or [])[:12]
        if isinstance(image, dict) and image.get("url")
    ]
    return {"ok": True, "name": version.get("name") or "", "images": images}


def build_custom_entry(
    *,
    source: str,
    title: str,
    file: str,
    url: str,
    architecture: str,
    kind: str = "checkpoint",
    bytes: int = 0,
    sha256: str | None = None,
    repo: str = "",
    repo_path: str = "",
    preview: str = "",
    repo_page: str = "",
    file_page: str = "",
    wan2gp_ids: list[str] | None = None,
    family: str = "",
    needs_token: bool = False,
) -> dict:
    """Turn one inspected file into a catalog entry the existing downloaders understand."""
    if source not in ("hf", "civitai"):
        raise ValueError("Unknown source")
    if not str(url or "").startswith("https://"):
        raise ValueError("A download URL is required")
    name = _safe_name(str(file or ""))
    if not architecture:
        raise ValueError("Pick the Wan2GP architecture first")
    ident = "custom_{}_{}".format(
        "hf" if source == "hf" else "civitai",
        slugify(f"{name}{title}"),
    )
    if not safe_id(ident):
        raise ValueError("Could not build a safe id")
    common = {
        "id": ident,
        "family": family,
        "name": str(title or name),
        "file": name,
        "repoPath": repo_path or file,
        "repo": repo if source == "hf" else "",
        "url": url,
        "sha256": sha256,
        "bytes": int(bytes or 0),
        "license": "user-added",
        "licenseNote": "Fetched directly from its source at your request; verify the licence yourself.",
        "wan2gpIds": list(wan2gp_ids or ([architecture] if architecture else [])),
        "defaultSelected": True,
        "custom": True,
        "preview": preview or "",
        "repoPage": repo_page or "",
        "filePage": file_page or url.replace("/resolve/main/", "/blob/main/"),
    }
    if kind == "lora":
        common.update(
            {
                "group": "custom",
                "pair": detect_pair(name),
                "note": f"Added by you from {repo_page}",
            }
        )
        return common
    return {
        **common,
        "name": f"{title} ({precision_of(name)})",
        "precisionToken": precision_of(name),
        "architecture": architecture,
        "description": f"Added by you from {repo_page}",
        "urls": [url],
        "urls2": [],
        "source": source,
        "prefetchDefault": False,
        "needsToken": bool(needs_token or source == "civitai"),
    }


def add_custom(entry: dict, path: str | Path | None = None) -> dict:
    kind = "loras" if str(entry.get("group")) == "custom" else "finetunes"
    custom = load_custom(path)
    items = [x for x in custom[kind] if str(x.get("id")) != str(entry.get("id"))]
    items.append(entry)
    custom[kind] = items
    save_custom(custom, path)
    return entry


# ---------------------------------------------------------------------------
# Live update checks
# ---------------------------------------------------------------------------


def _repo_path_from_url(url: str) -> str:
    try:
        parts = urlsplit(str(url)).path.split("/")
    except Exception:
        return ""
    if "resolve" in parts:
        idx = parts.index("resolve")
        return unquote("/".join(parts[idx + 2 :]))
    return ""


def _tree_file(tree: list[dict], path: str) -> dict | None:
    target = str(path or "").replace("\\", "/").lstrip("./")
    if not target:
        return None
    base = _basename(target)
    candidates = [x for x in tree if x.get("type") == "file"]
    for item in candidates:
        if str(item.get("path")) == target:
            return item
    for item in candidates:
        if _basename(str(item.get("path") or "")) == base:
            return item
    for item in candidates:
        if str(item.get("path") or "").endswith("/" + target):
            return item
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def live_update_report(
    manifest: dict,
    ckpts_dir: str | Path,
    lora_dir_for: Callable[[str], str | Path],
    finetunes_dir: str | Path,
    state: dict | None = None,
    *,
    custom: dict | None = None,
    token: str | None = None,
    site: str = "civitai.red",
    verify_hashes: bool = False,
) -> dict:
    """Compare local files with what each source serves today.

    Updates are pinned into ``state["pins"]`` and only then used by downloads,
    so an accepted update always carries the new hash and size.
    """
    state = state if state is not None else {}
    catalog = merged_catalog(manifest, custom)
    status = catalog_status(
        manifest, ckpts_dir, lora_dir_for, finetunes_dir, state, custom=custom
    )
    local_loras = {str(x.get("id")): x for x in status["loras"]}
    local_fins = {str(x.get("id")): x for x in status["finetunes"]}
    trees: dict[str, list[dict]] = {}
    items: list[dict] = []
    failed: list[str] = []

    def tree_for(repo: str) -> list[dict] | None:
        if not repo:
            return None
        if repo not in trees:
            try:
                trees[repo] = hf_tree(repo)
            except Exception as exc:
                failed.append(f"{repo}: {exc}")
                trees[repo] = []
        return trees[repo] or None

    for entry in catalog.get("loras", []):
        ident = str(entry.get("id"))
        repo = str(entry.get("repo") or "")
        rel = str(entry.get("repoPath") or _repo_path_from_url(entry.get("url") or ""))
        row = local_loras.get(ident) or {}
        item = {
            "id": ident,
            "kind": "lora",
            "name": entry.get("name"),
            "repo": repo,
            "path": row.get("path"),
            "localBytes": row.get("bytes") or 0,
        }
        tree = tree_for(repo)
        live = _tree_file(tree or [], rel)
        if not live:
            item["status"] = "unknown" if row.get("present") else "missing"
        else:
            item["liveBytes"] = int(live.get("size") or 0)
            item["liveSha"] = str((live.get("lfs") or {}).get("oid") or "")
            local_sha = ""
            if verify_hashes and row.get("present") and row.get("path"):
                try:
                    local_sha = sha256_file(row["path"])
                except Exception:
                    local_sha = ""
            item["localSha"] = local_sha
            if not row.get("present"):
                item["status"] = "missing"
            elif int(row.get("bytes") or 0) != item["liveBytes"]:
                item["status"] = "changed"
            elif item["liveSha"] and local_sha and local_sha != item["liveSha"]:
                item["status"] = "changed"
            else:
                item["status"] = "current"
        items.append(item)

    for entry in catalog.get("finetunes", []):
        ident = str(entry.get("id"))
        repo = str(entry.get("repo") or "")
        row = local_fins.get(ident) or {}
        assets = row.get("assets") or []
        by_role = {str(a.get("role")): a for a in assets}
        halves = [("high", (entry.get("urls") or [None])[0], by_role.get("model"))]
        if entry.get("urls2"):
            halves.append(("low", (entry.get("urls2") or [None])[0], by_role.get("second")))
        tree = tree_for(repo)
        for half, url, asset in halves:
            if not url:
                continue
            live = _tree_file(tree or [], _repo_path_from_url(str(url)))
            item = {
                "id": ident if half == "high" else f"{ident}#low",
                "parent": ident,
                "half": half,
                "kind": "finetune",
                "name": entry.get("name"),
                "repo": repo,
                "path": (asset or {}).get("path"),
                "localBytes": (asset or {}).get("bytes") or 0,
            }
            if not live:
                item["status"] = "unknown" if (asset or {}).get("present") else "missing"
            else:
                item["liveBytes"] = int(live.get("size") or 0)
                item["liveSha"] = str((live.get("lfs") or {}).get("oid") or "")
                present = bool((asset or {}).get("present"))
                if not present:
                    item["status"] = "missing"
                elif int((asset or {}).get("bytes") or 0) != item["liveBytes"]:
                    item["status"] = "changed"
                elif verify_hashes and item["liveSha"] and item.get("path"):
                    try:
                        current = sha256_file(item["path"])
                    except Exception:
                        current = ""
                    item["localSha"] = current
                    item["status"] = (
                        "changed" if current and current != item["liveSha"] else "current"
                    )
                else:
                    item["status"] = "current"
            items.append(item)

    for entry in catalog.get("civitai", []):
        ident = str(entry.get("id"))
        item = {
            "id": ident,
            "kind": "civitai",
            "name": entry.get("name"),
            "status": "unknown",
        }
        try:
            version_id = resolve_civitai_version(entry, site=site, token=token)
            item["liveVersion"] = version_id
            installed = (state.get("civitaiInstalled") or {}).get(ident) or {}
            applied = (state.get("civitaiPinned") or {}).get(ident)
            item["localVersion"] = applied
            if not installed.get("files"):
                item["status"] = "missing"
            elif applied and int(applied) != int(version_id):
                item["status"] = "changed"
            else:
                item["status"] = "current"
        except Exception as exc:
            item["error"] = str(exc)
        items.append(item)

    report = {
        "checkedAt": _now_iso(),
        "items": items,
        "failed": failed,
        "changed": sum(1 for x in items if x.get("status") in ("changed", "missing")),
    }
    state["updates"] = report
    return report


def apply_updates(
    state: dict, ids: Iterable[str]
) -> dict:
    """Pin the live hash/size of the selected update items."""
    wanted = {str(x) for x in ids}
    report = state.get("updates") or {}
    applied: list[dict] = []
    skipped: list[dict] = []
    pins = state.setdefault("pins", {})
    pinned_versions = state.setdefault("civitaiPinned", {})
    civitai_pins: dict[str, int | str] = {}
    for item in report.get("items") or []:
        ident = str(item.get("id"))
        if ident not in wanted:
            continue
        if item.get("kind") == "civitai":
            version = item.get("liveVersion")
            if not version:
                skipped.append({"id": ident, "reason": "no live version"})
                continue
            target = str(item.get("parent") or ident)
            pinned_versions[target] = version
            applied.append({"id": ident, "versionId": version})
            civitai_pins[target] = version
            continue
        if not item.get("liveSha"):
            skipped.append({"id": ident, "reason": "no live hash"})
            continue
        pins[ident] = {
            "sha256": item["liveSha"],
            "bytes": int(item.get("liveBytes") or 0),
            "appliedAt": _now_iso(),
        }
        applied.append(
            {
                "id": ident,
                "sha256": item["liveSha"],
                "bytes": int(item.get("liveBytes") or 0),
            }
        )
    return {
        "ok": True,
        "applied": applied,
        "skipped": skipped,
        "pins": pins,
        "civitaiPinned": civitai_pins,
    }


def update_report(
    manifest: dict,
    ckpts_dir: str | Path,
    lora_dir_for: Callable[[str], str | Path],
    finetunes_dir: str | Path,
    state: dict | None = None,
    *,
    verify_hashes: bool = False,
    custom: dict | None = None,
    pins: dict | None = None,
) -> dict:
    catalog = merged_catalog(manifest, custom)
    pins = pins if pins is not None else (state or {}).get("pins")
    st = catalog_status(
        manifest, ckpts_dir, lora_dir_for, finetunes_dir, state, custom=custom, pins=pins
    )
    loras: list[dict] = []
    for row in st["loras"]:
        status = "ok"
        if not row.get("present"):
            status = "missing"
        elif row.get("stale"):
            status = "stale-size"
        elif verify_hashes:
            expected = str(
                next(
                    (
                        x.get("sha256")
                        for x in catalog.get("loras", [])
                        if str(x.get("id")) == str(row.get("id"))
                    ),
                    "",
                )
                or ""
            ).lower()
            if expected and sha256_file(Path(row["path"])) != expected:
                status = "stale-hash"
        loras.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "status": status,
                "path": row.get("path"),
            }
        )
    finetunes: list[dict] = []
    for row in st["finetunes"]:
        statuses: list[str] = []
        if not row.get("recipeWritten"):
            statuses.append("missing-recipe")
        assets = row.get("assets") or []
        if not assets or not all(a.get("present") for a in assets):
            statuses.append("missing-weights")
        if any(a.get("stale") for a in assets):
            statuses.append("stale-size")
        if verify_hashes:
            entry = next(
                (
                    x
                    for x in catalog.get("finetunes", [])
                    if str(x.get("id")) == str(row.get("id"))
                ),
                None,
            )
            if entry:
                entry = pinned_finetune(entry, pins)
                for asset in assets:
                    if asset.get("present") and asset.get("path"):
                        digest = (
                            entry.get("sha256Low")
                            if asset.get("role") == "second"
                            else entry.get("sha256")
                        )
                        if digest and sha256_file(Path(asset["path"])) != str(digest).lower():
                            statuses.append("stale-hash")
                            break
        finetunes.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "status": ",".join(statuses) if statuses else "ok",
                "recipe": row.get("recipe"),
            }
        )
    civitai: list[dict] = []
    for row in st["civitai"]:
        status = "ok"
        if not row.get("installed"):
            status = "missing"
        elif row.get("present", 0) != row.get("expected", 0):
            status = "partial"
        civitai.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "status": status,
            }
        )
    return {"loras": loras, "finetunes": finetunes, "civitai": civitai}
