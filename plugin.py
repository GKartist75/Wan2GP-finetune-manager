import html
import json
import re as _re
import urllib.error
import urllib.parse
from pathlib import Path
import gradio as gr
# Lazy-imported: HfApi is imported inside functions that use it
# so the plugin can load even if huggingface_hub is missing.
from shared.utils.plugins import WAN2GPPlugin
from shared.gradio.local_file_picker import LocalFilePickerTextbox
import copy
import threading
import time

try:
    from . import fm_assets, fm_pack, fm_security
except ImportError:  # pragma: no cover - direct script/import fallback
    import fm_assets
    import fm_pack
    import fm_security


_MODEL_REF_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

# Runtime input paths do not belong in a finetune recipe. Keeping them out of
# the generated Wan2GP settings file prevents a registry entry from pointing
# the host at an arbitrary local file.
_PATH_SETTING_KEYS = {
    "lst_name",
    "image_start",
    "image_end",
    "image_refs",
    "image_mask",
    "image_guide",
    "video_source",
    "video_guide",
    "video_mask",
    "audio_guide",
    "audio_guide2",
    "audio_source",
    "depth_source",
    "pose_source",
    "mask_source",
}


def _safe_model_ref(model_id) -> str | None:
    """A finetune id usable as a file stem, or ``None``."""
    value = str(model_id or "").strip()
    if value.startswith("="):
        value = value[1:].strip()
    if not value or ".." in value or not _MODEL_REF_RE.fullmatch(value):
        return None
    return value


def _int_or_none(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _set_optional_int(mapping: dict, key: str, raw) -> None:
    number = _int_or_none(raw)
    if number is None:
        mapping.pop(key, None)
    else:
        mapping[key] = max(1, min(number, 100_000))


def _led(ok: bool, msg, *, size: int = 12) -> str:
    """An escaped inline validation badge for a Gradio HTML element."""
    text = html.escape(str(msg or ""), quote=True).replace("\n", "<br/>")
    color = "#16a34a" if ok else "#dc2626"
    icon = "\u2705" if ok else "\u274c"
    return f'<span style="font-size:{size}px;color:{color}">{icon} {text}</span>'


def _check_bare_filenames(data: dict) -> list[str]:
    """Scan model URL fields for bare filenames that need full download URLs.
    Returns a list of warnings.

    preload_URLs is exempt — it can contain bare model-type names that Wan2GP
    resolves internally from its own model database.
    """
    _URL_KEYS = {"URLs", "URLs2", "text_encoder_URLs", "VAE_URLs",
                 "loras", "custom_url_1", "custom_url_2", "custom_url_3"}
    m = data.get("model", {})
    warnings = []
    for key in _URL_KEYS:
        vals = m.get(key, [])
        if isinstance(vals, str):
            # A single string is either a model reference (valid) or a bare
            # filename/URL. Model references must not be flagged.
            if vals.strip().startswith("=") or _is_model_ref_text(vals.strip()):
                continue
            vals = [vals]
        if not isinstance(vals, list):
            continue
        for v in vals:
            if not v or not isinstance(v, str):
                continue
            v = v.strip()
            if not v:
                continue
            if v.startswith(("http://", "https://", "=")):
                continue
            if _is_model_ref_text(v):
                warnings.append(
                    f"'{key}' contains the model reference '{v}' inside a list — "
                    "Wan2GP only resolves a model reference when the field is a "
                    "single string, not a list entry"
                )
                continue
            warnings.append(f"'{key}' has a bare filename: '{v}' — needs a full https:// URL for Wan2GP auto-download")
    return warnings


def _validate_finetune_data(data, *, registry: bool = False) -> str | None:
    """Validate a loaded finetune dict before applying it to Wan2GP.

    Returns a human-readable error message if the data is unusable, or None
    if it is safe to pass to _write_settings_for_wan2gp(). Centralised so the
    registry-load, local-load and import paths all enforce the same contract
    (a dict with a `model` sub-object) instead of crashing deep inside
    _write_settings_for_wan2gp's `dict(data)`.

    ``registry=True`` also requires complete HTTPS download URLs. This is what
    stops a PR like ``Moody_Pro_Mix.json`` (a bare ``.safetensors`` filename
    plus copied CivitAI HTML) from being loaded as a working recipe.
    """
    if not isinstance(data, dict):
        return f"Finetune data is not a JSON object (got {type(data).__name__})"
    if "model" not in data:
        return "Finetune JSON is missing the required 'model' field"
    if not isinstance(data["model"], dict):
        return "'model' field is not a JSON object"
    if registry:
        m = data["model"]
        warnings = _check_bare_filenames(data)
        if warnings:
            return (
                "Registry entry has an incomplete download URL: "
                f"{warnings[0]}"
            )
        for key in (
            "URLs",
            "URLs2",
            "text_encoder_URLs",
            "VAE_URLs",
            "custom_url_1",
            "custom_url_2",
            "custom_url_3",
            "loras",
        ):
            values = m.get(key)
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip()
                if value.startswith("="):
                    continue
                if value.startswith(("http://", "https://")):
                    problem = fm_security.url_problem(value)
                    if problem:
                        return f"Registry entry has an unsafe URL in '{key}': {problem}"
                elif not _is_model_ref_text(value):
                    return (
                        f"Registry entry has a non-HTTPS source in '{key}': "
                        f"'{value}'"
                    )
    return None


def _write_settings_for_wan2gp(fid: str, data: dict) -> "Path":
    """Build a Wan2GP-compatible settings file from finetune data so
    load_settings_from_file() can parse it and trigger model switching.

    ALL top-level fields in `data` pass through verbatim -- this is already
    a complete Wan2GP settings JSON (prompt, activated_loras, guidance, steps,
    resolution, flow_shift, sample_solver, etc.).  Only `model_type` (the
    finetune ID) is added.  Unrecognized keys are ignored by Wan2GP.
    """
    settings = {k: v for k, v in data.items() if k not in _PATH_SETTING_KEYS}
    nested = settings.get("settings")
    if isinstance(nested, dict):
        settings["settings"] = {
            k: v for k, v in nested.items() if k not in _PATH_SETTING_KEYS
        }
    safe_id = _sanitize_fin_id(fid)
    # The model id must match the sanitized finetune filename, otherwise
    # load_settings_from_file() cannot find the model and silently falls back.
    settings["model_type"] = safe_id
    # Promote finetune-format model.loras / model.loras_multipliers to
    # Wan2GP-format top-level activated_loras / loras_multipliers.
    # If top-level already present they win; otherwise pull from model sub-object.
    m = data.get("model", {})
    if "activated_loras" not in settings and "loras" in m:
        loras = m["loras"]
        # A single string is a model-reference reuse (per FINETUNES.md), not a
        # LoRA filename; copying it into activated_loras would break it.
        if isinstance(loras, list):
            settings["activated_loras"] = loras
    if (
        "loras_multipliers" not in settings
        and "loras_multipliers" in m
        and not isinstance(m.get("loras"), str)
    ):
        lms = m["loras_multipliers"]
        if isinstance(lms, list):
            settings["loras_multipliers"] = " ".join(str(x) for x in lms)
        else:
            settings["loras_multipliers"] = str(lms)
    p = _settings_path_for(fid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(settings, indent=4), encoding="utf-8")
    return p


PlugIn_Name = "Finetune Manager"
PlugIn_Id = "FinetuneManager"
PLUGIN_VERSION = "4.0.0"

DEFAULT_REGISTRY = "https://huggingface.co/spaces/GKartist75/wan2gp-finetunes/raw/main"
REGISTRY_SPACE = "GKartist75/wan2gp-finetunes"
ADULT_REGISTRY_SPACE = "GKartist75/wan2gp-finetunes-adult"
FINETUNES_DIR = "finetunes"

# Registry sources: first = community/default upload target. Override with a
# ``registry_spaces`` list in config.json (repo ids, ``owner/name``).
_REGISTRY_SOURCES: list[str] = [REGISTRY_SPACE, ADULT_REGISTRY_SPACE]

_cfg_path = Path(__file__).parent / "config.json"
_cfg_dir = _cfg_path.parent

REGISTRY_TOKEN = ""
CIVITAI_NSFW_DEFAULT = False
CIVITAI_API_KEY = ""
CIVITAI_SITE_DEFAULT = "civitai.com"
ADULT_CONSENT = False


def _persist_config(key: str, value) -> None:
    """Update one key in config.json, preserving all other keys."""
    try:
        cfg = {}
        if _cfg_path.exists():
            cfg = json.loads(_cfg_path.read_text(encoding="utf-8"))
        cfg[key] = value
        _cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

if _cfg_path.exists():
    try:
        _cfg = json.loads(_cfg_path.read_text(encoding="utf-8"))
        REGISTRY_TOKEN = _cfg.get("registry_token", "")
        _rs = _cfg.get("registry_spaces")
        if isinstance(_rs, list):
            _rs = [str(x).strip() for x in _rs if str(x).strip()]
            if _rs:
                _REGISTRY_SOURCES[:] = _rs
        CIVITAI_NSFW_DEFAULT = bool(_cfg.get("civitai_nsfw", False))
        CIVITAI_API_KEY = str(_cfg.get("civitai_api_key", "") or "")
        CIVITAI_SITE_DEFAULT = str(_cfg.get("civitai_site", "civitai.com") or "civitai.com")
        ADULT_CONSENT = bool(_cfg.get("adult_consent", False))
    except Exception:
        pass
_PLACEHOLDER_TOKENS = {"test_token_abc", "hf_YOUR_TOKEN_HERE"}
_REGISTRY_TOKEN_VALID = bool(REGISTRY_TOKEN) and REGISTRY_TOKEN not in _PLACEHOLDER_TOKENS


def _open_config_dir():
    """Open the plugin's config directory in the system file manager."""
    import os
    import subprocess
    import sys
    d = str(_cfg_dir)
    if sys.platform == "win32":
        os.startfile(d)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", d])
    else:
        subprocess.Popen(["xdg-open", d])

_TOKEN_SETUP_GUIDE = f"""
<details style="margin:8px 0;padding:8px 12px;background:#1a1a2e;border:1px solid #eab308;border-radius:8px">
<summary style="cursor:pointer;font-weight:600;color:#eab308">⚠️ Registry Token Not Configured</summary>
<div style="margin-top:8px;color:#d1d5db;font-size:13px;line-height:1.6">
<p>Uploading to the registry and some registry features require a HuggingFace token.
Follow these steps:</p>
<ol>
<li>Go to <a href="https://huggingface.co/settings/tokens" target="_blank" style="color:#60a5fa">huggingface.co/settings/tokens</a></li>
<li>Click <strong>Create new token</strong> — any role works (Read is sufficient for browsing, Write for uploading)</li>
<li>Copy the token (starts with <code>hf_</code>)</li>
<li>Click the <strong>📂 Open plugin folder</strong> button below to open the folder, then open <code>config.json</code> in any text editor and set:</li>
</ol>
<pre style="background:#111827;padding:8px 12px;border-radius:6px;font-size:12px">{{
  "registry_token": "hf_YOUR_TOKEN_HERE"
}}</pre>
<p>Replace <code>hf_YOUR_TOKEN_HERE</code> with your actual token.
<strong>Note:</strong> <code>config.json</code> is in <code>.gitignore</code> — your token stays local.
<strong>Important:</strong> After adding the token, <strong>restart Wan2GP</strong> for the plugin to pick it up.</p>
<p>Alternatively, run <code>huggingface-cli login</code> in your terminal to authenticate globally.</p>
</div>
</details>
"""

# Shared with the built-in Finetune Editor
LORA_FILE_EXTENSIONS = {".safetensors", ".sft"}

# In-memory model id -> display name, populated once at UI build time.
# Validation/sync use this instead of reading JSON from disk on every
# dropdown change, which previously caused the slow "flashing" on load.
MODEL_INDEX: dict[str, str] = {}

# Small cache for Path.exists() — cleared at UI build time so it's fresh
# each module reload, but avoids re-stating the same paths on keystrokes.
DISK_EXISTS_CACHE: dict[str, bool] = {}

# ── Shared URL/LoRA field key lists (defined once, used in 5+ places) ──
URL_VALIDATION_KEYS = [
    "URLs",
    "URLs2",
    "text_encoder_URLs",
    "VAE_URLs",
    "preload_URLs",
    "custom_url_1",
    "custom_url_2",
    "custom_url_3",
    "loras",
    "finetune_source_model",
]
DOWNLOAD_URL_KEYS = [
    "URLs",
    "URLs2",
    "text_encoder_URLs",
    "VAE_URLs",
    "preload_URLs",
    "custom_url_1",
    "custom_url_2",
    "custom_url_3",
]
# Hosts normally used by registry and recipe downloads. Entries on other
# public HTTPS hosts are allowed, but Browse/Local badge them for review.
EXPECTED_DOWNLOAD_HOSTS = (
    "huggingface.co",
    "hf.co",
    "civitai.com",
    "civitai.red",
)
# Indices of URL/LoRA/path fields in ALL_INPUTS (editor tab)
URL_FIELD_IDX = [5, 6, 7, 8, 9, 10, 11, 12]


def _find_token_urls(data: dict) -> list[str]:
    """Return URL/LoRA values that embed a CivitAI `token=` query parameter.

    These work locally because WanGP downloads with the URL, but uploading the
    finetune would publish the API key. Upload paths call this to refuse.
    """
    m = data.get("model", {}) if isinstance(data, dict) else {}
    if not isinstance(m, dict):
        return []
    found = []
    for key in URL_VALIDATION_KEYS:
        vals = m.get(key, [])
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue
        for v in vals:
            if isinstance(v, str) and "token=" in v:
                found.append(f"{key}: {v}")
    return found


def _url_host(value) -> str | None:
    """Return the hostname of an http(s) URL, ignoring ``|subfolder``."""
    text = str(value or "").strip()
    if not text.casefold().startswith(("http://", "https://")):
        return None
    try:
        host = urllib.parse.urlsplit(text.partition("|")[0]).hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


def _unexpected_url_hosts(data) -> list[str]:
    """Download hosts outside the usual Hugging Face / CivitAI families.

    This is advisory only: the URL policy still allows any public HTTPS host.
    Bare model references and local paths are skipped.
    """
    m = data.get("model", {}) if isinstance(data, dict) else {}
    if not isinstance(m, dict):
        return []
    found: list[str] = []
    for key in URL_VALIDATION_KEYS:
        for value in fm_assets.source_strings(m.get(key)):
            if value.startswith("="):
                continue
            host = _url_host(value)
            if host and not fm_security.host_allowed(host, EXPECTED_DOWNLOAD_HOSTS):
                if host not in found:
                    found.append(host)
    return found


def _unexpected_host_warning(data) -> str:
    """Short badge text for unexpected download hosts, or an empty string."""
    hosts = _unexpected_url_hosts(data)
    if not hosts:
        return ""
    shown = ", ".join(hosts[:2])
    if len(hosts) > 2:
        shown += f" +{len(hosts) - 2}"
    return shown


def _unexpected_host_note(url: str) -> str:
    """Advisory text for one URL outside the expected download hosts."""
    host = _url_host(url)
    if host and not fm_security.host_allowed(host, EXPECTED_DOWNLOAD_HOSTS):
        return f"unexpected host '{host}' — verify it yourself"
    return ""


def _upload_host_problem(data) -> str | None:
    """Reject registry uploads whose download URLs use an unexpected host.

    Downloads and local saves remain permissive with a warning badge; only
    publishing to the shared registry is restricted to the usual hosts.
    """
    hosts = _unexpected_url_hosts(data)
    if not hosts:
        return None
    shown = ", ".join(hosts[:4])
    if len(hosts) > 4:
        shown += f" +{len(hosts) - 4} more"
    return (
        "Registry uploads are limited to huggingface.co, hf.co, civitai.com "
        f"and civitai.red (including subdomains); unexpected host(s): {shown}. "
        "You can still save locally and use the entry yourself."
    )


# ── Registry cache: avoids redundant N+1 fetches on rapid Refresh ──
_REGISTRY_CACHE: list[dict] | None = None
_REGISTRY_CACHE_TIME: float = 0
_REGISTRY_CACHE_TTL: float = 30.0  # seconds
_REGISTRY_CACHE_FAILURE_TTL: float = 300.0  # seconds - serve stale on error
_REGISTRY_FAILED_AT: float = 0


def _get_cached_registry(force: bool = False) -> list[dict]:
    """Return cached registry, fetching fresh only if stale or forced.
    On fetch failure, serves stale cache for up to _REGISTRY_CACHE_FAILURE_TTL
    and suppresses repeated refreshes for the same window."""
    global _REGISTRY_CACHE, _REGISTRY_CACHE_TIME, _REGISTRY_FAILED_AT
    now = time.time()
    if not force and _REGISTRY_CACHE is not None and (now - _REGISTRY_CACHE_TIME) < _REGISTRY_CACHE_TTL:
        return _REGISTRY_CACHE
    if (
        not force
        and _REGISTRY_CACHE is not None
        and _REGISTRY_FAILED_AT
        and (now - _REGISTRY_FAILED_AT) < _REGISTRY_CACHE_FAILURE_TTL
    ):
        return _REGISTRY_CACHE
    try:
        fins = _fetch_dynamic_registry_no_cache()
        _REGISTRY_FAILED_AT = 0
    except Exception:
        _REGISTRY_FAILED_AT = now
        # Fetch failed — serve stale cache if available and within failure TTL
        if _REGISTRY_CACHE is not None and (now - _REGISTRY_CACHE_TIME) < _REGISTRY_CACHE_FAILURE_TTL:
            return _REGISTRY_CACHE
        fins = []
    _REGISTRY_CACHE = fins
    _REGISTRY_CACHE_TIME = now
    return fins

# ═══════════════════════════════════════════════════════════════════
# Model families — the UI categories Wan2GP itself uses (LTX-2, MiniMax H3,
# Wan2.2, ...). A finetune's `model.architecture` is a *base model type*;
# its family is resolved by the matching Wan2GP family handler. An explicit
# `model.group` overrides the family, but only when it is a real family key.
# ═══════════════════════════════════════════════════════════════════

# Fallback catalogue (order, label) mirroring the host's query_family_infos()
# calls. The live host catalogue is preferred; this keeps the plugin usable
# when a host version does not expose `families_infos`.
_FAMILY_FALLBACK = {
    "wan": (0, "Wan2.1"),
    "wan2_2": (1, "Wan2.2"),
    "hunyuan": (20, "Hunyuan Video"),
    "hunyuan_1_5": (21, "Hunyuan Video 1.5"),
    "ltxv": (30, "LTX Video"),
    "ltx2": (40, "LTX-2"),
    "kandinsky5": (50, "Kandinsky 5"),
    "kandinsky6": (51, "Kandinsky 6"),
    "longcat": (60, "LongCat"),
    "magi_human": (62, "Magi Human"),
    "minimax_h3": (70, "MiniMax H3"),
    "flux": (1100, "Flux 1"),
    "flux2": (1101, "Flux 2"),
    "flux3": (1102, "Flux 3"),
    "qwen": (1110, "Qwen Image"),
    "qwen_image_2": (1111, "Qwen Image 2.1"),
    "z_image": (1120, "Z-Image"),
    "hidream": (1130, "HiDream"),
    "ideogram4": (1140, "Ideogram"),
    "krea2": (1150, "Krea 2"),
    "ming_image": (1160, "Ming Image 0.1"),
    "moss_image": (1161, "MOSS Image"),
    "sensenova": (1170, "SenseNova"),
    "hunyuan3d": (1300, "Hunyuan3D"),
    "music": (2195, "Music"),
    "tts": (2200, "TTS"),
    "trellis2": (3020, "TRELLIS.2"),
    "triposg": (3030, "TripoSG"),
    "triposplat": (3030, "TripoSplat"),
    "recon3d": (3030, "3D Reconstruction"),
    "animate_any_mesh": (3040, "AnimateAnyMesh"),
    "worldmirror": (3040, "WorldMirror"),
}

# Architecture prefix → family, used only when the host cannot resolve a type
# (older hosts, or registry entries for models that are not installed).
_FAMILY_PREFIX_FALLBACK = (
    ("minimax_h3", "minimax_h3"),
    ("ltx2", "ltx2"),
    ("ltxv", "ltxv"),
    ("hunyuan3d", "hunyuan3d"),
    ("hunyuan_1_5", "hunyuan_1_5"),
    ("hunyuan", "hunyuan"),
    ("kandinsky6", "kandinsky6"),
    ("kandinsky5", "kandinsky5"),
    ("k5_", "kandinsky5"),
    ("k6_", "kandinsky6"),
    ("longcat", "longcat"),
    ("magi", "magi_human"),
    ("flux3", "flux3"),
    ("flux2", "flux2"),
    ("flux", "flux"),
    ("qwen_image_2", "qwen_image_2"),
    ("qwen", "qwen"),
    ("z_image", "z_image"),
    ("hidream", "hidream"),
    ("ideogram", "ideogram4"),
    ("krea2", "krea2"),
    ("ming_image", "ming_image"),
    ("moss_image", "moss_image"),
    ("sensenova", "sensenova"),
    ("trellis", "trellis2"),
    ("triposplat", "triposplat"),
    ("triposg", "triposg"),
    ("animate_any_mesh", "animate_any_mesh"),
    ("worldmirror", "worldmirror"),
    ("da3", "recon3d"),
    ("moge", "recon3d"),
    ("mapanything", "recon3d"),
    ("ace_step", "music"),
    ("stable_audio", "music"),
    ("yue2", "music"),
    ("heartmula", "music"),
    ("minimax_music", "music"),
    ("chatterbox", "tts"),
    ("qwen3_tts", "tts"),
    ("index_tts", "tts"),
    ("kugelaudio", "tts"),
    ("omnivoice", "tts"),
    ("auk", "tts"),
    ("fish_", "tts"),
    ("moss_tts", "tts"),
    ("scenema_audio", "tts"),
    ("dramabox_audio", "tts"),
    ("wan", "wan"),
)

# Distinctive name/URL tokens used only to flag a suspicious architecture.
_FAMILY_TEXT_HINTS = (
    ("z-image", "z_image"),
    ("z_image", "z_image"),
    ("minimax", "minimax_h3"),
    ("ltx", "ltx2"),
    ("hunyuan", "hunyuan"),
    ("kandinsky", "kandinsky5"),
    ("longcat", "longcat"),
    ("magi", "magi_human"),
    ("flux", "flux"),
    ("qwen", "qwen"),
    ("krea", "krea2"),
    ("ideogram", "ideogram4"),
    ("hidream", "hidream"),
    ("sensenova", "sensenova"),
    ("ming_image", "ming_image"),
    ("moss_image", "moss_image"),
    ("trellis", "trellis2"),
    ("tripo", "triposg"),
)

_FAMILY_UNKNOWN = "unknown"


def _family_catalog(plugin=None) -> dict[str, tuple[int, str]]:
    """Return {family_key: (order, label)} from the host when available."""
    catalog: dict[str, tuple[int, str]] = {}
    infos = getattr(plugin, "families_infos", None) if plugin is not None else None
    if isinstance(infos, dict):
        for key, value in infos.items():
            if not key or key == _FAMILY_UNKNOWN:
                continue
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    order = int(value[0])
                except (TypeError, ValueError):
                    order = 9999
                catalog[str(key)] = (order, str(value[1]))
    return catalog or dict(_FAMILY_FALLBACK)


def _family_label(catalog: dict, key: str) -> str:
    if key in catalog:
        return catalog[key][1]
    if key == _FAMILY_UNKNOWN or not key:
        return "Unknown"
    return str(key)


def _family_choices(catalog: dict, keys=None) -> list[tuple[str, str]]:
    """Ordered (label, key) pairs, optionally limited to the given keys."""
    if keys is not None:
        keys = {key for key in keys if key in catalog}
    else:
        keys = set(catalog)
    ordered = sorted(
        keys, key=lambda key: (catalog[key][0], catalog[key][1].casefold())
    )
    return [(catalog[key][1], key) for key in ordered]


def _family_for_model(plugin, model) -> str:
    """Resolve a finetune/model dict to a Wan2GP family key.

    Mirrors Wan2GP: a valid explicit `group` wins, then the family handler
    registered for `architecture`, then a conservative prefix fallback."""
    m = model if isinstance(model, dict) else {}
    catalog = _family_catalog(plugin)
    group = str(m.get("group", "") or "").strip()
    if group in catalog:
        return group
    arch = str(m.get("architecture", "") or "").strip()
    if not arch:
        return _FAMILY_UNKNOWN
    getter = getattr(plugin, "get_model_family", None) if plugin is not None else None
    if callable(getter):
        try:
            family = getter(arch, for_ui=True)
        except TypeError:
            try:
                family = getter(arch)
            except Exception:
                family = None
        except Exception:
            family = None
        if family in catalog:
            return family
    low = arch.casefold()
    for prefix, family in _FAMILY_PREFIX_FALLBACK:
        if low.startswith(prefix) and family in catalog:
            return family
    return _FAMILY_UNKNOWN


def _family_text_hint(model) -> str:
    """Guess the family from the model name/URLs (used for warnings only)."""
    m = model if isinstance(model, dict) else {}
    parts = [str(m.get("name", "") or ""), str(m.get("finetune_source_model", "") or "")]
    urls = m.get("URLs", [])
    if isinstance(urls, str):
        urls = [urls]
    if isinstance(urls, list):
        parts.extend(str(u) for u in urls if isinstance(u, str))
    haystack = " ".join(parts).casefold()
    for token, family in _FAMILY_TEXT_HINTS:
        if token in haystack:
            return family
    return ""


def _family_warning(plugin, model, family: str) -> str:
    """Return a family label when the model name clearly contradicts the
    architecture-derived family, else an empty string."""
    hint = _family_text_hint(model)
    catalog = _family_catalog(plugin)
    if not hint or hint not in catalog or hint == family:
        return ""
    # Only flag against the generic Wan families or an unresolved type;
    # a Wan2.2 finetune mentioning "wan" in its name is not a mismatch.
    if family in ("wan", "wan2_2", _FAMILY_UNKNOWN):
        return catalog[hint][1]
    return ""


def _family_sort_key(catalog: dict, family: str, label: str):
    order, fam_label = catalog.get(family, (9999, str(family or "Unknown")))
    return (order, fam_label.casefold(), str(label or "").casefold())


def _architecture_choices(plugin) -> list[tuple[str, str]]:
    """(label, model_type) choices for the Create/Edit Architecture dropdown.

    Uses Wan2GP's own base-model definitions so the list always matches the
    running host, with the family label as a prefix (e.g. "LTX-2 · ltx2_22B").
    """
    catalog = _family_catalog(plugin)
    lister = getattr(plugin, "list_model_defs", None) if plugin is not None else None
    records = []
    if callable(lister):
        try:
            records = lister(finetune=False) or []
        except TypeError:
            try:
                records = lister() or []
            except Exception:
                records = []
        except Exception:
            records = []
    rows = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        model_type = str(record.get("model_type") or "").strip()
        if not model_type or model_type in seen:
            continue
        seen.add(model_type)
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("finetune"):
            continue
        family = metadata.get("family") or _family_for_model(plugin, record)
        rows.append((family, model_type))
    if not rows:
        # Older hosts: fall back to the files Wan2GP would load.
        rows = [
            (_family_for_model(plugin, {"architecture": mid}), mid)
            for mid, _label in _list_available_models()
        ]
    rows.sort(
        key=lambda row: (
            _family_sort_key(catalog, row[0], ""),
            row[1].casefold(),
        )
    )
    return [
        (f"{_family_label(catalog, family)} \u00b7 {model_type}", model_type)
        for family, model_type in rows
    ]


def _family_payload(plugin=None) -> dict:
    """Build the families.json payload served by the static HF Space page."""
    catalog = _family_catalog(plugin)
    families = {
        key: {"order": int(order), "label": str(label)}
        for key, (order, label) in catalog.items()
    }
    architectures = {}
    lister = getattr(plugin, "list_model_defs", None) if plugin is not None else None
    if callable(lister):
        try:
            records = lister(finetune=False) or []
        except TypeError:
            try:
                records = lister() or []
            except Exception:
                records = []
        except Exception:
            records = []
        for record in records:
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            if metadata.get("finetune"):
                continue
            model_type = str(record.get("model_type") or "").strip()
            family = str(metadata.get("family") or "").strip()
            if model_type and family in families:
                architectures[model_type] = family
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": families,
        "architectures": architectures,
    }


def _collect_url_entries(detail: dict) -> list[str]:
    """Collect all URL/LoRA/path entries from a finetune model dict for validation.
    Handles string/list fields and adds architecture as a model ref prefix."""
    m = detail.get("model", detail)
    entries = []
    for k in URL_VALIDATION_KEYS:
        v = m.get(k)
        if isinstance(v, str):
            entries.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    entries.append(item)
    arch = m.get("architecture", "")
    if arch and isinstance(arch, str) and arch.strip():
        entries.insert(0, f"={arch.strip()}")
    return entries


def _fmt_card_html_item(
    fid: str, name: str, arch: str, author: str, desc: str,
    urls: list[str], loras: list[str], ftags: list | str,
    is_selected: bool, sel_elem_id: str = "fm-selected",
    is_variant: bool = False, family_label: str = "",
    family_warning: str = "", is_pack: bool = False,
    security_warning: str = "", host_warning: str = "",
) -> str:
    """Build a single finetune card HTML fragment.
    Used by both the Browse and Local tab card builders."""
    c = " selected" if is_selected else ""
    safe_name = html.escape(str(name or "?"))
    safe_arch = html.escape(str(arch or ""))
    safe_author = html.escape(str(author or ""))
    safe_desc = html.escape(str(desc or "")[:300])
    safe_family = html.escape(str(family_label or ""))
    badge_parts = []
    if safe_family:
        badge_parts.append(f"<span class='fm-badge'>{safe_family}</span>")
    if safe_arch:
        badge_parts.append(
            f"<span class='fm-badge fm-badge-secondary'>{safe_arch}</span>"
        )
    if not badge_parts:
        badge_parts.append("<span class='fm-badge fm-badge-secondary'>Unknown</span>")
    if is_pack:
        badge_parts.append("<span class='fm-badge fm-badge-pack'>Adult Pack</span>")
    if is_variant:
        badge_parts.append("<span class='fm-badge fm-badge-secondary'>Variant</span>")
    if family_warning:
        badge_parts.append(
            "<span class='fm-badge fm-badge-warn' title='architecture does not "
            f"match the model name'>&#9888; looks like "
            f"{html.escape(str(family_warning))}</span>"
        )
    if security_warning:
        badge_parts.append(
            "<span class='fm-badge fm-badge-warn' title='This entry cannot be "
            "loaded until the download URL is fixed'>&#9888; unsafe entry</span>"
        )
    if host_warning:
        badge_parts.append(
            "<span class='fm-badge fm-badge-warn' title='Download host is not "
            "a usual Hugging Face or CivitAI host; verify it yourself'>&#9888; "
            f"check host: {html.escape(str(host_warning))}</span>"
        )
    tag = " ".join(badge_parts)
    if isinstance(ftags, str):
        ftags_list = [t.strip() for t in ftags.split(",") if t.strip()]
    elif isinstance(ftags, list):
        ftags_list = [str(t).strip() for t in ftags if str(t).strip()]
    else:
        ftags_list = []
    tags_badges = ""
    if ftags_list:
        tags_badges = '<div style="margin-top:2px">'
        for t in ftags_list[:4]:
            et = html.escape(t)
            tags_badges += f'<span class="fm-badge">{et}</span> '
        if len(ftags_list) > 4:
            tags_badges += (
                f'<span style="color:#9ca3af;font-size:10px">'
                f"+{len(ftags_list) - 4}</span>"
            )
        tags_badges += "</div>"
    onclick = (
        f"var r=(window.gradioApp?window.gradioApp():"
        f"document.querySelector('gradio-app'))||document;"
        f"r=r.shadowRoot||r;"
        f"var ta=r.querySelector('#{sel_elem_id} textarea');"
        f"if(ta){{ta.value={html.escape(json.dumps(fid), quote=True)};"
        f"ta.dispatchEvent(new Event('input',{{bubbles:true}}))}}"
    )

    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list):
        urls = []
    urls = [u for u in urls if isinstance(u, str) and u.strip()]
    files_html = ""
    if urls:
        shown = urls[:3]
        extra = len(urls) - 3
        link_parts = []
        for u in shown:
            eu = html.escape(u)
            if u.startswith(("http://", "https://")):
                link_parts.append(
                    f'<a href="{eu}" target="_blank" rel="noopener" '
                    f'style="word-break:break-all">{eu}</a>'
                )
            else:
                link_parts.append(
                    f'<span style="word-break:break-all">{eu}</span>'
                )
        files_html = '<div class="fm-card-files">' + " ".join(link_parts)
        if extra > 0:
            files_html += f' <span style="color:#9ca3af">+{extra} more</span>'
        files_html += "</div>"

    if isinstance(loras, str):
        loras = [loras]
    if not isinstance(loras, list):
        loras = []
    loras = [str(l) for l in loras if isinstance(l, str) and l.strip()]
    loras_html = ""
    if loras:
        shown_l = loras[:3]
        extra_l = len(loras) - 3
        loras_html = '<div class="fm-card-loras"><b>LoRAs:</b> '
        loras_html += " ".join(html.escape(l) for l in shown_l)
        if extra_l > 0:
            loras_html += f' <span style="color:#9ca3af">+{extra_l} more</span>'
        loras_html += "</div>"

    return (
        f'<div class="fm-card{c}" onclick="{html.escape(onclick, quote=True)}">'
        f"<div class='fm-card-title'>{safe_name}</div>"
        f"<div class='fm-card-meta'>{tag} {safe_author}</div>"
        f"<div class='fm-card-desc'>{safe_desc}</div>"
        f"{tags_badges}{files_html}{loras_html}"
        f"</div>"
    )


# ── Known model keys (shared between _build and _extract) ──
KNOWN_MODEL_KEYS = {
    "name",
    "architecture",
    "finetune_source_model",
    "description",
    "infos",
    "prompt_infos",
    "URLs",
    "URLs2",
    "text_encoder_URLs",
    "VAE_URLs",
    "custom_url_1",
    "custom_url_2",
    "custom_url_3",
    "modules",
    "preload_URLs",
    "loras",
    "loras_multipliers",
    "auto_quantize",
    "visible",
    "image_outputs",
    "resolutions",
    "resolutions_categories",
    "tags",
    "text_prompt_enhancer_instructions",
    "text_prompt_enhancer_max_tokens",
    "video_prompt_enhancer_instructions",
    "video_prompt_enhancer_max_tokens",
    "image_prompt_enhancer_instructions",
    "image_prompt_enhancer_max_tokens",
}
KNOWN_TOP_KEYS = {
    "num_inference_steps",
    "guidance_scale",
    "prompt",
    "sample_solver",
    "resolution",
}


def _check_disk_exists(path: str) -> bool:
    """Cached Path.exists() check.
    Uses DISK_EXISTS_CACHE to avoid re-stating the same path on keystrokes.
    """
    if not path or not isinstance(path, str):
        return False
    p = path.strip()
    if not p:
        return False
    if p not in DISK_EXISTS_CACHE:
        DISK_EXISTS_CACHE[p] = Path(p).exists()
    return DISK_EXISTS_CACHE[p]


def build_finetune_dict(
    name,
    arch,
    desc,
    fsm,
    urls,
    urls2,
    te,
    vae,
    pre,
    cu1,
    cu2,
    cu3,
    loras,
    lm,
    mods,
    aq,
    vis,
    img,
    res,
    resc,
    inf,
    pinf,
    steps,
    guid,
    solver,
    pe1,
    pe1t,
    pe2,
    pe2t,
    pe3,
    pe3t,
    tags="",
    prompt="",
    extra_data=None,
):
    """Build finetune model dict from form field values.
    Preserves original key ordering from extra_data when available,
    so round-trips (load → edit → upload → download) keep fields
    in their original positions.
    """

    # ── Utility: type-preserving copy helpers ──
    def _to_typed_multipliers(raw: str):
        # Split on whitespace only: commas are meaningful in step multipliers
        # such as "1,1.1;0" and must not be treated as separators.
        parts = raw.split()
        typed = []
        for p in parts:
            try:
                typed.append(int(p))
            except ValueError:
                try:
                    typed.append(float(p))
                except ValueError:
                    typed.append(p)
        return typed

    def _parse_resolutions(raw_res: str):
        res_out = []
        for l in raw_res.split("\n"):
            l = l.strip()
            if not l:
                continue
            if "x" in l.lower():
                match = _re.match(r"(\d+)\s*x\s*(\d+)", l, _re.IGNORECASE)
                if match:
                    res_out.append([l, f"{match.group(1)}x{match.group(2)}"])
                else:
                    res_out.append(l)
            else:
                res_out.append(l)
        return res_out

    # ── Build model dict ──
    orig_m = extra_data.get("model", {}) if isinstance(extra_data, dict) else {}
    if orig_m:
        # Seed with all original model keys (preserves their order)
        m: dict = {}
        for k, v in orig_m.items():
            m[k] = copy.deepcopy(v)
        # Overwrite with form values (keeps keys at original positions)
        m["name"] = name
        m["architecture"] = arch
        if fsm:
            m["finetune_source_model"] = fsm
        elif "finetune_source_model" in m:
            del m["finetune_source_model"]
        m["description"] = desc
        if inf:
            m["infos"] = inf
        elif "infos" in m:
            del m["infos"]
        if pinf:
            m["prompt_infos"] = pinf
        elif "prompt_infos" in m:
            del m["prompt_infos"]
        for key, raw in (
            ("URLs", urls),
            ("URLs2", urls2),
            ("text_encoder_URLs", te),
            ("VAE_URLs", vae),
        ):
            parsed = _parse_reference_or_lines(raw)
            if parsed is not None:
                m[key] = parsed
            else:
                m.pop(key, None)
        parsed_loras = _parse_reference_or_lines(loras)
        if parsed_loras is not None:
            m["loras"] = parsed_loras
        else:
            m.pop("loras", None)
        if cu1:
            m["custom_url_1"] = cu1
        elif "custom_url_1" in m:
            del m["custom_url_1"]
        if cu2:
            m["custom_url_2"] = cu2
        elif "custom_url_2" in m:
            del m["custom_url_2"]
        if cu3:
            m["custom_url_3"] = cu3
        elif "custom_url_3" in m:
            del m["custom_url_3"]
        if mods:
            m["modules"] = [x.strip() for x in mods.split(",") if x.strip()]
        elif "modules" in m:
            del m["modules"]
        # preload_URLs accepts URLs or a single model-reference string.
        parsed_pre = _parse_reference_or_lines(pre)
        if parsed_pre is not None:
            m["preload_URLs"] = parsed_pre
        else:
            m.pop("preload_URLs", None)
        if lm and isinstance(parsed_loras, list):
            m["loras_multipliers"] = _to_typed_multipliers(lm)
        elif "loras_multipliers" in m:
            del m["loras_multipliers"]
        if aq:
            m["auto_quantize"] = True
        elif "auto_quantize" in m:
            del m["auto_quantize"]
        if not vis:
            m["visible"] = False
        elif "visible" in m:
            del m["visible"]
        if img:
            m["image_outputs"] = True
        elif "image_outputs" in m:
            del m["image_outputs"]
        if tags:
            m["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        elif "tags" in m:
            del m["tags"]
        if res.strip():
            m["resolutions"] = _parse_resolutions(res)
        else:
            m.pop("resolutions", None)
        if resc:
            m["resolutions_categories"] = [x.strip() for x in resc.split("\n") if x.strip()]
        elif "resolutions_categories" in m:
            del m["resolutions_categories"]
        if pe1:
            m["text_prompt_enhancer_instructions"] = pe1
        elif "text_prompt_enhancer_instructions" in m:
            del m["text_prompt_enhancer_instructions"]
        _set_optional_int(m, "text_prompt_enhancer_max_tokens", pe1t)
        if pe2:
            m["video_prompt_enhancer_instructions"] = pe2
        elif "video_prompt_enhancer_instructions" in m:
            del m["video_prompt_enhancer_instructions"]
        _set_optional_int(m, "video_prompt_enhancer_max_tokens", pe2t)
        if pe3:
            m["image_prompt_enhancer_instructions"] = pe3
        elif "image_prompt_enhancer_instructions" in m:
            del m["image_prompt_enhancer_instructions"]
        _set_optional_int(m, "image_prompt_enhancer_max_tokens", pe3t)
    else:
        # No original data — build fresh in canonical order
        m: dict = {}
        m["name"] = name
        m["architecture"] = arch
        if fsm:
            m["finetune_source_model"] = fsm
        m["description"] = desc
        if inf:
            m["infos"] = inf
        if pinf:
            m["prompt_infos"] = pinf
        for key, raw in (
            ("URLs", urls),
            ("URLs2", urls2),
            ("text_encoder_URLs", te),
            ("VAE_URLs", vae),
        ):
            parsed = _parse_reference_or_lines(raw)
            if parsed is not None:
                m[key] = parsed
        if cu1:
            m["custom_url_1"] = cu1
        if cu2:
            m["custom_url_2"] = cu2
        if cu3:
            m["custom_url_3"] = cu3
        if mods:
            m["modules"] = [x.strip() for x in mods.split(",") if x.strip()]
        parsed_pre = _parse_reference_or_lines(pre)
        if parsed_pre is not None:
            m["preload_URLs"] = parsed_pre
        parsed_loras = _parse_reference_or_lines(loras)
        if parsed_loras is not None:
            m["loras"] = parsed_loras
        if lm and isinstance(parsed_loras, list):
            m["loras_multipliers"] = _to_typed_multipliers(lm)
        if aq:
            m["auto_quantize"] = True
        if not vis:
            m["visible"] = False
        if img:
            m["image_outputs"] = True
        if tags:
            m["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        if res.strip():
            m["resolutions"] = _parse_resolutions(res)
        if resc:
            m["resolutions_categories"] = [x.strip() for x in resc.split("\n") if x.strip()]
        if pe1:
            m["text_prompt_enhancer_instructions"] = pe1
        _set_optional_int(m, "text_prompt_enhancer_max_tokens", pe1t)
        if pe2:
            m["video_prompt_enhancer_instructions"] = pe2
        _set_optional_int(m, "video_prompt_enhancer_max_tokens", pe2t)
        if pe3:
            m["image_prompt_enhancer_instructions"] = pe3
        _set_optional_int(m, "image_prompt_enhancer_max_tokens", pe3t)

    # ── Top-level dict: preserve original key order ──
    orig_out = extra_data if isinstance(extra_data, dict) else {}
    if orig_out:
        # Seed with all original top-level keys in order, then overwrite
        out: dict = {}
        for k, v in orig_out.items():
            if k != "model":
                out[k] = copy.deepcopy(v)
        out["model"] = m
        # Overwrite form-provided fields (keeps their original positions)
        out["num_inference_steps"] = int(steps)
        out["guidance_scale"] = float(guid)
        if solver:
            out["sample_solver"] = solver
        else:
            out.pop("sample_solver", None)
        # Clearing the prompt must clear the stored default too.
        out["prompt"] = prompt
    else:
        # Fresh build — canonical order
        out: dict = {}
        out["model"] = m
        out["num_inference_steps"] = int(steps)
        out["guidance_scale"] = float(guid)
        if solver:
            out["sample_solver"] = solver
        if prompt:
            out["prompt"] = prompt

    # Keep the top-level (settings-format) LoRA copy in sync with model.loras.
    # Registry finetunes often carry both, and _write_settings_for_wan2gp
    # gives the top-level copy priority, so a stale copy silently wins.
    if isinstance(parsed_loras, list):
        out["activated_loras"] = parsed_loras
        if lm:
            out["loras_multipliers"] = " ".join(
                str(x) for x in _to_typed_multipliers(lm)
            )
        else:
            out.pop("loras_multipliers", None)
    elif isinstance(parsed_loras, str):
        # Single-string model-reference reuse: it belongs to model.loras only.
        out.pop("activated_loras", None)
        out.pop("loras_multipliers", None)
    else:
        out.pop("activated_loras", None)
        out.pop("loras_multipliers", None)

    return out




def extract_finetune_fields(d: dict) -> tuple:
    """Extract form field values from a finetune dict.
    Returns a tuple matching the _build parameter order (minus fin_id)."""
    m = d.get("model", {})

    def _val(k, fallback_k=None):
        v = d.get(k)
        if v is not None:
            if v or (not isinstance(v, (str, list, tuple))):
                return v
        if fallback_k is not None:
            v = m.get(fallback_k)
            if v is not None:
                return v
        return m.get(k, [])

    def _j(k, fallback_k=None):
        v = _val(k, fallback_k)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return "\n".join(v)
        return ""

    def _s(k, fallback_k=None):
        v = _val(k, fallback_k)
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return str(v) if v else ""

    def _c(k, fallback_k=None):
        v = _val(k, fallback_k)
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return str(v) if v else ""

    # Custom resolutions belong to the finetune definition (model.resolutions).
    # The top-level "resolution" is the author's current default and is not the
    # same field; showing it here used to replace the real custom list on save.
    res = ""
    for r in m.get("resolutions", []) or []:
        if isinstance(r, list) and len(r) >= 2:
            res += r[1] + "\n"
        elif isinstance(r, str):
            res += r + "\n"
    rc = "\n".join(m.get("resolutions_categories", []))
    st = d.get("num_inference_steps")
    if st is None:
        st = m.get("num_inference_steps")
    if st is None:
        st = 30
    gd = d.get("guidance_scale")
    if gd is None:
        gd = m.get("guidance_scale")
    if gd is None:
        gd = 5.0
    tags_raw = m.get("tags", [])
    if isinstance(tags_raw, list):
        tags_out = ", ".join(tags_raw)
    else:
        tags_out = str(tags_raw)
    solver = d.get("sample_solver", "")
    if not solver:
        solver = m.get("sample_solver", "")
    return _clean_utf8(
        (
            m.get("name", ""),
            m.get("architecture", ""),
            m.get("description", ""),
            m.get("finetune_source_model", ""),
            _j("URLs", "URLs"),
            _j("URLs2", "URLs2"),
            _j("text_encoder_URLs", "text_encoder_URLs"),
            _j("VAE_URLs", "VAE_URLs"),
            _j("preload_URLs", "preload_URLs"),
            _j("custom_url_1"),
            _j("custom_url_2"),
            _j("custom_url_3"),
            _j("activated_loras", "loras"),
            _s("loras_multipliers", "loras_multipliers"),
            _c("modules"),
            m.get("auto_quantize", False),
            m.get("visible", True),
            m.get("image_outputs", False),
            res.strip(),
            rc,
            m.get("infos", ""),
            m.get("prompt_infos", ""),
            st,
            gd,
            solver,
            m.get("text_prompt_enhancer_instructions", ""),
            m.get("text_prompt_enhancer_max_tokens", ""),
            m.get("video_prompt_enhancer_instructions", ""),
            m.get("video_prompt_enhancer_max_tokens", ""),
            m.get("image_prompt_enhancer_instructions", ""),
            m.get("image_prompt_enhancer_max_tokens", ""),
            tags_out,
            d.get("prompt", ""),
        )
    )


# ═══════════════════════════════════════════════════════════════════
# P0: URL Validation
# ═══════════════════════════════════════════════════════════════════


def _read_model_json(model_id: str) -> dict:
    """Read a model JSON file from defaults/ or finetunes/.
    Returns empty dict if not found or the id is not a safe file stem."""
    mid = _safe_model_ref(model_id)
    if not mid:
        return {}
    for base in ["defaults", FINETUNES_DIR]:
        p = Path(base) / f"{mid}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _list_available_models() -> list[tuple[str, str]]:
    """List all available model IDs (value, label)."""
    found = []
    seen = set()
    for base in ["defaults", FINETUNES_DIR]:
        d = Path(base)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            mid = f.stem
            if mid in seen:
                continue
            seen.add(mid)
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                name = data.get("model", {}).get("name", mid)
                found.append((mid, f"{name} ({mid})"))
            except Exception:
                found.append((mid, mid))
    return found


def _list_available_loras() -> list[tuple[str, str]]:
    """List available LoRA references from model JSONs.
    Returns [(model_id, label), ...] for models that define loras."""
    found = []
    seen = set()
    for base in ["defaults", FINETUNES_DIR]:
        d = Path(base)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            mid = f.stem
            if mid in seen:
                continue
            seen.add(mid)
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                m = data.get("model", {})
                loras = m.get("activated_loras", m.get("loras", []))
                if loras:
                    name = m.get("name", mid)
                    found.append((mid, f"{name} ({mid})"))
            except Exception:
                pass
    return found


def _pop_extra(vals: tuple) -> tuple[dict | None, tuple]:
    """Extract extra_data dict from the last element of vals if present.
    Form values are always strings/numbers/bools, so a trailing dict
    is assumed to be the extra_data state.
    Returns (extra_data, vals_without_extra)."""
    if vals and isinstance(vals[-1], dict):
        return vals[-1], vals[:-1]
    return None, vals


def _is_model_ref_text(value: str) -> bool:
    """True when a single line is a Wan2GP model id (not a URL/path).

    Wan2GP resolves a model reference only when the field value is a *string*;
    a list entry is treated as a filename.  The ref dropdown writes `=id`, but
    users also paste bare model ids, so recognise both forms.
    """
    value = (value or "").strip()
    if not value or value.startswith(("http://", "https://", "=")):
        return False
    if "/" in value or "\\" in value:
        return False
    if len(value) > 1 and value[1] == ":":
        return False
    if value in MODEL_INDEX:
        return True
    # Per docs/FINETUNES.md, a single bare text in these fields *is* a model
    # id. A bare filename (no directory) is the exception: keep it a list
    # entry so the bare-filename warning can flag it.
    suffix = Path(value).suffix.lower()
    if suffix in {
        ".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf",
        ".json", ".yaml", ".yml", ".txt",
    }:
        return False
    return True


def _parse_reference_or_lines(value: str):
    """Parse a URL/LoRA field, preserving Wan2GP's scalar model-reference form.

    Returns:
      - None for an empty field (the caller removes the key)
      - a string model id for a single `=id` / bare model id
      - a list of URLs or local paths otherwise
    """
    lines = [x.strip() for x in (value or "").split("\n") if x.strip()]
    if not lines:
        return None
    if len(lines) == 1:
        line = lines[0]
        if line.startswith("="):
            ref = line[1:].strip()
            return ref or None
        if _is_model_ref_text(line):
            return line
    return lines


def _get_model_urls(data: dict) -> list[str]:
    """Extract all download URLs from a model JSON dict."""
    m = data.get("model", {})
    urls = []
    for key in (
        "URLs",
        "URLs2",
        "text_encoder_URLs",
        "VAE_URLs",
        "preload_URLs",
        "loras",
    ):
        val = m.get(key, [])
        if isinstance(val, str):
            urls.append(val)
        elif isinstance(val, list):
            urls.extend(val)
    for key in ("custom_url_1", "custom_url_2", "custom_url_3"):
        val = m.get(key, "")
        if isinstance(val, str) and val.strip():
            urls.append(val.strip())
    return [u for u in urls if isinstance(u, str) and u.startswith(("http:", "https:"))]


def _ckpts_dir() -> Path:
    """The checkpoint root Wan2GP resolves files against."""
    from shared.utils import files_locator as fl

    try:
        return Path(fl.get_download_location())
    except Exception:
        return Path("ckpts")


def _lora_dir_for(architecture: str, plugin=None) -> Path:
    """Resolve the architecture's LoRA collection using Wan2GP's own rules."""
    arch = str(architecture or "").strip()
    getter = getattr(plugin, "get_lora_dir", None) if plugin is not None else None
    if callable(getter):
        try:
            resolved = getter(arch)
            if resolved:
                return Path(resolved)
        except Exception:
            pass
    try:
        from shared.lora_paths import resolve_lora_dir

        root_getter = getattr(plugin, "get_lora_root", None) if plugin is not None else None
        root = Path(root_getter()) if callable(root_getter) else Path("loras")
        return Path(resolve_lora_dir(arch or "wan", str(root), {}))
    except Exception:
        safe_arch = arch.replace("/", "").replace("\\", "").strip(".")
        return Path("loras") / (safe_arch or "wan")


def _asset_inventory(data: dict, plugin=None):
    model = data.get("model", {}) if isinstance(data, dict) else {}
    if not isinstance(model, dict):
        return []
    arch = str(model.get("architecture", "") or "")
    return fm_assets.build_inventory(
        model, _ckpts_dir(), _lora_dir_for(arch, plugin)
    )


def _check_model_download_status(data: dict, plugin=None) -> str:
    """Check which recipe files are on disk, including LoRAs and subfolders."""
    assets = _asset_inventory(data, plugin)
    if not assets:
        return ""
    return fm_assets.format_inventory(assets)



def _validate_model_ref(model_id: str, plugin=None) -> tuple[bool, str]:
    """Check if a model reference (=id) exists in defaults/ or finetunes/.
    Returns (is_valid, message) where message includes resolved URL(s)
    and download status."""
    if not model_id or not isinstance(model_id, str) or not model_id.strip():
        return True, ""
    mid = _safe_model_ref(model_id)
    if not mid:
        return False, "Unsafe model id (path separators or '..' are not allowed)"
    msg_parts = []
    if mid in MODEL_INDEX:
        name = MODEL_INDEX[mid]
        data = _read_model_json(mid)
        if data:
            urls = _get_model_urls(data)
            if urls:
                msg_parts.append(f"Found: {name}")
                msg_parts.append(f"\u2192 {urls[0]}")
                dl_status = _check_model_download_status(data, plugin)
                if dl_status:
                    msg_parts.append(dl_status)
                return True, "\n".join(msg_parts)
        return True, f"Found: {name}"
    data = _read_model_json(mid)
    if data:
        name = data.get("model", {}).get("name", mid)
        urls = _get_model_urls(data)
        if urls:
            msg_parts.append(f"Found: {name}")
            msg_parts.append(f"\u2192 {urls[0]}")
            dl_status = _check_model_download_status(data, plugin)
            if dl_status:
                msg_parts.append(dl_status)
            return True, "\n".join(msg_parts)
        return True, f"Found: {name}"
    return False, (
        f"'{mid}' not in Wan2GP model list \u2014 enter a URL or upload a file instead"
    )


def _validate_local_path(path: str) -> tuple[bool, str]:
    """Check if a local file path exists on disk."""
    if not path or not isinstance(path, str) or not path.strip():
        return True, ""
    p = Path(path.strip())
    if p.exists():
        return True, "Found on disk"
    return False, "File not found on disk"


def _validate_entry(
    value: str, plugin=None, *, network: bool = True
) -> tuple[bool, str]:
    """Validate a single URL, model reference (=id), or local path.

    ``network=False`` only performs format checks, so editor keystrokes never
    contact a community-supplied host.
    """
    if not value or not isinstance(value, str):
        return True, ""
    v = value.strip()
    if not v:
        return True, ""
    if v.startswith("="):
        return _validate_model_ref(v[1:], plugin)
    if v.startswith(("http:", "https:")):
        if not network:
            _remote, clean_url, _sub, _name = fm_assets.parse_source(v)
            problem = fm_security.url_problem(clean_url or v)
            if problem:
                return False, problem
            note = _unexpected_host_note(clean_url or v)
            if note:
                return True, f"URL format OK; {note}"
            return True, "URL format OK"
        return _validate_url(v)
    if v.startswith(("/", ".", "\\")) or (len(v) > 1 and v[1] == ":"):
        return _validate_local_path(v)
    # Could be a bare model ID stored in the URL/LoRA field
    # Check MODEL_INDEX first for a fast cache hit
    if v in MODEL_INDEX:
        return True, f"Model ref: {MODEL_INDEX[v]}"
    data = _read_model_json(v)
    if data:
        name = data.get("model", {}).get("name", v)
        urls = _get_model_urls(data)
        if urls:
            dl_status = _check_model_download_status(data, plugin)
            msg = f"Model ref: {name}\n\u2192 {urls[0]}"
            if dl_status:
                msg += f"\n{dl_status}"
            return True, msg
        return True, f"Model ref: {name}"
    return False, (
        f"'{v}' not in Wan2GP model list \u2014 enter a URL or upload a file instead"
    )


def _validate_urls(
    urls: list[str], plugin=None, *, network: bool = True
) -> list[tuple[str, bool, str]]:
    """Validate a list of URLs. Returns [(url, valid, message), ...].
    Skips non-URL entries."""
    results = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        valid, msg = _validate_entry(u, plugin, network=network)
        if msg or not valid:
            results.append((u, valid, msg))
    return results


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate that a URL is reachable via a guarded HEAD/GET request.

    The ``|subfolder`` / ``|%lora_dir`` suffix is a recipe instruction, not
    part of the URL, so it is stripped before the request. Redirects are
    checked against the same URL policy, which stops metadata validation from
    being used as an SSRF probe.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return True, ""
    is_remote, clean_url, subfolder, _name = fm_assets.parse_source(url.strip())
    if not is_remote or not clean_url:
        return True, "Not a remote URL (local path)"
    problem = fm_security.url_problem(clean_url)
    if problem:
        return False, problem
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinetuneManager/3.0)"}
    if CIVITAI_API_KEY and fm_assets.is_civitai_url(clean_url):
        headers.update(_civitai_api_headers())
    code, error = fm_security.probe_url(clean_url, headers=headers)
    if error:
        return False, error
    suffix = f" (saves to {subfolder})" if subfolder else ""
    note = _unexpected_host_note(clean_url)
    note = f"; {note}" if note else ""
    if code and code < 400:
        return True, f"HTTP {code}{suffix}{note}"
    if code == 404:
        return False, "HTTP 404 - Not Found"
    if code == 401:
        return False, "HTTP 401 - Unauthorized (may need token)"
    if code == 403:
        return False, "HTTP 403 - Forbidden (may need token)"
    return False, f"HTTP {code}"



def _build_url_validation_html(results: list[tuple[str, bool, str]]) -> str:
    """Build HTML displaying per-entry validation results."""
    if not results:
        return "<div style='color:#6b7280;padding:4px'>All URLs, model refs, and local paths look valid</div>"
    parts = ['<div style="max-height:300px;overflow-y:auto">']
    for url, valid, msg in results:
        color = "#16a34a" if valid else "#dc2626"
        icon = "\u2713" if valid else "\u2717"
        parts.append(
            f'<div style="padding:3px 4px;border-bottom:1px solid #f3f4f6;font-size:12px">'
            f'<span style="color:{color};font-weight:600">{icon}</span> '
            f'<span style="word-break:break-all">{html.escape(url)}</span> '
            f'<span style="color:{color};font-size:11px">{html.escape(msg)}</span>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════
# P1: Pre-Download
# ═══════════════════════════════════════════════════════════════════


def _download_asset(asset, plugin=None) -> str:
    """Download one inventory row to its resolved destination."""
    label = asset.filename or asset.source
    if asset.present:
        return f"Already exists: {asset.label}: {label}"
    if asset.error:
        return f"Failed: {asset.label}: {label} - {asset.error}"
    if not asset.downloadable or not asset.url or not asset.destination:
        return f"Skipped (local only): {asset.label}: {label}"
    dest = Path(asset.destination)
    try:
        # Recipe URLs may point at any public HTTPS host, but redirects and
        # DNS are checked by fm_security before a byte is written.
        fm_pack.download_verified(
            asset.url,
            dest,
            token=CIVITAI_API_KEY,
            allowed_hosts=None,
            label=asset.label,
        )
        if dest.is_file() and dest.stat().st_size > 0:
            return f"Downloaded {asset.label}: {dest}"
        return f"Failed: {asset.label}: {label} - download produced no file"
    except Exception as e:
        return f"Failed: {asset.label}: {label} - {e}"


def _download_finetune_files(data: dict, plugin=None) -> str:
    """Download every missing, downloadable asset in a finetune recipe.

    The inventory is the single source of truth: it already resolved
    ``|subfolder`` / ``|%lora_dir``, LoRA routing and model references.
    """
    assets = _asset_inventory(data, plugin)
    if not assets:
        return "No file assets found."
    lines = []
    for asset in assets:
        if asset.present:
            lines.append(f"Already exists: {asset.label}: {asset.filename}")
        else:
            lines.append(_download_asset(asset, plugin))
    DISK_EXISTS_CACHE.clear()
    # Re-read after the transfers so the summary reflects what is on disk now.
    summary = fm_assets.inventory_summary(_asset_inventory(data, plugin))
    lines.append(
        f"{summary['present']}/{summary['total']} present, "
        f"{summary['missing']} missing"
    )
    return "\n".join(lines)


CIVITAI_API = "https://civitai.com/api/v1"
# civitai.red is the adult front-end of the same service: same catalogue, same ids,
# same /enums, same nsfw=true behaviour, but it hands out download URLs on its own
# host. Mature files are gated behind a login on both hosts, which is what the
# optional civitai_api_key in config.json is for.
CIVITAI_SITES = ("civitai.com", "civitai.red")


def _civitai_site(value: str) -> str:
    """Normalise a site choice; anything unknown falls back to civitai.com."""
    site = (value or "").strip().lower()
    for prefix in ("https://", "http://"):
        if site.startswith(prefix):
            site = site[len(prefix):]
    site = site.strip("/")
    return site if site in CIVITAI_SITES else CIVITAI_SITES[0]


def _civitai_base(site: str) -> str:
    """API base URL for the chosen site."""
    return f"https://{_civitai_site(site)}/api/v1"


def _civitai_download_url(site: str, version_id) -> str:
    """Fallback download URL when the API response carries none."""
    return f"https://{_civitai_site(site)}/api/download/models/{version_id}"


def _civitai_api_headers() -> dict:
    """Request headers for CivitAI API calls; adds the API key when configured."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinetuneManager/3.0)"}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    return headers


CIVITAI_SITE_DEFAULT = _civitai_site(CIVITAI_SITE_DEFAULT)


CIVITAI_FALLBACK_ENUMS = {
    "ModelType": [
        "Checkpoint", "LORA", "LoCon", "DoRA", "TextualInversion", "Hypernetwork",
        "AestheticGradient", "Controlnet", "Upscaler", "MotionModule", "VAE",
        "TextEncoder", "Poses", "Wildcards", "Other",
    ],
    "BaseModel": [
        "Wan Video", "Wan Video 1.3B t2v", "Wan Video 14B t2v", "Wan Video 14B i2v 480p",
        "Wan Video 14B i2v 720p", "Wan Video 2.2 TI2V-5B", "Wan Video 2.2 I2V-A14B",
        "Wan Video 2.2 T2V-A14B", "LTXV", "LTXV2", "LTXV 2.3", "LTXV 2.5", "MiniMax H3",
        "Hunyuan Video", "Mochi", "CogVideoX", "Flux.1 S", "Flux.1 D", "Flux.1 Krea",
        "Flux.1 Kontext", "Flux.2 D", "Flux.2 Klein 9B", "Flux.2 Klein 9B-base",
        "Flux.2 Klein 4B", "Flux.2 Klein 4B-base", "Chroma", "Qwen", "Qwen 2.1",
        "ZImageTurbo", "ZImageBase", "Krea 2", "HiDream", "HiDream-O1", "Ideogram 4.0",
        "Ming Image Design 0.1", "ACE Audio", "MiniMax Music 3", "YuE2", "Illustrious",
        "NoobAI", "Pony", "Pony V7", "SDXL 1.0", "SDXL Lightning", "SDXL Turbo",
        "SD 1.5", "SD 2.1", "SD 3", "SD 3.5", "Other",
    ],
}
_CIVITAI_ENUM_KEYS = ("ModelType", "BaseModel")
_civitai_enums_cache: dict = {}
_civitai_enums_loading = False
_civitai_enums_lock = threading.Lock()


def _civitai_retry_delay(error, default: float, cap: float = 10.0) -> float:
    """Seconds to wait before retrying: honour Retry-After on a 429 response."""
    try:
        raw = (error.headers.get("Retry-After") or "").strip()
    except Exception:
        raw = ""
    if raw:
        try:
            return max(0.0, min(float(raw), cap))
        except ValueError:
            pass
    return default


def _civitai_get_json(url: str, timeout: int = 20, attempts: int = 3):
    """GET a CivitAI API URL as JSON, retrying the transient failures that
    otherwise abort a search: dropped TLS handshakes, resets, timeouts, HTTP 5xx
    and rate limiting (429, honouring Retry-After when the server sends one)."""
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            with fm_security.open_url(
                url,
                headers=_civitai_api_headers(),
                timeout=timeout,
                allowed_hosts=("civitai.com", "civitai.red"),
            ) as resp:
                body = resp.read(fm_security.MAX_JSON_BYTES + 1)
            if len(body) > fm_security.MAX_JSON_BYTES:
                raise ValueError("CivitAI response is too large")
            return json.loads(body.decode())
        except urllib.error.HTTPError as e:
            if last or (e.code != 429 and e.code < 500):
                raise
            delay = _civitai_retry_delay(e, 1.0) if e.code == 429 else 0.7 * (attempt + 1)
        except (urllib.error.URLError, OSError):
            if last:
                raise
            delay = 0.7 * (attempt + 1)
        time.sleep(delay)


def _civitai_enum_choices(key: str) -> list:
    """Dropdown choices for a CivitAI enum: live values once fetched, otherwise the
    built-in list. Never performs network I/O, so it is safe to call while the
    plugin UI is being built."""
    return ["All"] + (
        _civitai_enums_cache.get(key) or CIVITAI_FALLBACK_ENUMS.get(key, [])
    )


def _civitai_enum_updates(*current: str) -> list:
    """Refresh the Type / Base Model dropdowns to the live enum values while keeping
    the user's current selection. Applied after a search or when the tab is opened,
    so the filters upgrade from the built-in list without ever stalling the UI."""
    updates = []
    for value, key in zip(current, _CIVITAI_ENUM_KEYS):
        choices = _civitai_enum_choices(key)
        updates.append(
            gr.update(choices=choices, value=value if value in choices else "All")
        )
    return updates


def _civitai_load_enums() -> bool:
    """Fetch /enums once and cache it (one request fills every key).
    Returns True when live values were cached; never raises.
    Both hosts serve an identical enum payload, so this always uses the standard one."""
    try:
        data = _civitai_get_json(f"{CIVITAI_API}/enums", timeout=5, attempts=2)
    except Exception:
        return False
    found = {
        key: [v for v in (data.get(key) or []) if isinstance(v, str) and v]
        for key in _CIVITAI_ENUM_KEYS
    }
    if not all(found.values()):
        return False
    _civitai_enums_cache.update(found)
    return True


def _civitai_prefetch_enums() -> None:
    """Warm the enum cache on a background thread. Building the UI must never wait on
    CivitAI, so an unreachable or slow API costs the user nothing at startup."""
    global _civitai_enums_loading
    if _civitai_enums_cache:
        return
    with _civitai_enums_lock:
        if _civitai_enums_loading:
            return
        _civitai_enums_loading = True

    def _worker():
        global _civitai_enums_loading
        try:
            _civitai_load_enums()
        finally:
            _civitai_enums_loading = False

    threading.Thread(target=_worker, name="civitai-enums", daemon=True).start()


def _civitai_search(
    query: str,
    model_type: str = "",
    base_model: str = "",
    limit: int = 12,
    nsfw: bool = False,
    site: str = "civitai.com",
    sort: str = "Most Downloaded",
) -> dict:
    """Search CivitAI models. Returns raw API response dict with 'items' key.

    With a Type / Base Model filter, the API's Meilisearch `query` applies the
    filters to each page of text matches only, so a filtered search mostly came
    back empty. Filtered searches therefore merge three concurrent lookups: the
    `query` search paged by cursor, the server-side `tag` filter, and a local
    name/tag/creator match over the most downloaded models for those filters.

    `site` picks the host to query (civitai.com or civitai.red). Both serve the
    same catalogue; the difference is which host the returned download URLs use."""
    filters = [("nsfw", "true")] if nsfw else []
    if model_type and model_type != "All":
        filters.append(("types", model_type))
    if base_model and base_model != "All":
        filters.append(("baseModels", base_model))

    def _fetch(params):
        return _civitai_get_json(
            f"{_civitai_base(site)}/models?" + urllib.parse.urlencode(params)
        )

    if not any(key in ("types", "baseModels") for key, _ in filters):
        try:
            base_params = filters + [("sort", sort), ("limit", str(limit))]
            if query:
                base_params.insert(len(filters), ("query", query))
            return _fetch(base_params)
        except Exception as e:
            return {"error": str(e), "items": []}

    words = query.lower().split()

    def _matches(item):
        tags = [t for t in (item.get("tags") or []) if isinstance(t, str)]
        creator = (item.get("creator") or {}).get("username") or ""
        haystack = " ".join([item.get("name") or "", creator, *tags]).lower()
        return all(w in haystack for w in words)

    def _paged(params, pages, keep=None):
        found, cursor = [], None
        for _ in range(pages):
            data = _fetch(params + ([("cursor", cursor)] if cursor else []))
            found += [i for i in data.get("items", []) if keep is None or keep(i)]
            cursor = (data.get("metadata") or {}).get("nextCursor")
            if len(found) >= limit or not cursor:
                break
        return found

    lookups = [lambda: _paged(filters + [("sort", sort), ("limit", "100")], 2, _matches)]
    if query:
        lookups.insert(0, lambda: _paged(filters + [("query", query), ("limit", "100")], 3))
        lookups.insert(1, lambda: _paged(filters + [("tag", query.lower()), ("limit", "100")], 1))

    from concurrent.futures import ThreadPoolExecutor

    items, seen, errors = [], set(), []
    with ThreadPoolExecutor(max_workers=len(lookups)) as pool:
        futures = [pool.submit(fn) for fn in lookups]
    for future in futures:
        try:
            found = future.result()
        except Exception as e:
            errors.append(str(e))
            continue
        for item in found:
            if item.get("id") not in seen:
                seen.add(item.get("id"))
                items.append(item)
    if not items and errors:
        return {"error": errors[0], "items": []}
    return {"items": items[:limit]}


def _civitai_render_results(data: dict) -> str:
    """Render CivitAI search results as HTML cards."""
    items = data.get("items", []) if "error" not in data else []
    error = data.get("error", "")
    if error:
        return (
            f"<div style='color:#dc2626;padding:16px'>Error: {html.escape(error)}</div>"
        )
    if not items:
        return (
            "<div style='color:#9ca3af;text-align:center;padding:32px'>No results</div>"
        )

    parts = ['<div style="max-height:65vh;overflow-y:auto">']
    for model in items:
        if not isinstance(model, dict):
            continue
        name = html.escape(str(model.get("name") or "?"))
        mtype = html.escape(str(model.get("type") or "?"))
        desc = html.escape(str(model.get("description") or "")[:200])
        nsfw = model.get("nsfw", False)
        versions = model.get("modelVersions", [])
        if not isinstance(versions, list):
            versions = []

        nsfw_html = (
            ' <span style="color:#dc2626;font-size:10px">NSFW</span>' if nsfw else ""
        )
        parts.append(
            f'<div style="border:1px solid var(--border-color-primary,#e5e7eb);border-radius:8px;padding:10px;'
            f'margin-bottom:6px;background:var(--body-background-fill,#fff)">'
            f'<div style="font-size:13px;font-weight:600;color:var(--body-text-color,#111827)">'
            f"{name}{nsfw_html}"
            f' <span style="font-size:10px;color:var(--body-text-color-subdued,#6b7280);font-weight:400">({mtype})</span>'
            f"</div>"
            f'<div style="font-size:11px;color:var(--body-text-color,#374151);line-height:1.4;'
            f'max-height:2.6em;overflow:hidden">{desc}</div>'
        )

        if versions:
            parts.append(
                '<div style="margin-top:6px;padding-top:4px;border-top:1px solid var(--border-color-primary,#f3f4f6)">'
            )
            for v in versions:
                if not isinstance(v, dict):
                    continue
                vname = html.escape(str(v.get("name") or "?"))
                base = html.escape(str(v.get("baseModel") or ""))
                files = v.get("files", [])
                if not isinstance(files, list):
                    files = []
                primary = next(
                    (
                        f
                        for f in files
                        if isinstance(f, dict) and f.get("primary", False)
                    ),
                    files[0] if files and isinstance(files[0], dict) else None,
                )
                fname = html.escape(str(primary.get("name") or "") if primary else "")
                try:
                    fsize = float(primary.get("sizeKB") or 0) if primary else 0
                except (TypeError, ValueError):
                    fsize = 0
                size_str = f"{fsize / 1024:.1f}MB" if fsize else "?"
                parts.append(
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'padding:3px 0;font-size:12px">'
                    f'<span style="color:var(--body-text-color,#374151);flex:1">'
                    f"<b>{vname}</b> — {base} — {size_str}</span>"
                    f'<span style="color:var(--body-text-color-subdued,#6b7280);font-size:11px">{fname}</span>'
                    f"</div>"
                )
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _civitai_result_options(data: dict, site: str = "civitai.com") -> list[tuple[str, str]]:
    """Build the Version dropdown choices for a CivitAI search result.

    Each value is the JSON payload consumed by _civitai_extract_fill_data, so
    selection uses real Gradio state instead of script-injected HTML."""
    items = data.get("items", []) if "error" not in data else []
    options = []
    for model in items:
        model_name = model.get("name", "?") or "?"
        for version in model.get("modelVersions", []):
            label = (
                f"{model_name} — {version.get('name', '?') or '?'}"
                f" ({version.get('baseModel', '') or '?'})"
            )
            payload = json.dumps(
                {"model": model, "version": version, "site": _civitai_site(site)},
                default=str,
            )
            options.append((label, payload))
    return options


def _civitai_selection(civitai_json_str) -> dict | None:
    """Parse the JSON payload stored in the CivitAI result dropdown.

    Returns ``{"model", "version", "site"}`` or ``None`` when nothing valid is
    selected. Every action (fill, download, recipe) shares this validation so
    a stale or empty dropdown value produces a clear message.
    """
    if not civitai_json_str or civitai_json_str == "{}":
        return None
    if isinstance(civitai_json_str, dict):
        data = civitai_json_str
    else:
        try:
            data = json.loads(civitai_json_str)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    model = data.get("model")
    version = data.get("version")
    if not isinstance(model, dict) or not isinstance(version, dict):
        return None
    if not version.get("files") and not version.get("downloadUrl"):
        return None
    return {
        "model": model,
        "version": version,
        "site": _civitai_site(data.get("site") or CIVITAI_SITE_DEFAULT),
    }


def _civitai_selection_name(sel: dict) -> str:
    model = sel.get("model") or {}
    version = sel.get("version") or {}
    name = str(model.get("name") or "CivitAI model").strip()
    vname = str(version.get("name") or "").strip()
    return f"{name} \u2014 {vname}" if vname else name


def _civitai_recipe_id(sel: dict) -> str:
    model = sel.get("model") or {}
    version = sel.get("version") or {}
    return _sanitize_fin_id(
        f"civitai_{model.get('id') or 'model'}_{version.get('id') or 'version'}"
    )


# ── Hugging Face search engine ─────────────────────────────────────────────
# Same role as the CivitAI engine above: search, result cards, weight-file
# selection, editor fill and finetune.json recipes. The HF Hub API returns a
# plain list; the weight model is a models/{repo}?blobs=true call.

_HF_API = "https://huggingface.co/api"
# One set of labels drives both engines (ci_sort dropdown); the values map to
# each API's sort parameter.
_CIVITAI_SORTS = {
    "Trending": "Trending",
    "Most downloads": "Most Downloaded",
    "Most likes": "Most Liked",
    "Newest updated": "Newest",
}
_HF_SORTS = {
    "Trending": "trendingScore",
    "Most downloads": "downloads",
    "Most likes": "likes",
    "Newest updated": "lastModified",
}


def _hf_api_get(path: str, params=None, timeout: int = 20, attempts: int = 3):
    """GET a Hugging Face API endpoint as JSON with the same retry policy as
    the CivitAI helper (transient TLS/reset/5xx/429 failures)."""
    url = path if path.startswith("http") else f"{_HF_API}/{path.lstrip('/')}"
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    headers = _civitai_api_headers()
    if _REGISTRY_TOKEN_VALID:
        headers["Authorization"] = f"Bearer {REGISTRY_TOKEN}"
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            with fm_security.open_url(
                url,
                headers=headers,
                timeout=timeout,
                allowed_hosts=("huggingface.co", "hf.co"),
            ) as resp:
                body = resp.read(fm_security.MAX_JSON_BYTES + 1)
            if len(body) > fm_security.MAX_JSON_BYTES:
                raise ValueError("Hugging Face response is too large")
            return json.loads(body.decode())
        except urllib.error.HTTPError as e:
            if last or (e.code != 429 and e.code < 500):
                raise
            delay = _civitai_retry_delay(e, 1.0) if e.code == 429 else 0.7 * (attempt + 1)
        except (urllib.error.URLError, OSError):
            if last:
                raise
            delay = 0.7 * (attempt + 1)
        time.sleep(delay)


def _hf_search(query: str, sort: str = "Trending", adult_only: bool = False, limit: int = 24) -> dict:
    """Search Hugging Face models. ``adult_only`` restricts to repos tagged
    ``not-for-all-audiences`` (the Hub has no server-side 'exclude adult')."""
    params = [
        ("sort", _HF_SORTS.get(sort, "trendingScore")),
        ("direction", "-1"),
        ("limit", str(int(limit or 24))),
    ]
    query = (query or "").strip()
    if query:
        params.append(("search", query))
    if adult_only:
        params.append(("filter", "not-for-all-audiences"))
    items = _hf_api_get("models", params)
    if not isinstance(items, list):
        items = items.get("items", []) if isinstance(items, dict) else []
    return {"items": [m for m in items if isinstance(m, dict)]}


def _hf_render_results(items: list) -> str:
    """Render HF search results as HTML cards (name, metrics, tags)."""
    parts = ['<div style="max-height:65vh;overflow-y:auto">']
    for item in items[:60]:
        if not isinstance(item, dict):
            continue
        repo = str(item.get("id") or item.get("repoId") or "?")
        name = html.escape(repo)
        author = html.escape(str(repo.split("/", 1)[0] if "/" in repo else "?"))
        downloads = item.get("downloads")
        likes = item.get("likes")
        downloads = int(downloads) if isinstance(downloads, (int, float)) else 0
        likes = int(likes) if isinstance(likes, (int, float)) else 0
        updated = str(item.get("lastModified") or "")[:10]
        raw_tags = (item.get("tags") or [])[:5]
        tags = [html.escape(str(t)) for t in raw_tags if isinstance(t, str)]
        adult = "not-for-all-audiences" in tags
        adult_html = (
            ' <span style="color:#dc2626;font-size:10px">18+</span>'
            if adult
            else ""
        )
        tag_html = " · ".join(tags)
        extra = (
            f'<div style="font-size:10px;color:var(--body-text-color-subdued,#6b7280)">{tag_html}</div>'
            if tag_html
            else ""
        )
        parts.append(
            '<div style="border:1px solid var(--border-color-primary,#e5e7eb);'
            'border-radius:8px;padding:10px;margin-bottom:6px;'
            'background:var(--body-background-fill,#fff)">'
            '<div style="font-size:13px;font-weight:600;'
            'color:var(--body-text-color,#111827)">'
            f"{name}{adult_html}"
            '&nbsp;<span style="font-size:10px;font-weight:400;'
            f'color:var(--body-text-color-subdued,#6b7280)">{author}</span>'
            "</div>"
            '<div style="font-size:11px;color:var(--body-text-color,#374151)">'
            f"{downloads:,} downloads · {likes:,} likes · updated {updated}"
            "</div>"
            + extra
            + "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _hf_result_options(items: list) -> list:
    """Result dropdown choices for HF search results.

    The value is the JSON payload the pick and fill handlers share:
    ``{"engine":"hf","repo":id,"name":...}``."""
    options = []
    for item in items:
        if not isinstance(item, dict):
            continue
        repo = str(item.get("id") or item.get("repoId") or "").strip()
        if not repo:
            continue
        name = str(item.get("name") or repo.rsplit("/", 1)[-1] or repo)
        downloads = item.get("downloads")
        likes = item.get("likes")
        stats = []
        if isinstance(downloads, (int, float)):
            stats.append(f"{int(downloads)}d")
        if isinstance(likes, (int, float)):
            stats.append(f"{int(likes)}l")
        label = f"{name} — {repo}" + (f" ({', '.join(stats)})" if stats else "")
        options.append(
            (
                label,
                json.dumps(
                    {"engine": "hf", "repo": repo, "name": name}, ensure_ascii=False
                ),
            )
        )
    return options


def _hf_selection(civitai_json_str):
    """Parse a ``{"engine":"hf","repo":...}`` payload, or None when invalid."""
    if not civitai_json_str or civitai_json_str == "{}":
        return None
    try:
        data = (
            json.loads(civitai_json_str)
            if isinstance(civitai_json_str, str)
            else civitai_json_str
        )
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("engine") != "hf":
        return None
    repo = str(data.get("repo") or "").strip()
    if not repo or not _re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return None
    return {"repo": repo, "name": str(data.get("name") or "")}


def _hf_model_files(repo: str) -> list:
    """Weight files of one HF repo: ``{name,size,sha,url}`` entries.

    Uses ``blobs=true`` so each sibling carries the LFS SHA-256 and byte size.
    Only loadable weight extensions are kept (fm_assets.WEIGHT_EXTENSIONS)."""
    data = _hf_api_get(f"models/{repo}", params=[("blobs", "true")])
    siblings = data.get("siblings", []) if isinstance(data, dict) else []
    out = []
    for s in siblings:
        if not isinstance(s, dict):
            continue
        name = str(s.get("rfilename") or "").strip()
        if not name.lower().endswith(fm_assets.WEIGHT_EXTENSIONS):
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(name, safe='/')}"
        size = s.get("size") or (s.get("lfs") or {}).get("size") or 0
        sha = str((s.get("lfs") or {}).get("oid") or "").strip().lower()
        out.append(
            {
                "name": name,
                "size": int(size) if isinstance(size, (int, float)) else 0,
                "sha": sha if len(sha) == 64 else "",
                "url": url,
            }
        )
    out.sort(key=lambda f: (0 if "/" not in f["name"] else 1, f["name"]))
    return out


def _hf_human_size(size) -> str:
    size = float(size or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _hf_classify_file(name: str) -> str:
    """Map an HF weight file to the recipe field it belongs in."""
    low = str(name or "").lower()
    base = low.rsplit("/", 1)[-1]
    stem = low.split("/", 1)[-1] if "/" in low else low
    if "vae" in base or "/vae" in low:
        return "vae"
    if any(k in base or k in stem for k in ("text_encoder", "t5", "umt5", "clip")):
        return "te"
    return "main"


def _hf_fill_updates(
    payload,
    picked_names,
    arch="",
    current_name="",
    current_desc="",
    current_tags="",
    current_loras="",
    current_loras_multipliers="",
    force_kind: str = "Auto",
):
    """Build the Create/Edit updates (ALL_INPUTS[1:] layout) for an HF fill.

    Composite repos set URLs/URLs2 (HIGH/LOW pairs), text_encoder_URLs and
    VAE_URLs by keyword; LoRA-kind selections go to the LoRA list the way
    CivitAI LoRAs do. ``force_kind`` overrides the size/pair heuristic."""
    sel = payload if isinstance(payload, dict) else None
    if not sel or not sel.get("repo"):
        return _civitai_none_fill(
            "Select a Hugging Face result and pick weight files first."
        )
    repo = str(sel.get("repo"))
    name = str(sel.get("name") or repo.rsplit("/", 1)[-1])
    if not picked_names:
        return _civitai_none_fill(
            f"List {repo}'s weight files and pick at least one first."
        )
    by_name = {f["name"]: f for f in (sel.get("files") or [])}
    if any(n not in by_name for n in picked_names):
        return _civitai_none_fill(
            "The weight-file list is stale — open the repo again."
        )

    picked = [by_name[n] for n in picked_names]
    mains = [f for f in picked if _hf_classify_file(f["name"]) == "main"]
    te = [f for f in picked if _hf_classify_file(f["name"]) == "te"]
    vaes = [f for f in picked if _hf_classify_file(f["name"]) == "vae"]
    names = [f["name"] for f in mains]
    urls = [f["url"] for f in mains]
    high_url, low_url = fm_pack.split_halves(names, urls)
    big = max((f.get("size") or 0 for f in mains), default=0) >= 2 * 1024 * 1024 * 1024
    pair = bool(high_url and low_url)
    if force_kind == "LoRA":
        kind = "lora"
    elif force_kind == "Checkpoint":
        kind = "checkpoint"
    else:
        kind = "lora" if (len(picked) == 1 and not pair and not big) else "checkpoint"

    result = [gr.update()] * 33
    if not str(current_name or "").strip():
        result[0] = gr.update(value=name)
    if arch:
        result[1] = gr.update(value=str(arch))
    if not str(current_desc or "").strip():
        result[2] = gr.update(
            value=f"Hugging Face repo: {repo} — {len(picked)} weight file(s)"
        )
    if not str(current_tags or "").strip():
        result[31] = gr.update(value="huggingface")

    if kind == "lora":
        existing = [
            line.strip()
            for line in str(current_loras or "").splitlines()
            if line.strip()
        ]
        known = {l.split("#", 1)[0] for l in existing}
        appended = []
        for f in mains:
            if f["url"].split("#", 1)[0] in known:
                continue
            if len(existing) + len(appended) >= 10:
                break
            appended.append(f["url"])
        if not appended and not te and not vaes:
            return _civitai_none_fill("This LoRA URL is already in the LoRA list.")
        existing.extend(appended)
        mults = str(current_loras_multipliers or "").split()
        mults += ["1"] * (len(existing) - len(mults))
        result[12] = gr.update(value="\n".join(existing))
        result[13] = gr.update(value=" ".join(mults[: len(existing)]))
        status = f"Added {len(appended)} LoRA URL(s) from `{repo}` ({len(existing)}/10)."
    else:
        main_urls = list(high_url)
        main_urls += [
            f["url"] for f in mains if f["url"] not in high_url and f["url"] not in low_url
        ]
        if main_urls:
            result[4] = gr.update(value="\n".join(main_urls))
        if low_url:
            result[5] = gr.update(value="\n".join(low_url))
        if te:
            result[6] = gr.update(value="\n".join(f["url"] for f in te))
        if vaes:
            result[7] = gr.update(value="\n".join(f["url"] for f in vaes))
        status = f"Filled {len(picked)} file URL(s) from `{repo}`."
    return tuple(result + [status])


def _hf_recipe_payload(sel: dict, picked_names, architecture="", force_kind: str = "Auto"):
    """Finetune-JSON payload for an HF repo selection (same structure as
    CivitAI recipes: token-free public URLs, sha256 where available)."""
    repo = sel["repo"]
    name = str(sel.get("name") or repo.rsplit("/", 1)[-1])
    by_name = {f["name"]: f for f in (sel.get("files") or [])}
    picked = [by_name[n] for n in picked_names if n in by_name]
    if not picked:
        raise ValueError("no known weight files — open the repo again")

    mains = [f for f in picked if _hf_classify_file(f["name"]) == "main"]
    te = [f for f in picked if _hf_classify_file(f["name"]) == "te"]
    vaes = [f for f in picked if _hf_classify_file(f["name"]) == "vae"]
    names = [f["name"] for f in mains]
    urls = [f["url"] for f in mains]
    high_url, low_url = fm_pack.split_halves(names, urls)
    pair = bool(high_url and low_url)
    big = max((f.get("size") or 0 for f in mains), default=0) >= 2 * 1024 * 1024 * 1024
    if force_kind == "LoRA":
        kind = "lora"
    elif force_kind == "Checkpoint":
        kind = "checkpoint"
    else:
        kind = "lora" if (len(picked) == 1 and not pair and not big) else "checkpoint"

    entry = {"name": name}
    if architecture:
        entry["architecture"] = str(architecture)
    entry["description"] = f"Hugging Face repo: {repo} ({len(picked)} weight file(s))"
    urls_notes = []
    if kind == "lora":
        entry["loras"] = [f["url"] for f in mains]
        entry["loras_multipliers"] = [1] * len(entry["loras"])
    else:
        main_urls = list(high_url)
        main_urls += [
            f["url"]
            for f in mains
            if f["url"] not in high_url and f["url"] not in low_url
        ]
        entry["URLs"] = main_urls
        if low_url:
            entry["URLs2"] = low_url
    if te:
        entry["text_encoder_URLs"] = [f["url"] for f in te]
    if vaes:
        entry["VAE_URLs"] = [f["url"] for f in vaes]
    entry["tags"] = ["huggingface"]
    return {"model": entry}



def _state_model_type(state_value) -> str:
    """Best-effort read of the current Wan2GP model type from the host state."""
    if isinstance(state_value, str):
        try:
            state_value = json.loads(state_value)
        except (json.JSONDecodeError, TypeError):
            return ""
    if not isinstance(state_value, dict):
        return ""
    for key in ("model_type", "architecture"):
        value = state_value.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = state_value.get("settings")
    if isinstance(nested, dict):
        for key in ("model_type", "architecture"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    model = state_value.get("model")
    if isinstance(model, dict):
        value = model.get("architecture")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _civitai_none_fill(message: str) -> tuple:
    return tuple([gr.update()] * 33 + [message])


def _civitai_extract_fill_data(
    civitai_json_str,
    arch="",
    current_name="",
    current_desc="",
    current_tags="",
    current_loras="",
    current_loras_multipliers="",
) -> tuple:
    """Fill Create/Edit from a CivitAI result selection.

    Returns updates matching ``ALL_INPUTS[1:]`` plus a status message. The
    result dropdown stores the full model/version JSON, so no other field
    changes unless the user has left it empty. A missing selection returns a
    visible message instead of silently doing nothing.
    """
    sel = _civitai_selection(civitai_json_str)
    if sel is None:
        return _civitai_none_fill(
            "Select a result version first, then press *Fill Selected*."
        )
    model = sel["model"]
    version = sel["version"]
    site = sel["site"]
    kind = fm_pack.civitai_kind(model.get("type"))
    if kind is None:
        return _civitai_none_fill(
            f"CivitAI type '{model.get('type') or 'unknown'}' is not a "
            "checkpoint or LoRA, so there is no weight URL to fill."
        )
    files = fm_pack.civitai_weight_files(
        version.get("files") or [], primary_only=True
    )
    if not files:
        return _civitai_none_fill("This release has no downloadable weight file.")
    url = fm_pack.civitai_recipe_url(files[0], site)
    name = _civitai_selection_name(sel)

    result = [gr.update()] * 33
    if not str(current_name or "").strip():
        result[0] = gr.update(value=name)
    if arch:
        result[1] = gr.update(value=str(arch))
    if not str(current_desc or "").strip():
        desc = fm_pack.civitai_plain_text(model.get("description"))
        if desc:
            result[2] = gr.update(value=desc)
    if not str(current_tags or "").strip():
        tags = [
            str(tag).strip()
            for tag in (model.get("tags") or [])
            if isinstance(tag, str) and str(tag).strip()
        ]
        if tags:
            result[31] = gr.update(value=", ".join(tags))

    if kind == "lora":
        existing = [
            line.strip()
            for line in str(current_loras or "").splitlines()
            if line.strip()
        ]
        base_url = url.split("#", 1)[0]
        if any(line.split("#", 1)[0] == base_url for line in existing):
            status = f"'{name}' is already in the LoRA list."
        elif len(existing) >= 10:
            status = "The LoRA list is full (10 max); remove one first."
        else:
            existing.append(url)
            mults = str(current_loras_multipliers or "").split()
            mults += ["1"] * (len(existing) - len(mults))
            result[12] = gr.update(value="\n".join(existing))
            result[13] = gr.update(value=" ".join(mults[: len(existing)]))
            status = (
                f"Added LoRA URL from {name} ({len(existing)}/10). "
                "Open **Create / Edit \u2192 LoRAs** to review it."
            )
    else:
        result[4] = gr.update(value=url)
        status = (
            f"Set the main checkpoint URL from {name}. "
            "Open **Create / Edit \u2192 URLs** to review it."
        )
    return tuple(result + [status])


# ═══════════════════════════════════════════════════════════════════
# ponytail: inline helpers
# ═══════════════════════════════════════════════════════════════════


def _registry_raw_url(repo: str) -> str:
    """Raw-file base URL for a registry Space."""
    return f"https://huggingface.co/spaces/{repo}/raw/main"


def _registry_auth_headers() -> dict:
    """Authorization header for raw registry fetches (gated/private Spaces)."""
    if _REGISTRY_TOKEN_VALID:
        return {"Authorization": f"Bearer {REGISTRY_TOKEN}"}
    return {}


def _registry_target_for(data) -> str:
    """Upload target repo for a finetune def: adult-tagged entries go to the
    adult registry, everything else to the community Space."""
    tags = data.get("model", {}).get("tags") if isinstance(data, dict) else None
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    if isinstance(tags, list):
        for tag in tags:
            if str(tag or "").strip().lower() == "adult":
                return ADULT_REGISTRY_SPACE
    return _REGISTRY_SOURCES[0]


def _upload_token_problem() -> str | None:
    """Return why upload is unavailable, or ``None`` when a token exists."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return (
            "`huggingface_hub` is not installed in this Wan2GP environment. "
            "Install it with `pip install huggingface_hub` and restart Wan2GP."
        )
    if _REGISTRY_TOKEN_VALID:
        return None
    try:
        api = HfApi()
        user = api.whoami()
        if user and user.get("name"):
            return None
    except Exception:
        pass
    return (
        "\u26a0\ufe0f **No HF token configured.** "
        "Upload requires a HuggingFace token.\n\n"
        "1. Go to https://huggingface.co/settings/tokens\n"
        "2. Create a token (any role)\n"
        f"3. Edit `{_cfg_path}` and add:\n"
        '   `{"registry_token": "hf_..."}`\n'
        "Or run `huggingface-cli login` in your terminal."
    )


def _stamp_hf_username(json_data: dict) -> None:
    """Look up the authenticated HF user and stamp model.author in-place.
    Fails silently — upload proceeds without author if whoami() fails."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if _REGISTRY_TOKEN_VALID:
            api = HfApi(token=REGISTRY_TOKEN)
        user = api.whoami()
        username = user.get("name", "")
        if username:
            json_data.setdefault("model", {})["author"] = username
    except Exception:
        pass


def _publish_family_index(api, plugin=None, repo: str | None = None) -> None:
    """Refresh families.json on the HF Space (best effort).

    The static Space page uses it to group finetunes by Wan2GP model family.
    Failures are ignored: the finetune upload itself must not be affected.
    """
    try:
        payload = _family_payload(plugin)
        if not payload.get("architectures"):
            return
        api.upload_file(
            path_or_fileobj=json.dumps(payload, indent=2, sort_keys=True).encode(),
            path_in_repo="families.json",
            repo_id=repo or _REGISTRY_SOURCES[0],
            repo_type="space",
        )
    except Exception:
        pass


def _hf_upload(fin_id, json_data, plugin=None, target: str | None = None):
    """Upload a finetune JSON to the HF Space.
    Tries direct commit first; falls back to PR for community submissions.
    ``target`` selects the registry Space; default routes ``adult``-tagged
    entries to the adult registry, everything else to the community Space."""
    err = _validate_finetune_data(json_data, registry=True)
    if err:
        raise ValueError(err)
    host_problem = _upload_host_problem(json_data)
    if host_problem:
        raise ValueError(host_problem)
    token_urls = _find_token_urls(json_data)
    if token_urls:
        raise ValueError(
            f"{len(token_urls)} URL(s) embed a CivitAI API token (`token=`); "
            "remove the token before uploading."
        )
    repo = target or _registry_target_for(json_data)
    from huggingface_hub import HfApi
    api = HfApi()
    if _REGISTRY_TOKEN_VALID:
        api = HfApi(token=REGISTRY_TOKEN)
    safe_id = _sanitize_fin_id(fin_id)
    blob = json.dumps(json_data, indent=2).encode()
    direct_ok = False
    try:
        # Try direct commit first (works for Space owner)
        api.upload_file(
            path_or_fileobj=blob,
            path_in_repo=f"finetunes/{safe_id}.json",
            repo_id=repo,
            repo_type="space",
        )
        direct_ok = True
    except Exception:
        # Fallback: create a PR (works for community contributors)
        api.upload_file(
            path_or_fileobj=blob,
            path_in_repo=f"finetunes/{safe_id}.json",
            repo_id=repo,
            repo_type="space",
            create_pr=True,
        )
    if direct_ok:
        _publish_family_index(api, plugin, repo)
    return repo


def _fetch_registry_json(fin_id, source: str | None = None):
    fid = _safe_model_ref(fin_id)
    if not fid:
        raise ValueError("unsafe finetune id")
    sources = [source] if source in _REGISTRY_SOURCES else list(_REGISTRY_SOURCES)
    last_err: Exception | None = None
    for repo in sources:
        url = f"{_registry_raw_url(repo)}/finetunes/{fid}.json"
        try:
            with fm_security.open_url(
                url,
                headers={
                    "User-Agent": "Wan2GP-FinetuneManager/3.7",
                    **_registry_auth_headers(),
                },
                timeout=15,
                allowed_hosts=("huggingface.co", "hf.co"),
            ) as resp:
                body = resp.read(fm_security.MAX_JSON_BYTES + 1)
            if len(body) > fm_security.MAX_JSON_BYTES:
                raise ValueError("registry JSON is too large")
            return json.loads(body.decode("utf-8"))
        except Exception as exc:
            last_err = exc
    raise last_err if last_err else ValueError("no registry sources configured")


def _fetch_dynamic_registry_no_cache() -> list[dict]:
    """Dynamically list all finetunes from every registry Space by scanning
    actual files. No index.json needed -- always in sync. Entries from the
    first source win on id conflicts; every entry carries its
    ``registry_source``."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if _REGISTRY_TOKEN_VALID:
            api = HfApi(token=REGISTRY_TOKEN)
        listings = {}
        for repo in _REGISTRY_SOURCES:
            try:
                files = api.list_repo_files(repo_id=repo, repo_type="space")
            except Exception:
                continue
            listings[repo] = [
                f for f in files if f.startswith("finetunes/") and f.endswith(".json")
            ]
    except Exception:
        return []
    fins = []
    seen = set()
    for repo, fin_files in listings.items():
        for path in sorted(fin_files):
            fid = _safe_model_ref(path[len("finetunes/") : -len(".json")])
            if not fid or fid in seen:
                continue
            try:
                data = _fetch_registry_json(fid, repo)
                if not isinstance(data, dict):
                    continue
                m = data.get("model", {})
                if not isinstance(m, dict):
                    continue
                tags_entry = m.get("tags", [])
                if isinstance(tags_entry, str):
                    tags_entry = [t.strip() for t in tags_entry.split(",") if t.strip()]
                if isinstance(tags_entry, list):
                    tags_entry = [t for t in tags_entry if t]
                urls = m.get("URLs", [])
                if isinstance(urls, str):
                    urls = [urls]
                loras = m.get("loras", [])
                if isinstance(loras, str):
                    loras = [loras]
                entry = {
                    "id": fid,
                    "name": m.get("name", fid),
                    "author": m.get("author", "community"),
                    "version": m.get("version", "1.0.0"),
                    "architecture": m.get("architecture", ""),
                    "description": m.get("description", ""),
                    "source": m.get("finetune_source_model"),
                    "URLs": urls,
                    "loras": loras,
                    "loras_multipliers": m.get("loras_multipliers", []),
                    "tags": tags_entry,
                    "default_settings": {
                        "num_inference_steps": data.get(
                            "num_inference_steps", m.get("num_inference_steps", 30)
                        ),
                        "guidance_scale": data.get(
                            "guidance_scale", m.get("guidance_scale", 5.0)
                        ),
                    },
                    "registry_source": repo,
                    "raw": _clean_utf8(data),
                }
                security_warnings = _check_bare_filenames(data)
                if security_warnings:
                    entry["security_warning"] = security_warnings[0]
                host_warning = _unexpected_host_warning(data)
                if host_warning:
                    entry["host_warning"] = host_warning
                fins.append(entry)
                seen.add(fid)
            except Exception:
                continue
    return fins


def _clean_utf8(obj):
    """Recursively strip surrogate characters from any JSON-serializable obj.
    Prevents Gradio's json.dumps from raising 'surrogates not allowed'."""
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, dict):
        return {_clean_utf8(k): _clean_utf8(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_utf8(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clean_utf8(v) for v in obj)
    return obj


def _sanitize_fin_id(fin_id: str) -> str:
    """Strip path traversal characters from a finetune ID.
    Only allows alphanumeric, underscore, dot, and hyphen.
    """
    cleaned = _re.sub(r"[^A-Za-z0-9_.-]+", "_", fin_id.strip())
    cleaned = _re.sub(r"_+", "_", cleaned).strip("._-")
    if not cleaned:
        return "unnamed"
    # Windows device names (CON, NUL, COM1, ...) must not become file stems.
    if fm_assets.safe_file_name(f"{cleaned}.json") is None:
        return f"{cleaned}_finetune"
    return cleaned


def _settings_path_for(fin_id: str) -> Path:
    """Settings file path using the same sanitized id as the finetune file."""
    return Path("settings") / f"{_sanitize_fin_id(fin_id)}_settings.json"


def _resolve_finetune_path(fin_id: str) -> Path | None:
    """Resolve a finetune JSON path and verify it stays inside finetunes/.

    ``adultpack_foo`` and ``foo`` are the same pack recipe.  Try the exact id
    first, then its alias, so editing one never creates a duplicate file.
    """
    base = Path(FINETUNES_DIR).resolve()
    for stem in fm_assets.finetune_candidates(fin_id):
        if fm_assets.safe_file_name(f"{stem}.json") is None:
            continue
        p = (base / f"{stem}.json").resolve()
        try:
            p.relative_to(base)
        except ValueError:
            continue
        if p.is_file():
            return p
    # Return the exact path when nothing exists so callers can create it.
    exact_name = str(fin_id).strip()
    if fm_assets.safe_file_name(f"{exact_name}.json") is None:
        return None
    exact = (base / f"{exact_name}.json").resolve()
    try:
        exact.relative_to(base)
        return exact
    except ValueError:
        return None


def _existing_finetune_stems() -> set[str]:
    p = Path(FINETUNES_DIR)
    if not p.exists():
        return set()
    stems = {f.stem for f in p.glob("*.json")}
    # A pack recipe also reserves its bare id.
    for stem in list(stems):
        if fm_assets.is_pack_id(stem):
            stems.add(fm_assets.pack_display_id(stem))
    return stems


def _write_finetune(fin_id, data):
    Path(FINETUNES_DIR).mkdir(parents=True, exist_ok=True)
    safe_id = _sanitize_fin_id(fin_id)
    (Path(FINETUNES_DIR) / f"{safe_id}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


CARD_CSS = """
/* Open plugin folder button — matches the warning block style */
.fm-open-dir-btn { margin-top:0 !important }
.fm-open-dir-btn button {
  background:#374151 !important;
  color:#d1d5db !important;
  border:1px solid #4b5563 !important;
  border-radius:6px !important;
  font-size:13px !important;
  padding:8px 16px !important;
  cursor:pointer !important;
}
.fm-open-dir-btn button:hover {
  background:#4b5563 !important;
}
.fm-cards-container { max-height:65vh; overflow-y:auto; padding-right:6px; scroll-behavior:smooth }
.fm-card { border:1px solid var(--border-color-primary,#e5e7eb); border-radius:10px; padding:14px; margin-bottom:8px; cursor:pointer; transition:border-color .2s,box-shadow .2s,transform .15s; background:var(--body-background-fill,transparent) }
.fm-card:hover { border-color:#6366f1; box-shadow:0 2px 8px rgba(99,102,241,.15); transform:translateY(-1px) }
.fm-card:active { transform:translateY(0) }
.fm-card.selected { border-color:#6366f1; border-width:2px; background:color-mix(in srgb,var(--body-background-fill,transparent) 90%,#6366f1) }
.fm-card-title { font-size:14px; font-weight:600; color:var(--body-text-color,#111827); margin:0 0 3px; display:flex; align-items:center; gap:6px }
.fm-card-meta { font-size:11px; color:var(--body-text-color-subdued,#6b7280); margin:0 0 6px; display:flex; align-items:center; gap:6px; flex-wrap:wrap }
.fm-card-desc { font-size:12px; color:var(--body-text-color,#374151); margin:0 0 4px; line-height:1.5; max-height:3.8em; overflow:hidden; text-overflow:ellipsis }
.fm-card-files { font-size:11px; color:var(--body-text-color-subdued,#6b7280); margin:0; padding-top:6px; border-top:1px solid var(--border-color-primary,#f3f4f6); display:flex; flex-wrap:wrap; gap:4px }
.fm-card-files a { color:#6366f1; text-decoration:none; word-break:break-all; padding:2px 4px; border-radius:4px; transition:background .15s }
.fm-card-files a:hover { background:color-mix(in srgb,var(--body-background-fill,transparent) 95%,#6366f1); text-decoration:underline }
.fm-card-loras { font-size:11px; color:var(--body-text-color-subdued,#6b7280); margin:0; padding-top:4px; display:flex; flex-wrap:wrap; gap:4px; align-items:center }
.fm-badge { display:inline-block; font-size:10px; padding:2px 8px; border-radius:12px; background:var(--primary-200,#e0e7ff); color:var(--primary-500,#4338ca); margin-right:4px; font-weight:500; letter-spacing:.02em }
.fm-badge-secondary { background:var(--neutral-100,#f3f4f6); color:var(--neutral-600,#6b7280) }
.fm-badge-warn { background:color-mix(in srgb,#f59e0b 22%,transparent); color:#f59e0b }
.fm-badge-pack { background:color-mix(in srgb,#14b8a6 20%,transparent); color:#2dd4bf }
.fm-family-header { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:14px 0 6px; padding:0 2px; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--body-text-color-subdued,#6b7280) }
.fm-family-header:first-child { margin-top:0 }
.fm-family-header::after { content:""; flex:1; height:1px; background:var(--border-color-primary,#e5e7eb); order:1 }
.fm-family-header span:first-child { order:0 }
.fm-family-count { order:2; font-size:10px; font-weight:600; padding:1px 7px; border-radius:10px; background:color-mix(in srgb,#6366f1 18%,transparent); color:#818cf8; letter-spacing:0 }
/* JSON & Preview readable wrapping */
[data-testid="json"] .json-holder { max-height:450px !important; overflow-y:auto !important }
[data-testid="json"] .line .content { white-space:pre-wrap !important; word-break:break-word !important; overflow-wrap:break-word !important; line-height:1.65 !important; font-size:13px !important }
[data-testid="json"] .line, [data-testid="json"] .json-node { min-height:1.65em !important; height:auto !important; max-height:none !important }
/* CivitAI result cards */
/* Loading spinner overlay for buttons */
.fm-spinner { display:inline-block; width:12px; height:12px; border:2px solid var(--border-color-primary,#e5e7eb); border-top-color:#6366f1; border-radius:50%; animation:fm-spin .6s linear infinite; margin-right:6px; vertical-align:middle }
@keyframes fm-spin { to { transform:rotate(360deg) } }
"""


def _refresh_model_defs(plugin) -> None:
    """Ask Wan2GP to rescan finetunes after a pack recipe changes."""
    refresh = getattr(plugin, "refresh_model_defs", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass


class FinetuneManagerPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("refresh_model_defs")
        self.request_global("switch_to_model")
        self.request_global("load_settings_from_file")
        self.request_global("_model_choice_target_value")
        # Wan2GP's own model-family metadata (LTX-2, MiniMax H3, Wan2.2, ...).
        # All are optional: _family_catalog() falls back to a static catalogue.
        self.request_global("families_infos")
        self.request_global("get_model_family")
        self.request_global("list_model_defs")
        self.request_global("get_lora_dir")
        self.request_global("get_lora_root")
        self.request_component("state")
        self.request_component("model_choice_target")
        self.request_component("main_tabs")
        self.request_component("loras_choices")
        self.request_component("loras_multipliers")
        self.add_tab(
            tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui
        )

    def create_ui(self, api_session):
        gr.HTML(f"<style>{CARD_CSS}</style>")
        registry = gr.State([])
        gr.Markdown(
            f"**Finetune Manager v{PLUGIN_VERSION}** — "
            + " · ".join(
                f"[{('HF Registry' if repo == _REGISTRY_SOURCES[0] else repo)}]"
                f"(https://huggingface.co/spaces/{repo})"
                for repo in _REGISTRY_SOURCES
            )
        )
        if not _REGISTRY_TOKEN_VALID:
            gr.HTML(_TOKEN_SETUP_GUIDE)
            _open_dir_btn = gr.Button("📂 Open plugin folder", size="sm",
                elem_classes="fm-open-dir-btn")
            _open_dir_btn.click(fn=_open_config_dir, inputs=[], outputs=[])

        with gr.Tabs() as fm_tabs:
            # ── BROWSE ──
            with gr.TabItem("Browse", id="browse"):
                with gr.Row():
                    b_search = gr.Textbox(label="Search", scale=3, container=False)
                    b_family = gr.Dropdown(
                        label="Model family",
                        choices=[("All families", "All")],
                        value="All",
                        filterable=True,
                        allow_custom_value=True,
                        scale=1,
                    )
                    b_arch = gr.Dropdown(
                        label="Architecture",
                        choices=["All"],
                        value="All",
                        allow_custom_value=True,
                        filterable=True,
                        scale=1,
                    )
                    b_tag_filter = gr.Dropdown(
                        label="Tag",
                        choices=[],
                        value="",
                        allow_custom_value=True,
                        scale=1,
                    )
                    b_registry = gr.Dropdown(
                        label="Registry",
                        choices=[
                            ("All registries", "all"),
                            ("Community", "community"),
                            ("Adult Pack", "adult"),
                        ],
                        value="all",
                        allow_custom_value=False,
                        scale=1,
                    )
                    b_consent = gr.Checkbox(
                        label="I am 18 or older — show Adult Pack entries",
                        value=ADULT_CONSENT,
                        scale=0,
                        min_width=280,
                    )
                b_refresh = gr.Button("Refresh Registry")
                b_count = gr.Markdown("Click *Refresh* to load")
                b_sel_id = gr.State("")
                b_cards = gr.HTML(
                    "<div style='color:#9ca3af;text-align:center;"
                    "padding:32px'>Click Refresh</div>"
                )
                b_detail_raw = gr.State({})
                b_id_display = gr.Markdown("")
                b_detail = gr.JSON(label="Detail")
                with gr.Row():
                    b_validate_btn = gr.Button("Validate URLs", scale=1)
                b_validate_out = gr.HTML("")
                with gr.Row():
                    b_load = gr.Button("Download & Switch", variant="primary")
                    b_dl_only = gr.Button("Download")
                    b_improve = gr.Button("Improve / Create Variant", variant="primary")
                b_status = gr.Textbox(label="Status")

                def _enrich_registry(fins):
                    """Attach family key/label/warning to registry entries."""
                    catalog = _family_catalog(self)
                    for f in fins:
                        raw = f.get("raw")
                        m = raw.get("model", {}) if isinstance(raw, dict) else {}
                        if not isinstance(m, dict) or not m:
                            m = {
                                "architecture": f.get("architecture", ""),
                                "group": f.get("group", ""),
                                "name": f.get("name", ""),
                            }
                        family = _family_for_model(self, m)
                        f["family"] = family
                        f["family_label"] = _family_label(catalog, family)
                        f["family_warning"] = _family_warning(self, m, family)
                    return fins

                def _archs_for(fins, family):
                    archs = sorted({
                        str(f.get("architecture", "")).strip()
                        for f in fins
                        if str(f.get("architecture", "")).strip()
                        and (family in ("", "All") or f.get("family") == family)
                    })
                    return ["All"] + archs

                def _fmt_cards(
                    fins, search, arch, sel, tag_filter, family="All", registry="all",
                    consent=False,
                ):
                    ff = list(fins)
                    gated = not consent
                    if search:
                        s = search.lower()

                        def _match_tags(f):
                            tg = f.get("tags", [])
                            if isinstance(tg, str):
                                tg = [t.strip() for t in tg.split(",") if t.strip()]
                            if not isinstance(tg, list):
                                return False
                            return any(s in str(tag or "").lower() for tag in tg)

                        ff = [
                            f
                            for f in ff
                            if s in str(f.get("name") or "").lower()
                            or s in str(f.get("description") or "").lower()
                            or s in str(f.get("author") or "").lower()
                            or _match_tags(f)
                        ]
                    if family and family != "All":
                        ff = [f for f in ff if f.get("family") == family]
                    if registry and registry != "all":
                        want_adult = registry == "adult"
                        if want_adult and gated:
                            return (
                                "<div style='color:#9ca3af;text-align:center;"
                                "padding:32px'>Tick the consent checkbox above to "
                                "see Adult Pack entries</div>",
                                "**Adult Pack locked**",
                            )
                        ff = [
                            f
                            for f in ff
                            if str(f.get("registry_source") or "").endswith("-adult")
                            == want_adult
                        ]
                    elif gated:
                        ff = [
                            f
                            for f in ff
                            if not str(f.get("registry_source") or "")
                            .endswith("-adult")
                        ]
                    if arch and arch != "All":
                        ff = [f for f in ff if f.get("architecture") == arch]
                    if tag_filter:
                        tf = tag_filter.strip().lower()
                        if tf:

                            def _filter_tags(f):
                                tg = f.get("tags", [])
                                if isinstance(tg, str):
                                    tg = [t.strip() for t in tg.split(",") if t.strip()]
                                if not isinstance(tg, list):
                                    return False
                                return any(
                                    str(tag or "").strip().lower() == tf for tag in tg
                                )

                            ff = [f for f in ff if _filter_tags(f)]
                    if not ff:
                        return (
                            "<div style='color:#9ca3af;text-align:center;"
                            "padding:32px'>No results</div>",
                            "**0 matches**",
                        )
                    catalog = _family_catalog(self)
                    ff.sort(
                        key=lambda f: (
                            1
                            if str(f.get("registry_source") or "").endswith("-adult")
                            else 0,
                            _family_sort_key(
                                catalog,
                                f.get("family") or _FAMILY_UNKNOWN,
                                str(f.get("name", f.get("id", "")) or ""),
                            ),
                        )
                    )
                    counts = {}
                    for f in ff:
                        fam = f.get("family") or _FAMILY_UNKNOWN
                        counts[fam] = counts.get(fam, 0) + 1
                    _h = ['<div class="fm-cards-container">']
                    current_family = None
                    for f in ff:
                        fam = f.get("family") or _FAMILY_UNKNOWN
                        if fam != current_family:
                            _h.append(
                                '<div class="fm-family-header">'
                                f"<span>{html.escape(_family_label(catalog, fam))}</span>"
                                f'<span class="fm-family-count">{counts[fam]}</span>'
                                "</div>"
                            )
                            current_family = fam
                        fid = f.get("id", "")
                        _h.append(
                            _fmt_card_html_item(
                                fid=fid,
                                name=f.get("name", "?"),
                                arch=f.get("architecture", ""),
                                author=f.get("author", ""),
                                desc=f.get("description", ""),
                                urls=f.get("URLs", []),
                                loras=f.get("loras", []),
                                ftags=f.get("tags", []),
                                is_selected=(fid == sel),
                                sel_elem_id="fm-selected",
                                is_variant=bool(f.get("source")),
                                is_pack=str(f.get("registry_source") or "")
                                .endswith("-adult"),
                                family_label=_family_label(catalog, fam),
                                family_warning=f.get("family_warning", ""),
                                security_warning=f.get("security_warning", ""),
                                host_warning=f.get("host_warning", ""),
                            )
                        )
                    _h.append("</div>")
                    cnt_label = "match" if len(ff) == 1 else "matches"
                    return ("".join(_h), f"**{len(ff)} {cnt_label}**")

                def _all_tags(fins):
                    seen = set()
                    for f in fins:
                        tg = f.get("tags", [])
                        if isinstance(tg, str):
                            tg = [t.strip() for t in tg.split(",") if t.strip()]
                        if isinstance(tg, list):
                            for t in tg:
                                t = str(t or "").strip().lower()
                                if t:
                                    seen.add(t)
                    return sorted(seen)

                def _do_refresh(consent):
                    fins = _enrich_registry(_get_cached_registry(force=True))
                    html, cnt = _fmt_cards(fins, "", "All", "", "", consent=consent)
                    tags = _all_tags(fins)
                    catalog = _family_catalog(self)
                    present = {f.get("family", _FAMILY_UNKNOWN) for f in fins}
                    fam_choices = [("All families", "All")] + _family_choices(
                        catalog, present
                    )
                    if _FAMILY_UNKNOWN in present and "unknown" not in catalog:
                        fam_choices.append(("Unknown", _FAMILY_UNKNOWN))
                    return (
                        fins,
                        "",
                        html,
                        cnt,
                        gr.update(choices=tags, value=None),
                        gr.update(value=""),
                        gr.update(choices=_archs_for(fins, "All"), value="All"),
                        gr.update(choices=fam_choices, value="All"),
                    )

                b_refresh.click(
                    fn=_do_refresh,
                    inputs=[b_consent],
                    outputs=[
                        registry,
                        b_sel_id,
                        b_cards,
                        b_count,
                        b_tag_filter,
                        b_search,
                        b_arch,
                        b_family,
                    ],
                )

                def _filter(fins, search, arch, sel, tag_filter, family, registry, consent):
                    html, cnt = _fmt_cards(
                        fins, search, arch, sel, tag_filter, family, registry, consent
                    )
                    return html, cnt

                def _family_changed(fins, search, sel, tag_filter, family, registry, consent):
                    html, cnt = _fmt_cards(
                        fins, search, "All", sel, tag_filter, family, registry, consent
                    )
                    return (
                        html,
                        cnt,
                        gr.update(choices=_archs_for(fins, family), value="All"),
                    )

                b_search.input(
                    fn=_filter,
                    inputs=[
                        registry,
                        b_search,
                        b_arch,
                        b_sel_id,
                        b_tag_filter,
                        b_family,
                        b_registry,
                        b_consent,
                    ],
                    outputs=[b_cards, b_count],
                )
                b_arch.change(
                    fn=_filter,
                    inputs=[
                        registry,
                        b_search,
                        b_arch,
                        b_sel_id,
                        b_tag_filter,
                        b_family,
                        b_registry,
                        b_consent,
                    ],
                    outputs=[b_cards, b_count],
                )
                b_tag_filter.change(
                    fn=_filter,
                    inputs=[
                        registry,
                        b_search,
                        b_arch,
                        b_sel_id,
                        b_tag_filter,
                        b_family,
                        b_registry,
                        b_consent,
                    ],
                    outputs=[b_cards, b_count],
                )
                b_registry.change(
                    fn=_filter,
                    inputs=[
                        registry,
                        b_search,
                        b_arch,
                        b_sel_id,
                        b_tag_filter,
                        b_family,
                        b_registry,
                        b_consent,
                    ],
                    outputs=[b_cards, b_count],
                )

                def _consent_changed(
                    fins, search, arch, sel, tag_filter, family, registry, consent
                ):
                    _persist_config("adult_consent", bool(consent))
                    html, cnt = _fmt_cards(
                        fins, search, arch, sel, tag_filter, family, registry, consent
                    )
                    return html, cnt

                b_consent.change(
                    fn=_consent_changed,
                    inputs=[
                        registry,
                        b_search,
                        b_arch,
                        b_sel_id,
                        b_tag_filter,
                        b_family,
                        b_registry,
                        b_consent,
                    ],
                    outputs=[b_cards, b_count],
                )
                b_family.change(
                    fn=_family_changed,
                    inputs=[
                        registry,
                        b_search,
                        b_sel_id,
                        b_tag_filter,
                        b_family,
                        b_registry,
                        b_consent,
                    ],
                    outputs=[b_cards, b_count, b_arch],
                )

                fm_sel_input = gr.Textbox(visible=False, elem_id="fm-selected")

                def _browse_validate(fid, detail):
                    if not detail or not isinstance(detail, dict):
                        return (
                            "<div style='color:#6b7280;padding:4px'>"
                            "Select a card first</div>"
                        )
                    all_entries = _collect_url_entries(detail)
                    results = _validate_urls(all_entries, self)
                    return _build_url_validation_html(results)

                b_validate_btn.click(
                    fn=_browse_validate,
                    inputs=[b_sel_id, b_detail_raw],
                    outputs=[b_validate_out],
                )

                def _on_selected_auto_validate(fins, fid):
                    m = next((f for f in fins if f["id"] == fid), None)
                    if m and "raw" in m:
                        detail_raw = _clean_utf8(m["raw"])
                    else:
                        detail_raw = _clean_utf8(m if m else {})
                    # Auto-run validation on URL/LoRA fields only
                    if isinstance(detail_raw, dict):
                        all_entries = _collect_url_entries(detail_raw)
                        val_html = _build_url_validation_html(
                            _validate_urls(all_entries, self, network=False)
                        )
                    else:
                        val_html = ""
                    return (
                        fid,
                        detail_raw,
                        detail_raw,
                        f"**ID:** `{fid}.json`",
                        val_html,
                    )

                fm_sel_input.input(
                    fn=_on_selected_auto_validate,
                    inputs=[registry, fm_sel_input],
                    outputs=[
                        b_sel_id,
                        b_detail_raw,
                        b_detail,
                        b_id_display,
                        b_validate_out,
                    ],
                )

                def _load(state_value, fins, fid):
                    if not fid:
                        return "Select a card", gr.update(), gr.update()
                    m = next((f for f in fins if f["id"] == fid), None)
                    if not m:
                        return f"'{fid}' not found", gr.update(), gr.update()
                    try:
                        data = _fetch_registry_json(fid, m.get("registry_source"))
                    except Exception as e:
                        return f"Error: {e}", gr.update(), gr.update()
                    err = _validate_finetune_data(data, registry=True)
                    if err:
                        return f"Cannot load '{fid}': {err}", gr.update(), gr.update()
                    _write_finetune(fid, data)
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    settings_path = _write_settings_for_wan2gp(fid, data)
                    refresh, target, _ = (
                        self.load_settings_from_file(state_value, str(settings_path))
                        if hasattr(self, "load_settings_from_file") and settings_path.exists()
                        else (gr.update(), gr.update(), None)
                    )
                    target = _ensure_model_choice_target(target, fid)
                    return f"Loaded '{m.get('name', fid)}' and switched", target, gr.update()

                # ── Browse: Download (no switch) ──
                def _browse_dl(fins, fid):
                    if not fid:
                        return "Select a card"
                    m = next((f for f in fins if f["id"] == fid), None)
                    if not m:
                        return f"'{fid}' not found"
                    try:
                        data = _fetch_registry_json(fid, m.get("registry_source"))
                    except Exception as e:
                        return f"Error: {e}"
                    err = _validate_finetune_data(data, registry=True)
                    if err:
                        return f"Cannot download '{fid}': {err}"
                    _write_finetune(fid, data)
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    return f"Downloaded '{m.get('name', fid)}' locally"

                b_dl_only.click(
                    fn=_browse_dl, inputs=[registry, b_sel_id], outputs=[b_status]
                )

                # ── Browse: Improve / Create Variant → fill Create/Edit tab ──
                def _improve_to_editor(fins, fid):
                    if not fid:
                        return [gr.update()] * (len(ALL_INPUTS)) + [gr.update(), f"Select a card", {}]
                    m = next((f for f in fins if f["id"] == fid), None)
                    if not m:
                        return [gr.update()] * (len(ALL_INPUTS)) + [
                            gr.update(),
                            f"'{fid}' not found",
                            {},
                        ]
                    try:
                        data = _fetch_registry_json(fid, m.get("registry_source"))
                    except Exception as e:
                        return [gr.update()] * (len(ALL_INPUTS)) + [gr.update(), f"Error: {e}", {}]
                    err = _validate_finetune_data(data, registry=True)
                    if err:
                        return [gr.update()] * (len(ALL_INPUTS)) + [
                            gr.update(),
                            f"Cannot create variant: {err}",
                            {},
                        ]
                    # Preserve original finetune_source_model if it exists;
                    # only fall back to architecture if there's no source set.
                    model_dict = data.setdefault("model", {})
                    if not model_dict.get("finetune_source_model"):
                        model_dict["finetune_source_model"] = model_dict.get(
                            "architecture", ""
                        )
                    vid = _unique_id(f"{fid}_variant")
                    _write_finetune(vid, data)
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    data = _clean_utf8(data)
                    # Same pattern as _fill(): (id,) + _extract(data) + [tab_switch, status, extra_data]
                    fill_values = list((vid,) + _extract(data))
                    return fill_values + [
                        gr.update(selected="editor"),
                        f"Variant '{vid}' loaded into Create/Edit tab",
                        data,
                    ]

            # ── CREATE / EDIT ──
            with gr.TabItem("Create / Edit", id="editor"):
                gr.Markdown(
                    "### Create or Edit \u2014 "
                    "matches the built-in Finetune Editor (Alt+F)"
                )

                with gr.Row():
                    fin_import_file = gr.File(
                        label="Import JSON", file_types=[".json"], scale=2
                    )
                    fin_import_btn = gr.Button("Fill from JSON", scale=1)
                with gr.Row():
                    fin_fsm = gr.Textbox(
                        label="Source Model (finetune_source_model)", scale=3
                    )
                    fin_source_info = gr.Textbox(
                        label="Source Model Info", interactive=False, lines=1, scale=2
                    )
                with gr.Row():
                    fin_id = gr.Textbox(label="ID (filename)", scale=4)
                    fin_auto_id = gr.Checkbox(
                        label="auto", value=True, scale=1, min_width=60
                    )
                    fin_arch = gr.Dropdown(
                        label="Architecture",
                        scale=3,
                        choices=_architecture_choices(self),
                        allow_custom_value=True,
                        filterable=True,
                    )
                with gr.Row():
                    fin_name = gr.Textbox(label="Name", scale=2)
                    fin_tags = gr.Textbox(
                        label="Tags (comma-separated)",
                        lines=1,
                        scale=2,
                        placeholder="character, style, anime, photorealistic, ...",
                    )
                fin_desc = gr.Textbox(label="Description", lines=2)
                fin_prompt = gr.Textbox(
                    label="Prompt",
                    lines=2,
                    placeholder="Enter the default prompt for this finetune...",
                )

                fin_lora_root = gr.Textbox(value="", visible=False)
                fin_creator_source_mode = gr.Radio(
                    label="Create New Finetune",
                    choices=[
                        (
                            "Using Current Model (via Improve / Create Variant)",
                            "current",
                        )
                    ],
                    value="current",
                    visible=False,
                )
                fin_use_current_settings = gr.Checkbox(
                    label="Use Current Model Settings as Default Settings", value=False
                )

                with gr.Tabs():
                    with gr.Tab("URLs"):
                        gr.Markdown(
                            "**Override only what's different from the base model.** "
                            "Main Checkpoints = the finetune weights. "
                            "Text Encoder, VAE, Preload are usually **inherited** from the base model — leave blank unless your finetune replaces them."
                        )
                        available_models = _list_available_models()
                        model_choices = [
                            (label, mid) for mid, label in available_models
                        ]
                        # Cache model id -> name so validation/sync never hits disk.
                        MODEL_INDEX.clear()
                        DISK_EXISTS_CACHE.clear()
                        MODEL_INDEX.update(
                            {mid: name for mid, name in available_models}
                        )

                        def _ensure_model_choice_target(target, fid):
                            """Ensure model_choice_target is a unique string to force Gradio re-render."""
                            if not isinstance(target, str):
                                if hasattr(self, "_model_choice_target_value"):
                                    return self._model_choice_target_value(fid)
                                return gr.update()
                            return target

                        # URL fields as labeled rows (matching LoRA tab pattern)
                        _URL_ENTRIES = [
                            ("Main Checkpoints", True),
                            ("Secondary Checkpoints", True),
                            ("Text Encoder", True),
                            ("VAE", True),
                            ("Preload URLs", True),
                            ("Custom URL 1", False),
                            ("Custom URL 2", False),
                            ("Custom URL 3", False),
                        ]

                        fin_urls_list = []
                        fin_urls_refs_list = []
                        fin_urls_vals_list = []

                        for _i, (_label, _multi) in enumerate(_URL_ENTRIES):
                            with gr.Row():
                                _ref = gr.Dropdown(
                                    choices=model_choices,
                                    label=_label,
                                    scale=1,
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                                _url = gr.Textbox(
                                    label="URL / path (or =modelid)",
                                    lines=2 if _multi else 1,
                                    scale=4,
                                )
                            _val = gr.HTML("")
                            fin_urls_refs_list.append(_ref)
                            fin_urls_list.append(_url)
                            fin_urls_vals_list.append(_val)

                        # Unpack to named variables for backward compatibility
                        (
                            fin_urls_ref,
                            fin_urls2_ref,
                            fin_te_urls_ref,
                            fin_vae_urls_ref,
                            fin_preload_ref,
                            fin_custom_url_1_ref,
                            fin_custom_url_2_ref,
                            fin_custom_url_3_ref,
                        ) = fin_urls_refs_list

                        (
                            fin_urls,
                            fin_urls2,
                            fin_te_urls,
                            fin_vae_urls,
                            fin_preload,
                            fin_custom_url_1,
                            fin_custom_url_2,
                            fin_custom_url_3,
                        ) = fin_urls_list

                        (
                            fin_urls_val,
                            fin_urls2_val,
                            fin_te_urls_val,
                            fin_vae_urls_val,
                            fin_preload_val,
                            fin_custom_url_1_val,
                            fin_custom_url_2_val,
                            fin_custom_url_3_val,
                        ) = fin_urls_vals_list

                        with gr.Row():
                            fin_validate_urls_btn = gr.Button(
                                "Validate All URLs", scale=1
                            )
                            fin_download_btn = gr.Button(
                                "Download Missing Files", scale=1, variant="primary"
                            )
                        fin_validate_out = gr.HTML("")
                        fin_download_out = gr.Textbox(
                            label="Download Status", lines=4, max_lines=10
                        )

                    with gr.Tab("LoRAs"):
                        available_loras = _list_available_loras()
                        lora_choices = [(label, mid) for mid, label in available_loras]

                        gr.Markdown(
                            "**Always Loaded LoRAs** \u2014 up to 10. "
                            "Multipliers are applied in the same order as "
                            "the LoRAs above."
                        )

                        fin_lora_urls = []
                        fin_lora_mults = []
                        fin_lora_refs = []
                        fin_lora_vals = []
                        for _i in range(10):
                            with gr.Row():
                                _ref = gr.Dropdown(
                                    choices=lora_choices,
                                    label=f"LoRA {_i + 1}",
                                    scale=1,
                                    allow_custom_value=True,
                                    interactive=True,
                                )
                                _lora_picker = LocalFilePickerTextbox(
                                    label="URL / path (or =modelid)",
                                    file_extensions=LORA_FILE_EXTENSIONS,
                                    multiselect=False,
                                    popup_title=f"Select LoRA {_i + 1}",
                                    lines=1,
                                    textbox_scale=4,
                                )
                                _url = _lora_picker.mount()
                                _mult = gr.Textbox(
                                    label="Multiplier", value="1", lines=1, scale=1
                                )
                            _val = gr.HTML("")
                            fin_lora_urls.append(_url)
                            fin_lora_mults.append(_mult)
                            fin_lora_refs.append(_ref)
                            fin_lora_vals.append(_val)

                        # Hidden accumulators that feed _build()/_extract().
                        fin_loras = gr.Textbox(visible=False)
                        fin_lmults = gr.Textbox(visible=False)

                        _FILE_EXTS = (
                            ".safetensors",
                            ".pt",
                            ".pth",
                            ".bin",
                            ".ckpt",
                            ".zip",
                            ".tar",
                            ".gz",
                            ".tgz",
                            ".onnx",
                            ".tar.gz",
                            ".gguf",
                        )

                        def _looks_like_file(name):
                            lower = name.lower()
                            return any(lower.endswith(e) for e in _FILE_EXTS)

                        def _lora_ref_for(u):
                            u = (u or "").strip()
                            if not u:
                                return None
                            mid = u[1:].strip() if u.startswith("=") else u
                            if mid in MODEL_INDEX:
                                return mid
                            # If user explicitly used =, trust the intent
                            if u.startswith("="):
                                return mid
                            # If it looks like a file path, don't show in dropdown
                            if _looks_like_file(mid):
                                return None
                            if (
                                not u.startswith(("http:", "https:", "/", ".", "\\"))
                                and "/" not in mid
                                and not (len(mid) > 1 and mid[1] == ":")
                            ):
                                return mid
                            return None

                        def _lora_val_for(u):
                            # Cheap, no network: model-ref via cache, local path
                            # via disk check. Full URL reachability is covered
                            # by the "Validate All URLs" button (keeps load <1s).
                            u = (u or "").strip()
                            if not u:
                                return ""
                            if u.startswith("="):
                                mid = u[1:].strip()
                                if mid in MODEL_INDEX:
                                    data = _read_model_json(mid)
                                    urls = _get_model_urls(data) if data else []
                                    msg = MODEL_INDEX[mid]
                                    if urls:
                                        msg += f"\n\u2192 {urls[0]}"
                                    return _led(True, msg, size=11)
                                return _led(
                                    False,
                                    f"{mid} not in list \u2014 enter a URL or upload a file instead",
                                    size=11,
                                )
                            if u.startswith(("http:", "https:")):
                                problem = fm_security.url_problem(u)
                                if problem:
                                    return _led(False, problem, size=11)
                                return _led(
                                    True,
                                    "URL format OK \u2014 use Validate button to check",
                                    size=11,
                                )
                            if (
                                not u.startswith(("/", ".", "\\"))
                                and "/" not in u
                                and "\\" not in u
                                and not (len(u) > 1 and u[1] == ":")
                            ):
                                if u in MODEL_INDEX:
                                    data = _read_model_json(u)
                                    urls = _get_model_urls(data) if data else []
                                    msg = MODEL_INDEX[u]
                                    if urls:
                                        msg += f"\n\u2192 {urls[0]}"
                                    return _led(True, msg, size=11)
                                if _looks_like_file(u) and _check_disk_exists(u):
                                    return _led(True, "Found on disk", size=11)
                                return _led(
                                    False,
                                    f"{u} not in list \u2014 enter a URL or upload a file instead",
                                    size=11,
                                )
                            if _check_disk_exists(u):
                                return _led(True, "Found on disk", size=11)
                            return _led(False, "File not found", size=11)

                        def _lora_compute(urls, mults):
                            loras_out = [
                                (urls[i] or "").strip()
                                for i in range(10)
                                if (urls[i] or "").strip()
                            ]
                            mults_out = [
                                (mults[i] or "").strip() or "1"
                                for i in range(10)
                                if (urls[i] or "").strip()
                            ]
                            loras_str = "\n".join(loras_out)
                            lm_str = " ".join(mults_out)
                            refs = [_lora_ref_for(urls[i]) for i in range(10)]
                            vals = [_lora_val_for(urls[i]) for i in range(10)]
                            return loras_str, lm_str, refs, vals

                        def _split_to_rows(loras_str, mults_str):
                            raw_u = (
                                loras_str.split("\n")
                                if isinstance(loras_str, str)
                                else list(loras_str or [])
                            )
                            urls = [
                                (raw_u[i].strip() if i < len(raw_u) else "")
                                for i in range(10)
                            ]
                            raw_m = (
                                mults_str.split()
                                if isinstance(mults_str, str)
                                else list(mults_str or [])
                            )
                            mults = [
                                (raw_m[i].strip() if i < len(raw_m) else "1")
                                for i in range(10)
                            ]
                            _, _, refs, vals = _lora_compute(urls, mults)
                            return urls + mults + refs + vals

                        def _on_lora_url_change(i, *vals20):
                            urls = list(vals20[:10])
                            mults = list(vals20[10:])
                            ls, lm, refs, vals = _lora_compute(urls, mults)
                            return [ls, lm] + refs + vals

                        def _on_lora_ref_change(i, mid, *vals20):
                            urls = list(vals20[:10])
                            mults = list(vals20[10:])
                            if mid:
                                urls[i] = f"={mid}"
                            ls, lm, refs, vals = _lora_compute(urls, mults)
                            return [urls[i], ls, lm] + refs + vals

                        def _on_lora_mult_change(i, *vals20):
                            urls = list(vals20[:10])
                            mults = list(vals20[10:])
                            ls, lm, refs, vals = _lora_compute(urls, mults)
                            return [ls, lm] + refs + vals

                        _lora_row_inputs = fin_lora_urls + fin_lora_mults
                        _lora_row_outputs = (
                            [fin_loras, fin_lmults] + fin_lora_refs + fin_lora_vals
                        )
                        for _i in range(10):
                            _ref = fin_lora_refs[_i]
                            _url = fin_lora_urls[_i]
                            _mult = fin_lora_mults[_i]
                            _url.change(
                                fn=lambda *a, i=_i: _on_lora_url_change(i, *a),
                                inputs=_lora_row_inputs,
                                outputs=_lora_row_outputs,
                                queue=False,
                            )
                            _ref.change(
                                fn=lambda mid, *a, i=_i: _on_lora_ref_change(
                                    i, mid, *a
                                ),
                                inputs=[_ref] + _lora_row_inputs,
                                outputs=[_url] + _lora_row_outputs,
                                queue=False,
                            )
                            _mult.change(
                                fn=lambda m, *a, i=_i: _on_lora_mult_change(i, m, *a),
                                inputs=[_mult] + _lora_row_inputs,
                                outputs=_lora_row_outputs,
                                queue=False,
                            )
                    with gr.Tab("Resolutions"):
                        fin_rescat = gr.Textbox(
                            label="Resolution Category Conditions (OR per line)",
                            lines=4,
                            placeholder=">=720&<=1440",
                        )
                        gr.HTML(
                            "<div style='font-size:11px;color:#6b7280;"
                            "margin:-8px 0 8px'>Auto-builds resolutions from "
                            "the global presets that match these rules "
                            "(e.g. <code>&gt;=720&amp;&lt;=1440</code>). "
                            "Separate lines are OR.</div>"
                        )
                        fin_res = gr.Textbox(
                            label="Custom Resolutions (one WxH per line)",
                            lines=4,
                            placeholder="1024x2048",
                        )
                        gr.HTML(
                            "<div style='font-size:11px;color:#6b7280;"
                            "margin:-8px 0 8px'>Explicit resolutions added "
                            "<b>on top of</b> the categories above. "
                            "Example: <code>1024x2048</code> \u2192 saved as "
                            "<code>1024x2048 (1:2)</code>.</div>"
                        )
                    with gr.Tab("Help"):
                        fin_infos = gr.Textbox(
                            label="Model Infos (markdown)",
                            lines=8,
                            elem_classes=["wangp-markdown-editor"],
                        )
                        fin_pinfos = gr.Textbox(
                            label="Prompt Help (markdown)",
                            lines=8,
                            elem_classes=["wangp-markdown-editor"],
                        )
                    with gr.Tab("Prompt Enhancer"):
                        with gr.Column():
                            fin_pe_txt = gr.Textbox(
                                label="System Prompt (Text)", lines=4
                            )
                            fin_pe_txt_tok = gr.Textbox(
                                label="Max Tokens \u2014 Text (empty = auto)", lines=1
                            )
                        with gr.Column():
                            fin_pe_vid = gr.Textbox(
                                label="System Prompt (Video)", lines=4
                            )
                            fin_pe_vid_tok = gr.Textbox(
                                label="Max Tokens \u2014 Video (empty = auto)", lines=1
                            )
                        with gr.Column():
                            fin_pe_img = gr.Textbox(
                                label="System Prompt (Image)", lines=4
                            )
                            fin_pe_img_tok = gr.Textbox(
                                label="Max Tokens \u2014 Image (empty = auto)", lines=1
                            )
                        with gr.Row():
                            fin_pe_load_src = gr.Button(
                                "\U0001f441 Load from Source Model", size="sm"
                            )
                    with gr.Tab("Settings"):
                        fin_modules = gr.Textbox(
                            label="Modules (comma-separated)", lines=1
                        )
                        with gr.Row():
                            fin_autoq = gr.Checkbox(label="auto_quantize", scale=1)
                            fin_visible = gr.Checkbox(
                                label="visible", value=True, scale=1
                            )
                            fin_imgout = gr.Checkbox(label="image_outputs", scale=1)
                            fin_solver = gr.Dropdown(
                                label="Solver",
                                scale=3,
                                choices=[
                                    "euler",
                                    "dpm",
                                    "ddim",
                                    "pndm",
                                    "lcm",
                                    "dpmpp_2m",
                                    "dpmpp_sde",
                                    "unipc",
                                ],
                                value="euler",
                                allow_custom_value=True,
                            )
                        with gr.Row():
                            fin_steps = gr.Slider(
                                label="Steps",
                                scale=1,
                                minimum=1,
                                maximum=200,
                                value=30,
                                step=1,
                            )
                            fin_guidance = gr.Slider(
                                label="Guidance",
                                scale=1,
                                minimum=1.0,
                                maximum=30.0,
                                value=5.0,
                                step=0.5,
                            )

                fin_preview = gr.JSON(label="Preview")
                fin_mode = gr.State("creator")
                fin_extra_data = gr.State(value={})

                with gr.Row(visible=True) as fin_creator_actions:
                    fin_create = gr.Button("Create", variant="primary")
                    fin_save_switch = gr.Button("Save & Switch", variant="primary")
                    fin_save_up = gr.Button("Save & Upload")
                    fin_cancel_create = gr.Button("Cancel")
                with gr.Row(visible=False) as fin_editor_actions:
                    fin_save = gr.Button("Save Locally", variant="primary")
                    fin_save_switch_ed = gr.Button("Save & Switch", variant="primary")
                    fin_save_up_ed = gr.Button("Save & Upload")
                    fin_export = gr.DownloadButton("Export", value=None)
                    fin_del = gr.Button("Delete", variant="stop")
                    fin_cancel_edit = gr.Button("Cancel")
                fin_status = gr.Textbox(label="Status")
                fin_del_confirm = gr.Row(visible=False)
                with fin_del_confirm:
                    fin_del_confirm_btn = gr.Button("Confirm Delete", variant="stop")
                    fin_del_cancel_btn = gr.Button("Cancel")

                def _build(
                    name,
                    arch,
                    desc,
                    fsm,
                    urls,
                    urls2,
                    te,
                    vae,
                    pre,
                    cu1,
                    cu2,
                    cu3,
                    loras,
                    lm,
                    mods,
                    aq,
                    vis,
                    img,
                    res,
                    resc,
                    inf,
                    pinf,
                    steps,
                    guid,
                    solver,
                    pe1,
                    pe1t,
                    pe2,
                    pe2t,
                    pe3,
                    pe3t,
                    tags="",
                    prompt="",
                    extra_data=None,
                ):
                    return _clean_utf8(
                        build_finetune_dict(
                            name,
                            arch,
                            desc,
                            fsm,
                            urls,
                            urls2,
                            te,
                            vae,
                            pre,
                            cu1,
                            cu2,
                            cu3,
                            loras,
                            lm,
                            mods,
                            aq,
                            vis,
                            img,
                            res,
                            resc,
                            inf,
                            pinf,
                            steps,
                            guid,
                            solver,
                            pe1,
                            pe1t,
                            pe2,
                            pe2t,
                            pe3,
                            pe3t,
                            tags=tags,
                            prompt=prompt,
                            extra_data=extra_data,
                        )
                    )

                ALL_INPUTS = [
                    fin_id,
                    fin_name,
                    fin_arch,
                    fin_desc,
                    fin_fsm,
                    fin_urls,
                    fin_urls2,
                    fin_te_urls,
                    fin_vae_urls,
                    fin_preload,
                    fin_custom_url_1,
                    fin_custom_url_2,
                    fin_custom_url_3,
                    fin_loras,
                    fin_lmults,
                    fin_modules,
                    fin_autoq,
                    fin_visible,
                    fin_imgout,
                    fin_res,
                    fin_rescat,
                    fin_infos,
                    fin_pinfos,
                    fin_steps,
                    fin_guidance,
                    fin_solver,
                    fin_pe_txt,
                    fin_pe_txt_tok,
                    fin_pe_vid,
                    fin_pe_vid_tok,
                    fin_pe_img,
                    fin_pe_img_tok,
                    fin_tags,
                    fin_prompt,
                ]

                _URL_REFS = [
                    fin_urls_ref,
                    fin_urls2_ref,
                    fin_te_urls_ref,
                    fin_vae_urls_ref,
                    fin_preload_ref,
                    fin_custom_url_1_ref,
                    fin_custom_url_2_ref,
                    fin_custom_url_3_ref,
                ]
                _URL_VALS = [
                    fin_urls_val,
                    fin_urls2_val,
                    fin_te_urls_val,
                    fin_vae_urls_val,
                    fin_preload_val,
                    fin_custom_url_1_val,
                    fin_custom_url_2_val,
                    fin_custom_url_3_val,
                ]
                _URL_INPUTS = [
                    fin_urls,
                    fin_urls2,
                    fin_te_urls,
                    fin_vae_urls,
                    fin_preload,
                    fin_custom_url_1,
                    fin_custom_url_2,
                    fin_custom_url_3,
                ]

                def _sync_refs_from_loaded(*url_values):
                    """After loading a finetune, sync ref dropdowns to match URL values
                    and show validation LED for each field.
                    Returns (9 ref updates, 9 validation HTML updates)."""
                    ref_results = []
                    val_results = []
                    for val in url_values:
                        if not val or not val.strip():
                            ref_results.append(gr.update(value=None))
                            val_results.append(gr.update(value=""))
                            continue
                        first_line = val.strip().split("\n")[0].strip()
                        if first_line.startswith("="):
                            mid = first_line[1:].strip()
                            ok, msg = _validate_model_ref(mid, self)
                            val_results.append(gr.update(value=_led(ok, msg)))
                            ref_results.append(gr.update(value=mid))
                        elif (
                            "/" not in first_line
                            and "\\" not in first_line
                            and not first_line.startswith(("http:", "https:", "."))
                            and not (len(first_line) > 1 and first_line[1] == ":")
                        ):
                            ok, msg = _validate_model_ref(first_line, self)
                            val_results.append(gr.update(value=_led(ok, msg)))
                            ref_results.append(
                                gr.update(value=first_line if ok else None)
                            )
                        else:
                            ref_results.append(gr.update(value=None))
                            lines = [
                                l.strip() for l in val.strip().split("\n") if l.strip()
                            ]
                            parts = []
                            for line in lines:
                                # No-network validation — only format checks + cached disk.
                                # Full HEAD reachability is covered by Validate button.
                                if line.startswith(("http:", "https:")):
                                    parts.append(_led(True, "URL format OK \u2014 use Validate button to check", size=11))
                                elif _check_disk_exists(line):
                                    parts.append(_led(True, "Found on disk", size=11))
                                else:
                                    parts.append(
                                        _led(
                                            False,
                                            f"{line} \u2014 use Validate button to check",
                                            size=11,
                                        )
                                    )
                            val_results.append(gr.update(value="<br>".join(parts)))
                    return tuple(ref_results) + tuple(val_results)

                # Wire Download & Switch → load into main Media Generator tab
                b_load.click(
                    fn=_load,
                    inputs=[self.state, registry, b_sel_id],
                    outputs=[b_status, self.model_choice_target, self.main_tabs],
                )

                # Wire Improve / Create Variant button (needs ALL_INPUTS defined)
                b_improve.click(
                    fn=_improve_to_editor,
                    inputs=[registry, b_sel_id],
                    outputs=ALL_INPUTS + [fm_tabs, b_status, fin_extra_data],
                ).then(
                    fn=lambda: gr.update(value=False), inputs=[], outputs=[fin_auto_id]
                ).then(
                    fn=_sync_refs_from_loaded,
                    inputs=_URL_INPUTS,
                    outputs=_URL_REFS + _URL_VALS,
                ).then(
                    fn=_split_to_rows,
                    inputs=[fin_loras, fin_lmults],
                    outputs=fin_lora_urls
                    + fin_lora_mults
                    + fin_lora_refs
                    + fin_lora_vals,
                )

                def _preview(*vals):
                    # vals: ALL_INPUTS + [fin_extra_data]
                    extra = vals[-1] if len(vals) > len(ALL_INPUTS) else None
                    return _build(*vals[1 : len(ALL_INPUTS)], extra_data=extra)

                for f in ALL_INPUTS:
                    f.change(
                        fn=_preview,
                        inputs=list(ALL_INPUTS) + [fin_extra_data],
                        outputs=[fin_preview],
                    )

                def _sanitize_id(text):
                    return _sanitize_fin_id(str(text or ""))

                def _generate_id(name, source):
                    # Name-first: the finetune name is the dominant, readable
                    # part of the filename; the source model is a suffix used
                    # only for grouping (matches Wan2GP's base-model prefix habit).
                    name_text = str(name or "").strip()
                    src = _sanitize_id(source or "").lower()
                    if name_text:
                        words = _re.findall(r"[A-Za-z0-9]+", name_text)[:4]
                        name_slug = (
                            _sanitize_id("_".join(words).casefold()) if words else ""
                        )
                        if name_slug:
                            return _sanitize_id(
                                f"{name_slug}_{src}" if src else name_slug
                            )
                    return _sanitize_id(f"{src}_finetune" if src else "finetune")

                def _unique_id(base):
                    existing = _existing_finetune_stems()
                    c = base
                    i = 1
                    while c in existing:
                        c = f"{base}_{i}"
                        i += 1
                    return c

                def _auto_id(name, fsm, auto_id, current_id):
                    if auto_id:
                        base = _unique_id(_generate_id(name, fsm))
                        return gr.update(value=base, interactive=False)
                    return gr.update(interactive=True)

                fin_auto_id.change(
                    fn=_auto_id,
                    inputs=[fin_name, fin_fsm, fin_auto_id, fin_id],
                    outputs=[fin_id],
                    queue=False,
                )
                fin_name.input(
                    fn=_auto_id,
                    inputs=[fin_name, fin_fsm, fin_auto_id, fin_id],
                    outputs=[fin_id],
                    queue=False,
                )

                def _update_source_info(fsm, arch):
                    src = str(fsm or "").strip() or str(arch or "").strip()
                    if src:
                        return src
                    return "(no source model)"

                fin_fsm.change(
                    fn=_update_source_info,
                    inputs=[fin_fsm, fin_arch],
                    outputs=[fin_source_info],
                    queue=False,
                )
                fin_arch.change(
                    fn=_update_source_info,
                    inputs=[fin_fsm, fin_arch],
                    outputs=[fin_source_info],
                    queue=False,
                )

                def _switch_mode(fsm):
                    if fsm and str(fsm).strip():
                        return (
                            "editor",
                            gr.update(visible=False),
                            gr.update(visible=True),
                        )
                    return (
                        "creator",
                        gr.update(visible=True),
                        gr.update(visible=False),
                    )

                fin_fsm.change(
                    fn=_switch_mode,
                    inputs=[fin_fsm],
                    outputs=[fin_mode, fin_creator_actions, fin_editor_actions],
                    queue=False,
                )

                def _extract(d):
                    return extract_finetune_fields(d)

                def _fill(file):
                    n_inputs = len(ALL_INPUTS)
                    if file is None:
                        return [gr.update()] * n_inputs + ["Select a file", {}]
                    s = (
                        file
                        if isinstance(file, str)
                        else file.get("name") or file.get("path")
                    )
                    if not s or not Path(s).exists():
                        return [gr.update()] * n_inputs + ["File not found", {}]
                    try:
                        data = json.loads(Path(s).read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as e:
                        return [gr.update()] * n_inputs + [f"Invalid: {e}", {}]
                    err = _validate_finetune_data(data)
                    if err:
                        return [gr.update()] * n_inputs + [f"Cannot load: {err}", {}]
                    vid = Path(s).stem
                    data = _clean_utf8(data)
                    return list((vid,) + _extract(data)) + [
                        "Filled from " + Path(s).name,
                        data,
                    ]

                fin_import_btn.click(
                    fn=_fill,
                    inputs=[fin_import_file],
                    outputs=ALL_INPUTS + [fin_status, fin_extra_data],
                ).then(
                    fn=lambda: gr.update(value=False), inputs=[], outputs=[fin_auto_id]
                ).then(
                    fn=_sync_refs_from_loaded,
                    inputs=_URL_INPUTS,
                    outputs=_URL_REFS + _URL_VALS,
                ).then(
                    fn=_split_to_rows,
                    inputs=[fin_loras, fin_lmults],
                    outputs=fin_lora_urls
                    + fin_lora_mults
                    + fin_lora_refs
                    + fin_lora_vals,
                )

                def _create_action(state_value, id_, *vals):
                    extra_data, remaining = _pop_extra(vals)
                    if extra_data is not None:
                        vals = remaining
                    if not id_:
                        return "Enter an ID", gr.update(), gr.update()
                    safe_id = _sanitize_fin_id(id_)
                    data = _build(*vals, extra_data=extra_data)
                    _write_finetune(safe_id, data)
                    settings_path = _write_settings_for_wan2gp(safe_id, data)
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    refresh, target, _ = (
                        self.load_settings_from_file(state_value, str(settings_path))
                        if hasattr(self, "load_settings_from_file") and settings_path.exists()
                        else (gr.update(), gr.update(), None)
                    )
                    target = _ensure_model_choice_target(target, safe_id)
                    return f"Created {safe_id}", target, gr.update()

                fin_create.click(
                    fn=_create_action,
                    inputs=[self.state] + list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_status, self.model_choice_target, self.main_tabs],
                )
                fin_save_switch.click(
                    fn=_create_action,
                    inputs=[self.state] + list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_status, self.model_choice_target, self.main_tabs],
                )
                fin_save_switch_ed.click(
                    fn=_create_action,
                    inputs=[self.state] + list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_status, self.model_choice_target, self.main_tabs],
                )

                def _export_action(id_, *vals):
                    extra_data, remaining = _pop_extra(vals)
                    if extra_data is not None:
                        vals = remaining
                    if not id_:
                        return None, "Enter an ID first"
                    data = _build(*vals, extra_data=extra_data)
                    safe_id = _sanitize_fin_id(id_)
                    _write_finetune(safe_id, data)
                    path = str(Path(FINETUNES_DIR) / f"{safe_id}.json")
                    return path, f"Exported {safe_id}"

                fin_export.click(
                    fn=_export_action,
                    inputs=list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_export, fin_status],
                )

                def _save_action(id_, *vals):
                    extra_data, remaining = _pop_extra(vals)
                    if extra_data is not None:
                        vals = remaining
                    if not id_:
                        return "Enter an ID"
                    safe_id = _sanitize_fin_id(id_)
                    _write_finetune(safe_id, _build(*vals, extra_data=extra_data))
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    return f"Saved {safe_id}"

                fin_save.click(
                    fn=_save_action,
                    inputs=list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_status],
                )

                def _save_up_action(id_, *vals):
                    extra_data, remaining = _pop_extra(vals)
                    if extra_data is not None:
                        vals = remaining
                    if not id_:
                        return "Enter an ID"
                    token_problem = _upload_token_problem()
                    if token_problem:
                        return token_problem
                    data = _build(*vals, extra_data=extra_data)
                    err = _validate_finetune_data(data, registry=True)
                    if err:
                        return f"Cannot save & upload: {err}"
                    host_problem = _upload_host_problem(data)
                    if host_problem:
                        return f"Cannot save & upload: {host_problem}"
                    bare_warnings = _check_bare_filenames(data)
                    if bare_warnings:
                        return "⚠️ Cannot save & upload:\n" + "\n".join(bare_warnings)
                    safe_id = _sanitize_fin_id(id_)
                    _stamp_hf_username(data)
                    _write_finetune(safe_id, data)
                    token_urls = _find_token_urls(data)
                    if token_urls:
                        keys = sorted({entry.split(":", 1)[0] for entry in token_urls})
                        return (
                            f"Saved locally as '{safe_id}', but upload was blocked: "
                            f"{len(token_urls)} URL(s) in {', '.join(keys)} embed your "
                            "CivitAI API token (`token=`). Uploading would publish the key. "
                            "Replace those URLs with non-token download links first."
                        )
                    try:
                        target = _hf_upload(safe_id, data, self)
                        if (
                            hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs
                        ):
                            self.refresh_model_defs()
                        return f"Saved & uploaded {safe_id} → {target}"
                    except Exception as e:
                        return f"Saved but upload failed: {e}"

                fin_save_up.click(
                    fn=_save_up_action,
                    inputs=list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_status],
                )
                fin_save_up_ed.click(
                    fn=_save_up_action,
                    inputs=list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_status],
                )

                def _load_src_enhancer(fsm):
                    if not fsm:
                        return [gr.update()] * 6
                    raw = _read_model_json(fsm)
                    m = raw.get("model", {})
                    return (
                        m.get("text_prompt_enhancer_instructions", ""),
                        m.get("text_prompt_enhancer_max_tokens", ""),
                        m.get("video_prompt_enhancer_instructions", ""),
                        m.get("video_prompt_enhancer_max_tokens", ""),
                        m.get("image_prompt_enhancer_instructions", ""),
                        m.get("image_prompt_enhancer_max_tokens", ""),
                    )

                fin_pe_load_src.click(
                    fn=_load_src_enhancer,
                    inputs=[fin_fsm],
                    outputs=[
                        fin_pe_txt,
                        fin_pe_txt_tok,
                        fin_pe_vid,
                        fin_pe_vid_tok,
                        fin_pe_img,
                        fin_pe_img_tok,
                    ],
                )

                def _delete_action(id_):
                    if not id_:
                        return "Enter an ID", {}
                    p = _resolve_finetune_path(id_)
                    if not p or not p.exists():
                        return "Not found", {}
                    p.unlink()
                    try:
                        _settings_path_for(id_).unlink(missing_ok=True)
                    except OSError:
                        pass
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    return f"Deleted {id_}", {}

                fin_del.click(
                    fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                    outputs=[fin_del, fin_del_confirm],
                    queue=False,
                )
                fin_del_cancel_btn.click(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    outputs=[fin_del, fin_del_confirm],
                    queue=False,
                )
                fin_del_confirm_btn.click(
                    fn=_delete_action,
                    inputs=[fin_id],
                    outputs=[fin_status, fin_extra_data],
                ).then(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    outputs=[fin_del, fin_del_confirm],
                    queue=False,
                )

                def _cancel_action():
                    blanks = [gr.update(value="") for _ in ALL_INPUTS]
                    blanks[0] = gr.update(value="", interactive=True)
                    return (
                        blanks
                        + [gr.update(value="")]
                        + [gr.update(visible=True), gr.update(visible=False)]
                        + [{}]
                    )

                fin_cancel_create.click(
                    fn=_cancel_action,
                    outputs=(
                        ALL_INPUTS
                        + [fin_status]
                        + [fin_creator_actions, fin_editor_actions]
                        + [fin_extra_data]
                    ),
                    queue=False,
                ).then(
                    fn=_sync_refs_from_loaded,
                    inputs=_URL_INPUTS,
                    outputs=_URL_REFS + _URL_VALS,
                ).then(
                    fn=_split_to_rows,
                    inputs=[fin_loras, fin_lmults],
                    outputs=fin_lora_urls
                    + fin_lora_mults
                    + fin_lora_refs
                    + fin_lora_vals,
                )
                fin_cancel_edit.click(
                    fn=_cancel_action,
                    outputs=(
                        ALL_INPUTS
                        + [fin_status]
                        + [fin_creator_actions, fin_editor_actions]
                        + [fin_extra_data]
                    ),
                    queue=False,
                ).then(
                    fn=_sync_refs_from_loaded,
                    inputs=_URL_INPUTS,
                    outputs=_URL_REFS + _URL_VALS,
                ).then(
                    fn=_split_to_rows,
                    inputs=[fin_loras, fin_lmults],
                    outputs=fin_lora_urls
                    + fin_lora_mults
                    + fin_lora_refs
                    + fin_lora_vals,
                )

                # Indices of URL/LoRA/path fields in ALL_INPUTS — see URL_FIELD_IDX

                def _editor_validate(*vals):
                    # Extract raw text from URL/LoRA/path fields before _build
                    raw_entries = []
                    # URL fields (indices 5-12 in ALL_INPUTS)
                    for i in URL_FIELD_IDX:
                        if i < len(vals) and isinstance(vals[i], str):
                            for line in vals[i].split("\n"):
                                line = line.strip()
                                if line:
                                    raw_entries.append(line)
                    # LoRA URLs from the hidden fin_loras accumulator (index 13)
                    if len(vals) > 13 and isinstance(vals[13], str):
                        for line in vals[13].split("\n"):
                            line = line.strip()
                            if line:
                                raw_entries.append(line)
                    # Also check model ref from fsm field (index 4)
                    if len(vals) > 4 and isinstance(vals[4], str) and vals[4].strip():
                        raw_entries.insert(0, f"={vals[4].strip()}")
                    # Aggregate results (for fin_validate_out)
                    all_results = _validate_urls(raw_entries, self)
                    aggregate_html = _build_url_validation_html(all_results)
                    # Per-field validation HTML (for each URL row's val element)
                    per_field = []
                    for i in URL_FIELD_IDX:  # ALL_INPUTS URL/LoRA fields
                        field_val = (
                            vals[i]
                            if i < len(vals) and isinstance(vals[i], str)
                            else ""
                        )
                        lines = [l.strip() for l in field_val.split("\n") if l.strip()]
                        if not lines:
                            per_field.append(gr.update(value=""))
                            continue
                        field_parts = []
                        for line in lines:
                            valid, msg = _validate_entry(line, self)
                            field_parts.append(_led(valid, msg, size=11))
                        per_field.append(gr.update(value="<br>".join(field_parts)))
                    return tuple([aggregate_html] + per_field)

                fin_validate_urls_btn.click(
                    fn=_editor_validate,
                    inputs=list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_validate_out] + _URL_VALS,
                )

                def _editor_download(*vals):
                    extra_data, remaining = _pop_extra(vals)
                    if extra_data is not None:
                        vals = remaining
                    data = _build(*vals[1:], extra_data=extra_data)
                    return _download_finetune_files(data, self)

                fin_download_btn.click(
                    fn=_editor_download,
                    inputs=list(ALL_INPUTS) + [fin_extra_data],
                    outputs=[fin_download_out],
                )

                def _make_insert_handler(tb):
                    """Drop-down changed: set textbox to `=model_id` and validate."""

                    def _handler(mid, current):
                        if not mid:
                            return (
                                gr.update(),
                                gr.update(value=None),
                                gr.update(value=""),
                            )
                        val_text = f"={mid}"
                        # Guard: if textbox already represents this model, skip to break loop
                        current_clean = (current or "").strip().lstrip("=")
                        if current_clean == mid:
                            return (gr.update(), gr.update(), gr.update())
                        # Validate the model ref
                        ok, msg = _validate_model_ref(mid, self)
                        val_html = _led(ok, msg)
                        return (
                            gr.update(value=val_text),
                            gr.update(value=None),
                            gr.update(value=val_html),
                        )

                    return _handler

                def _textbox_to_dropdown(current):
                    """URL textbox changed: detect model ref and auto-select dropdown.
                    Does lightweight validation (model ref cache, local path, URL format).
                    HTTP availability is checked by the 'Validate All URLs' button."""
                    if not current or not current.strip():
                        return (gr.update(value=None), gr.update(value=""))
                    first_line = current.strip().split("\n")[0].strip()
                    mid = None
                    if first_line.startswith("="):
                        mid = first_line[1:].strip()
                    elif (
                        "/" not in first_line
                        and "\\" not in first_line
                        and not first_line.startswith(("http:", "https:", "."))
                        and not (len(first_line) > 1 and first_line[1] == ":")
                    ):
                        mid = first_line
                    if mid:
                        ok, msg = _validate_model_ref(mid, self)
                        val_html = _led(ok, msg)
                        return (
                            gr.update(value=mid if ok else None),
                            gr.update(value=val_html),
                        )
                    # Custom URL or local path — lightweight (no network)
                    lines = [l.strip() for l in current.split("\n") if l.strip()]
                    parts = []
                    for line in lines:
                        if line.startswith(("http:", "https:")):
                            problem = fm_security.url_problem(line)
                            if problem:
                                parts.append(_led(False, problem, size=11))
                            else:
                                parts.append(
                                    _led(
                                        True,
                                        "URL format OK \u2014 use Validate button "
                                        "to check availability",
                                        size=11,
                                    )
                                )
                        elif line.startswith(("/", ".", "\\")) or (
                            len(line) > 1 and line[1] == ":"
                        ):
                            ok, msg = _validate_local_path(line)
                            parts.append(_led(ok, msg, size=11))
                        else:
                            # Bare string — treat as potential model ref
                            data = _read_model_json(line)
                            if data:
                                name = data.get("model", {}).get("name", line)
                                urls = _get_model_urls(data)
                                msg = f"Model ref: {name}"
                                if urls:
                                    msg += f"\n\u2192 {urls[0]}"
                                parts.append(_led(True, msg, size=11))
                            else:
                                parts.append(
                                    _led(
                                        False,
                                        f"'{line}' not in list \u2014 enter a URL "
                                        "or upload a file instead",
                                        size=11,
                                    )
                                )
                    val_html = "<br>".join(parts)
                    return (gr.update(value=None), gr.update(value=val_html))

                # Wire each URL field: dropdown → textbox, textbox → dropdown, + validation
                _URL_FIELDS = [
                    (fin_urls_ref, fin_urls, fin_urls_val),
                    (fin_urls2_ref, fin_urls2, fin_urls2_val),
                    (fin_te_urls_ref, fin_te_urls, fin_te_urls_val),
                    (fin_vae_urls_ref, fin_vae_urls, fin_vae_urls_val),
                    (fin_preload_ref, fin_preload, fin_preload_val),
                    (fin_custom_url_1_ref, fin_custom_url_1, fin_custom_url_1_val),
                    (fin_custom_url_2_ref, fin_custom_url_2, fin_custom_url_2_val),
                    (fin_custom_url_3_ref, fin_custom_url_3, fin_custom_url_3_val),
                ]
                for ref, tb, val_html in _URL_FIELDS:
                    tb.change(
                        fn=_textbox_to_dropdown,
                        inputs=[tb],
                        outputs=[ref, val_html],
                    )
                    ref.change(
                        fn=_make_insert_handler(tb),
                        inputs=[ref, tb],
                        outputs=[tb, ref, val_html],
                    )

                with gr.Accordion("Search models — fill into editor (CivitAI / Hugging Face)", open=False):
                    gr.Markdown(
                        "Search **CivitAI** or **Hugging Face** — sort by "
                        "trending, downloads, likes or recency — then **fill the "
                        "editor fields** or **create a finetune.json** recipe "
                        "(token-free URLs; Wan2GP fetches the weights when the "
                        "finetune is loaded). CivitAI's **Site** picks "
                        "`civitai.com` or its adult front-end `civitai.red`; "
                        "mature models need *Include NSFW*."
                    )
                    with gr.Row():
                        ci_search = gr.Textbox(
                            label="Search",
                            scale=3,
                            container=False,
                            placeholder="e.g. fantasy landscape, character style, author/repo...",
                        )
                        ci_sort = gr.Dropdown(
                            label="Sort by",
                            choices=list(_CIVITAI_SORTS),
                            value="Most downloads",
                            scale=1,
                            min_width=140,
                        )
                        ci_engine = gr.Dropdown(
                            label="Source",
                            choices=["CivitAI", "Hugging Face"],
                            value="CivitAI",
                            scale=1,
                            min_width=130,
                        )
                        ci_search_btn = gr.Button("Search", variant="primary", scale=1)
                    with gr.Row():
                        ci_type = gr.Dropdown(
                            label="Type",
                            choices=_civitai_enum_choices("ModelType"),
                            value="All",
                            allow_custom_value=True,
                            scale=1,
                        )
                        ci_base = gr.Dropdown(
                            label="Base Model",
                            choices=_civitai_enum_choices("BaseModel"),
                            filterable=True,
                            value="All",
                            allow_custom_value=True,
                            scale=1,
                        )
                        ci_site = gr.Dropdown(
                            label="Site",
                            choices=list(CIVITAI_SITES),
                            value=CIVITAI_SITE_DEFAULT,
                            scale=1,
                            min_width=120,
                        )
                        ci_nsfw = gr.Checkbox(
                            label="Include NSFW",
                            value=CIVITAI_NSFW_DEFAULT,
                            scale=0,
                            min_width=120,
                        )
                    with gr.Row():
                        ci_pick = gr.Dropdown(
                            label="Result version",
                            choices=[],
                            value=None,
                            interactive=True,
                            allow_custom_value=True,
                            scale=3,
                            container=False,
                        )
                        ci_load_btn = gr.Button(
                            "Fill into editor", variant="primary", scale=1
                        )
                        ci_clear_btn = gr.Button("Clear", scale=1)
                    with gr.Row(visible=False) as hf_files_row:
                        hf_files = gr.Dropdown(
                            label="Weight files (Hugging Face)",
                            choices=[],
                            value=None,
                            interactive=True,
                            multiselect=True,
                            allow_custom_value=False,
                            scale=3,
                            container=False,
                        )
                        hf_kind = gr.Dropdown(
                            label="Add as",
                            choices=["Auto", "Checkpoint", "LoRA"],
                            value="Auto",
                            scale=1,
                            min_width=120,
                            filterable=False,
                        )
                    hf_payload_store = gr.State({})
                    with gr.Row():
                        ci_arch = gr.Dropdown(
                            label="Target architecture (LoRA folder / recipe model)",
                            choices=_architecture_choices(self),
                            filterable=True,
                            allow_custom_value=True,
                            scale=3,
                        )
                        ci_all_files = gr.Checkbox(
                            label="All weight files (CivitAI only)",
                            value=False,
                            scale=1,
                            min_width=170,
                        )
                        ci_recipe_btn = gr.Button("Create finetune.json", scale=1)
                    ci_status = gr.Markdown(
                        "Enter a search term to find models on CivitAI"
                    )
                    ci_action_status = gr.Markdown("")
                    ci_results = gr.HTML(
                        "<div style='color:#9ca3af;text-align:center;"
                        "padding:32px'>Click Search</div>"
                    )

                    def _engine_changed(engine):
                        """Switch the row between the two engines."""
                        is_ci = engine != "Hugging Face"
                        return (
                            gr.update(visible=is_ci),  # ci_type
                            gr.update(visible=is_ci),  # ci_base
                            gr.update(visible=is_ci),  # ci_site
                            (
                                gr.update(visible=True, label="Include NSFW")
                                if is_ci
                                else gr.update(
                                    visible=True, label="Adult-flagged only"
                                )
                            ),  # ci_nsfw
                            gr.update(visible=False),  # hf_files_row
                            (
                                "Enter a search term to find models on CivitAI"
                                if is_ci
                                else "Empty search lists Hub-wide trending"
                            ),
                            "<div style='color:#9ca3af;text-align:center;"
                            "padding:32px'>Click Search</div>",
                            gr.update(choices=[], value=None),  # ci_pick
                        )

                    ci_engine.change(
                        fn=_engine_changed,
                        inputs=[ci_engine],
                        outputs=[
                            ci_type,
                            ci_base,
                            ci_site,
                            ci_nsfw,
                            hf_files_row,
                            ci_status,
                            ci_results,
                            ci_pick,
                        ],
                    )

                    def _pick_changed(civitai_json_str, engine):
                        """CivitAI payloads need nothing; HF selections stream the
                        repo's weight files into their dropdown plus a payload
                        store the fill/recipe handlers share."""
                        if engine != "Hugging Face":
                            return gr.update(), gr.update(), ""
                        sel = _hf_selection(civitai_json_str)
                        if sel is None:
                            return gr.update(), gr.update(), "Select a result first."
                        try:
                            files = _hf_model_files(sel["repo"])
                        except Exception as exc:  # noqa: BLE001 - status text
                            return (
                                gr.update(),
                                gr.update(choices=[], value=None),
                                f"Error: {exc}",
                            )
                        if not files:
                            return (
                                gr.update(),
                                gr.update(choices=[], value=None),
                                f"`{sel['repo']}` has no weight files "
                                "(.safetensors/.gguf/.pth/.ckpt).",
                            )
                        payload = dict(sel)
                        payload["files"] = files
                        opts = [
                            (
                                f"{f['name'].rsplit('/', 1)[-1]}"
                                f" ({_hf_human_size(f['size'])})",
                                f["name"],
                            )
                            for f in files
                        ]
                        return (
                            payload,
                            gr.update(choices=opts, value=None),
                            f"**{len(files)} weight file(s)** in `{sel['repo']}` "
                            "— pick one or a HIGH/LOW pair, then Fill.",
                        )

                    ci_pick.change(
                        fn=_pick_changed,
                        inputs=[ci_pick, ci_engine],
                        outputs=[hf_payload_store, hf_files, ci_status],
                    )

                    def _ci_target_arch(arch, state_value):
                        """Resolve the target architecture: explicit choice, then the
                        current Wan2GP model, then Wan2GP's first installed model."""
                        value = str(arch or "").strip() or _state_model_type(state_value)
                        if value:
                            return value
                        choices = _architecture_choices(self)
                        return str(choices[0][1]) if choices else ""

                    def _ci_hint(exc) -> str:
                        text = str(exc).lower()
                        if any(
                            marker in text
                            for marker in (
                                "text/html",
                                "401",
                                "403",
                                "unauthorized",
                                "forbidden",
                                "login",
                            )
                        ):
                            return (
                                "\n\nCivitAI gates this file. Add `civitai_api_key` to "
                                f"`{_cfg_path}` or download it manually from the site."
                            )
                        return ""


                    def _ci_do_search(
                        query, mtype, base, nsfw, site, engine, sort
                    ):
                        query = (query or "").strip()
                        sort = sort if sort in _CIVITAI_SORTS else "Most downloads"
                        if engine == "Hugging Face":
                            if not query:
                                pass  # empty query = Hub-wide trending browse
                            try:
                                data = _hf_search(
                                    query,
                                    sort=sort,
                                    adult_only=bool(nsfw),
                                    limit=24,
                                )
                            except Exception as exc:  # noqa: BLE001 - status text
                                return (
                                    "<div style='color:#dc2626;padding:16px'>"
                                    f"Error: {html.escape(str(exc))}</div>",
                                    f"Error: {html.escape(str(exc))}",
                                    gr.update(choices=[], value=None),
                                    gr.update(),
                                    gr.update(),
                                )
                            items = data["items"]
                            html_out = _hf_render_results(items)
                            options = _hf_result_options(items)
                            label = (
                                f"'{html.escape(query)}'" if query else "Hub-wide trending"
                            )
                            status = (
                                f"**{len(options)} results** for {label} "
                                f"({sort.lower()})"
                            )
                            if bool(nsfw):
                                status += " · adult-flagged only"
                            return (
                                html_out,
                                status,
                                gr.update(choices=options, value=None),
                                gr.update(),
                                gr.update(),
                            )
                        site = _civitai_site(site)
                        filtered = mtype not in ("", "All", None) or base not in ("", "All", None)
                        _civitai_prefetch_enums()
                        enum_updates = _civitai_enum_updates(mtype, base)
                        if not query and not filtered:
                            return (
                                "<div style='color:#9ca3af;"
                                "text-align:center;padding:32px'>"
                                "Enter a search term or pick a Type / Base Model</div>",
                                "Enter a search term or pick a Type / Base Model",
                                gr.update(choices=[], value=None),
                            ) + tuple(enum_updates)
                        data = _civitai_search(
                            query,
                            model_type=mtype,
                            base_model=base,
                            nsfw=bool(nsfw),
                            site=site,
                            sort=_CIVITAI_SORTS.get(sort, "Most Downloaded"),
                        )
                        html_out = _civitai_render_results(data)
                        options = _civitai_result_options(data, site)
                        count = len(data.get("items", [])) if "error" not in data else 0
                        label = f"'{html.escape(query)}'" if query else _CIVITAI_SORTS.get(sort, sort)
                        status = (
                            f"**{count} results** for {label} ({sort.lower()})"
                            if count
                            else f"No results found: {html.escape(data['error'])}"
                            if "error" in data
                            else "No results found"
                        )
                        if count and (site != "civitai.com" or nsfw):
                            status += f" · {site}" + (" · NSFW included" if nsfw else "")
                        return (
                            html_out,
                            status,
                            gr.update(choices=options, value=None),
                        ) + tuple(enum_updates)

                    ci_search_btn.click(
                        fn=_ci_do_search,
                        inputs=[
                            ci_search,
                            ci_type,
                            ci_base,
                            ci_nsfw,
                            ci_site,
                            ci_engine,
                            ci_sort,
                        ],
                        outputs=[ci_results, ci_status, ci_pick, ci_type, ci_base],
                    )
                    ci_search.submit(
                        fn=_ci_do_search,
                        inputs=[
                            ci_search,
                            ci_type,
                            ci_base,
                            ci_nsfw,
                            ci_site,
                            ci_engine,
                            ci_sort,
                        ],
                        outputs=[ci_results, ci_status, ci_pick, ci_type, ci_base],
                    )

                    def _ci_fill(
                        engine,
                        civitai_json_str,
                        hf_store,
                        hf_files_sel,
                        hf_kind_sel,
                        arch,
                        current_name,
                        current_desc,
                        current_tags,
                        current_loras,
                        current_mults,
                    ):
                        if engine == "Hugging Face":
                            return _hf_fill_updates(
                                hf_store,
                                hf_files_sel,
                                arch=arch,
                                current_name=current_name,
                                current_desc=current_desc,
                                current_tags=current_tags,
                                current_loras=current_loras,
                                current_loras_multipliers=current_mults,
                                force_kind=hf_kind_sel or "Auto",
                            )
                        return _civitai_extract_fill_data(
                            civitai_json_str,
                            arch=arch,
                            current_name=current_name,
                            current_desc=current_desc,
                            current_tags=current_tags,
                            current_loras=current_loras,
                            current_loras_multipliers=current_mults,
                        )

                    ci_load_btn.click(
                        fn=_ci_fill,
                        inputs=[
                            ci_engine,
                            ci_pick,
                            hf_payload_store,
                            hf_files,
                            hf_kind,
                            ci_arch,
                            fin_name,
                            fin_desc,
                            fin_tags,
                            fin_loras,
                            fin_lmults,
                        ],
                        outputs=ALL_INPUTS[1:] + [ci_action_status],
                    ).then(
                        fn=_sync_refs_from_loaded,
                        inputs=_URL_INPUTS,
                        outputs=_URL_REFS + _URL_VALS,
                    ).then(
                        fn=_split_to_rows,
                        inputs=[fin_loras, fin_lmults],
                        outputs=fin_lora_urls
                        + fin_lora_mults
                        + fin_lora_refs
                        + fin_lora_vals,
                    )


                    def _ci_create_recipe(
                        state_value,
                        engine,
                        civitai_json_str,
                        hf_store,
                        hf_files_sel,
                        hf_kind_sel,
                        arch,
                        all_files,
                    ):
                        arch_used = _ci_target_arch(arch, state_value)
                        if not arch_used:
                            return (
                                "Choose a target architecture for the recipe.",
                                gr.update(),
                            )
                        if engine == "Hugging Face":
                            if not hf_store.get("repo") or not hf_files_sel:
                                return (
                                    "Pick a Hugging Face result and its weight "
                                    "files first.",
                                    gr.update(),
                                )
                            try:
                                data = _hf_recipe_payload(
                                    hf_store,
                                    hf_files_sel,
                                    architecture=arch_used,
                                    force_kind=hf_kind_sel or "Auto",
                                )
                            except Exception as exc:  # noqa: BLE001 - status text
                                return (
                                    f"**Cannot create recipe:** {exc}",
                                    gr.update(value=arch_used or None),
                                )
                            safe_id = _unique_id(
                                _sanitize_fin_id(
                                    f"hf_{hf_store['repo'].replace('/', '_')}"
                                )
                            )
                            _write_finetune(safe_id, data)
                            _refresh_model_defs(self)
                            url_count = len(data["model"].get("URLs", [])) + len(
                                data["model"].get("URLs2", [])
                            ) + len(data["model"].get("loras", []))
                            filename = Path(FINETUNES_DIR) / f"{safe_id}.json"
                            return (
                                f"**Created `{filename}`** with {url_count} "
                                "download URL(s). Wan2GP fetches the weights "
                                "when this finetune is loaded and saves each "
                                "under its real filename.",
                                gr.update(value=arch_used),
                            )
                        sel = _civitai_selection(civitai_json_str)
                        if sel is None:
                            return (
                                "Select a result version first, then press "
                                "*Create finetune.json*.",
                                gr.update(),
                            )
                        try:
                            data = fm_pack.civitai_recipe(
                                sel["model"],
                                sel["version"],
                                architecture=arch_used,
                                primary_only=not bool(all_files),
                                site=sel["site"],
                            )
                        except Exception as exc:  # noqa: BLE001 - shown in status
                            return (
                                f"**Cannot create recipe:** {exc}",
                                gr.update(value=arch_used or None),
                            )
                        safe_id = _unique_id(_civitai_recipe_id(sel))
                        _write_finetune(safe_id, data)
                        _refresh_model_defs(self)
                        url_count = len(
                            fm_pack.civitai_download_urls(
                                sel["version"],
                                site=sel["site"],
                                primary_only=not bool(all_files),
                            )
                        )
                        filename = Path(FINETUNES_DIR) / f"{safe_id}.json"
                        return (
                            f"**Created `{filename}`** with {url_count} download "
                            "URL(s). Wan2GP fetches the weights when this finetune "
                            "is loaded and saves each under its real filename. The "
                            "recipe stores the public URL only.",
                            gr.update(value=arch_used),
                        )

                    ci_recipe_btn.click(
                        fn=_ci_create_recipe,
                        inputs=[
                            self.state,
                            ci_engine,
                            ci_pick,
                            hf_payload_store,
                            hf_files,
                            hf_kind,
                            ci_arch,
                            ci_all_files,
                        ],
                        outputs=[ci_action_status, ci_arch],
                    )

                    def _ci_clear(engine):
                        return (
                            gr.update(visible=engine != "Hugging Face"),
                            "<div style='color:#9ca3af;text-align:center;"
                            "padding:32px'>Click Search</div>",
                            gr.update(choices=[], value=None),
                            {},
                            gr.update(choices=[], value=None),
                            "Enter a search term to find models",
                            "",
                        )

                    ci_clear_btn.click(
                        fn=_ci_clear,
                        inputs=[ci_engine],
                        outputs=[
                            hf_files_row,
                            ci_results,
                            ci_pick,
                            hf_payload_store,
                            hf_files,
                            ci_status,
                            ci_action_status,
                        ],
                    )

                with gr.Accordion("Fill from Hugging Face", open=False):
                    gr.Markdown(
                        "Paste an HF model URL or `author/repo` id, list its weight "
                        "files, then fill the editor URLs and name."
                    )
                    with gr.Row():
                        hf_repo_box = gr.Textbox(
                            label="HF model",
                            scale=3,
                            container=False,
                            placeholder="https://huggingface.co/author/repo  or  author/repo",
                        )
                        hf_list_btn = gr.Button("List files", scale=0)
                    with gr.Row():
                        hf_files_dd = gr.Dropdown(
                            label="Weight file(s)",
                            choices=[],
                            value=None,
                            multiselect=True,
                            interactive=True,
                            allow_custom_value=True,
                            scale=3,
                        )
                        hf_apply_btn = gr.Button("Fill into editor", scale=1)
                    hf_status = gr.Markdown("")

                    def _hf_repo_id(text):
                        t = str(text or "").strip()
                        m = _re.search(
                            r"(?:huggingface\.co|hf\.co)/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
                            t,
                        )
                        if m:
                            return m.group(1).rstrip("/#").removesuffix("/blob").removesuffix("/tree")
                        if _re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", t):
                            return t
                        return ""

                    def _hf_list(repo_text):
                        repo = _hf_repo_id(repo_text)
                        if not repo:
                            return (
                                gr.update(choices=[], value=None),
                                "Enter an HF model URL or `author/repo` id.",
                            )
                        try:
                            from huggingface_hub import HfApi
                            api = HfApi(token=REGISTRY_TOKEN) if _REGISTRY_TOKEN_VALID else HfApi()
                            info = api.model_info(repo, files_metadata=True)
                            weights = [
                                s.rfilename
                                for s in (info.siblings or [])
                                if str(s.rfilename).lower().endswith(
                                    (".safetensors", ".gguf", ".pth", ".ckpt")
                                )
                            ]
                            if not weights:
                                return (
                                    gr.update(choices=[], value=None),
                                    f"`{repo}` has no weight files (.safetensors/.gguf/.pth/.ckpt).",
                                )
                            options = [(f, f) for f in weights]
                            return (
                                gr.update(choices=options, value=None),
                                f"**{len(weights)} weight file(s)** in `{repo}` — pick one or a HIGH/LOW pair.",
                            )
                        except Exception as exc:
                            return gr.update(choices=[], value=None), f"Error: {exc}"

                    hf_list_btn.click(
                        fn=_hf_list,
                        inputs=[hf_repo_box],
                        outputs=[hf_files_dd, hf_status],
                    )
                    hf_repo_box.submit(
                        fn=_hf_list,
                        inputs=[hf_repo_box],
                        outputs=[hf_files_dd, hf_status],
                    )

                    def _hf_apply(repo_text, files, current_name):
                        repo = _hf_repo_id(repo_text)
                        picked = [str(f).strip() for f in (files or []) if str(f).strip()]
                        if not repo or not picked:
                            return (
                                gr.update(),
                                gr.update(),
                                "List files and pick at least one weight file first.",
                            )
                        urls = "\n".join(
                            f"https://huggingface.co/{repo}/resolve/main/{f}"
                            for f in picked
                        )
                        last = picked[-1].rsplit("/", 1)[-1]
                        name = (
                            gr.update(value=last)
                            if not str(current_name or "").strip()
                            else gr.update()
                        )
                        return (
                            gr.update(value=urls),
                            name,
                            f"Filled {len(picked)} URL(s) from `{repo}`.",
                        )

                    hf_apply_btn.click(
                        fn=_hf_apply,
                        inputs=[hf_repo_box, hf_files_dd, fin_name],
                        outputs=[fin_urls, fin_name, hf_status],
                    ).then(
                        fn=_sync_refs_from_loaded,
                        inputs=_URL_INPUTS,
                        outputs=_URL_REFS + _URL_VALS,
                    )

            # ── LOCAL ──
            with gr.TabItem("Local", id="local"):
                with gr.Row():
                    l_search = gr.Textbox(
                        label="Search", scale=3, container=False,
                        placeholder="Filter by name, description or ID...",
                    )
                    l_family = gr.Dropdown(
                        label="Model family",
                        choices=[("All families", "All")],
                        value="All",
                        filterable=True,
                        allow_custom_value=True,
                        scale=1,
                    )
                    l_arch = gr.Dropdown(
                        label="Architecture",
                        choices=["All"],
                        value="All",
                        allow_custom_value=True,
                        filterable=True,
                        scale=1,
                    )
                l_refresh = gr.Button("Refresh", variant="primary")
                l_entries = gr.State([])
                l_count = gr.Markdown("Click *Refresh* to list local finetunes")
                l_sel_id = gr.Textbox(visible=False, elem_id="l-selected", value="")
                l_cards = gr.HTML(
                    "<div style='color:#9ca3af;text-align:center;"
                    "padding:32px'>Click Refresh to list local finetunes</div>"
                )
                l_detail = gr.JSON(label="Content")
                with gr.Row():
                    l_load = gr.Button("Load & Switch", variant="primary")
                    l_del = gr.Button("Delete")
                    l_up = gr.Button("Upload to Registry")
                l_del_confirm = gr.Row(visible=False)
                with l_del_confirm:
                    l_del_confirm_btn = gr.Button("Confirm Delete", variant="stop")
                    l_del_cancel_btn = gr.Button("Cancel")
                with gr.Row():
                    l_imp_file = gr.File(label="Import .json", file_types=[".json"])
                    l_imp_btn = gr.Button("Import")
                l_export_btn = gr.DownloadButton("Export JSON", value=None)
                l_status = gr.Textbox(label="Status")

            # ── CIVITAI ──
            # ── SHARED LOCAL HANDLERS ──
            def _loc_detail(fid):
                if not fid:
                    return {}
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return {"error": "not found"}
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    return {"error": f"invalid JSON: {e}"}
                return _clean_utf8(data)

            def _scan_local_entries():
                """Read every local finetune once and attach its model family."""
                p = Path(FINETUNES_DIR)
                entries = []
                if not p.exists():
                    return entries
                catalog = _family_catalog(self)
                for fpath in sorted(p.glob("*.json")):
                    fid = fpath.stem
                    try:
                        data = json.loads(fpath.read_text(encoding="utf-8"))
                    except Exception as e:
                        entries.append({
                            "id": fid,
                            "pack_id": fm_assets.pack_display_id(fid),
                            "is_pack": fm_assets.is_pack_id(fid),
                            "name": fid,
                            "broken": f"Corrupt or invalid JSON: {e}",
                            "family": _FAMILY_UNKNOWN,
                            "family_label": _family_label(catalog, _FAMILY_UNKNOWN),
                        })
                        continue
                    err = _validate_finetune_data(data)
                    if err:
                        entries.append({
                            "id": fid,
                            "pack_id": fm_assets.pack_display_id(fid),
                            "is_pack": fm_assets.is_pack_id(fid),
                            "name": fid,
                            "broken": err,
                            "family": _FAMILY_UNKNOWN,
                            "family_label": _family_label(catalog, _FAMILY_UNKNOWN),
                        })
                        continue
                    m = data.get("model", {})
                    family = _family_for_model(self, m)
                    entries.append({
                        "id": fid,
                        "pack_id": fm_assets.pack_display_id(fid),
                        "is_pack": fm_assets.is_pack_id(fid),
                        "name": m.get("name", fid),
                        "model": m,
                        "family": family,
                        "family_label": _family_label(catalog, family),
                        "family_warning": _family_warning(self, m, family),
                        "host_warning": _unexpected_host_warning(data),
                    })
                return entries

            def _render_local(entries, search="", family="All", arch="All"):
                ff = list(entries or [])
                if search:
                    s = search.strip().casefold()
                    ff = [
                        e for e in ff
                        if s in str(e.get("name", "")).casefold()
                        or s in str(e.get("id", "")).casefold()
                        or s in str(e.get("pack_id", "")).casefold()
                        or s in str((e.get("model") or {}).get("description", "")).casefold()
                    ]
                if family and family != "All":
                    ff = [e for e in ff if e.get("family") == family]
                if arch and arch != "All":
                    ff = [
                        e for e in ff
                        if str((e.get("model") or {}).get("architecture", "")) == arch
                    ]
                if not ff:
                    return (
                        "<div style='color:#9ca3af;text-align:center;"
                        "padding:32px'>No local finetunes match</div>",
                        "**0 local finetunes**",
                    )
                catalog = _family_catalog(self)
                ff.sort(
                    key=lambda e: _family_sort_key(
                        catalog, e.get("family") or _FAMILY_UNKNOWN, e.get("name", "")
                    )
                )
                counts = {}
                for e in ff:
                    fam = e.get("family") or _FAMILY_UNKNOWN
                    counts[fam] = counts.get(fam, 0) + 1
                _h = ['<div class="fm-cards-container">']
                current_family = None
                for e in ff:
                    fam = e.get("family") or _FAMILY_UNKNOWN
                    if fam != current_family:
                        _h.append(
                            '<div class="fm-family-header">'
                            f"<span>{html.escape(_family_label(catalog, fam))}</span>"
                            f'<span class="fm-family-count">{counts[fam]}</span>'
                            "</div>"
                        )
                        current_family = fam
                    fid = e.get("id", "")
                    if e.get("broken"):
                        _h.append(
                            f'<div class="fm-card fm-card--broken">'
                            f'<div class="fm-card-name" style="color:#f87171">'
                            f'&#9888; {html.escape(fid)}</div>'
                            f'<div class="fm-card-desc">'
                            f'{html.escape(str(e["broken"]))}</div></div>'
                        )
                        continue
                    m = e.get("model") or {}
                    _h.append(
                        _fmt_card_html_item(
                            fid=fid,
                            name=e.get("name", fid),
                            arch=m.get("architecture", ""),
                            author=m.get("author", "local"),
                            desc=m.get("description", ""),
                            urls=m.get("URLs", []),
                            loras=m.get("loras", []),
                            ftags=m.get("tags", []),
                            is_selected=False,
                            sel_elem_id="l-selected",
                            is_variant=bool(m.get("finetune_source_model")),
                            family_label=e.get("family_label", ""),
                            family_warning=e.get("family_warning", ""),
                            is_pack=bool(e.get("is_pack")),
                            host_warning=e.get("host_warning", ""),
                        )
                    )
                _h.append("</div>")
                plural = "" if len(ff) == 1 else "s"
                return ("".join(_h), f"**{len(ff)} local finetune{plural}**")

            def _local_family_updates(entries):
                catalog = _family_catalog(self)
                present = {e.get("family", _FAMILY_UNKNOWN) for e in entries}
                choices = [("All families", "All")] + _family_choices(catalog, present)
                if _FAMILY_UNKNOWN in present and "unknown" not in catalog:
                    choices.append(("Unknown", _FAMILY_UNKNOWN))
                archs = sorted({
                    str((e.get("model") or {}).get("architecture", "")).strip()
                    for e in entries
                    if str((e.get("model") or {}).get("architecture", "")).strip()
                })
                return (
                    gr.update(choices=choices, value="All"),
                    gr.update(choices=["All"] + archs, value="All"),
                )

            def _refresh_local():
                entries = _scan_local_entries()
                html_out, count = _render_local(entries)
                family_update, arch_update = _local_family_updates(entries)
                return entries, html_out, count, family_update, arch_update

            def _local_filter(entries, search, family, arch):
                html_out, count = _render_local(entries, search, family, arch)
                return html_out, count

            def _local_family_changed(entries, search, family):
                html_out, count = _render_local(entries, search, family, "All")
                archs = sorted({
                    str((e.get("model") or {}).get("architecture", "")).strip()
                    for e in (entries or [])
                    if str((e.get("model") or {}).get("architecture", "")).strip()
                    and (family in ("", "All") or e.get("family") == family)
                })
                return html_out, count, gr.update(choices=["All"] + archs, value="All")

            l_refresh.click(
                fn=_refresh_local,
                outputs=[l_entries, l_cards, l_count, l_family, l_arch],
            )
            l_search.input(
                fn=_local_filter,
                inputs=[l_entries, l_search, l_family, l_arch],
                outputs=[l_cards, l_count],
            )
            l_arch.change(
                fn=_local_filter,
                inputs=[l_entries, l_search, l_family, l_arch],
                outputs=[l_cards, l_count],
            )
            l_family.change(
                fn=_local_family_changed,
                inputs=[l_entries, l_search, l_family],
                outputs=[l_cards, l_count, l_arch],
            )
            l_sel_id.input(fn=_loc_detail, inputs=[l_sel_id], outputs=[l_detail])

            def _loc_load(state_value, fid):
                if not fid:
                    return "Select one", gr.update(), gr.update()
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return "Not found", gr.update(), gr.update()
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception as e:
                    return f"Invalid JSON in '{fid}': {e}", gr.update(), gr.update()
                err = _validate_finetune_data(data)
                if err:
                    return f"Cannot load '{fid}': {err}", gr.update(), gr.update()
                bare_warnings = _check_bare_filenames(data)
                if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                    self.refresh_model_defs()
                settings_path = _write_settings_for_wan2gp(fid, data)
                if hasattr(self, "load_settings_from_file"):
                    refresh, target, _ = self.load_settings_from_file(state_value, str(settings_path))
                else:
                    refresh, target = gr.update(), gr.update()
                # load_settings_from_file returns gr.update() for model_choice_target
                # when the model type falls back to the current model (because the
                # finetune ID isn't in base model types).  In that case the form
                target = _ensure_model_choice_target(target, fid)
                msg = f"Switched to '{fid}'"
                if bare_warnings:
                    msg += "\n\n⚠️ " + "\n".join(bare_warnings)
                return msg, target, gr.update()

            l_load.click(
                fn=_loc_load,
                inputs=[self.state, l_sel_id],
                outputs=[l_status, self.model_choice_target, self.main_tabs],
            )

            def _loc_del(fid):
                if not fid:
                    return "Select one"
                p = _resolve_finetune_path(fid)
                if p and p.exists():
                    p.unlink()
                else:
                    return "Not found"
                try:
                    _settings_path_for(fid).unlink(missing_ok=True)
                except OSError:
                    pass
                DISK_EXISTS_CACHE.clear()
                if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                    self.refresh_model_defs()
                return f"Deleted {fid}"

            l_del.click(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                outputs=[l_del, l_del_confirm],
                queue=False,
            )
            l_del_cancel_btn.click(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                outputs=[l_del, l_del_confirm],
                queue=False,
            )
            l_del_confirm_btn.click(
                fn=_loc_del, inputs=[l_sel_id], outputs=[l_status]
            ).then(
                fn=_refresh_local,
                outputs=[l_entries, l_cards, l_count, l_family, l_arch],
            ).then(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                outputs=[l_del, l_del_confirm],
                queue=False,
            )

            def _loc_export_detail(fid):
                if not fid:
                    return {"error": "not found"}, None
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return {"error": "not found"}, None
                data = json.loads(p.read_text(encoding="utf-8"))
                return _clean_utf8(data), str(p)

            l_export_btn.click(
                fn=_loc_export_detail,
                inputs=[l_sel_id],
                outputs=[l_detail, l_export_btn],
            )

            def _preview_import_file(file):
                if file is None:
                    return {}
                s = (
                    file
                    if isinstance(file, str)
                    else file.get("name") or file.get("path")
                )
                if not s or not Path(s).exists():
                    return {"error": "File not found"}
                try:
                    data = json.loads(Path(s).read_text(encoding="utf-8"))
                    return _clean_utf8(data)
                except Exception as e:
                    return {"error": f"Invalid JSON: {e}"}

            l_imp_file.change(
                fn=_preview_import_file,
                inputs=[l_imp_file],
                outputs=[l_detail],
            )

            def _loc_import(state_value, file):
                def _failed(message):
                    entries, cards_html, count, fam_u, arch_u = _refresh_local()
                    return (
                        message, gr.update(), gr.update(),
                        entries, cards_html, count, fam_u, arch_u,
                    )

                if file is None:
                    return _failed("Select a .json")
                s = (
                    file
                    if isinstance(file, str)
                    else file.get("name") or file.get("path")
                )
                if not s or not Path(s).exists():
                    return _failed("Not found")
                try:
                    data = json.loads(Path(s).read_text(encoding="utf-8"))
                except Exception as e:
                    return _failed(f"Invalid: {e}")
                err = _validate_finetune_data(data, registry=True)
                if err:
                    return _failed(f"Cannot import: {err}")
                fid = _sanitize_fin_id(Path(s).stem)
                existing = _resolve_finetune_path(fid)
                if existing and existing.exists():
                    return _failed(
                        f"'{fid}' already exists — delete it first or rename the file"
                    )
                _write_finetune(fid, data)
                # Build a Wan2GP-compatible settings file and delegate to
                # Wan2GP's built-in load_settings_from_file for model switching.
                settings_path = _write_settings_for_wan2gp(fid, data)
                refresh, target, _ = (
                    self.load_settings_from_file(state_value, str(settings_path))
                    if hasattr(self, "load_settings_from_file") and settings_path.exists()
                    else (gr.update(), gr.update(), None)
                )
                target = _ensure_model_choice_target(target, fid)
                entries, cards_html, count, fam_u, arch_u = _refresh_local()
                return (
                    f"Imported '{fid}'", target, gr.update(),
                    entries, cards_html, count, fam_u, arch_u,
                )

            l_imp_btn.click(
                fn=_loc_import,
                inputs=[self.state, l_imp_file],
                outputs=[
                    l_status, self.model_choice_target, self.main_tabs,
                    l_entries, l_cards, l_count, l_family, l_arch,
                ],
            )

            def _up(fid):
                if not fid:
                    return "Select one"
                token_problem = _upload_token_problem()
                if token_problem:
                    return token_problem
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return "Not found"
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception as e:
                    return f"Cannot upload: invalid JSON ({e})"
                err = _validate_finetune_data(data, registry=True)
                if err:
                    return f"Cannot upload: {err}"
                host_problem = _upload_host_problem(data)
                if host_problem:
                    return f"Cannot upload: {host_problem}"
                token_urls = _find_token_urls(data)
                if token_urls:
                    return (
                        f"Cannot upload '{fid}': {len(token_urls)} URL(s) embed your "
                        "CivitAI API token (`token=`). Remove the token before uploading "
                        "or it will be published with the finetune."
                    )
                _stamp_hf_username(data)
                target = _hf_upload(fid, data, self)
                return f"Uploaded '{fid}' → {target}"

            l_up.click(fn=_up, inputs=[l_sel_id], outputs=[l_status])
