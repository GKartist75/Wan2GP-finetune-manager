import html
import json
import re as _re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import requests
import gradio as gr
# Lazy-imported: HfApi is imported inside functions that use it
# so the plugin can load even if huggingface_hub is missing.
from shared.utils.plugins import WAN2GPPlugin
from shared.gradio.local_file_picker import LocalFilePickerTextbox
import copy
import threading
import time

try:
    from . import fm_assets, fm_pack
except ImportError:  # pragma: no cover - direct script/import fallback
    import fm_assets
    import fm_pack


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


def _validate_finetune_data(data) -> str | None:
    """Validate a loaded finetune dict before applying it to Wan2GP.

    Returns a human-readable error message if the data is unusable, or None
    if it is safe to pass to _write_settings_for_wan2gp(). Centralised so the
    registry-load, local-load and import paths all enforce the same contract
    (a dict with a `model` sub-object) instead of crashing deep inside
    _write_settings_for_wan2gp's `dict(data)`.
    """
    if not isinstance(data, dict):
        return f"Finetune data is not a JSON object (got {type(data).__name__})"
    if "model" not in data:
        return "Finetune JSON is missing the required 'model' field"
    if not isinstance(data["model"], dict):
        return "'model' field is not a JSON object"
    return None


def _write_settings_for_wan2gp(fid: str, data: dict) -> "Path":
    """Build a Wan2GP-compatible settings file from finetune data so
    load_settings_from_file() can parse it and trigger model switching.

    ALL top-level fields in `data` pass through verbatim -- this is already
    a complete Wan2GP settings JSON (prompt, activated_loras, guidance, steps,
    resolution, flow_shift, sample_solver, etc.).  Only `model_type` (the
    finetune ID) is added.  Unrecognized keys are ignored by Wan2GP.
    """
    settings = dict(data)
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
PLUGIN_VERSION = "3.7.0"

DEFAULT_REGISTRY = "https://huggingface.co/spaces/GKartist75/wan2gp-finetunes/raw/main"
REGISTRY_SPACE = "GKartist75/wan2gp-finetunes"
FINETUNES_DIR = "finetunes"

_cfg_path = Path(__file__).parent / "config.json"
_cfg_dir = _cfg_path.parent
_pack_state_path = _cfg_dir / fm_pack.STATE_NAME

REGISTRY_TOKEN = ""
CIVITAI_NSFW_DEFAULT = False
CIVITAI_API_KEY = ""
CIVITAI_SITE_DEFAULT = "civitai.com"
if _cfg_path.exists():
    try:
        _cfg = json.loads(_cfg_path.read_text(encoding="utf-8"))
        REGISTRY_TOKEN = _cfg.get("registry_token", "")
        CIVITAI_NSFW_DEFAULT = bool(_cfg.get("civitai_nsfw", False))
        CIVITAI_API_KEY = str(_cfg.get("civitai_api_key", "") or "")
        CIVITAI_SITE_DEFAULT = str(_cfg.get("civitai_site", "civitai.com") or "civitai.com")
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
    for key in DOWNLOAD_URL_KEYS + ["loras"]:
        vals = m.get(key, [])
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue
        for v in vals:
            if isinstance(v, str) and "token=" in v:
                found.append(f"{key}: {v}")
    return found

# ── Registry cache: avoids redundant N+1 fetches on rapid Refresh ──
_REGISTRY_CACHE: list[dict] | None = None
_REGISTRY_CACHE_TIME: float = 0
_REGISTRY_CACHE_TTL: float = 30.0  # seconds
_REGISTRY_CACHE_FAILURE_TTL: float = 300.0  # seconds — serve stale on error


def _get_cached_registry(force: bool = False) -> list[dict]:
    """Return cached registry, fetching fresh only if stale or forced.
    On fetch failure, serves stale cache for up to _REGISTRY_CACHE_FAILURE_TTL."""
    global _REGISTRY_CACHE, _REGISTRY_CACHE_TIME
    now = time.time()
    if not force and _REGISTRY_CACHE is not None and (now - _REGISTRY_CACHE_TIME) < _REGISTRY_CACHE_TTL:
        return _REGISTRY_CACHE
    try:
        fins = _fetch_dynamic_registry_no_cache()
    except Exception:
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
) -> str:
    """Build a single finetune card HTML fragment.
    Used by both the Browse and Local tab card builders."""
    c = " selected" if is_selected else ""
    safe_name = html.escape(name or "?")
    safe_arch = html.escape(arch or "")
    safe_author = html.escape(author or "")
    safe_desc = html.escape((desc or "")[:300])
    safe_family = html.escape(family_label or "")
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
            f"match the model name'>&#9888; looks like {html.escape(family_warning)}</span>"
        )
    tag = " ".join(badge_parts)
    if isinstance(ftags, str):
        ftags_list = [t.strip() for t in ftags.split(",") if t.strip()]
    elif isinstance(ftags, list):
        ftags_list = [t.strip() for t in ftags if t.strip()]
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
        f"if(ta){{ta.value={json.dumps(fid)};"
        f"ta.dispatchEvent(new Event('input',{{bubbles:true}}))}}"
    )

    if isinstance(urls, str):
        urls = [urls]
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
        if pe1t:
            m["text_prompt_enhancer_max_tokens"] = int(pe1t)
        elif "text_prompt_enhancer_max_tokens" in m:
            del m["text_prompt_enhancer_max_tokens"]
        if pe2:
            m["video_prompt_enhancer_instructions"] = pe2
        elif "video_prompt_enhancer_instructions" in m:
            del m["video_prompt_enhancer_instructions"]
        if pe2t:
            m["video_prompt_enhancer_max_tokens"] = int(pe2t)
        elif "video_prompt_enhancer_max_tokens" in m:
            del m["video_prompt_enhancer_max_tokens"]
        if pe3:
            m["image_prompt_enhancer_instructions"] = pe3
        elif "image_prompt_enhancer_instructions" in m:
            del m["image_prompt_enhancer_instructions"]
        if pe3t:
            m["image_prompt_enhancer_max_tokens"] = int(pe3t)
        elif "image_prompt_enhancer_max_tokens" in m:
            del m["image_prompt_enhancer_max_tokens"]
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
        if pe1t:
            m["text_prompt_enhancer_max_tokens"] = int(pe1t)
        if pe2:
            m["video_prompt_enhancer_instructions"] = pe2
        if pe2t:
            m["video_prompt_enhancer_max_tokens"] = int(pe2t)
        if pe3:
            m["image_prompt_enhancer_instructions"] = pe3
        if pe3t:
            m["image_prompt_enhancer_max_tokens"] = int(pe3t)

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
    Returns empty dict if not found."""
    for base in ["defaults", FINETUNES_DIR]:
        p = Path(base) / f"{model_id}.json"
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
        return Path("loras") / (arch or "wan")


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
    mid = model_id.strip()
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


def _validate_entry(value: str, plugin=None) -> tuple[bool, str]:
    """Validate a single URL, model reference (=id), or local path.
    Returns (is_valid, message)."""
    if not value or not isinstance(value, str):
        return True, ""
    v = value.strip()
    if not v:
        return True, ""
    if v.startswith("="):
        return _validate_model_ref(v[1:], plugin)
    if v.startswith(("http:", "https:")):
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


def _validate_urls(urls: list[str], plugin=None) -> list[tuple[str, bool, str]]:
    """Validate a list of URLs. Returns [(url, valid, message), ...].
    Skips non-URL entries."""
    results = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        valid, msg = _validate_entry(u, plugin)
        if msg or not valid:
            results.append((u, valid, msg))
    return results


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate that a URL is reachable via HEAD request.

    The ``|subfolder`` / ``|%lora_dir`` suffix is a recipe instruction, not
    part of the URL, so it is stripped before the request.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return True, ""
    is_remote, clean_url, subfolder, _name = fm_assets.parse_source(url.strip())
    if not is_remote or not clean_url:
        return True, "Not a remote URL (local path)"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FinetuneManager/3.0)"}
        if CIVITAI_API_KEY and fm_assets.is_civitai_url(clean_url):
            headers.update(_civitai_api_headers())
        req = urllib.request.Request(clean_url, method="HEAD", headers=headers)
        # Some servers reject HEAD on /resolve/main/ - try GET as fallback
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                code = resp.getcode()
        except urllib.error.HTTPError as e:
            code = e.code
            if code in (405, 501, 403):
                get_req = urllib.request.Request(clean_url, method="GET", headers=headers)
                get_req.add_header("Range", "bytes=0-0")
                try:
                    with urllib.request.urlopen(get_req, timeout=15) as get_resp:
                        code = get_resp.getcode()
                except urllib.error.HTTPError as e2:
                    code = e2.code
        suffix = f" (saves to {subfolder})" if subfolder else ""
        if code and code < 400:
            return True, f"HTTP {code}{suffix}"
        elif code and code == 404:
            return False, "HTTP 404 - Not Found"
        elif code and code == 401:
            return False, "HTTP 401 - Unauthorized (may need token)"
        elif code and code == 403:
            return False, "HTTP 403 - Forbidden (may need token)"
        else:
            return False, f"HTTP {code}"
    except urllib.error.URLError as e:
        return False, f"Connection error: {e.reason}"
    except Exception as e:
        return False, f"Error: {e}"



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
    from shared.utils.download import download_file

    dest = Path(asset.destination)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Wan2GP's downloader cannot send an Authorization header; CivitAI's
        # documented query-token form is attached only for CivitAI hosts.
        download_file(_civitai_with_token(asset.url), str(dest))
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


def _civitai_with_token(url: str) -> str:
    """Attach the configured CivitAI API key to a download URL. Wan2GP fetches the
    URL itself and cannot send an Authorization header, so the query-parameter form
    (documented by CivitAI for download tools) is the only option here. The key is
    only ever attached to CivitAI hosts."""
    if not CIVITAI_API_KEY or not url:
        return url
    parts = urllib.parse.urlsplit(url)
    if parts.hostname not in CIVITAI_SITES:
        return url
    if "token" in urllib.parse.parse_qs(parts.query):
        return url
    query = parts.query + "&" if parts.query else ""
    query += "token=" + urllib.parse.quote(CIVITAI_API_KEY, safe="")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, query, parts.fragment)
    )


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
            req = urllib.request.Request(url, headers=_civitai_api_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
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
            return _fetch(filters + [("query", query), ("limit", str(limit))])
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

    lookups = [lambda: _paged(filters + [("sort", "Most Downloaded"), ("limit", "100")], 2, _matches)]
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
        name = html.escape(model.get("name", "?") or "?")
        mtype = html.escape(model.get("type", "?") or "?")
        desc = html.escape((model.get("description", "") or "")[:200])
        nsfw = model.get("nsfw", False)
        versions = model.get("modelVersions", [])

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
                vname = html.escape(v.get("name", "?") or "?")
                base = html.escape(v.get("baseModel", "") or "")
                files = v.get("files", [])
                primary = next(
                    (f for f in files if f.get("primary", False)),
                    files[0] if files else None,
                )
                fname = html.escape(primary.get("name", "") if primary else "")
                fsize = primary.get("sizeKB", 0) if primary else 0
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


def _civitai_extract_fill_data(civitai_json_str: str) -> tuple:
    """Extract download URL from a CivitAI model/version selection.
    Only fills the URL field (main checkpoints or loras depending on type).
    All other fields are left unchanged via gr.update().
    Returns a tuple matching the ALL_INPUTS order (minus the ID at index 0),
    plus a status message.
    """
    if not civitai_json_str or civitai_json_str == "{}":
        return tuple([gr.update()] * 33 + [""])

    try:
        sel = json.loads(civitai_json_str)
    except (json.JSONDecodeError, TypeError):
        return tuple([gr.update()] * 33 + [""])

    model = sel.get("model", {})
    version = sel.get("version", {})
    site = _civitai_site(sel.get("site") or CIVITAI_SITE_DEFAULT)
    model_name = model.get("name", "?") or "?"

    files = version.get("files", [])
    primary = next(
        (f for f in files if f.get("primary", False)), files[0] if files else None
    )
    dl_url = (
        primary.get("downloadUrl", version.get("downloadUrl", ""))
        if primary
        else version.get("downloadUrl", "")
    )
    if not dl_url:
        dl_url = _civitai_download_url(site, version.get("id", ""))
    dl_url = _civitai_with_token(dl_url)
    keyed = " (API key attached)" if CIVITAI_API_KEY and "token=" in dl_url else ""

    mtype = model.get("type", "").upper()
    # Put URL in main checkpoints for non-LoRA, loras field for LoRA
    if mtype in ("LORA", "LOCON", "DORA"):
        # URL goes to loras field (index 12 in return tuple = ALL_INPUTS[13])
        result = [gr.update()] * 33
        result[12] = gr.update(value=dl_url)
        status = f"LoRA URL from {model_name}{keyed}"
    else:
        # URL goes to main checkpoints field (index 4 in return tuple = ALL_INPUTS[5])
        result = [gr.update()] * 33
        result[4] = gr.update(value=dl_url)
        status = f"Download URL from {model_name}{keyed}"

    return tuple(result + [status])


# ═══════════════════════════════════════════════════════════════════
# ponytail: inline helpers
# ═══════════════════════════════════════════════════════════════════


def _check_upload_token() -> bool:
    """Check if we have a valid HF token for registry uploads.
    Returns True if either config has a real token or the HF hub cache has one."""
    from huggingface_hub import HfApi
    if _REGISTRY_TOKEN_VALID:
        return True
    try:
        api = HfApi()
        user = api.whoami()
        return bool(user and user.get("name"))
    except Exception:
        return False


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


def _publish_family_index(api, plugin=None) -> None:
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
            repo_id=REGISTRY_SPACE,
            repo_type="space",
        )
    except Exception:
        pass


def _hf_upload(fin_id, json_data, plugin=None):
    """Upload a finetune JSON to the HF Space.
    Tries direct commit first; falls back to PR for community submissions."""
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
            repo_id=REGISTRY_SPACE,
            repo_type="space",
        )
        direct_ok = True
    except Exception:
        # Fallback: create a PR (works for community contributors)
        api.upload_file(
            path_or_fileobj=blob,
            path_in_repo=f"finetunes/{safe_id}.json",
            repo_id=REGISTRY_SPACE,
            repo_type="space",
            create_pr=True,
        )
    if direct_ok:
        _publish_family_index(api, plugin)


def _fetch_registry_json(fin_id):
    r = requests.get(f"{DEFAULT_REGISTRY}/finetunes/{fin_id}.json", timeout=10)
    r.raise_for_status()
    return r.json()


def _fetch_dynamic_registry_no_cache() -> list[dict]:
    """Dynamically list all finetunes from the HF Space by scanning
    actual files. No index.json needed -- always in sync."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if _REGISTRY_TOKEN_VALID:
            api = HfApi(token=REGISTRY_TOKEN)
        files = api.list_repo_files(repo_id=REGISTRY_SPACE, repo_type="space")
        fin_files = [
            f for f in files if f.startswith("finetunes/") and f.endswith(".json")
        ]
    except Exception:
        return []
    fins = []
    for path in sorted(fin_files):
        fid = path[len("finetunes/") : -len(".json")]
        try:
            r = requests.get(f"{DEFAULT_REGISTRY}/finetunes/{fid}.json", timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            m = data.get("model", {})
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
                "raw": _clean_utf8(data),
            }
            fins.append(entry)
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
    return cleaned if cleaned else "unnamed"


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
        p = (base / f"{stem}.json").resolve()
        try:
            p.relative_to(base)
        except ValueError:
            continue
        if p.is_file():
            return p
    # Return the exact path when nothing exists so callers can create it.
    exact = (base / f"{str(fin_id).strip()}.json").resolve()
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


def _pack_lora_root(plugin) -> Path:
    getter = getattr(plugin, "get_lora_root", None) if plugin is not None else None
    if callable(getter):
        try:
            root = getter()
            if root:
                return Path(root)
        except Exception:
            pass
    return Path("loras")


def _pack_md_cell(value) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _pack_status_md(manifest, ckpts_dir, lora_dir_for, finetunes_dir, state, custom=None) -> str:
    st = fm_pack.catalog_status(
        manifest, ckpts_dir, lora_dir_for, finetunes_dir, state, custom=custom
    )
    loras = st["loras"]
    fins = st["finetunes"]
    civ = st["civitai"]
    l_ok = sum(1 for x in loras if x.get("present"))
    l_bytes = sum(int(x.get("bytes") or 0) for x in loras)
    l_exp = sum(int(x.get("expected") or 0) for x in loras)
    f_ok = sum(1 for x in fins if x.get("weightsPresent"))
    f_rec = sum(1 for x in fins if x.get("recipeWritten"))
    c_ok = sum(1 for x in civ if x.get("installed"))
    lines = [
        f"**LoRAs:** {l_ok}/{len(loras)} installed - "
        f"{fm_pack.human_bytes(l_bytes)} / {fm_pack.human_bytes(l_exp)}",
        f"**Checkpoints:** {f_rec}/{len(fins)} recipes written - "
        f"{f_ok}/{len(fins)} weight sets present",
        f"**CivitAI:** {c_ok}/{len(civ)} installed",
    ]
    for group in manifest.get("groups", []):
        gid = str(group.get("id") or "")
        rows = [x for x in loras if x.get("group") == gid]
        if rows:
            ok = sum(1 for x in rows if x.get("present"))
            lines.append(f"- {group.get('title') or gid}: {ok}/{len(rows)}")
    own = [x for x in loras + fins if x.get("custom")]
    if own:
        ok = sum(1 for x in own if x.get("present") or x.get("weightsPresent"))
        lines.append(f"- Added by you: {ok}/{len(own)}")
    return "\n".join(lines)


def _pack_details_md(manifest, ckpts_dir, lora_dir_for, finetunes_dir, state, custom=None) -> str:
    st = fm_pack.catalog_status(
        manifest, ckpts_dir, lora_dir_for, finetunes_dir, state, custom=custom
    )
    lines = [
        "#### Checkpoints and recipes",
        "| Name | Architecture | Recipe | Weights |",
        "|---|---|---|---|",
    ]
    for row in st["finetunes"]:
        assets = row.get("assets") or []
        present = sum(1 for a in assets if a.get("present"))
        lines.append(
            "| {name} | {arch} | {recipe} | {present}/{total} |".format(
                name=_pack_md_cell(row.get("name")),
                arch=_pack_md_cell(row.get("architecture")),
                recipe="yes" if row.get("recipeWritten") else "no",
                present=present,
                total=len(assets),
            )
        )
    lines += [
        "",
        "#### CivitAI",
        "| Name | Installed | Files |",
        "|---|---|---|",
    ]
    for row in st["civitai"]:
        lines.append(
            "| {name} | {installed} | {present}/{total} |".format(
                name=_pack_md_cell(row.get("name")),
                installed="yes" if row.get("installed") else "no",
                present=row.get("present", 0),
                total=row.get("expected", 0),
            )
        )
    return "\n".join(lines)


def _pack_inventory_md(plugin) -> str:
    root = _pack_lora_root(plugin)
    inv = fm_pack.lora_inventory(root)
    total = sum(int(x.get("bytes") or 0) for x in inv)
    lines = [
        f"**{len(inv)} LoRA weight files** in `{root}` - "
        f"{fm_pack.human_bytes(total)}"
    ]
    if inv:
        lines.append("")
        for item in inv[:120]:
            mark = "pack" if item.get("known") else ""
            lines.append(
                f"- `{item.get('rel')}` - {fm_pack.human_bytes(item.get('bytes') or 0)}"
                + (f" ({mark})" if mark else "")
            )
        if len(inv) > 120:
            lines.append(f"- ... and {len(inv) - 120} more")
    return "\n".join(lines)


def _fmt_update_report(report: dict) -> str:
    lines = ["#### Update check"]
    for section, title in (
        ("loras", "LoRAs"),
        ("finetunes", "Checkpoints"),
        ("civitai", "CivitAI"),
    ):
        rows = report.get(section) or []
        bad = [r for r in rows if r.get("status") not in ("ok",)]
        lines.append(f"- **{title}:** {len(rows) - len(bad)}/{len(rows)} ok")
        for row in bad[:40]:
            lines.append(f"  - `{row.get('id')}` - {row.get('status')}")
        if len(bad) > 40:
            lines.append(f"  - ... and {len(bad) - 40} more")
    return "\n".join(lines)


def _fmt_live_report(report: dict) -> str:
    items = report.get("items") or []
    changed = [x for x in items if x.get("status") in ("changed", "missing")]
    lines = [
        f"#### Live check - {report.get('checkedAt') or ''}",
        f"- **{len(items) - len(changed)}/{len(items)}** current; "
        f"**{len(changed)}** need attention.",
    ]
    for item in changed[:60]:
        lines.append(
            f"  - `{item.get('id')}` ({item.get('kind')}) - {item.get('status')}"
        )
    if len(changed) > 60:
        lines.append(f"  - ... and {len(changed) - 60} more")
    unknown = [x for x in items if x.get("status") == "unknown"]
    if unknown:
        lines.append(
            f"- {len(unknown)} could not be compared (source unreachable, or the "
            "file is not in the repo tree)."
        )
    for err in (report.get("failed") or [])[:10]:
        lines.append(f"  - ! {err}")
    if not items:
        lines.append("- Nothing in the catalog to check.")
    return "\n".join(lines)


def _pack_media_html(urls, title="") -> str:
    items = []
    for url in urls or []:
        u = str(url or "").strip()
        if not u.startswith(("https://", "http://")):
            continue
        safe = html.escape(u, quote=True)
        if u.lower().endswith((".mp4", ".webm", ".mov")):
            items.append(
                f"<video src=\"{safe}\" controls muted loop "
                "style=\"max-width:240px;max-height:180px;border-radius:8px\"></video>"
            )
        else:
            items.append(
                f"<img src=\"{safe}\" loading=\"lazy\" alt=\"{html.escape(title)}\" "
                "style=\"max-width:240px;max-height:180px;object-fit:cover;border-radius:8px\">"
            )
    if not items:
        return ""
    return (
        "<div style=\"display:flex;flex-wrap:wrap;gap:8px;padding:4px 0\">"
        + "".join(items)
        + "</div>"
    )


def _pack_catalog_rows(st: dict, catalog: dict) -> list[dict]:
    """Flatten catalog status into one row per selectable item."""
    entries = {str(x.get("id")): x for x in catalog.get("loras", [])}
    entries.update({str(x.get("id")): x for x in catalog.get("finetunes", [])})
    entries.update({str(x.get("id")): x for x in catalog.get("civitai", [])})
    rows: list[dict] = []
    for row in st.get("loras") or []:
        entry = entries.get(str(row.get("id"))) or {}
        state = "missing"
        if row.get("present"):
            state = "stale" if row.get("stale") else "installed"
        rows.append(
            {
                "kind": "LoRA",
                "id": str(row.get("id")),
                "pick": f"lora:{row.get('id')}",
                "name": row.get("name") or "",
                "family": row.get("family") or entry.get("family") or "",
                "arch": "",
                "bytes": int(row.get("expected") or row.get("bytes") or 0),
                "state": state,
                "custom": bool(entry.get("custom")),
                "entry": entry,
            }
        )
    for row in st.get("finetunes") or []:
        entry = entries.get(str(row.get("id"))) or {}
        assets = row.get("assets") or []
        present = sum(1 for a in assets if a.get("present"))
        if present == len(assets) and row.get("recipeWritten"):
            state = "installed"
        elif present == len(assets) and assets:
            state = "no recipe"
        elif present:
            state = "partial"
        else:
            state = "missing"
        if any(a.get("stale") for a in assets):
            state = "stale"
        rows.append(
            {
                "kind": "Checkpoint",
                "id": str(row.get("id")),
                "pick": f"finetune:{row.get('id')}",
                "name": row.get("name") or "",
                "family": row.get("family") or entry.get("family") or "",
                "arch": row.get("architecture") or "",
                "bytes": int(entry.get("bytes") or 0) + int(entry.get("bytesLow") or 0),
                "state": state,
                "custom": bool(entry.get("custom")),
                "entry": entry,
            }
        )
    for row in st.get("civitai") or []:
        entry = entries.get(str(row.get("id"))) or {}
        state = "missing"
        if row.get("installed"):
            state = "installed"
        elif row.get("present"):
            state = "partial"
        rows.append(
            {
                "kind": "CivitAI",
                "id": str(row.get("id")),
                "pick": f"civitai:{row.get('id')}",
                "name": row.get("name") or "",
                "family": entry.get("family") or "",
                "arch": entry.get("architecture") or "",
                "bytes": int(entry.get("bytesEstimate") or 0),
                "state": state,
                "custom": bool(entry.get("custom")),
                "entry": entry,
            }
        )
    return rows


def _filter_catalog_rows(rows: list[dict], query: str, family: str, show: str) -> list[dict]:
    q = str(query or "").strip().lower()
    family = str(family or "all")
    show = str(show or "all")
    out = []
    for row in rows:
        haystack = " ".join(
            [str(row.get("name") or ""), str(row.get("id") or ""), str(row.get("arch") or "")]
        ).lower()
        if q and q not in haystack:
            continue
        if family != "all" and str(row.get("family") or "") != family:
            continue
        state = str(row.get("state") or "")
        if show == "missing" and state == "installed":
            continue
        if show == "installed" and state != "installed":
            continue
        if show == "custom" and not row.get("custom"):
            continue
        out.append(row)
    return out


def _pack_table_value(rows: list[dict]) -> list[list[str]]:
    return [
        [
            str(r.get("kind") or ""),
            str(r.get("name") or ""),
            str(r.get("family") or ""),
            str(r.get("arch") or ""),
            fm_pack.human_bytes(r.get("bytes") or 0),
            str(r.get("state") or ""),
            "yes" if r.get("custom") else "",
        ]
        for r in rows
    ]


def _pack_pick_choices(rows: list[dict]) -> list[tuple[str, str]]:
    return [
        (
            f"{r.get('name')} - {r.get('kind')} - {r.get('state')}"
            + (" - added by you" if r.get("custom") else ""),
            str(r.get("pick")),
        )
        for r in rows
    ]


def _pack_detail_md(row: dict | None) -> str:
    if not row:
        return "Select an item above."
    entry = row.get("entry") or {}
    lines = [
        f"#### {row.get('name')}",
        f"- **Kind:** {row.get('kind')}",
        f"- **Family / architecture:** {row.get('family') or '?'} / {row.get('arch') or '?'}",
        f"- **Size:** {fm_pack.human_bytes(row.get('bytes') or 0)}",
        f"- **State:** {row.get('state')}",
    ]
    page = str(entry.get("repoPage") or "")
    if page.startswith("https://"):
        lines.append(f"- **Source:** [{page}]({page})")
    file_page = str(entry.get("filePage") or "")
    if file_page.startswith("https://"):
        lines.append(f"- **File:** [{entry.get('file')}]({file_page})")
    if entry.get("licenseNote"):
        lines.append(f"- **Licence:** {entry.get('licenseNote')}")
    assets = entry.get("assets") if isinstance(entry.get("assets"), list) else None
    if assets:
        lines.append("")
        lines.append("**Files**")
        for asset in assets:
            status = "present" if asset.get("present") else "missing"
            lines.append(f"- `{asset.get('path') or asset.get('source')}` - {status}")
    return "\n".join(lines)


def _build_pack_tab(plugin) -> None:
    """Add the curated adult-pack catalog as a native Gradio tab."""
    manifest = fm_pack.load_manifest()
    state = fm_pack.load_state(_pack_state_path, manifest)

    def _catalog() -> dict:
        return fm_pack.merged_catalog(manifest, fm_pack.load_custom())

    ckpts_dir = lambda: _ckpts_dir()
    lora_dir_for = lambda arch: _lora_dir_for(arch, plugin)
    finetunes_dir = Path(FINETUNES_DIR)
    active = bool(state.get("enabled") and state.get("ageConfirmed"))
    civ_active = bool(active and state.get("civitaiEnabled"))

    gr.Markdown(
        "### Adult Pack\n"
        "Curated checkpoints and LoRAs that Wan2GP can load. Weights stay under "
        "`adult-pack` folders; checkpoints are written as "
        "`finetunes/adultpack_*.json` recipes. Adult content is optional and "
        "off by default."
    )
    with gr.Row():
        pack_enabled = gr.Checkbox(
            label="Enable the adult pack", value=bool(state.get("enabled"))
        )
        pack_age = gr.Checkbox(
            label="I am 18 or older and it is legal where I live",
            value=bool(state.get("ageConfirmed")),
        )
    pack_gate_msg = gr.Markdown(
        "**Adult pack enabled.** Nothing downloads until you press a download button."
        if active
        else "The adult pack is off. Tick both boxes to show the catalog."
    )

    def _state_from(enabled, age, civ_enabled, lora_sel, ft_sel, civ_sel) -> None:
        state["enabled"] = bool(enabled)
        state["ageConfirmed"] = bool(age)
        state["civitaiEnabled"] = bool(civ_enabled)
        state["selections"]["loras"] = [str(x) for x in (lora_sel or [])]
        state["selections"]["finetunes"] = [str(x) for x in (ft_sel or [])]
        state["selections"]["civitai"] = [str(x) for x in (civ_sel or [])]
        fm_pack.save_state(state, _pack_state_path)

    with gr.Column(visible=active) as pack_body:
        with gr.Row():
            pack_refresh = gr.Button("Refresh status")
            pack_defaults = gr.Button("Select recommended")
            pack_download = gr.Button("Download selected", variant="primary")
            pack_recipes = gr.Button("Write selected recipes")
            pack_check_updates = gr.Button("Check updates")
            pack_verify_hashes = gr.Checkbox(label="Verify hashes", value=False, scale=0)
        catalog = _catalog()
        pack_status = gr.Markdown(
            _pack_status_md(
                manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=catalog
            )
            if active
            else ""
        )

        lora_choices = [
            (
                f"{entry.get('name')} - {fm_pack.human_bytes(entry.get('bytes') or 0)}"
                + (f" - {entry.get('pair')} half" if entry.get("pair") else ""),
                str(entry.get("id")),
            )
            for entry in catalog.get("loras", [])
        ]
        fin_choices = [
            (
                f"{entry.get('name')} - {entry.get('architecture')} - "
                f"{fm_pack.human_bytes(int(entry.get('bytes') or 0) + int(entry.get('bytesLow') or 0))}",
                str(entry.get("id")),
            )
            for entry in catalog.get("finetunes", [])
        ]
        civ_choices = [
            (
                f"{entry.get('name')} - {entry.get('architecture') or entry.get('family')}",
                str(entry.get("id")),
            )
            for entry in catalog.get("civitai", [])
        ]
        lora_ids = set(str(x) for x in state["selections"].get("loras", []))
        ft_ids = set(str(x) for x in state["selections"].get("finetunes", []))
        civ_ids = set(str(x) for x in state["selections"].get("civitai", []))

        def _pack_arch_choices() -> list[tuple[str, str]]:
            rows = _architecture_choices(plugin)
            if rows:
                return rows
            return [
                (
                    f"{entry.get('label') or entry.get('id')} \u00b7 "
                    f"{entry.get('architecture')}",
                    str(entry.get("architecture")),
                )
                for entry in manifest.get("archMap", [])
                if entry.get("architecture")
            ]

        initial_rows = _pack_catalog_rows(
            fm_pack.catalog_status(
                manifest,
                ckpts_dir(),
                lora_dir_for,
                finetunes_dir,
                state,
                custom=catalog,
            ),
            catalog,
        )

        # A dropdown rejects values that are not in its choices, so the
        # handlers below must only ever echo an architecture that is listed.
        arch_values = {v for _, v in _pack_arch_choices()}

        with gr.Tabs():
            with gr.Tab("Catalog"):
                gr.Markdown(
                    "Every curated and user-added item with its live state. "
                    "Filter the list, pick an item, then act on just that one."
                )
                with gr.Row():
                    cat_q = gr.Textbox(label="Filter", scale=3, container=False)
                    cat_family = gr.Dropdown(
                        label="Family",
                        choices=[("All families", "all")]
                        + sorted(
                            {
                                (str(x.get("family")), str(x.get("family")))
                                for x in catalog.get("loras", [])
                                + catalog.get("finetunes", [])
                                + catalog.get("civitai", [])
                                if x.get("family")
                            }
                        ),
                        value="all",
                        scale=1,
                    )
                    cat_show = gr.Dropdown(
                        label="Show",
                        choices=["all", "missing", "installed", "custom"],
                        value="all",
                        scale=1,
                    )
                cat_table = gr.Dataframe(
                    headers=[
                        "Kind", "Name", "Family", "Architecture",
                        "Size", "State", "Added by you",
                    ],
                    value=_pack_table_value(initial_rows),
                    interactive=False,
                    wrap=True,
                )
                cat_pick = gr.Dropdown(label="Item", choices=[], filterable=True)
                with gr.Row():
                    cat_download = gr.Button("Download this", variant="primary")
                    cat_recipe = gr.Button("Write recipe")
                    cat_gallery = gr.Button("Load gallery")
                    cat_delete = gr.Button("Delete this", variant="stop")
                cat_detail = gr.Markdown("Select an item above.")
                cat_media = gr.HTML("")
                cat_log = gr.Textbox(label="Result", interactive=False)

            with gr.Tab("Find & Add"):
                gr.Markdown(
                    "Search HuggingFace or CivitAI, inspect the files inside, and "
                    "add one to your own list. Added items behave exactly like the "
                    "curated ones: same download, recipe and delete paths."
                )
                with gr.Row():
                    find_source = gr.Dropdown(
                        label="Source",
                        choices=[("HuggingFace", "hf"), ("CivitAI", "civitai")],
                        value="hf",
                        scale=1,
                    )
                    find_mode = gr.Dropdown(
                        label="Mode",
                        choices=[("Search", "search"), ("Browse", "browse")],
                        value="search",
                        scale=1,
                    )
                    find_query = gr.Textbox(label="Query", scale=3, container=False)
                    find_kind = gr.Dropdown(
                        label="Type",
                        choices=["any", "lora", "checkpoint", "gguf"],
                        value="any",
                        scale=1,
                    )
                    find_sort = gr.Dropdown(
                        label="Sort",
                        choices=["trending", "downloads", "likes", "newest"],
                        value="trending",
                        scale=1,
                    )
                    find_period = gr.Dropdown(
                        label="Period",
                        choices=["week", "month", "year"],
                        value="week",
                        scale=1,
                    )
                    find_limit = gr.Slider(5, 50, 24, step=1, label="Limit", scale=1)
                find_run = gr.Button("Search / Browse", variant="primary")
                find_table = gr.Dataframe(
                    headers=[
                        "Title", "Family", "Architecture",
                        "Downloads", "Updated", "Id",
                    ],
                    value=[],
                    interactive=False,
                    wrap=True,
                )
                find_pick = gr.Dropdown(label="Result", choices=[], filterable=True)
                with gr.Row():
                    find_inspect = gr.Button("Inspect")
                    find_version = gr.Dropdown(
                        label="CivitAI version", choices=[], scale=2
                    )
                    find_arch = gr.Dropdown(
                        label="Wan2GP architecture",
                        choices=_pack_arch_choices(),
                        filterable=True,
                        scale=2,
                    )
                find_files = gr.Dataframe(
                    headers=["File", "Kind", "Size", "Half", "SHA-256"],
                    value=[],
                    interactive=False,
                    wrap=True,
                )
                find_file_pick = gr.Dropdown(label="File", choices=[], filterable=True)
                with gr.Row():
                    find_kind_override = gr.Dropdown(
                        label="Add as",
                        choices=["auto", "lora", "checkpoint"],
                        value="auto",
                        scale=1,
                    )
                    find_add = gr.Button("Add selected file", variant="primary")
                find_media = gr.HTML("")
                find_status = gr.Textbox(label="Status", interactive=False)
                with gr.Accordion("Add by link", open=False):
                    gr.Markdown(
                        "Paste a HuggingFace repo (`owner/name` or its URL), a "
                        "CivitAI model URL, or a CivitAI model id."
                    )
                    with gr.Row():
                        find_url = gr.Textbox(label="Repo / URL / id", scale=3)
                        find_url_arch = gr.Dropdown(
                            label="Wan2GP architecture",
                            choices=_pack_arch_choices(),
                            filterable=True,
                            scale=2,
                        )
                    find_url_btn = gr.Button("Resolve and inspect")

            with gr.Tab("LoRAs"):
                gr.Markdown(
                    "Select LoRAs to download. HIGH/LOW pairs are selected "
                    "together automatically. They land in the collection "
                    "Wan2GP scans, under `adult-pack/`."
                )
                pack_lora_sel = gr.Dropdown(
                    label="LoRAs",
                    choices=lora_choices,
                    value=[x for x in lora_ids if x in {v for _, v in lora_choices}],
                    multiselect=True,
                    filterable=True,
                )
                with gr.Row():
                    pack_inv_btn = gr.Button("Refresh LoRA inventory")
                    pack_delete_loras = gr.Button("Delete selected LoRAs")
                pack_inv = gr.Markdown("")
            with gr.Tab("Checkpoints and recipes"):
                gr.Markdown(
                    "Checkpoint entries become `adultpack_*.json` recipes. "
                    "Downloading the weights and writing the recipe are separate "
                    "actions, so you can inspect a recipe before fetching tens of GB."
                )
                pack_ft_sel = gr.Dropdown(
                    label="Checkpoints / recipes",
                    choices=fin_choices,
                    value=[x for x in ft_ids if x in {v for _, v in fin_choices}],
                    multiselect=True,
                    filterable=True,
                )
                pack_details = gr.Markdown(
                    _pack_details_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog())
                    if active
                    else ""
                )
                pack_delete_fins = gr.Button("Delete selected recipes/weights")
            with gr.Tab("CivitAI"):
                token_state = (
                    "configured"
                    if CIVITAI_API_KEY
                    else "not configured - public files still download"
                )
                gr.Markdown(
                    "CivitAI is a separate opt-in. It uses `civitai.red` for adult "
                    f"content. API token: **{token_state}**. Add `civitai_api_key` "
                    "to the plugin's `config.json` for gated releases."
                )
                pack_civ_enabled = gr.Checkbox(
                    label="Enable CivitAI downloads",
                    value=bool(state.get("civitaiEnabled")),
                )
                with gr.Column(visible=civ_active) as pack_civ_body:
                    pack_civ_sel = gr.Dropdown(
                        label="CivitAI releases",
                        choices=civ_choices,
                        value=[x for x in civ_ids if x in {v for _, v in civ_choices}],
                        multiselect=True,
                        filterable=True,
                    )
                    pack_civ_download = gr.Button(
                        "Download CivitAI selected", variant="primary"
                    )
                    pack_delete_civ = gr.Button("Delete selected CivitAI")
                    pack_civ_status = gr.Markdown("")
            with gr.Tab("Updates"):
                gr.Markdown(
                    "Compare local files with the catalog, then with what the "
                    "sources serve today. **Check online** reads each repo's live "
                    "file size and hash; **Apply selected** pins them; "
                    "**Download selected** fetches the new bytes."
                )
                with gr.Row():
                    pack_live_updates = gr.Button("Check online", variant="primary")
                    pack_apply_updates = gr.Button("Apply selected")
                    pack_download_updates = gr.Button("Download selected")
                update_sel = gr.Dropdown(
                    label="Update items",
                    choices=[],
                    multiselect=True,
                    filterable=True,
                )
                pack_update = gr.Markdown("")
        pack_log = gr.Textbox(label="Pack log", lines=10, interactive=False)

        all_inputs = [
            pack_enabled,
            pack_age,
            pack_civ_enabled,
            pack_lora_sel,
            pack_ft_sel,
            pack_civ_sel,
        ]

        def _gate(enabled, age, civ_enabled, lora_sel, ft_sel, civ_sel):
            _state_from(enabled, age, civ_enabled, lora_sel, ft_sel, civ_sel)
            on = bool(enabled and age)
            return (
                gr.update(visible=on),
                gr.update(visible=bool(on and civ_enabled)),
                "**Adult pack enabled.** Nothing downloads until you press a download button."
                if on
                else "The adult pack is off. Tick both boxes to show the catalog.",
                _pack_status_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog())
                if on
                else "",
            )

        def _save(enabled, age, civ_enabled, lora_sel, ft_sel, civ_sel):
            _state_from(enabled, age, civ_enabled, lora_sel, ft_sel, civ_sel)

        pack_enabled.change(_gate, inputs=all_inputs, outputs=[pack_body, pack_civ_body, pack_gate_msg, pack_status])
        pack_age.change(_gate, inputs=all_inputs, outputs=[pack_body, pack_civ_body, pack_gate_msg, pack_status])
        pack_civ_enabled.change(_gate, inputs=all_inputs, outputs=[pack_body, pack_civ_body, pack_gate_msg, pack_status])
        pack_lora_sel.change(_save, inputs=all_inputs, outputs=[])
        pack_ft_sel.change(_save, inputs=all_inputs, outputs=[])
        pack_civ_sel.change(_save, inputs=all_inputs, outputs=[])

        def _defaults():
            loras = [str(x.get("id")) for x in manifest.get("loras", []) if x.get("defaultSelected")]
            loras = fm_pack.selected_ids_with_pairs(manifest, loras)
            fins = [str(x.get("id")) for x in manifest.get("finetunes", []) if x.get("defaultSelected")]
            civs = [str(x.get("id")) for x in manifest.get("civitai", []) if x.get("defaultSelected")]
            _state_from(
                state.get("enabled"),
                state.get("ageConfirmed"),
                state.get("civitaiEnabled"),
                loras,
                fins,
                civs,
            )
            return (
                gr.update(value=loras),
                gr.update(value=fins),
                gr.update(value=civs),
                "Recommended selection loaded. Review it, then download.",
            )

        pack_defaults.click(
            _defaults,
            inputs=[],
            outputs=[pack_lora_sel, pack_ft_sel, pack_civ_sel, pack_log],
        )

        def _refresh():
            return (
                _pack_status_md(
                    manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state,
                    custom=_catalog(),
                ),
                _pack_details_md(
                    manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state,
                    custom=_catalog(),
                ),
                _pack_inventory_md(plugin),
                *_cat_snapshot("", "all", "all", ""),
            )

        pack_refresh.click(
            _refresh,
            inputs=[],
            outputs=[
                pack_status, pack_details, pack_inv,
                cat_table, cat_pick, cat_detail, cat_media,
            ],
        )
        pack_inv_btn.click(
            _refresh,
            inputs=[],
            outputs=[
                pack_status, pack_details, pack_inv,
                cat_table, cat_pick, cat_detail, cat_media,
            ],
        )

        def _download(enabled, age, civ_enabled, lora_sel, ft_sel, civ_sel, progress=gr.Progress()):
            if not (enabled and age):
                return "Enable the 18+ gate first.", "", "", ""
            _state_from(enabled, age, civ_enabled, lora_sel, ft_sel, civ_sel)
            lora_ids = fm_pack.selected_ids_with_pairs(_catalog(), lora_sel or [])
            ft_ids = [str(x) for x in (ft_sel or [])]
            civ_ids = [str(x) for x in (civ_sel or [])] if civ_enabled else []
            loras = {str(x.get("id")): x for x in _catalog().get("loras", [])}
            fins = {str(x.get("id")): x for x in _catalog().get("finetunes", [])}
            civs = {str(x.get("id")): x for x in _catalog().get("civitai", [])}
            total = fm_pack.selected_total_bytes(_catalog(), lora_ids, ft_ids)
            for ident in civ_ids:
                total += int((civs.get(ident) or {}).get("bytesEstimate") or 0)
            done = 0
            log: list[str] = []

            def cb(n, _t, label):
                nonlocal done
                progress(
                    (min(done + int(n or 0), total or done + int(n or 0)), total or done + int(n or 0)),
                    desc=label,
                )

            for ident in lora_ids:
                entry = loras.get(ident)
                if not entry:
                    continue
                try:
                    res = fm_pack.fetch_lora(
                        entry,
                        lora_dir_for,
                        token=CIVITAI_API_KEY,
                        progress=cb,
                        pins=state.get("pins"),
                    )
                    done += int(res.get("bytes") or 0)
                    log.append(f"[OK] LoRA {entry.get('name')}: {res.get('status')}")
                except Exception as exc:
                    log.append(f"[FAIL] LoRA {entry.get('name')}: {exc}")
            for ident in ft_ids:
                entry = fins.get(ident)
                if not entry:
                    continue
                results = fm_pack.fetch_finetune(
                    entry,
                    ckpts_dir(),
                    token=CIVITAI_API_KEY,
                    progress=cb,
                    pins=state.get("pins"),
                )
                for res in results:
                    done += int(res.get("bytes") or 0)
                    if res.get("ok") is False or res.get("error"):
                        log.append(f"[FAIL] {entry.get('name')}: {res.get('error')}")
                    else:
                        log.append(f"[OK] {entry.get('name')}: {res.get('status')}")
            for ident in civ_ids:
                spec = civs.get(ident)
                if not spec:
                    continue
                try:
                    res = fm_pack.fetch_civitai(
                        spec,
                        ckpts_dir=ckpts_dir(),
                        lora_dir_for=lora_dir_for,
                        finetunes_dir=finetunes_dir,
                        site="civitai.red",
                        token=CIVITAI_API_KEY,
                        progress=cb,
                        pinned=(state.get("civitaiPinned") or {}).get(ident),
                    )
                    done += int(res.get("bytes") or 0)
                    state.setdefault("civitaiPinned", {})[ident] = res.get("versionId")
                    state.setdefault("civitaiInstalled", {})[ident] = {
                        "files": res.get("paths") or [],
                        "kind": res.get("kind"),
                        "bytes": res.get("bytes") or 0,
                        "versionId": res.get("versionId"),
                    }
                    log.append(f"[OK] CivitAI {spec.get('name')}: {res.get('kind')}")
                except Exception as exc:
                    log.append(f"[FAIL] CivitAI {spec.get('name')}: {exc}")
            fm_pack.save_state(state, _pack_state_path)
            return (
                "\n".join(log) or "Nothing selected.",
                _pack_status_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _pack_details_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                "",
            )

        pack_download.click(
            _download,
            inputs=all_inputs,
            outputs=[pack_log, pack_status, pack_details, pack_civ_status],
            show_progress="full",
        )

        def _recipes(enabled, age, ft_sel):
            if not (enabled and age):
                return "Enable the 18+ gate first.", "", "", ""
            fins = {str(x.get("id")): x for x in _catalog().get("finetunes", [])}
            log: list[str] = []
            for ident in [str(x) for x in (ft_sel or [])]:
                entry = fins.get(ident)
                if not entry:
                    continue
                try:
                    p = fm_pack.write_recipe(entry, finetunes_dir)
                    log.append(f"[OK] wrote {p}")
                except Exception as exc:
                    log.append(f"[FAIL] {entry.get('name')}: {exc}")
            return (
                "\n".join(log) or "Nothing selected.",
                _pack_status_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _pack_details_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                "",
            )

        pack_recipes.click(
            _recipes,
            inputs=[pack_enabled, pack_age, pack_ft_sel],
            outputs=[pack_log, pack_status, pack_details, pack_civ_status],
        )

        def _civ_download(enabled, age, civ_enabled, civ_sel, progress=gr.Progress()):
            if not (enabled and age):
                return "Enable the 18+ gate first.", "", "", ""
            if not civ_enabled:
                return "Enable CivitAI downloads first.", "", "", ""
            civs = {str(x.get("id")): x for x in _catalog().get("civitai", [])}
            log: list[str] = []
            for ident in [str(x) for x in (civ_sel or [])]:
                spec = civs.get(ident)
                if not spec:
                    continue
                try:
                    res = fm_pack.fetch_civitai(
                        spec,
                        ckpts_dir=ckpts_dir(),
                        lora_dir_for=lora_dir_for,
                        finetunes_dir=finetunes_dir,
                        site="civitai.red",
                        token=CIVITAI_API_KEY,
                        progress=lambda n, t, label: progress(
                            (n, t or n), desc=label
                        ),
                        pinned=(state.get("civitaiPinned") or {}).get(ident),
                    )
                    state.setdefault("civitaiPinned", {})[ident] = res.get("versionId")
                    state.setdefault("civitaiInstalled", {})[ident] = {
                        "files": res.get("paths") or [],
                        "kind": res.get("kind"),
                        "bytes": res.get("bytes") or 0,
                        "versionId": res.get("versionId"),
                    }
                    log.append(f"[OK] CivitAI {spec.get('name')}: {res.get('kind')}")
                except Exception as exc:
                    log.append(f"[FAIL] CivitAI {spec.get('name')}: {exc}")
            fm_pack.save_state(state, _pack_state_path)
            return (
                "\n".join(log) or "Nothing selected.",
                _pack_status_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _pack_details_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                "\n".join(log) or "Nothing selected.",
            )

        pack_civ_download.click(
            _civ_download,
            inputs=[pack_enabled, pack_age, pack_civ_enabled, pack_civ_sel],
            outputs=[pack_log, pack_status, pack_details, pack_civ_status],
            show_progress="full",
        )

        def _delete_loras(enabled, age, lora_sel):
            if not (enabled and age):
                return "Enable the 18+ gate first.", "", "", ""
            loras = {str(x.get("id")): x for x in _catalog().get("loras", [])}
            log: list[str] = []
            for ident in fm_pack.selected_ids_with_pairs(_catalog(), lora_sel or []):
                entry = loras.get(ident)
                if not entry:
                    continue
                try:
                    res = fm_pack.delete_lora(entry, lora_dir_for)
                    if res.get("removed"):
                        log.append(f"[OK] deleted {entry.get('name')}")
                    else:
                        log.append(f"[SKIP] {entry.get('name')}: not present")
                except Exception as exc:
                    log.append(f"[FAIL] {entry.get('name')}: {exc}")
            fm_pack.save_state(state, _pack_state_path)
            return (
                "\n".join(log) or "Nothing selected.",
                _pack_status_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _pack_details_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _fmt_update_report(
                    fm_pack.update_report(
                        manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state
                    )
                ),
            )

        pack_delete_loras.click(
            _delete_loras,
            inputs=[pack_enabled, pack_age, pack_lora_sel],
            outputs=[pack_log, pack_status, pack_details, pack_update],
        )

        def _delete_fins(enabled, age, ft_sel):
            if not (enabled and age):
                return "Enable the 18+ gate first.", "", "", ""
            fins = {str(x.get("id")): x for x in _catalog().get("finetunes", [])}
            log: list[str] = []
            for ident in [str(x) for x in (ft_sel or [])]:
                entry = fins.get(ident)
                if not entry:
                    continue
                try:
                    res = fm_pack.delete_finetune(entry, ckpts_dir(), finetunes_dir)
                    parts = []
                    if res.get("removedRecipe"):
                        parts.append("recipe")
                    if res.get("removedWeights"):
                        parts.append(f"{len(res['removedWeights'])} weight(s)")
                    log.append(
                        f"[OK] {entry.get('name')}: deleted {', '.join(parts) if parts else 'nothing present'}"
                    )
                except Exception as exc:
                    log.append(f"[FAIL] {entry.get('name')}: {exc}")
            return (
                "\n".join(log) or "Nothing selected.",
                _pack_status_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _pack_details_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _fmt_update_report(
                    fm_pack.update_report(
                        manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state
                    )
                ),
            )

        pack_delete_fins.click(
            _delete_fins,
            inputs=[pack_enabled, pack_age, pack_ft_sel],
            outputs=[pack_log, pack_status, pack_details, pack_update],
        )

        def _delete_civ(enabled, age, civ_enabled, civ_sel):
            if not (enabled and age):
                return "Enable the 18+ gate first.", "", "", ""
            if not civ_enabled:
                return "Enable CivitAI downloads first.", "", "", ""
            civs = {str(x.get("id")): x for x in _catalog().get("civitai", [])}
            log: list[str] = []
            for ident in [str(x) for x in (civ_sel or [])]:
                spec = civs.get(ident)
                if not spec:
                    continue
                try:
                    res = fm_pack.delete_civitai(
                        spec, state, ckpts_dir(), lora_dir_for, finetunes_dir
                    )
                    parts = []
                    if res.get("removedFiles"):
                        parts.append(f"{len(res['removedFiles'])} file(s)")
                    if res.get("removedRecipe"):
                        parts.append("recipe")
                    log.append(
                        f"[OK] {spec.get('name')}: deleted {', '.join(parts) if parts else 'nothing present'}"
                    )
                except Exception as exc:
                    log.append(f"[FAIL] {spec.get('name')}: {exc}")
            fm_pack.save_state(state, _pack_state_path)
            return (
                "\n".join(log) or "Nothing selected.",
                _pack_status_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _pack_details_md(manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state, custom=_catalog()),
                _fmt_update_report(
                    fm_pack.update_report(
                        manifest, ckpts_dir(), lora_dir_for, finetunes_dir, state
                    )
                ),
            )

        pack_delete_civ.click(
            _delete_civ,
            inputs=[pack_enabled, pack_age, pack_civ_enabled, pack_civ_sel],
            outputs=[pack_log, pack_status, pack_details, pack_update],
        )

        def _check_updates(verify):
            report = fm_pack.update_report(
                manifest,
                ckpts_dir(),
                lora_dir_for,
                finetunes_dir,
                state,
                verify_hashes=bool(verify),
                custom=_catalog(),
            )
            return _fmt_update_report(report)

        pack_check_updates.click(
            _check_updates,
            inputs=[pack_verify_hashes],
            outputs=[pack_update],
        )

        # -- Updates: compare with the live sources, pin, download -------
        def _fetch_one(row, progress):
            """Download one catalog row; mirrors the Catalog tab action."""
            entry = row["entry"]
            if row["kind"] == "LoRA":
                res = fm_pack.fetch_lora(
                    entry,
                    lora_dir_for,
                    token=CIVITAI_API_KEY,
                    pins=state.get("pins"),
                    progress=lambda n, t, label: progress((n, t or n), desc=label),
                )
                return f"[OK] {entry.get('name')}: {res.get('status')}"
            if row["kind"] == "Checkpoint":
                results = fm_pack.fetch_finetune(
                    entry,
                    ckpts_dir(),
                    token=CIVITAI_API_KEY,
                    pins=state.get("pins"),
                    progress=lambda n, t, label: progress((n, t or n), desc=label),
                )
                failed = [r for r in results if r.get("ok") is False or r.get("error")]
                if failed:
                    raise RuntimeError(str(failed[0].get("error")))
                return f"[OK] {entry.get('name')}: weights present"
            res = fm_pack.fetch_civitai(
                entry,
                ckpts_dir=ckpts_dir(),
                lora_dir_for=lora_dir_for,
                finetunes_dir=finetunes_dir,
                site="civitai.red",
                token=CIVITAI_API_KEY,
                progress=lambda n, t, label: progress((n, t or n), desc=label),
                pinned=(state.get("civitaiPinned") or {}).get(row["id"]),
            )
            state.setdefault("civitaiPinned", {})[row["id"]] = res.get("versionId")
            state.setdefault("civitaiInstalled", {})[row["id"]] = {
                "files": res.get("paths") or [],
                "kind": res.get("kind"),
                "bytes": res.get("bytes") or 0,
                "versionId": res.get("versionId"),
            }
            return f"[OK] {entry.get('name')}: {res.get('kind')}"

        def _entry_rows():
            rows, _cat = _rows_now()
            return {r["id"]: r for r in rows}

        def _live_updates():
            report = fm_pack.live_update_report(
                _catalog(),
                ckpts_dir(),
                lora_dir_for,
                finetunes_dir,
                state,
                token=CIVITAI_API_KEY,
            )
            changed = [r for r in report.get("items", []) if r.get("status") != "current"]
            choices = [
                (f"{r.get('name')} - {r.get('kind')} - {r.get('status')}", str(r.get("id")))
                for r in changed
            ]
            return _fmt_live_report(report), gr.update(choices=choices, value=[])

        pack_live_updates.click(
            _live_updates,
            inputs=[],
            outputs=[pack_update, update_sel],
        )

        def _apply_updates(ids):
            ids = [str(x) for x in (ids or [])]
            if not ids:
                return "Pick update items first.", *_cat_snapshot("", "all", "all", "")
            applied = fm_pack.apply_updates(state, ids)
            fm_pack.save_state(state, _pack_state_path)
            lines = [
                f"[PIN] {k} -> {v.get('sha256', '')[:12]} ({v.get('bytes')} B)"
                for k, v in sorted((applied.get("pins") or {}).items())
            ]
            lines += [
                f"[PIN] civitai {k} -> v{v}"
                for k, v in sorted((applied.get("civitaiPinned") or {}).items())
            ]
            for skip in applied.get("skipped") or []:
                lines.append(f"[SKIP] {skip.get('id')}: {skip.get('reason')}")
            if not (applied.get("applied") or applied.get("skipped")):
                lines.append("Nothing to pin - run Check online first.")
            return (
                "\n".join(lines),
                *_cat_snapshot("", "all", "all", ""),
            )

        pack_apply_updates.click(
            _apply_updates,
            inputs=[update_sel],
            outputs=[pack_update, cat_table, cat_pick, cat_detail, cat_media],
        )

        def _download_updates(ids, progress=gr.Progress()):
            ids = [str(x) for x in (ids or [])]
            if not ids:
                return "Pick update items first.", *_cat_snapshot("", "all", "all", "")
            rows = _entry_rows()
            log: list[str] = []
            for raw in ids:
                row = rows.get(str(raw).partition("#")[0])
                if not row:
                    log.append(f"[SKIP] {raw}: not in the catalog")
                    continue
                try:
                    log.append(_fetch_one(row, progress))
                except Exception as exc:
                    log.append(f"[FAIL] {row.get('name')}: {exc}")
            fm_pack.save_state(state, _pack_state_path)
            return (
                "\n".join(log) or "Nothing to download.",
                *_cat_snapshot("", "all", "all", ""),
            )

        pack_download_updates.click(
            _download_updates,
            inputs=[update_sel],
            outputs=[pack_update, cat_table, cat_pick, cat_detail, cat_media],
            show_progress="full",
        )

        # ── Catalog: filter, select one item, act on it ──────────────────
        def _rows_now():
            cat = _catalog()
            st = fm_pack.catalog_status(
                manifest,
                ckpts_dir(),
                lora_dir_for,
                finetunes_dir,
                state,
                custom=cat,
            )
            return _pack_catalog_rows(st, cat), cat

        def _cat_snapshot(query, family, show, pick):
            rows, _cat = _rows_now()
            shown = _filter_catalog_rows(rows, query, family, show)
            choices = _pack_pick_choices(shown)
            valid = {value for _, value in choices}
            pick = pick if pick in valid else None
            row = next((r for r in shown if r["pick"] == pick), None)
            media = (
                _pack_media_html([row["entry"].get("preview")], row.get("name"))
                if row
                else ""
            )
            return (
                _pack_table_value(shown),
                gr.update(choices=choices, value=pick),
                _pack_detail_md(row),
                media,
            )

        def _cat_filter(query, family, show, pick):
            return _cat_snapshot(query, family, show, pick)

        cat_q.change(
            _cat_filter,
            inputs=[cat_q, cat_family, cat_show, cat_pick],
            outputs=[cat_table, cat_pick, cat_detail, cat_media],
        )
        cat_family.change(
            _cat_filter,
            inputs=[cat_q, cat_family, cat_show, cat_pick],
            outputs=[cat_table, cat_pick, cat_detail, cat_media],
        )
        cat_show.change(
            _cat_filter,
            inputs=[cat_q, cat_family, cat_show, cat_pick],
            outputs=[cat_table, cat_pick, cat_detail, cat_media],
        )

        def _cat_select(pick):
            rows, _cat = _rows_now()
            row = next((r for r in rows if r["pick"] == pick), None)
            return (
                _pack_detail_md(row),
                _pack_media_html([row["entry"].get("preview")], row.get("name"))
                if row
                else "",
            )

        cat_pick.change(
            _cat_select,
            inputs=[cat_pick],
            outputs=[cat_detail, cat_media],
        )

        def _cat_row(pick):
            rows, cat = _rows_now()
            return next((r for r in rows if r["pick"] == pick), None), cat

        def _cat_download_one(pick, query, family, show, progress=gr.Progress()):
            row, _cat = _cat_row(pick)
            if not row:
                return "Pick an item first.", *_cat_snapshot(query, family, show, pick)
            try:
                msg = _fetch_one(row, progress)
                fm_pack.save_state(state, _pack_state_path)
            except Exception as exc:
                msg = f"[FAIL] {row['entry'].get('name')}: {exc}"
            return (
                msg,
                *_cat_snapshot(query, family, show, pick),
            )

        cat_download.click(
            _cat_download_one,
            inputs=[cat_pick, cat_q, cat_family, cat_show],
            outputs=[cat_log, cat_table, cat_pick, cat_detail, cat_media],
            show_progress="full",
        )

        def _cat_recipe_one(pick, query, family, show):
            row, _cat = _cat_row(pick)
            if not row:
                return "Pick a checkpoint first.", *_cat_snapshot(query, family, show, pick)
            if row["kind"] != "Checkpoint":
                return (
                    "Recipes only exist for checkpoints.",
                    *_cat_snapshot(query, family, show, pick),
                )
            try:
                p = fm_pack.write_recipe(row["entry"], finetunes_dir)
                msg = f"[OK] wrote {p}"
            except Exception as exc:
                msg = f"[FAIL] {row['entry'].get('name')}: {exc}"
            return (msg, *_cat_snapshot(query, family, show, pick))

        cat_recipe.click(
            _cat_recipe_one,
            inputs=[cat_pick, cat_q, cat_family, cat_show],
            outputs=[cat_log, cat_table, cat_pick, cat_detail, cat_media],
        )

        def _cat_delete_one(pick, query, family, show):
            row, _cat = _cat_row(pick)
            if not row:
                return "Pick an item first.", *_cat_snapshot(query, family, show, pick)
            entry = row["entry"]
            try:
                if row["kind"] == "LoRA":
                    res = fm_pack.delete_lora(entry, lora_dir_for)
                    done = bool(res.get("removed"))
                elif row["kind"] == "Checkpoint":
                    res = fm_pack.delete_finetune(entry, ckpts_dir(), finetunes_dir)
                    done = bool(res.get("removedWeights") or res.get("removedRecipe"))
                else:
                    res = fm_pack.delete_civitai(
                        entry, state, ckpts_dir(), lora_dir_for, finetunes_dir
                    )
                    done = bool(res.get("removedFiles") or res.get("removedRecipe"))
                if entry.get("custom"):
                    fm_pack.drop_custom(row["id"])
                    done = True
                fm_pack.save_state(state, _pack_state_path)
                msg = (
                    f"[OK] {entry.get('name')}: deleted"
                    if done
                    else f"[SKIP] {entry.get('name')}: nothing present"
                )
            except Exception as exc:
                msg = f"[FAIL] {entry.get('name')}: {exc}"
            return (
                msg,
                *_cat_snapshot(query, family, show, "" if entry.get("custom") else pick),
            )

        cat_delete.click(
            _cat_delete_one,
            inputs=[cat_pick, cat_q, cat_family, cat_show],
            outputs=[cat_log, cat_table, cat_pick, cat_detail, cat_media],
        )

        def _cat_gallery(pick):
            row, _cat = _cat_row(pick)
            if not row:
                return "", "Pick an item first."
            entry = row["entry"]
            urls = []
            if entry.get("preview"):
                urls.append(entry["preview"])
            try:
                if row["kind"] == "CivitAI":
                    data = fm_pack.civitai_gallery(entry, token=CIVITAI_API_KEY)
                    urls += [x.get("url") for x in data.get("images") or []]
                else:
                    repo = str(entry.get("repo") or "")
                    if not repo:
                        page = str(entry.get("repoPage") or "")
                        marker = "huggingface.co/"
                        if marker in page:
                            repo = page.split(marker, 1)[1].split("/")[:2]
                            repo = "/".join(repo)
                    if repo:
                        data = fm_pack.hf_gallery(repo)
                        urls += [x.get("url") for x in data.get("images") or []]
                        urls += [x.get("url") for x in data.get("videos") or []]
                html_out = _pack_media_html(urls, row.get("name"))
                return (
                    html_out,
                    "" if html_out else "No gallery images found for this item.",
                )
            except Exception as exc:
                return "", f"[FAIL] gallery: {exc}"

        cat_gallery.click(
            _cat_gallery,
            inputs=[cat_pick],
            outputs=[cat_media, cat_log],
        )

        # ── Find & Add ──────────────────────────────────────────────────
        find_items_state = gr.State([])
        find_info_state = gr.State({})

        def _find_run(source, mode, query, kind, sort, period, limit):
            try:
                if mode == "browse":
                    items = (
                        fm_pack.civitai_browse(
                            manifest,
                            sort=sort,
                            period=period,
                            kind=kind,
                            limit=int(limit),
                            token=CIVITAI_API_KEY,
                        )
                        if source == "civitai"
                        else fm_pack.hf_browse(
                            manifest, sort=sort, kind=kind, limit=int(limit)
                        )
                    )
                else:
                    items = (
                        fm_pack.civitai_search(
                            manifest,
                            query,
                            kind=kind,
                            limit=int(limit),
                            token=CIVITAI_API_KEY,
                        )
                        if source == "civitai"
                        else fm_pack.hf_search(
                            manifest, query, kind=kind, limit=int(limit)
                        )
                    )
            except Exception as exc:
                return (
                    [],
                    [],
                    gr.update(choices=[], value=None),
                    f"[FAIL] {exc}",
                )
            rows = [
                [
                    x.get("title") or "",
                    x.get("family") or "",
                    x.get("architecture") or "",
                    x.get("downloads") or 0,
                    x.get("updated") or x.get("releasedAt") or "",
                    x.get("id"),
                ]
                for x in items
            ]
            choices = [
                (f"{x.get('title')} - {x.get('family')} - {x.get('architecture')}", str(i))
                for i, x in enumerate(items)
            ]
            note = (
                "Nothing matched - try a broader query or another type."
                if not items
                else f"{len(items)} result(s). Pick one, then Inspect."
            )
            return items, rows, gr.update(choices=choices, value=None), note

        find_run.click(
            _find_run,
            inputs=[
                find_source,
                find_mode,
                find_query,
                find_kind,
                find_sort,
                find_period,
                find_limit,
            ],
            outputs=[find_items_state, find_table, find_pick, find_status],
        )

        def _find_inspect(idx, items, arch, version):
            try:
                item = (items or [])[int(idx)]
            except Exception:
                return (
                    {},
                    [],
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    "",
                    gr.update(),
                    "Pick a result first.",
                )
            chosen_arch = str(arch or item.get("architecture") or "")
            try:
                info = fm_pack.inspect_source(
                    manifest,
                    item.get("source"),
                    str(item.get("id")),
                    architecture=chosen_arch,
                    version=version or item.get("versionId"),
                    token=CIVITAI_API_KEY,
                )
            except Exception as exc:
                return (
                    {},
                    [],
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    "",
                    gr.update(),
                    f"[FAIL] {exc}",
                )
            files = info.get("files") or []
            rows = [
                [
                    f.get("file"),
                    f.get("kind"),
                    fm_pack.human_bytes(f.get("bytes") or 0),
                    f.get("pair") or "",
                    str(f.get("sha256") or "")[:16],
                ]
                for f in files
            ]
            file_choices = [
                (
                    f"{f.get('file')} - {f.get('kind')} - "
                    f"{fm_pack.human_bytes(f.get('bytes') or 0)}",
                    str(i),
                )
                for i, f in enumerate(files)
            ]
            versions = info.get("versions") or []
            version_choices = [
                (
                    f"{v.get('name') or v.get('id')} - {v.get('baseModel') or ''} "
                    f"({v.get('fileCount')} file(s))",
                    str(v.get("id")),
                )
                for v in versions
            ]
            urls = [info.get("preview")] + [
                x.get("url") for x in info.get("samples") or []
            ]
            note = (
                f"{info.get('title')} - {len(files)} loadable file(s). "
                "Pick a file, then Add."
                if files
                else "No loadable files found (needs .safetensors or .gguf)."
            )
            return (
                info,
                rows,
                gr.update(choices=file_choices, value=None),
                gr.update(
                    choices=version_choices,
                    value=str(info.get("versionId") or "") or None,
                ),
                _pack_media_html(urls, info.get("title")),
                gr.update(value=chosen_arch if chosen_arch in arch_values else None),
                note,
            )

        find_inspect.click(
            _find_inspect,
            inputs=[find_pick, find_items_state, find_arch, find_version],
            outputs=[
                find_info_state,
                find_files,
                find_file_pick,
                find_version,
                find_media,
                find_arch,
                find_status,
            ],
        )
        find_version.change(
            _find_inspect,
            inputs=[find_pick, find_items_state, find_arch, find_version],
            outputs=[
                find_info_state,
                find_files,
                find_file_pick,
                find_version,
                find_media,
                find_arch,
                find_status,
            ],
        )

        def _find_add(info, file_idx, kind_override, arch):
            if not info:
                return "Inspect a result first.", *_cat_snapshot("", "all", "all", "")
            files = info.get("files") or []
            try:
                f = files[int(file_idx)]
            except Exception:
                return "Pick a file first.", *_cat_snapshot("", "all", "all", "")
            chosen_arch = str(arch or info.get("architecture") or "")
            if not chosen_arch:
                return "Pick the Wan2GP architecture first.", *_cat_snapshot(
                    "", "all", "all", ""
                )
            kind = f.get("kind") if kind_override == "auto" else kind_override
            source = str(info.get("source") or "")
            family = fm_pack.family_for_arch(manifest, chosen_arch)
            if not family:
                hit = fm_pack.match_arch(manifest, str(info.get("title") or ""))
                family = str((hit or {}).get("id") or "")
            try:
                entry = fm_pack.build_custom_entry(
                    source=source,
                    title=str(info.get("title") or ""),
                    file=str(f.get("file") or ""),
                    url=str(f.get("url") or ""),
                    architecture=chosen_arch,
                    kind=str(kind or "checkpoint"),
                    bytes=int(f.get("bytes") or 0),
                    sha256=f.get("sha256"),
                    repo=str(info.get("id") or "") if source == "hf" else "",
                    repo_path=str(f.get("file") or ""),
                    preview=str(info.get("preview") or ""),
                    repo_page=str(info.get("repoPage") or ""),
                    file_page="",
                    wan2gp_ids=[chosen_arch],
                    family=family,
                    needs_token=bool(f.get("needsToken")),
                )
                fm_pack.add_custom(entry)
                msg = (
                    f"[OK] added {entry.get('name')} - fetch it from the Catalog tab."
                )
                pick = ("lora:" if entry.get("group") == "custom" else "finetune:") + str(
                    entry.get("id")
                )
            except Exception as exc:
                return f"[FAIL] {exc}", *_cat_snapshot("", "all", "all", "")
            return (msg, *_cat_snapshot("", "all", "all", pick))

        find_add.click(
            _find_add,
            inputs=[find_info_state, find_file_pick, find_kind_override, find_arch],
            outputs=[
                find_status,
                cat_table,
                cat_pick,
                cat_detail,
                cat_media,
            ],
        )

        def _find_url_resolve(text, arch):
            parsed = fm_pack.parse_source_url(text)
            if not parsed:
                return (
                    {},
                    [],
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    "",
                    None,
                    "Paste a HuggingFace repo/URL, a CivitAI model URL, or a CivitAI model id.",
                )
            source, ident = parsed
            try:
                info = fm_pack.inspect_source(
                    manifest,
                    source,
                    ident,
                    architecture=str(arch or ""),
                    token=CIVITAI_API_KEY,
                )
            except Exception as exc:
                return (
                    {},
                    [],
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    "",
                    None,
                    f"[FAIL] {exc}",
                )
            files = info.get("files") or []
            rows = [
                [
                    f.get("file"),
                    f.get("kind"),
                    fm_pack.human_bytes(f.get("bytes") or 0),
                    f.get("pair") or "",
                    str(f.get("sha256") or "")[:16],
                ]
                for f in files
            ]
            file_choices = [
                (
                    f"{f.get('file')} - {f.get('kind')} - "
                    f"{fm_pack.human_bytes(f.get('bytes') or 0)}",
                    str(i),
                )
                for i, f in enumerate(files)
            ]
            version_choices = [
                (
                    f"{v.get('name') or v.get('id')} - {v.get('baseModel') or ''} "
                    f"({v.get('fileCount')} file(s))",
                    str(v.get("id")),
                )
                for v in info.get("versions") or []
            ]
            urls = [info.get("preview")] + [
                x.get("url") for x in info.get("samples") or []
            ]
            note = (
                f"Found {info.get('title')} ({source}) - {len(files)} file(s). "
                "Pick one and Add, or change the architecture first."
                if files
                else "No loadable files found (needs .safetensors or .gguf)."
            )
            return (
                info,
                rows,
                gr.update(choices=file_choices, value=None),
                gr.update(
                    choices=version_choices,
                    value=str(info.get("versionId") or "") or None,
                ),
                _pack_media_html(urls, info.get("title")),
                str(arch or info.get("architecture") or "") or None,
                note,
            )

        find_url_btn.click(
            _find_url_resolve,
            inputs=[find_url, find_url_arch],
            outputs=[
                find_info_state,
                find_files,
                find_file_pick,
                find_version,
                find_media,
                find_arch,
                find_status,
            ],
        )

    for line in manifest.get("legal", []):
        gr.Markdown(f"- {line}")


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
            f"[HF Registry](https://huggingface.co/spaces/{REGISTRY_SPACE})"
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

                def _fmt_cards(fins, search, arch, sel, tag_filter, family="All"):
                    ff = list(fins)
                    if search:
                        s = search.lower()

                        def _match_tags(f):
                            tg = f.get("tags", [])
                            if isinstance(tg, str):
                                tg = [t.strip() for t in tg.split(",") if t.strip()]
                            if not isinstance(tg, list):
                                return False
                            return any(s in (tag or "").lower() for tag in tg)

                        ff = [
                            f
                            for f in ff
                            if s in f.get("name", "").lower()
                            or s in (f.get("description", "") or "").lower()
                            or s in f.get("author", "").lower()
                            or _match_tags(f)
                        ]
                    if family and family != "All":
                        ff = [f for f in ff if f.get("family") == family]
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
                                    (tag or "").strip().lower() == tf for tag in tg
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
                        key=lambda f: _family_sort_key(
                            catalog,
                            f.get("family") or _FAMILY_UNKNOWN,
                            f.get("name", f.get("id", "")),
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
                                family_label=_family_label(catalog, fam),
                                family_warning=f.get("family_warning", ""),
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
                                t = t.strip().lower()
                                if t:
                                    seen.add(t)
                    return sorted(seen)

                def _do_refresh():
                    fins = _enrich_registry(_get_cached_registry(force=True))
                    html, cnt = _fmt_cards(fins, "", "All", "", "")
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

                def _filter(fins, search, arch, sel, tag_filter, family):
                    html, cnt = _fmt_cards(fins, search, arch, sel, tag_filter, family)
                    return html, cnt

                def _family_changed(fins, search, sel, tag_filter, family):
                    html, cnt = _fmt_cards(
                        fins, search, "All", sel, tag_filter, family
                    )
                    return (
                        html,
                        cnt,
                        gr.update(choices=_archs_for(fins, family), value="All"),
                    )

                b_search.input(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id, b_tag_filter, b_family],
                    outputs=[b_cards, b_count],
                )
                b_arch.change(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id, b_tag_filter, b_family],
                    outputs=[b_cards, b_count],
                )
                b_tag_filter.change(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id, b_tag_filter, b_family],
                    outputs=[b_cards, b_count],
                )
                b_family.change(
                    fn=_family_changed,
                    inputs=[registry, b_search, b_sel_id, b_tag_filter, b_family],
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
                            _validate_urls(all_entries, self)
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
                        data = _fetch_registry_json(fid)
                    except Exception as e:
                        return f"Error: {e}", gr.update(), gr.update()
                    err = _validate_finetune_data(data)
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
                        data = _fetch_registry_json(fid)
                    except Exception as e:
                        return f"Error: {e}"
                    err = _validate_finetune_data(data)
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
                        data = _fetch_registry_json(fid)
                    except Exception as e:
                        return [gr.update()] * (len(ALL_INPUTS)) + [gr.update(), f"Error: {e}", {}]
                    err = _validate_finetune_data(data)
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

            # ── PACK ──
            with gr.TabItem("Pack", id="pack"):
                _build_pack_tab(self)

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
                            # Model reference =id
                            if u.startswith("="):
                                mid = u[1:].strip()
                                if mid in MODEL_INDEX:
                                    data = _read_model_json(mid)
                                    if data:
                                        urls = _get_model_urls(data)
                                        if urls:
                                            return (
                                                f'<span style="font-size:11px">'
                                                f"\u2705 {MODEL_INDEX[mid]}"
                                                f"<br/>&nbsp;&nbsp;\u2192 "
                                                f"{urls[0]}</span>"
                                            )
                                    return (
                                        f'<span style="font-size:11px">'
                                        f"\u2705 {MODEL_INDEX[mid]}</span>"
                                    )
                                return (
                                    f'<span style="font-size:11px;'
                                    f'color:#dc2626">\u274c {mid} not in '
                                    f"list \u2014 enter a URL or upload a file instead</span>"
                                )
                            # HTTP/HTTPS URL
                            if u.startswith(("http:", "https:")):
                                return (
                                    f'<span style="font-size:11px">'
                                    f"\u2705 URL format OK \u2014 use "
                                    f"Validate button to check</span>"
                                )
                            # Bare model ID (not a path, not a URL)
                            if (
                                not u.startswith(("/", ".", "\\"))
                                and "/" not in u
                                and "\\" not in u
                                and not (len(u) > 1 and u[1] == ":")
                            ):
                                if u in MODEL_INDEX:
                                    data = _read_model_json(u)
                                    if data:
                                        urls = _get_model_urls(data)
                                        if urls:
                                            return (
                                                f'<span style="font-size:11px">'
                                                f"\u2705 {MODEL_INDEX[u]}"
                                                f"<br/>&nbsp;&nbsp;\u2192 "
                                                f"{urls[0]}</span>"
                                            )
                                    return (
                                        f'<span style="font-size:11px">'
                                        f"\u2705 {MODEL_INDEX[u]}</span>"
                                    )
                                if _looks_like_file(u):
                                    if _check_disk_exists(u):
                                        return (
                                            f'<span style="font-size:11px">'
                                            f"\u2705 Found on disk</span>"
                                        )
                                return (
                                    f'<span style="font-size:11px;'
                                    f'color:#dc2626">\u274c {u} not in '
                                    f"list \u2014 enter a URL or upload a file instead</span>"
                                )
                            # Local path
                            if _check_disk_exists(u):
                                return (
                                    f'<span style="font-size:11px">'
                                    f"\u2705 Found on disk</span>"
                                )
                            return (
                                f'<span style="font-size:11px;'
                                f'color:#dc2626">\u274c File not found'
                                f"</span>"
                            )

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
                            icon = "\u2705" if ok else "\u274c"
                            val_results.append(
                                gr.update(
                                    value=f"<span style='font-size:12px'>{icon} {msg}</span>"
                                )
                            )
                            ref_results.append(gr.update(value=mid))
                        elif (
                            "/" not in first_line
                            and "\\" not in first_line
                            and not first_line.startswith(("http:", "https:", "."))
                            and not (len(first_line) > 1 and first_line[1] == ":")
                        ):
                            ok, msg = _validate_model_ref(first_line, self)
                            icon = "\u2705" if ok else "\u274c"
                            val_results.append(
                                gr.update(
                                    value=f"<span style='font-size:12px'>{icon} {msg}</span>"
                                )
                            )
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
                                    parts.append(
                                        f'<span style="font-size:11px">'
                                        f"\u2705 URL format OK \u2014 use "
                                        f"Validate button to check</span>"
                                    )
                                elif _check_disk_exists(line):
                                    parts.append(
                                        f'<span style="font-size:11px">'
                                        f"\u2705 Found on disk</span>"
                                    )
                                else:
                                    parts.append(
                                        f'<span style="font-size:11px;'
                                        f'color:#dc2626">\u274c {line} \u2014 '
                                        f"use Validate button to check</span>"
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
                    if not _check_upload_token():
                        return (
                            "⚠️ **No HF token configured.** "
                            "Upload requires a HuggingFace token.\n\n"
                            "1. Go to https://huggingface.co/settings/tokens\n"
                            "2. Create a token (any role)\n"
                            f"3. Edit `{_cfg_path}` and add:\n"
                            '   `{"registry_token": "hf_..."}`\n'
                            "Or run `huggingface-cli login` in your terminal."
                        )
                    data = _build(*vals, extra_data=extra_data)
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
                        _hf_upload(safe_id, data, self)
                        if (
                            hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs
                        ):
                            self.refresh_model_defs()
                        return f"Saved & uploaded {safe_id}"
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
                            color = "#16a34a" if valid else "#dc2626"
                            icon = "\u2713" if valid else "\u2717"
                            field_parts.append(
                                f"<span style='font-size:11px;color:{color}'>"
                                f"{icon} {msg}</span>"
                            )
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
                        icon = "\u2705" if ok else "\u274c"
                        val_html = f"<span style='font-size:12px'>{icon} {msg}</span>"
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
                        icon = "\u2705" if ok else "\u274c"
                        val_html = f"<span style='font-size:12px'>{icon} {msg}</span>"
                        return (
                            gr.update(value=mid if ok else None),
                            gr.update(value=val_html),
                        )
                    # Custom URL or local path — lightweight (no network)
                    lines = [l.strip() for l in current.split("\n") if l.strip()]
                    parts = []
                    for line in lines:
                        if line.startswith(("http:", "https:")):
                            parts.append(
                                "<span style='font-size:11px'>"
                                "\u2705 URL format OK — use Validate button "
                                "to check availability</span>"
                            )
                        elif line.startswith(("/", ".", "\\")) or (
                            len(line) > 1 and line[1] == ":"
                        ):
                            ok, msg = _validate_local_path(line)
                            icon = "\u2705" if ok else "\u274c"
                            parts.append(
                                f"<span style='font-size:11px'>{icon} {msg}</span>"
                            )
                        else:
                            # Bare string — treat as potential model ref
                            data = _read_model_json(line)
                            if data:
                                name = data.get("model", {}).get("name", line)
                                urls = _get_model_urls(data)
                                if urls:
                                    parts.append(
                                        "<span style='font-size:11px'>"
                                        f"\u2705 Model ref: {name}"
                                        f"<br/>&nbsp;&nbsp;\u2192 {urls[0]}</span>"
                                    )
                                else:
                                    parts.append(
                                        "<span style='font-size:11px'>"
                                        f"\u2705 Model ref: {name}</span>"
                                    )
                            else:
                                parts.append(
                                    "<span style='font-size:11px;color:#dc2626'>"
                                    f"\u274c '{line}' not in list — "
                                    "enter a URL or upload a file instead</span>"
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
            with gr.TabItem("CivitAI", id="civitai") as ci_tab:
                gr.Markdown(
                    "Search CivitAI for models and auto-fill into Create/Edit. "
                    "**Site** picks the host: `civitai.com` or its adult front-end "
                    "`civitai.red` (same catalogue; mature models need *Include NSFW* "
                    "on either one)."
                )
                with gr.Row():
                    ci_search = gr.Textbox(
                        label="Search",
                        scale=3,
                        container=False,
                        placeholder="e.g. fantasy landscape, character style...",
                    )
                    ci_type = gr.Dropdown(
                        label="Type",
                        choices=_civitai_enum_choices("ModelType"),
                        value="All",
                        scale=1,
                    )
                    ci_base = gr.Dropdown(
                        label="Base Model",
                        choices=_civitai_enum_choices("BaseModel"),
                        filterable=True,
                        value="All",
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
                    ci_search_btn = gr.Button("Search", variant="primary", scale=1)
                with gr.Row():
                    ci_pick = gr.Dropdown(
                        label="Result version",
                        choices=[],
                        value=None,
                        interactive=True,
                        scale=3,
                        container=False,
                    )
                    ci_load_btn = gr.Button(
                        "Fill Selected into Create/Edit", variant="primary", scale=1
                    )
                    ci_clear_btn = gr.Button("Clear", scale=1)
                ci_status = gr.Markdown("Enter a search term to find models on CivitAI")
                ci_results = gr.HTML(
                    "<div style='color:#9ca3af;text-align:center;"
                    "padding:32px'>Click Search</div>"
                )

                def _ci_refresh_filters(mtype, base):
                    """Upgrade the filters to CivitAI's live enum values when the tab is
                    opened. The UI is built from the built-in list, and the fetch runs on
                    a background thread, so nothing here waits on the network."""
                    _civitai_prefetch_enums()
                    return _civitai_enum_updates(mtype, base)

                ci_tab.select(
                    fn=_ci_refresh_filters,
                    inputs=[ci_type, ci_base],
                    outputs=[ci_type, ci_base],
                )

                def _ci_do_search(query, mtype, base, nsfw=False, site="civitai.com"):
                    query = (query or "").strip()
                    site = _civitai_site(site)
                    filtered = mtype not in ("", "All", None) or base not in ("", "All", None)
                    _civitai_prefetch_enums()
                    enum_updates = _civitai_enum_updates(mtype, base)
                    if not query and not filtered:
                        return (
                            "<div style='color:#9ca3af;"
                            "text-align:center;padding:32px'>"
                            "Enter a search term or pick a Type / Base Model</div>",
                            "",
                            gr.update(choices=[], value=None),
                        ) + tuple(enum_updates)
                    data = _civitai_search(
                        query, model_type=mtype, base_model=base, nsfw=bool(nsfw), site=site
                    )
                    html_out = _civitai_render_results(data)
                    options = _civitai_result_options(data, site)
                    count = len(data.get("items", [])) if "error" not in data else 0
                    label = f"'{html.escape(query)}'" if query else "most downloaded"
                    status = (
                        f"**{count} results** for {label}"
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
                    inputs=[ci_search, ci_type, ci_base, ci_nsfw, ci_site],
                    outputs=[ci_results, ci_status, ci_pick, ci_type, ci_base],
                )
                ci_search.submit(
                    fn=_ci_do_search,
                    inputs=[ci_search, ci_type, ci_base, ci_nsfw, ci_site],
                    outputs=[ci_results, ci_status, ci_pick, ci_type, ci_base],
                )

                def _ci_fill(civitai_json_str):
                    return _civitai_extract_fill_data(civitai_json_str)

                ci_load_btn.click(
                    fn=_ci_fill,
                    inputs=[ci_pick],
                    outputs=ALL_INPUTS[1:] + [ci_status],
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

                def _ci_clear():
                    return (
                        "<div style='color:#9ca3af;text-align:center;"
                        "padding:32px'>Click Search</div>",
                        gr.update(choices=[], value=None),
                        "Enter a search term to find models on CivitAI",
                    )

                ci_clear_btn.click(
                    fn=_ci_clear, outputs=[ci_results, ci_pick, ci_status]
                )

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
                err = _validate_finetune_data(data)
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
                if not _check_upload_token():
                    return (
                        "⚠️ **No HF token configured.** "
                        "Upload requires a HuggingFace token.\n\n"
                        "1. Go to https://huggingface.co/settings/tokens\n"
                        "2. Create a token (any role)\n"
                        f"3. Edit `{_cfg_path}` and add:\n"
                        '   `{"registry_token": "hf_..."}`\n'
                        "Or run `huggingface-cli login` in your terminal."
                    )
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return "Not found"
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception as e:
                    return f"Cannot upload: invalid JSON ({e})"
                err = _validate_finetune_data(data)
                if err:
                    return f"Cannot upload: {err}"
                token_urls = _find_token_urls(data)
                if token_urls:
                    return (
                        f"Cannot upload '{fid}': {len(token_urls)} URL(s) embed your "
                        "CivitAI API token (`token=`). Remove the token before uploading "
                        "or it will be published with the finetune."
                    )
                _stamp_hf_username(data)
                _hf_upload(fid, data, self)
                return f"Uploaded '{fid}'"

            l_up.click(fn=_up, inputs=[l_sel_id], outputs=[l_status])
