import html
import json
import re as _re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import requests
import gradio as gr
from huggingface_hub import HfApi
from shared.utils.plugins import WAN2GPPlugin
from shared.gradio.local_file_picker import LocalFilePickerTextbox
import copy
import time

PlugIn_Name = "Finetune Manager"
PlugIn_Id = "FinetuneManager"
PLUGIN_VERSION = "3.5.0"

DEFAULT_REGISTRY = "https://huggingface.co/spaces/GKartist75/wan2gp-finetunes/raw/main"
REGISTRY_SPACE = "GKartist75/wan2gp-finetunes"
FINETUNES_DIR = "finetunes"

_cfg_path = Path(__file__).parent / "config.json"
REGISTRY_TOKEN = ""
if _cfg_path.exists():
    try:
        REGISTRY_TOKEN = json.loads(_cfg_path.read_text(encoding="utf-8")).get(
            "registry_token", ""
        )
    except Exception:
        pass
print(f"[FM] v{PLUGIN_VERSION} Token:{bool(REGISTRY_TOKEN)}")

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

# ── Registry cache: avoids redundant N+1 fetches on rapid Refresh ──
_REGISTRY_CACHE: list[dict] | None = None
_REGISTRY_CACHE_TIME: float = 0
_REGISTRY_CACHE_TTL: float = 30.0  # seconds

def _get_cached_registry(force: bool = False) -> list[dict]:
    """Return cached registry, fetching fresh only if stale or forced."""
    global _REGISTRY_CACHE, _REGISTRY_CACHE_TIME
    now = time.time()
    if not force and _REGISTRY_CACHE is not None and (now - _REGISTRY_CACHE_TIME) < _REGISTRY_CACHE_TTL:
        return _REGISTRY_CACHE
    fins = _fetch_dynamic_registry_no_cache()
    _REGISTRY_CACHE = fins
    _REGISTRY_CACHE_TIME = now
    return fins

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
    is_variant: bool = False,
) -> str:
    """Build a single finetune card HTML fragment.
    Used by both the Browse and Local tab card builders."""
    c = " selected" if is_selected else ""
    safe_name = html.escape(name or "?")
    safe_arch = html.escape(arch or "")
    safe_author = html.escape(author or "")
    safe_desc = html.escape((desc or "")[:300])
    tag = "Variant" if is_variant else safe_arch
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
    fid_safe = (
        fid.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    )
    onclick = (
        f"var r=(window.gradioApp?window.gradioApp():"
        f"document.querySelector('gradio-app'))||document;"
        f"r=r.shadowRoot||r;"
        f"var ta=r.querySelector('#{sel_elem_id} textarea');"
        f"if(ta){{ta.value='{fid_safe}';"
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
            link_parts.append(
                f'<a href="{eu}" target="_blank" rel="noopener" '
                f'style="word-break:break-all">{eu}</a>'
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
        f'<div class="fm-card{c}" onclick="{onclick}">'
        f"<div class='fm-card-title'>{safe_name}</div>"
        f"<div class='fm-card-meta'>"
        f"<span class='fm-badge'>{html.escape(tag)}</span> {safe_author}</div>"
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
        parts = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
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
        parsed_urls = _parse_maybe_scalar(urls)
        if parsed_urls is not None:
            m["URLs"] = parsed_urls
        parsed_urls2 = _parse_maybe_scalar(urls2)
        if parsed_urls2 is not None:
            m["URLs2"] = parsed_urls2
        parsed_te = _parse_maybe_scalar(te)
        if parsed_te is not None:
            m["text_encoder_URLs"] = parsed_te
        parsed_vae = _parse_maybe_scalar(vae)
        if parsed_vae is not None:
            m["VAE_URLs"] = parsed_vae
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
        # preload_URLs: preserve original string/list type
        orig_pre = orig_m.get("preload_URLs")
        if isinstance(orig_pre, str):
            m["preload_URLs"] = pre if pre else orig_pre
        else:
            parsed_pre = _parse_maybe_scalar(pre)
            if parsed_pre is not None:
                m["preload_URLs"] = parsed_pre
            elif "preload_URLs" in m:
                del m["preload_URLs"]
        parsed_loras = _parse_maybe_scalar(loras)
        if parsed_loras is not None:
            m["loras"] = parsed_loras
        if lm:
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
        if res:
            orig_had_res = "resolutions" in orig_m
            if orig_had_res:
                m["resolutions"] = _parse_resolutions(res)
        elif "resolutions" in m:
            del m["resolutions"]
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
        parsed_urls = _parse_maybe_scalar(urls)
        if parsed_urls is not None:
            m["URLs"] = parsed_urls
        parsed_urls2 = _parse_maybe_scalar(urls2)
        if parsed_urls2 is not None:
            m["URLs2"] = parsed_urls2
        parsed_te = _parse_maybe_scalar(te)
        if parsed_te is not None:
            m["text_encoder_URLs"] = parsed_te
        parsed_vae = _parse_maybe_scalar(vae)
        if parsed_vae is not None:
            m["VAE_URLs"] = parsed_vae
        if cu1:
            m["custom_url_1"] = cu1
        if cu2:
            m["custom_url_2"] = cu2
        if cu3:
            m["custom_url_3"] = cu3
        if mods:
            m["modules"] = [x.strip() for x in mods.split(",") if x.strip()]
        parsed_pre = _parse_maybe_scalar(pre)
        if parsed_pre is not None:
            m["preload_URLs"] = parsed_pre
        parsed_loras = _parse_maybe_scalar(loras)
        if parsed_loras is not None:
            m["loras"] = parsed_loras
        if lm:
            m["loras_multipliers"] = _to_typed_multipliers(lm)
        if aq:
            m["auto_quantize"] = True
        if not vis:
            m["visible"] = False
        if img:
            m["image_outputs"] = True
        if tags:
            m["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        if res:
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
        gd = float(guid)
        orig_gd = orig_out.get("guidance_scale")
        if isinstance(orig_gd, int):
            out["guidance_scale"] = int(gd)
        else:
            out["guidance_scale"] = gd
        if solver:
            out["sample_solver"] = solver
        if prompt:
            out["prompt"] = prompt
    else:
        # Fresh build — canonical order
        out: dict = {}
        out["model"] = m
        out["num_inference_steps"] = int(steps)
        gd = float(guid)
        out["guidance_scale"] = gd
        if solver:
            out["sample_solver"] = solver
        if prompt:
            out["prompt"] = prompt

    return out


def _clean_surrogates(val):
    """Strip surrogate characters (U+D800-U+DFFF) from strings to prevent orjson crashes."""
    if isinstance(val, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in val):
        return val.translate({i: None for i in range(0xD800, 0xE000)})
    return val


def _clean_tuple(t):
    """Recursively strip surrogates from all strings in a nested tuple/list/dict."""
    return tuple(_clean_surrogates(v) if isinstance(v, str) else v for v in t)


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

    res = d.get("resolution", "")
    if not res:
        for r in m.get("resolutions", []):
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
    return _clean_tuple(
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


# Markdown toolbar JS (from built-in Finetune Editor)
MD_TOOLBAR_JS = """<script>
(function(){
if(window.__fmMdToolbar)return;window.__fmMdToolbar=true;
function root(){
if(window.gradioApp)return window.gradioApp();
var app=document.querySelector('gradio-app');
return app?(app.shadowRoot||app):document;
}
function snippet(a){
var m={bold:{p:"**",s:"**",x:"bold text"},italic:{p:"*",s:"*",x:"italic text"},heading:{p:"## ",s:"",x:"Heading"},list:{p:"- ",s:"",x:"item"},link:{p:"[",s:"](https://)",x:"link text"},code:{p:"`",s:"`",x:"code"}};
return m[a]||{p:"",s:"",x:""};
}
function tbHTML(){
return '<div style="display:flex;gap:4px;padding:3px 0">'+
'<button type=button data-md-action=bold style="width:26px;height:24px;border:1px solid var(--border-color-primary,#ccc);border-radius:4px;background:var(--button-secondary-background-fill,#f5f5f5);cursor:pointer;font-size:12px"><b>B</b></button>'+
'<button type=button data-md-action=italic style="width:26px;height:24px;border:1px solid var(--border-color-primary,#ccc);border-radius:4px;background:var(--button-secondary-background-fill,#f5f5f5);cursor:pointer;font-size:12px"><i>I</i></button>'+
'<button type=button data-md-action=heading style="width:26px;height:24px;border:1px solid var(--border-color-primary,#ccc);border-radius:4px;background:var(--button-secondary-background-fill,#f5f5f5);cursor:pointer;font-size:12px">H</button>'+
'<button type=button data-md-action=list style="width:26px;height:24px;border:1px solid var(--border-color-primary,#ccc);border-radius:4px;background:var(--button-secondary-background-fill,#f5f5f5);cursor:pointer;font-size:12px">\u2022</button>'+
'<button type=button data-md-action=link style="width:26px;height:24px;border:1px solid var(--border-color-primary,#ccc);border-radius:4px;background:var(--button-secondary-background-fill,#f5f5f5);cursor:pointer;font-size:12px">\u2197</button>'+
'<button type=button data-md-action=code style="width:26px;height:24px;border:1px solid var(--border-color-primary,#ccc);border-radius:4px;background:var(--button-secondary-background-fill,#f5f5f5);cursor:pointer;font-size:12px">`</button></div>';
}
function inject(){
var r=root();if(!r)return;
r.querySelectorAll('.wangp-markdown-editor').forEach(function(el){
if(el.dataset.fmToolbar)return;el.dataset.fmToolbar='1';
var tb=document.createElement('div');tb.innerHTML=tbHTML();
tb.addEventListener('click',function(e){
var btn=e.target.closest('[data-md-action]');if(!btn)return;e.preventDefault();
var action=btn.getAttribute('data-md-action');
var textarea=el.querySelector('textarea');if(!textarea)return;
var s=snippet(action);var st=textarea.selectionStart,en=textarea.selectionEnd;
var sel=textarea.value.slice(st,en)||s.x;
textarea.setRangeText(s.p+sel+s.s,st,en,'select');
textarea.dispatchEvent(new Event('input',{bubbles:true}));
textarea.dispatchEvent(new Event('change',{bubbles:true}));
});
el.parentNode.insertBefore(tb,el);
});
}
inject();
new MutationObserver(inject).observe(root()||document.body,{childList:true,subtree:true});
})();
</script>"""


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


def _parse_maybe_scalar(value: str) -> list[str] | None:
    """Parse a multi-line input. Always returns a list (never a scalar)
    so that a single-element list in the original JSON round-trips as
    a list, not as a bare string.
    - None if the input is empty after stripping."""
    lines = [x.strip() for x in (value or "").split("\n") if x.strip()]
    if not lines:
        return None
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


def _check_model_download_status(data: dict) -> str:
    """Check which model files are already on disk.
    Returns a multi-line status string or empty string."""
    from shared.utils import files_locator as fl

    m = data.get("model", {})
    url_keys = ("URLs", "URLs2", "text_encoder_URLs", "VAE_URLs", "preload_URLs")
    statuses = []
    for key in url_keys:
        raw = m.get(key, [])
        if isinstance(raw, str):
            raw = [raw]
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                continue
            entry = entry.strip()
            fname = entry.rsplit("/", 1)[-1].split("?")[0]
            if not fname:
                continue
            try:
                existing = fl.locate_file(fname, error_if_none=False)
                if existing:
                    statuses.append(f"\U0001F4E6 {fname} — on disk")
                else:
                    statuses.append(f"\u2b07\ufe0f {fname} — needs download")
            except Exception:
                statuses.append(f"\u2753 {fname} — could not check")
    if not statuses:
        return ""
    return "\n".join(statuses)


def _validate_model_ref(model_id: str) -> tuple[bool, str]:
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
                dl_status = _check_model_download_status(data)
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
            dl_status = _check_model_download_status(data)
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


def _validate_entry(value: str) -> tuple[bool, str]:
    """Validate a single URL, model reference (=id), or local path.
    Returns (is_valid, message)."""
    if not value or not isinstance(value, str):
        return True, ""
    v = value.strip()
    if not v:
        return True, ""
    if v.startswith("="):
        return _validate_model_ref(v[1:])
    if v.startswith(("http:", "https:")):
        return _validate_url(v)
    if v.startswith(("/", ".", "\\")) or (len(v) > 1 and v[1] == ":"):
        return _validate_local_path(v)
    # Could be a bare model ID stored by _parse_maybe_scalar
    # Check MODEL_INDEX first for a fast cache hit
    if v in MODEL_INDEX:
        return True, f"Model ref: {MODEL_INDEX[v]}"
    data = _read_model_json(v)
    if data:
        name = data.get("model", {}).get("name", v)
        urls = _get_model_urls(data)
        if urls:
            dl_status = _check_model_download_status(data)
            msg = f"Model ref: {name}\n\u2192 {urls[0]}"
            if dl_status:
                msg += f"\n{dl_status}"
            return True, msg
        return True, f"Model ref: {name}"
    return False, (
        f"'{v}' not in Wan2GP model list \u2014 enter a URL or upload a file instead"
    )


def _validate_urls(urls: list[str]) -> list[tuple[str, bool, str]]:
    """Validate a list of URLs. Returns [(url, valid, message), ...].
    Skips non-URL entries."""
    results = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        valid, msg = _validate_entry(u)
        if msg or not valid:
            results.append((u, valid, msg))
    return results


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate that a URL is reachable via HEAD request.
    Returns (is_valid, message).
    """
    if not url or not isinstance(url, str) or not url.strip():
        return True, ""
    url = url.strip()
    if not url.startswith(("http:", "https:")):
        return True, "Not a remote URL (local path)"
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0 (compatible; FinetuneManager/3.0)")
        # Some servers reject HEAD on /resolve/main/ — try GET as fallback
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            code = resp.getcode()
        except urllib.error.HTTPError as e:
            code = e.code
            if code in (405, 501, 403):
                get_req = urllib.request.Request(url, method="GET")
                get_req.add_header(
                    "User-Agent", "Mozilla/5.0 (compatible; FinetuneManager/3.0)"
                )
                get_req.add_header("Range", "bytes=0-0")
                try:
                    get_resp = urllib.request.urlopen(get_req, timeout=15)
                    code = get_resp.getcode()
                except urllib.error.HTTPError as e2:
                    code = e2.code
        if code and code < 400:
            return True, f"HTTP {code}"
        elif code and code == 404:
            return False, "HTTP 404 — Not Found"
        elif code and code == 401:
            return False, "HTTP 401 — Unauthorized (may need token)"
        elif code and code == 403:
            return False, "HTTP 403 — Forbidden (may need token)"
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


def _download_finetune_file(url: str, target_dir: str | None = None) -> str:
    """Download a single file from a URL into the checkpoints directory.
    Returns a status string.
    """
    url = url.strip()
    if not url.startswith(("http:", "https:")):
        return f"Skipped (not remote): {url}"
    fname = url.rsplit("/", 1)[-1].split("?")[0]
    if not fname:
        fname = "download"
    from shared.utils.download import download_file
    from shared.utils import files_locator as fl

    if target_dir:
        dest_dir = Path(target_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = str(dest_dir / fname)
    else:
        dest = fl.get_download_location(fname)
    try:
        existing = fl.locate_file(fname, error_if_none=False)
        if existing:
            return f"Already exists: {fname}"
        download_file(url, dest)
        return f"Downloaded: {fname}"
    except Exception as e:
        return f"Failed: {fname} — {e}"


def _download_finetune_files(data: dict) -> str:
    """Download all URL-based files referenced in a finetune model dict.
    Returns a multi-line status string.
    """
    from shared.utils import files_locator as fl

    m = data.get("model", {})
    url_keys = DOWNLOAD_URL_KEYS
    all_urls = []
    for k in url_keys:
        v = m.get(k, []) or []
        if isinstance(v, str):
            v = [v]
        all_urls.extend(v)
    loras = m.get("loras", []) or []
    if isinstance(loras, str):
        loras = [loras]
    all_urls.extend(loras)

    lines = []
    for url in all_urls:
        if not url or not isinstance(url, str) or not url.strip():
            continue
        result = _download_finetune_file(url)
        lines.append(result)
    if not lines:
        return "No remote URLs found to download"
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# P1: CivitAI Integration
# ═══════════════════════════════════════════════════════════════════

CIVITAI_API = "https://civitai.com/api/v1"

CIVITAI_BASE_MODEL_MAP = {
    "SDXL 1.0": "sdxl",
    "SDXL 0.9": "sdxl",
    "SD 1.5": "sd15",
    "SD 1.4": "sd15",
    "SD 2.0": "sd2",
    "SD 2.1": "sd2",
    "Pony": "pony",
    "SDXL Turbo": "sdxl_turbo",
    "SD 3": "sd3",
    "SD 3.5": "sd35",
    "Flux.1 D": "flux_dev",
    "Flux.1 S": "flux_schnell",
    "Flux": "flux",
}


def _civitai_search(
    query: str, model_type: str = "", base_model: str = "", limit: int = 12
) -> dict:
    """Search CivitAI models. Returns raw API response dict with 'items' key."""
    params = [("query", query), ("limit", str(limit))]
    if model_type and model_type != "All":
        params.append(("types", model_type))
    if base_model and base_model != "All":
        params.append(("baseModels", base_model))
    url = f"{CIVITAI_API}/models?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (compatible; FinetuneManager/3.0)")
        resp = urllib.request.urlopen(req, timeout=20)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e), "items": []}


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
        mid = model.get("id", 0)
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
                vid = v.get("id", 0)
                vname = html.escape(v.get("name", "?") or "?")
                base = html.escape(v.get("baseModel", "") or "")
                files = v.get("files", [])
                primary = next(
                    (f for f in files if f.get("primary", False)),
                    files[0] if files else None,
                )
                fname = html.escape(primary.get("name", "") if primary else "")
                fsize = primary.get("sizeKB", 0) if primary else 0
                dl_url = html.escape(
                    v.get(
                        "downloadUrl", f"https://civitai.com/api/download/models/{vid}"
                    )
                )
                size_str = f"{fsize / 1024:.1f}MB" if fsize else "?"
                _civ_model_json = json.dumps(model, default=str).replace("'", "&#39;")
                _civ_ver_json = json.dumps(v, default=str).replace("'", "&#39;")
                parts.append(
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'padding:3px 0;font-size:12px">'
                    f'<span style="color:var(--body-text-color,#374151);flex:1">'
                    f"<b>{vname}</b> — {base} — {size_str}</span>"
                    f'<span style="color:var(--body-text-color-subdued,#6b7280);font-size:11px">{fname}</span>'
                    f'<button class="civitai-use-btn" '
                    f"data-civitai-model='{_civ_model_json}' "
                    f"data-civitai-version='{_civ_ver_json}' "
                    f'style="padding:2px 10px;border:1px solid #6366f1;'
                    f"border-radius:4px;background:var(--primary-100,#eef2ff);"
                    f'color:var(--primary-500,#4338ca);cursor:pointer;font-size:11px">Use</button>'
                    f"</div>"
                )
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")
    parts.append("""<script>
(function(){
if(window.__fmCivitaiHandler)return;window.__fmCivitaiHandler=true;
document.addEventListener('click',function(e){
var btn=e.target.closest('.civitai-use-btn');
if(!btn)return;e.preventDefault();
var version=JSON.parse(btn.getAttribute('data-civitai-version')||'{}');
var model=JSON.parse(btn.getAttribute('data-civitai-model')||'{}');
var data=JSON.stringify({model:model,version:version});
var r=(window.gradioApp?window.gradioApp():document.querySelector('gradio-app'))||document;r=r.shadowRoot||r;
var ta=r.querySelector('#civitai-selected textarea');
if(ta){ta.value=data;ta.dispatchEvent(new Event('input',{bubbles:true}));}
});
})();
</script>""")
    return "".join(parts)


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
        dl_url = f"https://civitai.com/api/download/models/{version.get('id', '')}"

    mtype = model.get("type", "").upper()
    # Put URL in main checkpoints for non-LoRA, loras field for LoRA
    if mtype in ("LORA", "DOJO"):
        # URL goes to loras field (index 12 in return tuple = ALL_INPUTS[13])
        result = [gr.update()] * 33
        result[12] = gr.update(value=dl_url)
        status = f"LoRA URL from {model_name}"
    else:
        # URL goes to main checkpoints field (index 4 in return tuple = ALL_INPUTS[5])
        result = [gr.update()] * 33
        result[4] = gr.update(value=dl_url)
        status = f"Download URL from {model_name}"

    return tuple(result + [status])


# ═══════════════════════════════════════════════════════════════════
# ponytail: inline helpers
# ═══════════════════════════════════════════════════════════════════


def _check_upload_token() -> bool:
    """Check if we have a valid HF token for registry uploads.
    Returns True if either config has a real token or the HF hub cache has one."""
    if REGISTRY_TOKEN and REGISTRY_TOKEN != "test_token_abc":
        return True
    try:
        api = HfApi()
        user = api.whoami()
        return bool(user and user.get("name"))
    except Exception:
        return False


def _hf_upload(fin_id, json_data):
    """Upload a finetune JSON to the HF Space.
    No index.json management -- the Browse tab discovers files dynamically."""
    api = HfApi()
    if REGISTRY_TOKEN and REGISTRY_TOKEN != "test_token_abc":
        api = HfApi(token=REGISTRY_TOKEN)
    safe_id = _sanitize_fin_id(fin_id)
    api.upload_file(
        path_or_fileobj=json.dumps(json_data, indent=2).encode(),
        path_in_repo=f"finetunes/{safe_id}.json",
        repo_id=REGISTRY_SPACE,
        repo_type="space",
    )


def _fetch_registry_json(fin_id):
    r = requests.get(f"{DEFAULT_REGISTRY}/finetunes/{fin_id}.json", timeout=10)
    r.raise_for_status()
    return r.json()


def _fetch_dynamic_registry_no_cache() -> list[dict]:
    """Dynamically list all finetunes from the HF Space by scanning
    actual files. No index.json needed -- always in sync."""
    try:
        api = HfApi()
        if REGISTRY_TOKEN and REGISTRY_TOKEN != "test_token_abc":
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
    return obj


def _sanitize_fin_id(fin_id: str) -> str:
    """Strip path traversal characters from a finetune ID.
    Only allows alphanumeric, underscore, dot, and hyphen.
    """
    cleaned = _re.sub(r"[^A-Za-z0-9_.-]+", "_", fin_id.strip())
    cleaned = _re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned if cleaned else "unnamed"


def _resolve_finetune_path(fin_id: str) -> Path | None:
    """Resolve a finetune JSON path and verify it's within FINETUNES_DIR.
    Returns None if the path escapes the directory."""
    base = Path(FINETUNES_DIR).resolve()
    p = (base / f"{fin_id}.json").resolve()
    try:
        p.relative_to(base)
        return p
    except ValueError:
        return None


def _write_finetune(fin_id, data):
    Path(FINETUNES_DIR).mkdir(parents=True, exist_ok=True)
    safe_id = _sanitize_fin_id(fin_id)
    (Path(FINETUNES_DIR) / f"{safe_id}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


CARD_CSS = """
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
/* JSON & Preview readable wrapping */
[data-testid="json"] .json-holder { max-height:450px !important; overflow-y:auto !important }
[data-testid="json"] .line .content { white-space:pre-wrap !important; word-break:break-word !important; overflow-wrap:break-word !important; line-height:1.65 !important; font-size:13px !important }
[data-testid="json"] .line, [data-testid="json"] .json-node { min-height:1.65em !important; height:auto !important; max-height:none !important }
/* CivatAI result cards */
.civitai-use-btn:hover { background:#6366f1 !important; color:#fff !important }
/* Loading spinner overlay for buttons */
.fm-spinner { display:inline-block; width:12px; height:12px; border:2px solid var(--border-color-primary,#e5e7eb); border-top-color:#6366f1; border-radius:50%; animation:fm-spin .6s linear infinite; margin-right:6px; vertical-align:middle }
@keyframes fm-spin { to { transform:rotate(360deg) } }
"""


class FinetuneManagerPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("refresh_model_defs")
        self.request_global("switch_to_model")
        self.request_component("state")
        self.request_component("model_choice_target")
        self.request_component("main_tabs")
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

        with gr.Tabs() as fm_tabs:
            # ── BROWS┬ ──
            with gr.TabItem("Browse", id="browse"):
                with gr.Row():
                    b_search = gr.Textbox(label="Search", scale=3, container=False)
                    b_arch = gr.Dropdown(
                        label="Architecture",
                        choices=[
                            "All",
                            "t2v",
                            "i2v",
                            "vace_14B",
                            "hunyuan",
                            "hunyuan_i2v",
                        ],
                        value="All",
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

                def _fmt_cards(fins, search, arch, sel, tag_filter):
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
                    _h = ['<div class="fm-cards-container">']
                    for f in ff:
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
                    fins = _get_cached_registry(force=True)
                    html, cnt = _fmt_cards(fins, "", "All", "", "")
                    tags = _all_tags(fins)
                    return (
                        fins,
                        "",
                        html,
                        cnt,
                        gr.update(choices=tags, value=""),
                        gr.update(value=""),
                        gr.update(value="All"),
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
                    ],
                )

                def _filter(fins, search, arch, sel, tag_filter):
                    html, cnt = _fmt_cards(fins, search, arch, sel, tag_filter)
                    return html, cnt

                b_search.input(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id, b_tag_filter],
                    outputs=[b_cards, b_count],
                )
                b_arch.change(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id, b_tag_filter],
                    outputs=[b_cards, b_count],
                )
                b_tag_filter.change(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id, b_tag_filter],
                    outputs=[b_cards, b_count],
                )

                fm_sel_input = gr.Textbox(visible=False, elem_id="fm-selected")

                def _browse_validate(fid, detail):
                    if not detail or not isinstance(detail, dict):
                        return (
                            "<div style='color:#6b7280;padding:4px'>"
                            "Select a card first</div>"
                        )
                    all_entries = _collect_url_entries(detail)
                    results = _validate_urls(all_entries)
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
                            _validate_urls(all_entries)
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

                def _load(fins, fid):
                    if not fid:
                        return "Select a card", gr.update(), gr.update()
                    m = next((f for f in fins if f["id"] == fid), None)
                    if not m:
                        return f"'{fid}' not found", gr.update(), gr.update()
                    try:
                        data = _fetch_registry_json(fid)
                    except Exception as e:
                        return f"Error: {e}", gr.update(), gr.update()
                    _write_finetune(fid, data)
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    t, tab = (
                        self.switch_to_model(fid, False)
                        if hasattr(self, "switch_to_model")
                        else (gr.update(), gr.update())
                    )
                    return f"Loaded '{m.get('name', fid)}' and switched", t, tab

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
                gr.HTML(MD_TOOLBAR_JS, visible=False)
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
                        choices=["t2v", "i2v", "vace_14B", "hunyuan", "hunyuan_i2v"],
                        allow_custom_value=True,
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
                    fin_save_up = gr.Button("Save & Upload")
                    fin_cancel_create = gr.Button("Cancel")
                with gr.Row(visible=False) as fin_editor_actions:
                    fin_save = gr.Button("Save Locally", variant="primary")
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
                            ok, msg = _validate_model_ref(mid)
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
                            ok, msg = _validate_model_ref(first_line)
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
                    inputs=[registry, b_sel_id],
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
                    v = _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
                    v = _re.sub(r"_+", "_", v).strip("._-")
                    return v

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
                    p = Path(FINETUNES_DIR)
                    existing = (
                        {f.stem for f in p.glob("*.json")} if p.exists() else set()
                    )
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
                fin_name.change(
                    fn=_auto_id,
                    inputs=[fin_name, fin_fsm, fin_auto_id, fin_id],
                    outputs=[fin_id],
                    queue=False,
                )
                fin_name.blur(
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
                    if "model" not in data:
                        return [gr.update()] * n_inputs + ["Missing model", {}]
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

                def _create_action(id_, *vals):
                    extra_data, remaining = _pop_extra(vals)
                    if extra_data is not None:
                        vals = remaining
                    if not id_:
                        return "Enter an ID", gr.update(), gr.update()
                    data = _build(*vals, extra_data=extra_data)
                    _write_finetune(id_, data)
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    t, tab = (
                        self.switch_to_model(id_, False)
                        if hasattr(self, "switch_to_model")
                        else (gr.update(), gr.update())
                    )
                    return f"Created {id_}", t, tab

                fin_create.click(
                    fn=_create_action,
                    inputs=list(ALL_INPUTS) + [fin_extra_data],
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
                    _write_finetune(id_, _build(*vals, extra_data=extra_data))
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    return f"Saved {id_}"

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
                        return "No HF token available — run `huggingface-cli login` or set registry_token in config.json"
                    data = _build(*vals, extra_data=extra_data)
                    _write_finetune(id_, data)
                    try:
                        _hf_upload(id_, data)
                        if (
                            hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs
                        ):
                            self.refresh_model_defs()
                        return f"Saved & uploaded {id_}"
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
                    # Check MODEL_INDEX first for a fast cache hit
                    if fsm in MODEL_INDEX:
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
                        return "Enter an ID"
                    p = _resolve_finetune_path(id_)
                    if not p or not p.exists():
                        return "Not found"
                    p.unlink()
                    if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                        self.refresh_model_defs()
                    return f"Deleted {id_}"

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
                    fn=_delete_action, inputs=[fin_id], outputs=[fin_status]
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
                    )

                fin_cancel_create.click(
                    fn=_cancel_action,
                    outputs=(
                        ALL_INPUTS
                        + [fin_status]
                        + [fin_creator_actions, fin_editor_actions]
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

                # Indices of URL/LoRA/path fields in ALL_INPUTS — now module-level URL_FIELD_IDX

                def _editor_validate(*vals):
                    # Extract raw text from URL/LoRA/path fields before _build
                    raw_entries = []
                    for i in URL_FIELD_IDX:
                        if i < len(vals) and isinstance(vals[i], str):
                            for line in vals[i].split("\n"):
                                line = line.strip()
                                if line:
                                    raw_entries.append(line)
                    # Also check model ref from fsm field (index 4)
                    if len(vals) > 4 and isinstance(vals[4], str) and vals[4].strip():
                        raw_entries.insert(0, f"={vals[4].strip()}")
                    # Aggregate results (for fin_validate_out)
                    all_results = _validate_urls(raw_entries)
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
                            valid, msg = _validate_entry(line)
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
                    return _download_finetune_files(data)

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
                        ok, msg = _validate_model_ref(mid)
                        icon = "\u2705" if ok else "\u274c"
                        val_html = f"<span style='font-size:12px'>{icon} {msg}</span>"
                        return (
                            gr.update(value=val_text),
                            gr.update(value=None),
                            gr.update(value=val_html),
                        )

                    return _handler

                def _textbox_to_dropdown(current, *refs):
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
                        ok, msg = _validate_model_ref(mid)
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
                        inputs=[tb, ref],
                        outputs=[ref, val_html],
                    )
                    ref.change(
                        fn=_make_insert_handler(tb),
                        inputs=[ref, tb],
                        outputs=[tb, ref, val_html],
                    )

            # ── LOCAL ──
            with gr.TabItem("Local", id="local"):
                l_refresh = gr.Button("Refresh", variant="primary")
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
                l_status = gr.Textbox(label="Status")

            # ── UPLOAD ──
            # ── CIVITAI ──
            with gr.TabItem("CivitAI", id="civitai"):
                gr.Markdown("Search CivitAI for models and auto-fill into Create/Edit.")
                with gr.Row():
                    ci_search = gr.Textbox(
                        label="Search",
                        scale=3,
                        container=False,
                        placeholder="e.g. fantasy landscape, character style...",
                    )
                    ci_type = gr.Dropdown(
                        label="Type",
                        choices=[
                            "All",
                            "LORA",
                            "Checkpoint",
                            "TextualInversion",
                            "Hypernetwork",
                            "AestheticGradient",
                            "ControlNet",
                            "Poses",
                            "Wildcards",
                            "MotionModule",
                            "VAE",
                        ],
                        value="All",
                        scale=1,
                    )
                    ci_base = gr.Dropdown(
                        label="Base Model",
                        choices=[
                            "All",
                            "SDXL 1.0",
                            "SDXL 0.9",
                            "SD 1.5",
                            "SD 2.0",
                            "SD 2.1",
                            "Pony",
                            "Flux.1 D",
                            "Flux.1 S",
                            "SD 3",
                            "SD 3.5",
                        ],
                        value="All",
                        scale=1,
                    )
                    ci_search_btn = gr.Button("Search", variant="primary", scale=1)
                with gr.Row():
                    ci_load_btn = gr.Button(
                        "Fill Selected into Create/Edit", variant="primary", scale=1
                    )
                    ci_clear_btn = gr.Button("Clear", scale=1)
                ci_status = gr.Markdown("Enter a search term to find models on CivitAI")
                ci_results = gr.HTML(
                    "<div style='color:#9ca3af;text-align:center;"
                    "padding:32px'>Click Search</div>"
                )
                ci_selected = gr.Textbox(
                    visible=False, elem_id="civitai-selected", value="{}"
                )

                def _ci_do_search(query, mtype, base):
                    if not query or not query.strip():
                        return (
                            "<div style='color:#9ca3af;"
                            "text-align:center;padding:32px'>"
                            "Enter a search term</div>",
                            "",
                        )
                    data = _civitai_search(
                        query.strip(), model_type=mtype, base_model=base
                    )
                    html_out = _civitai_render_results(data)
                    count = len(data.get("items", [])) if "error" not in data else 0
                    status = (
                        f"**{count} results** for '{html.escape(query.strip())}'"
                        if count
                        else "No results found"
                    )
                    return html_out, status

                ci_search_btn.click(
                    fn=_ci_do_search,
                    inputs=[ci_search, ci_type, ci_base],
                    outputs=[ci_results, ci_status],
                )
                ci_search.submit(
                    fn=_ci_do_search,
                    inputs=[ci_search, ci_type, ci_base],
                    outputs=[ci_results, ci_status],
                )

                def _ci_fill(civitai_json_str):
                    return _civitai_extract_fill_data(civitai_json_str)

                ci_load_btn.click(
                    fn=_ci_fill,
                    inputs=[ci_selected],
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
                        "{}",
                        "Enter a search term to find models on CivitAI",
                    )

                ci_clear_btn.click(
                    fn=_ci_clear, outputs=[ci_results, ci_selected, ci_status]
                )

            # ── SHARED LOCAL HANDLERS ──
            def _loc_list():
                p = Path(FINETUNES_DIR)
                if not p.exists():
                    return [], "*No finetunes directory*"
                fs = sorted(p.glob("*.json"))
                plural = "" if len(fs) == 1 else "s"
                return (
                    [(f.stem, f.stem) for f in fs],
                    f"**{len(fs)} local finetune{plural}**",
                )

            def _loc_detail(fid):
                if not fid:
                    return {}
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return {"error": "not found"}
                data = json.loads(p.read_text(encoding="utf-8"))
                return _clean_utf8(data)

            def _fmt_local_cards():
                p = Path(FINETUNES_DIR)
                if not p.exists():
                    return (
                        "<div style='color:#9ca3af;text-align:center;"
                        "padding:32px'>No finetunes directory</div>",
                        "*No finetunes directory*",
                    )
                fs = sorted(p.glob("*.json"))
                if not fs:
                    return (
                        "<div style='color:#9ca3af;text-align:center;"
                        "padding:32px'>No local finetunes</div>",
                        "**0 local finetunes**",
                    )
                _h = ['<div class="fm-cards-container">']
                for fpath in fs:
                    fid = fpath.stem
                    try:
                        data = json.loads(fpath.read_text(encoding="utf-8"))
                    except Exception:
                        data = {}
                    m = data.get("model", {})
                    _h.append(
                        _fmt_card_html_item(
                            fid=fid,
                            name=m.get("name", fid),
                            arch=m.get("architecture", ""),
                            author=m.get("author", "local"),
                            desc=m.get("description", ""),
                            urls=m.get("URLs", []),
                            loras=m.get("loras", []),
                            ftags=m.get("tags", []),
                            is_selected=False,
                            sel_elem_id="l-selected",
                            is_variant=bool(m.get("finetune_source_model")),
                        )
                    )
                _h.append("</div>")
                plural = "" if len(fs) == 1 else "s"
                return ("".join(_h), f"**{len(fs)} local finetune{plural}**")

            l_refresh.click(fn=_fmt_local_cards, outputs=[l_cards, l_status])
            l_sel_id.input(fn=_loc_detail, inputs=[l_sel_id], outputs=[l_detail])

            def _loc_load(fid):
                if not fid:
                    return "Select one", gr.update(), gr.update()
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return "Not found", gr.update(), gr.update()
                if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                    self.refresh_model_defs()
                t, tab = (
                    self.switch_to_model(fid, False)
                    if hasattr(self, "switch_to_model")
                    else (gr.update(), gr.update())
                )
                return f"Switched to '{fid}'", t, tab

            l_load.click(
                fn=_loc_load,
                inputs=[l_sel_id],
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
                fn=_fmt_local_cards, outputs=[l_cards, l_status]
            ).then(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                outputs=[l_del, l_del_confirm],
                queue=False,
            )

            l_export_btn = gr.DownloadButton("Export JSON", value=None)

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

            def _loc_import(file):
                if file is None:
                    cards_html, _ = _fmt_local_cards()
                    return "Select a .json", gr.update(), gr.update(), cards_html
                s = (
                    file
                    if isinstance(file, str)
                    else file.get("name") or file.get("path")
                )
                if not s or not Path(s).exists():
                    cards_html, _ = _fmt_local_cards()
                    return "Not found", gr.update(), gr.update(), cards_html
                try:
                    data = json.loads(Path(s).read_text(encoding="utf-8"))
                except Exception as e:
                    cards_html, _ = _fmt_local_cards()
                    return f"Invalid: {e}", gr.update(), gr.update(), cards_html
                if "model" not in data:
                    cards_html, _ = _fmt_local_cards()
                    return ("Missing 'model'", gr.update(), gr.update(), cards_html)
                fid = Path(s).stem
                existing_path = Path(FINETUNES_DIR) / f"{fid}.json"
                overwrite_note = (
                    " (overwrote existing)" if existing_path.exists() else ""
                )
                _write_finetune(fid, data)
                if hasattr(self, "refresh_model_defs") and self.refresh_model_defs:
                    self.refresh_model_defs()
                t, tab = (
                    self.switch_to_model(fid, False)
                    if hasattr(self, "switch_to_model")
                    else (gr.update(), gr.update())
                )
                cards_html, _ = _fmt_local_cards()
                return f"Imported '{fid}'{overwrite_note}", t, tab, cards_html

            l_imp_btn.click(
                fn=_loc_import,
                inputs=[l_imp_file],
                outputs=[l_status, self.model_choice_target, self.main_tabs, l_cards],
            )

            def _up(fid):
                if not fid:
                    return "Select one"
                if not _check_upload_token():
                    return "No HF token available — run `huggingface-cli login` or set registry_token in config.json"
                p = _resolve_finetune_path(fid)
                if not p or not p.exists():
                    return "Not found"
                _hf_upload(fid, json.loads(p.read_text(encoding="utf-8")))
                return f"Uploaded '{fid}'"

            l_up.click(fn=_up, inputs=[l_sel_id], outputs=[l_status])
