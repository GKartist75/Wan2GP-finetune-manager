import html
import json
import re as _re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import requests
import gradio as gr
from huggingface_hub import HfApi
from shared.utils.plugins import WAN2GPPlugin
from shared.gradio.local_file_picker import CHECKPOINT_FILE_EXTENSIONS, LocalFilePickerTextbox

PlugIn_Name = "Finetune Manager"
PlugIn_Id = "FinetuneManager"
PLUGIN_VERSION = "3.2.0"

DEFAULT_REGISTRY = "https://huggingface.co/spaces/GKartist75/wan2gp-finetunes/raw/main"
REGISTRY_SPACE = "GKartist75/wan2gp-finetunes"
FINETUNES_DIR = "finetunes"

_cfg_path = Path(__file__).parent / "config.json"
REGISTRY_TOKEN = ""
if _cfg_path.exists():
    try:
        REGISTRY_TOKEN = json.loads(_cfg_path.read_text(encoding="utf-8")).get("registry_token", "")
    except Exception:
        pass
print(f"[FM] v{PLUGIN_VERSION} Token:{bool(REGISTRY_TOKEN)}")

# Shared with the built-in Finetune Editor
LORA_FILE_EXTENSIONS = {".safetensors", ".sft"}

# Markdown toolbar JS (from built-in Finetune Editor)
MD_TOOLBAR_JS = """<script>
(function(){
if(window.__fmMdToolbar)return;window.__fmMdToolbar=true;
function snippet(a){
const m={bold:{p:"**",s:"**",x:"bold text"},italic:{p:"*",s:"*",x:"italic text"},heading:{p:"## ",s:"",x:"Heading"},list:{p:"- ",s:"",x:"item"},link:{p:"[",s:"](https://)",x:"link text"},code:{p:"`",s:"`",x:"code"}};
return m[a]||{p:"",s:"",x:""};
}
function toolbarHTML(){
return '<div style="display:flex;gap:4px;padding:3px 0">'+
'<button type="button" data-md-action="bold" style="width:26px;height:24px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:12px"><b>B</b></button>'+
'<button type="button" data-md-action="italic" style="width:26px;height:24px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:12px"><i>I</i></button>'+
'<button type="button" data-md-action="heading" style="width:26px;height:24px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:12px">H</button>'+
'<button type="button" data-md-action="list" style="width:26px;height:24px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:12px">\u2022</button>'+
'<button type="button" data-md-action="link" style="width:26px;height:24px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:12px">\u2197</button>'+
'<button type="button" data-md-action="code" style="width:26px;height:24px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:12px">`</button></div>';
}
function injectToolbars(){
document.querySelectorAll('.wangp-markdown-editor').forEach(function(el){
if(el.dataset.fmToolbar)return;el.dataset.fmToolbar='1';
var tb=document.createElement('div');tb.innerHTML=toolbarHTML();
tb.addEventListener('click',function(e){
var btn=e.target.closest('[data-md-action]');if(!btn)return;e.preventDefault();
var action=btn.getAttribute('data-md-action');
var textarea=el.querySelector('textarea');if(!textarea)return;
var s=snippet(action);var start=textarea.selectionStart,end=textarea.selectionEnd;
var selected=textarea.value.slice(start,end)||s.x;
textarea.setRangeText(s.p+selected+s.s,start,end,'select');
textarea.dispatchEvent(new Event('input',{bubbles:true}));
textarea.dispatchEvent(new Event('change',{bubbles:true}));
});
el.parentNode.insertBefore(tb,el);
});
}
injectToolbars();
new MutationObserver(injectToolbars).observe(document.body,{childList:true,subtree:true});
})();
</script>"""


# ═══════════════════════════════════════════════════════════════════
# P0: URL Validation
# ═══════════════════════════════════════════════════════════════════

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
                get_req.add_header("User-Agent", "Mozilla/5.0 (compatible; FinetuneManager/3.0)")
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


def _validate_urls(urls: list[str]) -> list[tuple[str, bool, str]]:
    """Validate a list of URLs. Returns [(url, valid, message), ...]."""
    results = []
    for u in urls:
        u = u.strip()
        if not u or not u.startswith(("http:", "https:")):
            continue
        valid, msg = _validate_url(u)
        results.append((u, valid, msg))
    return results


def _build_url_validation_html(results: list[tuple[str, bool, str]]) -> str:
    """Build HTML displaying per-URL validation results."""
    if not results:
        return "<div style='color:#6b7280;padding:4px'>No remote URLs to validate</div>"
    parts = ['<div style="max-height:300px;overflow-y:auto">']
    for url, valid, msg in results:
        color = "#16a34a" if valid else "#dc2626"
        icon = "\u2713" if valid else "\u2717"
        short = url.rstrip("/").split("/")[-1] if "/" in url else url
        if len(short) > 60:
            short = short[:57] + "..."
        parts.append(
            f'<div style="padding:3px 4px;border-bottom:1px solid #f3f4f6;font-size:12px">'
            f'<span style="color:{color};font-weight:600">{icon}</span> '
            f'<span title="{html.escape(url)}">{html.escape(short)}</span> '
            f'<span style="color:{color};font-size:11px">{html.escape(msg)}</span>'
            f'</div>'
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
    url_keys = ["URLs", "URLs2", "text_encoder_URLs", "VAE_URLs", "preload_URLs",
                "custom_url_1", "custom_url_2", "custom_url_3"]
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


def _civitai_search(query: str, model_type: str = "", base_model: str = "", limit: int = 12) -> dict:
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
        return f"<div style='color:#dc2626;padding:16px'>Error: {html.escape(error)}</div>"
    if not items:
        return "<div style='color:#9ca3af;text-align:center;padding:32px'>No results</div>"

    parts = ['<div style="max-height:65vh;overflow-y:auto">']
    for model in items:
        mid = model.get("id", 0)
        name = html.escape(model.get("name", "?") or "?")
        mtype = html.escape(model.get("type", "?") or "?")
        desc = html.escape((model.get("description", "") or "")[:200])
        nsfw = model.get("nsfw", False)
        versions = model.get("modelVersions", [])

        nsfw_html = (' <span style="color:#dc2626;'
                    'font-size:10px">NSFW</span>' if nsfw else '')
        parts.append(
            f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px;'
            f'margin-bottom:6px;background:#fff">'
            f'<div style="font-size:13px;font-weight:600;color:#111827">'
            f'{name}{nsfw_html}'
            f' <span style="font-size:10px;color:#6b7280;font-weight:400">({mtype})</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#374151;line-height:1.4;'
            f'max-height:2.6em;overflow:hidden">{desc}</div>'
        )

        if versions:
            parts.append('<div style="margin-top:6px;padding-top:4px;border-top:1px solid #f3f4f6">')
            for v in versions:
                vid = v.get("id", 0)
                vname = html.escape(v.get("name", "?") or "?")
                base = html.escape(v.get("baseModel", "") or "")
                files = v.get("files", [])
                primary = next((f for f in files if f.get("primary", False)), files[0] if files else None)
                fname = html.escape(primary.get("name", "") if primary else "")
                fsize = primary.get("sizeKB", 0) if primary else 0
                dl_url = html.escape(v.get("downloadUrl",
                    f"https://civitai.com/api/download/models/{vid}"))
                size_str = f"{fsize / 1024:.1f}MB" if fsize else "?"
                parts.append(
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'padding:3px 0;font-size:12px">'
                    f'<span style="color:#374151;flex:1">'
                    f'<b>{vname}</b> — {base} — {size_str}</span>'
                    f'<span style="color:#6b7280;font-size:11px">{fname}</span>'
                    f'<button class="civitai-use-btn" '
                    f'data-civitai-model=\'{json.dumps(model, default=str)}\' '
                    f'data-civitai-version=\'{json.dumps(v, default=str)}\' '
                    f'style="padding:2px 10px;border:1px solid #6366f1;'
                    f'border-radius:4px;background:#eef2ff;'
                    f'color:#4338ca;cursor:pointer;font-size:11px">Use</button>'
                    f'</div>'
                )
            parts.append('</div>')
        parts.append('</div>')
    parts.append('</div>')
    parts.append("""<script>
(function(){
if(window.__fmCivitaiHandler)return;window.__fmCivitaiHandler=true;
document.addEventListener('click',function(e){
var btn=e.target.closest('.civitai-use-btn');
if(!btn)return;e.preventDefault();
var version=JSON.parse(btn.getAttribute('data-civitai-version')||'{}');
var model=JSON.parse(btn.getAttribute('data-civitai-model')||'{}');
var data=JSON.stringify({model:model,version:version});
var ta=document.querySelector('#civitai-selected textarea');
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
        return tuple([gr.update()] * 32 + [""])

    try:
        sel = json.loads(civitai_json_str)
    except (json.JSONDecodeError, TypeError):
        return tuple([gr.update()] * 32 + [""])

    model = sel.get("model", {})
    version = sel.get("version", {})
    model_name = model.get("name", "?") or "?"

    files = version.get("files", [])
    primary = next((f for f in files if f.get("primary", False)),
                   files[0] if files else None)
    dl_url = primary.get("downloadUrl", version.get("downloadUrl", "")) if primary else version.get("downloadUrl", "")
    if not dl_url:
        dl_url = f"https://civitai.com/api/download/models/{version.get('id', '')}"

    mtype = model.get("type", "").upper()
    # Put URL in main checkpoints for non-LoRA, loras field for LoRA
    if mtype in ("LORA", "DOJO"):
        # URL goes to loras field (index 12 in ALL_INPUTS, index 12 in return tuple)
        result = [gr.update()] * 32
        result[12] = gr.update(value=dl_url)
        status = f"LoRA URL from {model_name}"
    else:
        # URL goes to main checkpoints field (index 4 in return tuple)
        result = [gr.update()] * 32
        result[4] = gr.update(value=dl_url)
        status = f"Download URL from {model_name}"

    return tuple(result + [status])


# ═══════════════════════════════════════════════════════════════════
# ponytail: inline helpers
# ═══════════════════════════════════════════════════════════════════

def _hf_upload(fin_id, json_data):
    """Upload a finetune JSON to the HF Space.
    No index.json management -- the Browse tab discovers files dynamically."""
    api = HfApi(token=REGISTRY_TOKEN)
    api.upload_file(
        path_or_fileobj=json.dumps(json_data, indent=2).encode(),
        path_in_repo=f"finetunes/{fin_id}.json",
        repo_id=REGISTRY_SPACE, repo_type="space")


def _fetch_registry_json(fin_id):
    r = requests.get(
        f"{DEFAULT_REGISTRY}/finetunes/{fin_id}.json", timeout=10)
    r.raise_for_status()
    return r.json()


def _fetch_dynamic_registry():
    """Dynamically list all finetunes from the HF Space by scanning
    actual files. No index.json needed -- always in sync."""
    try:
        api = HfApi()
        files = api.list_repo_files(
            repo_id=REGISTRY_SPACE, repo_type="space")
        fin_files = [
            f for f in files
            if f.startswith("finetunes/") and f.endswith(".json")]
    except Exception:
        return []
    fins = []
    for path in sorted(fin_files):
        fid = path[len("finetunes/"):-len(".json")]
        try:
            r = requests.get(
                f"{DEFAULT_REGISTRY}/finetunes/{fid}.json", timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            m = data.get("model", {})
            tags_entry = m.get("tags", [])
            if isinstance(tags_entry, str):
                tags_entry = [t.strip() for t in
                              tags_entry.split(",") if t.strip()]
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
                        "num_inference_steps",
                        m.get("num_inference_steps", 30)),
                    "guidance_scale": data.get(
                        "guidance_scale",
                        m.get("guidance_scale", 5.0))
                },
                "raw": data
            }
            fins.append(entry)
        except Exception:
            continue
    return fins


def _write_finetune(fin_id, data):
    Path(FINETUNES_DIR).mkdir(parents=True, exist_ok=True)
    (Path(FINETUNES_DIR) / f"{fin_id}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")


CARD_CSS = """
.fm-cards-container { max-height:65vh; overflow-y:auto; padding-right:6px }
.fm-card { border:1px solid #e5e7eb; border-radius:8px; padding:12px; margin-bottom:6px; cursor:pointer; background:#fff; transition:border-color .15s,box-shadow .15s }
.fm-card:hover { border-color:#6366f1; box-shadow:0 1px 4px rgba(99,102,241,.12) }
.fm-card.selected { border-color:#6366f1; background:#eef2ff }
.fm-card-title { font-size:14px; font-weight:600; color:#111827; margin:0 0 2px }
.fm-card-meta { font-size:11px; color:#6b7280; margin:0 0 4px }
.fm-card-desc { font-size:12px; color:#374151; margin:0 0 4px; line-height:1.4; max-height:3.8em; overflow:hidden; text-overflow:ellipsis }
.fm-card-files { font-size:11px; color:#6b7280; margin:0; padding-top:4px; border-top:1px solid #f3f4f6 }
.fm-card-files a { color:#6366f1; text-decoration:none; word-break:break-all }
.fm-card-files a:hover { text-decoration:underline }
.fm-card-loras { font-size:11px; color:#6b7280; margin:0; padding-top:2px }
.fm-badge { display:inline-block; font-size:10px; padding:1px 6px; border-radius:10px; background:#e0e7ff; color:#4338ca; margin-right:4px }
/* JSON & Preview readable wrapping */
[data-testid="json"] .json-holder { max-height:450px !important; overflow-y:auto !important }
[data-testid="json"] .line .content { white-space:pre-wrap !important; word-break:break-word !important; overflow-wrap:break-word !important; line-height:1.65 !important; font-size:13px !important }
[data-testid="json"] .line, [data-testid="json"] .json-node { min-height:1.65em !important; height:auto !important; max-height:none !important }
/* CivatAI result cards */
.civitai-use-btn:hover { background:#6366f1 !important; color:#fff !important }
"""


class FinetuneManagerPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("refresh_model_defs")
        self.request_global("switch_to_model")
        self.request_component("state")
        self.request_component("model_choice_target")
        self.request_component("main_tabs")
        self.add_tab(
            tab_id=PlugIn_Id,
            label=PlugIn_Name,
            component_constructor=self.create_ui
        )

    def create_ui(self, api_session):
        gr.HTML(f"<style>{CARD_CSS}</style>")
        registry = gr.State([])
        gr.Markdown(f"**Finetune Manager v{PLUGIN_VERSION}** — "
                    f"[HF Registry](https://huggingface.co/spaces/{REGISTRY_SPACE})")

        with gr.Tabs() as fm_tabs:

            # ── BROWS┬ ──
            with gr.TabItem("Browse", id="browse"):
                with gr.Row():
                    b_search = gr.Textbox(label="Search", scale=3,
                                          container=False)
                    b_arch = gr.Dropdown(
                        label="Architecture",
                        choices=["All", "t2v", "i2v", "vace_14B",
                                 "hunyuan", "hunyuan_i2v"],
                        value="All", scale=1)
                    b_tag_filter = gr.Dropdown(
                        label="Tag", choices=[], value="",
                        allow_custom_value=True, scale=1)
                b_refresh = gr.Button("Refresh Registry")
                b_count = gr.Markdown("Click *Refresh* to load")
                b_sel_id = gr.State("")
                b_cards = gr.HTML(
                    "<div style='color:#9ca3af;text-align:center;"
                    "padding:32px'>Click Refresh</div>")
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
                                tg = [t.strip() for t in tg.split(",")
                                      if t.strip()]
                            if not isinstance(tg, list):
                                return False
                            return any(s in (tag or "").lower()
                                       for tag in tg)
                        ff = [f for f in ff
                              if s in f.get("name", "").lower()
                              or s in (f.get("description", "") or "").lower()
                              or s in f.get("author", "").lower()
                              or _match_tags(f)]
                    if arch and arch != "All":
                        ff = [f for f in ff
                              if f.get("architecture") == arch]
                    if tag_filter:
                        tf = tag_filter.strip().lower()
                        if tf:
                            def _filter_tags(f):
                                tg = f.get("tags", [])
                                if isinstance(tg, str):
                                    tg = [t.strip() for t in tg.split(",")
                                          if t.strip()]
                                if not isinstance(tg, list):
                                    return False
                                return any((tag or "").strip().lower() == tf
                                           for tag in tg)
                            ff = [f for f in ff if _filter_tags(f)]
                    if not ff:
                        return ("<div style='color:#9ca3af;text-align:center;"
                                "padding:32px'>No results</div>",
                                "**0 matches**")
                    _h = ['<div class="fm-cards-container">']
                    for f in ff:
                        fid = f.get("id", "")
                        c = " selected" if fid == sel else ""
                        name = html.escape(f.get("name", "?") or "?")
                        arch_s = html.escape(
                            f.get("architecture", "") or "")
                        author = html.escape(
                            f.get("author", "") or "")
                        desc = html.escape(
                            (f.get("description", "") or "")[:300])
                        tag = "Variant" if f.get("source") else arch_s
                        ftags = f.get("tags", [])
                        if isinstance(ftags, str):
                            ftags = [t.strip() for t in ftags.split(",")
                                     if t.strip()]
                        tags_badges = ""
                        if ftags:
                            tags_badges = '<div style="margin-top:2px">'
                            for t in ftags[:4]:
                                et = html.escape(t.strip())
                                tags_badges += (
                                    f'<span class="fm-badge">{et}</span> ')
                            if len(ftags) > 4:
                                tags_badges += (
                                    f'<span style="color:#9ca3af;'
                                    f'font-size:10px">'
                                    f'+{len(ftags)-4}</span>')
                            tags_badges += '</div>'
                        fid_safe = (fid.replace("\\", "\\\\")
                                       .replace("'", "\\'")
                                       .replace('"', '\\"'))
                        onclick = (
                            "document.querySelector('#fm-selected textarea')"
                            f".value='{fid_safe}';"
                            "document.querySelector('#fm-selected textarea')"
                            ".dispatchEvent(new Event('input',"
                            "{bubbles:true}))")
                        urls = f.get("URLs", [])
                        if isinstance(urls, str):
                            urls = [urls]
                        files_html = ""
                        if urls:
                            shown = urls[:3]
                            extra = len(urls) - 3
                            link_parts = []
                            for u in shown:
                                eu = html.escape(u)
                                short = (u.rstrip("/").split("/")[-1]
                                         if "/" in u else u)
                                if len(short) > 50:
                                    short = short[:47] + "..."
                                link_parts.append(
                                    f'<a href="{eu}" target="_blank" '
                                    f'rel="noopener">'
                                    f'{html.escape(short)}</a>')
                            files_html = ('<div class="fm-card-files">'
                                          + " ".join(link_parts))
                            if extra > 0:
                                files_html += (
                                    f' <span style="color:#9ca3af">'
                                    f'+{extra} more</span>')
                            files_html += '</div>'
                        loras = f.get("loras", [])
                        if isinstance(loras, str):
                            loras = [loras]
                        loras_html = ""
                        if loras:
                            shown_l = loras[:3]
                            extra_l = len(loras) - 3
                            loras_html = (
                                '<div class="fm-card-loras">'
                                '<b>LoRAs:</b> ')
                            loras_html += " ".join(
                                html.escape(l) for l in shown_l)
                            if extra_l > 0:
                                loras_html += (
                                    f' <span style="color:#9ca3af">'
                                    f'+{extra_l} more</span>')
                            loras_html += '</div>'
                        _h.append(
                            f'<div class="fm-card{c}" onclick="{onclick}">'
                            f"<div class='fm-card-title'>{name}</div>"
                            f"<div class='fm-card-meta'>"
                            f"<span class='fm-badge'>"
                            f"{html.escape(tag)}</span> {author}</div>"
                            f"<div class='fm-card-desc'>{desc}</div>"
                            f"{tags_badges}{files_html}{loras_html}"
                            f"</div>")
                    _h.append('</div>')
                    cnt_label = "match" if len(ff) == 1 else "matches"
                    return ("".join(_h),
                            f"**{len(ff)} {cnt_label}**")

                def _all_tags(fins):
                    seen = set()
                    for f in fins:
                        tg = f.get("tags", [])
                        if isinstance(tg, str):
                            tg = [t.strip() for t in tg.split(",")
                                  if t.strip()]
                        if isinstance(tg, list):
                            for t in tg:
                                t = t.strip().lower()
                                if t:
                                    seen.add(t)
                    return sorted(seen)

                def _do_refresh():
                    fins = _fetch_dynamic_registry()
                    html, cnt = _fmt_cards(fins, "", "All", "", "")
                    tags = _all_tags(fins)
                    return (fins, "", html, cnt,
                            gr.update(choices=tags, value=""))

                b_refresh.click(
                    fn=_do_refresh,
                    outputs=[registry, b_sel_id, b_cards, b_count,
                             b_tag_filter])

                def _filter(fins, search, arch, sel, tag_filter):
                    html, cnt = _fmt_cards(
                        fins, search, arch, sel, tag_filter)
                    return html, cnt

                b_search.input(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id,
                            b_tag_filter],
                    outputs=[b_cards, b_count])
                b_arch.change(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id,
                            b_tag_filter],
                    outputs=[b_cards, b_count])
                b_tag_filter.change(
                    fn=_filter,
                    inputs=[registry, b_search, b_arch, b_sel_id,
                            b_tag_filter],
                    outputs=[b_cards, b_count])

                fm_sel_input = gr.Textbox(visible=False,
                                          elem_id="fm-selected")

                def _on_selected(fins, fid):
                    m = next((f for f in fins if f["id"] == fid), None)
                    if m and "raw" in m:
                        detail = m["raw"]
                    else:
                        detail = m if m else {}
                    return fid, detail

                fm_sel_input.input(
                    fn=_on_selected,
                    inputs=[registry, fm_sel_input],
                    outputs=[b_sel_id, b_detail])

                def _browse_validate(fid, detail):
                    if not detail or not isinstance(detail, dict):
                        return ("<div style='color:#6b7280;padding:4px'>"
                                "Select a card first</div>")
                    if "model" in detail:
                        m = detail["model"]
                    else:
                        m = detail
                    urls = m.get("URLs", [])
                    if isinstance(urls, str):
                        urls = [urls]
                    loras = m.get("loras", [])
                    if isinstance(loras, str):
                        loras = [loras]
                    all_urls = list(urls) + list(loras)
                    results = _validate_urls(all_urls)
                    return _build_url_validation_html(results)

                b_validate_btn.click(
                    fn=_browse_validate,
                    inputs=[b_sel_id, b_detail],
                    outputs=[b_validate_out])

                def _on_selected_clear_validate(fins, fid):
                    m = next((f for f in fins if f["id"] == fid), None)
                    if m and "raw" in m:
                        detail = m["raw"]
                    else:
                        detail = m if m else {}
                    return fid, detail, ""

                fm_sel_input.input(
                    fn=_on_selected_clear_validate,
                    inputs=[registry, fm_sel_input],
                    outputs=[b_sel_id, b_detail, b_validate_out])

                def _load(fins, fid):
                    if not fid:
                        return "Select a card", gr.update(), gr.update()
                    m = next((f for f in fins if f["id"] == fid), None)
                    if not m:
                        return (f"'{fid}' not found",
                                gr.update(), gr.update())
                    try:
                        data = _fetch_registry_json(fid)
                    except Exception as e:
                        return f"Error: {e}", gr.update(), gr.update()
                    _write_finetune(fid, data)
                    if (hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs):
                        self.refresh_model_defs()
                    t, tab = (
                        self.switch_to_model(fid, False)
                        if hasattr(self, "switch_to_model")
                        else (gr.update(), gr.update()))
                    return f"Loaded '{m.get('name', fid)}'", t, tab

                b_load.click(
                    fn=_load,
                    inputs=[registry, b_sel_id],
                    outputs=[b_status, self.model_choice_target,
                             self.main_tabs])

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
                    if (hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs):
                        self.refresh_model_defs()
                    return f"Downloaded '{m.get('name', fid)}' locally"

                b_dl_only.click(
                    fn=_browse_dl,
                    inputs=[registry, b_sel_id],
                    outputs=[b_status])

                # ── Browse: Improve / Create Variant → fill Create/Edit tab ──
                # (click wiring is done after ALL_INPUTS is defined below)
                def _improve_to_editor(fins, fid, inputs_len):
                    if not fid:
                        return [gr.update()] * (inputs_len + 1) + [gr.update()]
                    m = next((f for f in fins if f["id"] == fid), None)
                    if not m:
                        return [gr.update()] * (inputs_len + 1) + [f"'{fid}' not found"]
                    try:
                        data = _fetch_registry_json(fid)
                    except Exception as e:
                        return [gr.update()] * (inputs_len + 1) + [f"Error: {e}"]
                    data.setdefault("model", {})["finetune_source_model"] = (
                        data.get("model", {}).get("architecture", ""))
                    vid = f"{fid}_variant"
                    _write_finetune(vid, data)
                    if (hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs):
                        self.refresh_model_defs()
                    fill_vals = list(_extract(data))
                    result = [gr.update()] * (inputs_len)
                    result[0] = gr.update(value=vid)
                    if len(fill_vals) >= 31:
                        for j in range(min(31, inputs_len - 1)):
                            result[j + 1] = gr.update(value=fill_vals[j])
                    return result + [
                        gr.update(selected="editor"),
                        f"Variant '{vid}' loaded into Create/Edit tab"
                    ]

            # ── CREATE / EDIT ──
            with gr.TabItem("Create / Edit", id="editor"):
                gr.HTML(MD_TOOLBAR_JS, visible=False)
                gr.Markdown(
                    "### Create or Edit \u2014 "
                    "matches the built-in Finetune Editor (Alt+F)")

                with gr.Row():
                    fin_import_file = gr.File(
                        label="Import JSON", file_types=[".json"], scale=2)
                    fin_import_btn = gr.Button("Fill from JSON", scale=1)
                fin_fsm = gr.Textbox(
                    label="Source Model (finetune_source_model)")
                with gr.Row():
                    fin_id = gr.Textbox(label="ID (filename)", scale=7)
                    fin_auto_id = gr.Checkbox(
                        label="auto", value=True, scale=1, min_width=80)
                fin_name = gr.Textbox(label="Name")
                fin_arch = gr.Dropdown(
                    label="Architecture",
                    choices=["t2v", "i2v", "vace_14B", "hunyuan",
                             "hunyuan_i2v"],
                    allow_custom_value=True)
                fin_desc = gr.Textbox(label="Description", lines=3)
                fin_tags = gr.Textbox(
                    label="Tags (comma-separated)", lines=1,
                    placeholder="character, style, anime, "
                                "photorealistic, ...")

                fin_lora_root = gr.Textbox(value="", visible=False)
                fin_creator_source_mode = gr.Radio(
                    label="Create New Finetune",
                    choices=[
                        ("Using Current Model "
                         "(via Improve / Create Variant)", "current")],
                    value="current",
                    visible=False)
                fin_use_current_settings = gr.Checkbox(
                    label="Use Current Model Settings as "
                          "Default Settings",
                    value=False)

                with gr.Tabs():
                    with gr.Tab("URLs"):
                        fin_urls = LocalFilePickerTextbox(
                            label="Main Checkpoints",
                            file_extensions=CHECKPOINT_FILE_EXTENSIONS,
                            multiselect=True,
                            popup_title="Select Checkpoints").mount()
                        fin_urls2 = LocalFilePickerTextbox(
                            label="Secondary Checkpoints",
                            file_extensions=CHECKPOINT_FILE_EXTENSIONS,
                            multiselect=True,
                            popup_title="Select Checkpoints").mount()
                        fin_te_urls = LocalFilePickerTextbox(
                            label="Text Encoder Checkpoints",
                            file_extensions=CHECKPOINT_FILE_EXTENSIONS,
                            multiselect=True,
                            popup_title="Select Text Encoder").mount()
                        fin_vae_urls = gr.Textbox(
                            label="VAE_URLs (one/line)", lines=2)
                        fin_preload = gr.Textbox(
                            label="preload_URLs (one/line)", lines=2)
                        fin_custom_url_1 = gr.Textbox(
                            label="custom_url_1", lines=1)
                        fin_custom_url_2 = gr.Textbox(
                            label="custom_url_2", lines=1)
                        fin_custom_url_3 = gr.Textbox(
                            label="custom_url_3", lines=1)
                        with gr.Row():
                            fin_validate_urls_btn = gr.Button(
                                "Validate URLs", scale=1)
                            fin_download_btn = gr.Button(
                                "Download Missing Files", scale=1,
                                variant="primary")
                        fin_validate_out = gr.HTML("")
                        fin_download_out = gr.Textbox(
                            label="Download Status", lines=4, max_lines=10)
                    with gr.Tab("LoRAs"):
                        fin_loras = LocalFilePickerTextbox(
                            label="Always Loaded LoRAs",
                            file_extensions=LORA_FILE_EXTENSIONS,
                            multiselect=True,
                            popup_title="Select LoRA Files",
                            default_dir_input=fin_lora_root,
                            compress_root_input=fin_lora_root).mount()
                        fin_lmults = gr.Textbox(
                            label="LoRAs Multipliers (space-separated)",
                            lines=1)
                    with gr.Tab("Resolutions"):
                        fin_rescat = gr.Textbox(
                            label="Resolution Categories Conditions "
                                  "(OR per line)",
                            lines=4,
                            placeholder=">=720&<=1440")
                        gr.HTML(
                            "<div style='font-size:11px;color:#6b7280;"
                            "margin:-8px 0 8px'>Example: "
                            "<code>&gt;=720&amp;&lt;=1440</code> "
                            "keeps 720p\u20131440p. "
                            "Separate lines are OR.</div>")
                        fin_res = gr.Textbox(
                            label="Custom Resolutions "
                                  "(one WxH per line)",
                            lines=4, placeholder="1024x2048")
                        gr.HTML(
                            "<div style='font-size:11px;color:#6b7280;"
                            "margin:-8px 0 8px'>Example: "
                            "<code>1024x2048</code> \u2192 saved as "
                            "<code>1024x2048 (1:2)</code>.</div>")
                    with gr.Tab("Help"):
                        fin_infos = gr.Textbox(
                            label="Model Infos (markdown)", lines=8,
                            elem_classes=["wangp-markdown-editor"])
                        fin_pinfos = gr.Textbox(
                            label="Prompt Help (markdown)", lines=8,
                            elem_classes=["wangp-markdown-editor"])
                    with gr.Tab("Prompt Enhancer"):
                        with gr.Column():
                            fin_pe_txt = gr.Textbox(
                                label="System Prompt (Text)", lines=4)
                            fin_pe_txt_tok = gr.Textbox(
                                label="Max Tokens \u2014 Text "
                                      "(empty = auto)", lines=1)
                        with gr.Column():
                            fin_pe_vid = gr.Textbox(
                                label="System Prompt (Video)", lines=4)
                            fin_pe_vid_tok = gr.Textbox(
                                label="Max Tokens \u2014 Video "
                                      "(empty = auto)", lines=1)
                        with gr.Column():
                            fin_pe_img = gr.Textbox(
                                label="System Prompt (Image)", lines=4)
                            fin_pe_img_tok = gr.Textbox(
                                label="Max Tokens \u2014 Image "
                                      "(empty = auto)", lines=1)
                    with gr.Tab("Settings"):
                        fin_modules = gr.Textbox(
                            label="Modules (comma-separated)")
                        fin_autoq = gr.Checkbox(label="auto_quantize")
                        fin_visible = gr.Checkbox(
                            label="visible", value=True)
                        fin_imgout = gr.Checkbox(label="image_outputs")
                        fin_steps = gr.Slider(
                            label="num_inference_steps",
                            minimum=1, maximum=200, value=30, step=1)
                        fin_guidance = gr.Slider(
                            label="guidance_scale",
                            minimum=1.0, maximum=30.0, value=5.0, step=0.5)

                fin_preview = gr.JSON(label="Preview")
                fin_mode = gr.State("creator")

                with gr.Row(visible=True) as fin_creator_actions:
                    fin_create = gr.Button("Create", variant="primary")
                    fin_create_new = gr.Button("Create & New")
                    fin_cancel_create = gr.Button("Cancel")
                with gr.Row(visible=False) as fin_editor_actions:
                    fin_save = gr.Button(
                        "Save Locally", variant="primary")
                    fin_export = gr.DownloadButton("Export", value=None)
                    fin_save_up = gr.Button("Save & Upload")
                    fin_del = gr.Button("Delete", variant="stop")
                    fin_cancel_edit = gr.Button("Cancel")
                fin_status = gr.Textbox(label="Status")
                fin_del_confirm = gr.Row(visible=False)
                with fin_del_confirm:
                    fin_del_confirm_btn = gr.Button(
                        "Confirm Delete", variant="stop")
                    fin_del_cancel_btn = gr.Button("Cancel")

                def _build(name, arch, desc, fsm, urls, urls2,
                           te, vae, pre, cu1, cu2, cu3,
                           loras, lm, mods, aq, vis, img, res, resc,
                           inf, pinf, steps, guid,
                           pe1, pe1t, pe2, pe2t, pe3, pe3t,
                           tags=""):
                    m = {"name": name,
                         "architecture": arch,
                         "description": desc}
                    if fsm:
                        m["finetune_source_model"] = fsm
                    if urls:
                        m["URLs"] = [x.strip()
                                     for x in urls.split("\n") if x.strip()]
                    if urls2:
                        m["URLs2"] = [x.strip()
                                      for x in urls2.split("\n") if x.strip()]
                    if te:
                        m["text_encoder_URLs"] = [
                            x.strip() for x in te.split("\n") if x.strip()]
                    if vae:
                        m["VAE_URLs"] = [
                            x.strip() for x in vae.split("\n") if x.strip()]
                    if pre:
                        m["preload_URLs"] = [
                            x.strip() for x in pre.split("\n") if x.strip()]
                    if cu1:
                        m["custom_url_1"] = cu1
                    if cu2:
                        m["custom_url_2"] = cu2
                    if cu3:
                        m["custom_url_3"] = cu3
                    if loras:
                        parts = [x.strip()
                                 for x in loras.split("\n") if x.strip()]
                        m["loras"] = parts
                    if lm:
                        m["loras_multipliers"] = [
                            x.strip()
                            for x in lm.replace(",", " ").split() if x.strip()]
                    if mods:
                        m["modules"] = [
                            x.strip() for x in mods.split(",") if x.strip()]
                    if aq:
                        m["auto_quantize"] = True
                    if not vis:
                        m["visible"] = False
                    if img:
                        m["image_outputs"] = True
                    if tags:
                        m["tags"] = [t.strip()
                                     for t in tags.split(",") if t.strip()]
                    if res:
                        out = []
                        for l in res.split("\n"):
                            l = l.strip()
                            if not l:
                                continue
                            if "x" in l.lower():
                                match = _re.match(
                                    r'(\d+)\s*x\s*(\d+)', l, _re.IGNORECASE)
                                if match:
                                    out.append(
                                        [l,
                                         f"{match.group(1)}x{match.group(2)}"])
                                else:
                                    out.append(l)
                            else:
                                out.append(l)
                        m["resolutions"] = out
                    if resc:
                        m["resolutions_categories"] = [
                            x.strip()
                            for x in resc.split("\n") if x.strip()]
                    if inf:
                        m["infos"] = inf
                    if pinf:
                        m["prompt_infos"] = pinf
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
                    return dict(model=m,
                                num_inference_steps=int(steps),
                                guidance_scale=float(guid))

                ALL_INPUTS = [
                    fin_id, fin_name, fin_arch, fin_desc, fin_fsm,
                    fin_urls, fin_urls2, fin_te_urls,
                    fin_vae_urls, fin_preload,
                    fin_custom_url_1, fin_custom_url_2, fin_custom_url_3,
                    fin_loras, fin_lmults, fin_modules,
                    fin_autoq, fin_visible, fin_imgout,
                    fin_res, fin_rescat,
                    fin_infos, fin_pinfos,
                    fin_steps, fin_guidance,
                    fin_pe_txt, fin_pe_txt_tok,
                    fin_pe_vid, fin_pe_vid_tok,
                    fin_pe_img, fin_pe_img_tok,
                    fin_tags,
                ]

                # Wire Improve / Create Variant button (needs ALL_INPUTS defined)
                b_improve.click(
                    fn=_improve_to_editor,
                    inputs=[registry, b_sel_id, gr.State(len(ALL_INPUTS))],
                    outputs=ALL_INPUTS + [fm_tabs, b_status])

                def _preview(*vals):
                    return _build(*vals[1:])

                for f in ALL_INPUTS:
                    f.change(
                        fn=_preview,
                        inputs=ALL_INPUTS,
                        outputs=[fin_preview])

                def _sanitize_id(text):
                    v = _re.sub(
                        r'[^A-Za-z0-9_.-]+', '_',
                        str(text or '').strip())
                    v = _re.sub(r'_+', '_', v).strip('._-')
                    return v

                def _generate_id(name, source):
                    base = (_sanitize_id(source or '').lower()
                            or 'finetune')
                    name_text = str(name or '').strip()
                    if not name_text:
                        return f"{base}_finetune"
                    words = _re.findall(r'[A-Za-z0-9]+', name_text)
                    suffix = ('_'.join(words[:2]).casefold()
                              if words else 'finetune')
                    return (_sanitize_id(f"{base}_{suffix}")
                            if suffix else base)

                def _unique_id(base):
                    p = Path(FINETUNES_DIR)
                    existing = ({f.stem for f in p.glob('*.json')}
                                if p.exists() else set())
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
                    outputs=[fin_id], queue=False)
                fin_name.input(
                    fn=_auto_id,
                    inputs=[fin_name, fin_fsm, fin_auto_id, fin_id],
                    outputs=[fin_id], queue=False)
                fin_name.change(
                    fn=_auto_id,
                    inputs=[fin_name, fin_fsm, fin_auto_id, fin_id],
                    outputs=[fin_id], queue=False)
                fin_name.blur(
                    fn=_auto_id,
                    inputs=[fin_name, fin_fsm, fin_auto_id, fin_id],
                    outputs=[fin_id], queue=False)

                def _switch_mode(fsm):
                    if fsm and str(fsm).strip():
                        return ("editor",
                                gr.update(visible=False),
                                gr.update(visible=True))
                    return ("creator",
                            gr.update(visible=True),
                            gr.update(visible=False))

                fin_fsm.change(
                    fn=_switch_mode,
                    inputs=[fin_fsm],
                    outputs=[fin_mode, fin_creator_actions,
                             fin_editor_actions],
                    queue=False)

                def _extract(d):
                    m = d.get("model", {})

                    def _j(k):
                        v = m.get(k, [])
                        if isinstance(v, str):
                            return v
                        return "\n".join(v)

                    def _s(k):
                        return " ".join(str(x) for x in m.get(k, []))

                    def _c(k):
                        return ",".join(str(x) for x in m.get(k, []))
                    res = ""
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
                    return (
                        m.get("name", ""),
                        m.get("architecture", ""),
                        m.get("description", ""),
                        m.get("finetune_source_model", ""),
                        _j("URLs"), _j("URLs2"),
                        _j("text_encoder_URLs"),
                        _j("VAE_URLs"), _j("preload_URLs"),
                        _j("custom_url_1"),
                        _j("custom_url_2"),
                        _j("custom_url_3"),
                        _j("loras"), _s("loras_multipliers"),
                        _c("modules"),
                        m.get("auto_quantize", False),
                        m.get("visible", True),
                        m.get("image_outputs", False),
                        res.strip(), rc,
                        m.get("infos", ""),
                        m.get("prompt_infos", ""),
                        st, gd,
                        m.get("text_prompt_enhancer_instructions", ""),
                        m.get("text_prompt_enhancer_max_tokens", ""),
                        m.get("video_prompt_enhancer_instructions", ""),
                        m.get("video_prompt_enhancer_max_tokens", ""),
                        m.get("image_prompt_enhancer_instructions", ""),
                        m.get("image_prompt_enhancer_max_tokens", ""),
                        tags_out,
                    )

                def _fill(file):
                    n_inputs = len(ALL_INPUTS)
                    if file is None:
                        return [gr.update()] * n_inputs + ["Select a file"]
                    s = (file if isinstance(file, str)
                         else file.get("name") or file.get("path"))
                    if not s or not Path(s).exists():
                        return [gr.update()] * n_inputs + ["File not found"]
                    try:
                        data = json.loads(
                            Path(s).read_text(encoding="utf-8"))
                    except Exception as e:
                        return ([gr.update()] * n_inputs
                                + [f"Invalid: {e}"])
                    if "model" not in data:
                        return ([gr.update()] * n_inputs
                                + ["Missing model"])
                    vid = Path(s).stem
                    return list((vid,) + _extract(data)) + [
                        "Filled from " + Path(s).name]

                fin_import_btn.click(
                    fn=_fill,
                    inputs=[fin_import_file],
                    outputs=ALL_INPUTS + [fin_status])

                def _create_action(id_, *vals):
                    if not id_:
                        return "Enter an ID"
                    data = _build(*vals)
                    _write_finetune(id_, data)
                    if (hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs):
                        self.refresh_model_defs()
                    t, tab = (
                        self.switch_to_model(id_, False)
                        if hasattr(self, "switch_to_model")
                        else (gr.update(), gr.update()))
                    return f"Created {id_}", t, tab

                fin_create.click(
                    fn=_create_action, inputs=ALL_INPUTS,
                    outputs=[fin_status, self.model_choice_target,
                             self.main_tabs])

                def _create_new_action(id_, *vals):
                    if not id_:
                        return "Enter an ID"
                    data = _build(*vals)
                    _write_finetune(id_, data)
                    if (hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs):
                        self.refresh_model_defs()
                    blanks = [gr.update(value="") for _ in ALL_INPUTS]
                    blanks[0] = gr.update(value="", interactive=False)
                    return ([f"Created {id_}. Ready for new entry."]
                            + blanks)

                fin_create_new.click(
                    fn=_create_new_action, inputs=ALL_INPUTS,
                    outputs=[fin_status] + ALL_INPUTS)

                def _export_action(id_, *vals):
                    if not id_:
                        return None, "Enter an ID first"
                    data = _build(*vals)
                    _write_finetune(id_, data)
                    path = str(Path(FINETUNES_DIR) / f"{id_}.json")
                    return path, f"Exported {id_}"

                fin_export.click(
                    fn=_export_action, inputs=ALL_INPUTS,
                    outputs=[fin_export, fin_status])

                def _save_action(id_, *vals):
                    if not id_:
                        return "Enter an ID"
                    _write_finetune(id_, _build(*vals))
                    if (hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs):
                        self.refresh_model_defs()
                    return f"Saved {id_}"

                fin_save.click(
                    fn=_save_action, inputs=ALL_INPUTS,
                    outputs=[fin_status])

                def _save_up_action(id_, *vals):
                    if not id_:
                        return "Enter an ID"
                    if not REGISTRY_TOKEN:
                        return "Missing config.json"
                    data = _build(*vals)
                    _write_finetune(id_, data)
                    try:
                        _hf_upload(id_, data)
                        if (hasattr(self, "refresh_model_defs")
                                and self.refresh_model_defs):
                            self.refresh_model_defs()
                        return f"Saved & uploaded {id_}"
                    except Exception as e:
                        return f"Saved but upload failed: {e}"

                fin_save_up.click(
                    fn=_save_up_action, inputs=ALL_INPUTS,
                    outputs=[fin_status])

                def _delete_action(id_):
                    if not id_:
                        return "Enter an ID"
                    p = Path(FINETUNES_DIR) / f"{id_}.json"
                    if not p.exists():
                        return "Not found"
                    p.unlink()
                    if (hasattr(self, "refresh_model_defs")
                            and self.refresh_model_defs):
                        self.refresh_model_defs()
                    return f"Deleted {id_}"

                fin_del.click(
                    fn=lambda: (gr.update(visible=False),
                                gr.update(visible=True)),
                    outputs=[fin_del, fin_del_confirm], queue=False)
                fin_del_cancel_btn.click(
                    fn=lambda: (gr.update(visible=True),
                                gr.update(visible=False)),
                    outputs=[fin_del, fin_del_confirm], queue=False)
                fin_del_confirm_btn.click(
                    fn=_delete_action, inputs=[fin_id],
                    outputs=[fin_status]).then(
                    fn=lambda: (gr.update(visible=True),
                                gr.update(visible=False)),
                    outputs=[fin_del, fin_del_confirm], queue=False)

                def _cancel_action():
                    blanks = [gr.update(value="") for _ in ALL_INPUTS]
                    blanks[0] = gr.update(value="", interactive=True)
                    return (blanks
                            + [gr.update(value="")]
                            + [gr.update(visible=True),
                               gr.update(visible=False)])

                fin_cancel_create.click(
                    fn=_cancel_action,
                    outputs=(ALL_INPUTS + [fin_status]
                             + [fin_creator_actions, fin_editor_actions]),
                    queue=False)
                fin_cancel_edit.click(
                    fn=_cancel_action,
                    outputs=(ALL_INPUTS + [fin_status]
                             + [fin_creator_actions, fin_editor_actions]),
                    queue=False)

                def _editor_validate(*vals):
                    data = _build(*vals[1:])
                    m = data.get("model", {})
                    url_keys = ["URLs", "URLs2", "text_encoder_URLs",
                                "VAE_URLs", "preload_URLs",
                                "custom_url_1", "custom_url_2",
                                "custom_url_3"]
                    all_urls = []
                    for k in url_keys:
                        v = m.get(k, [])
                        if isinstance(v, str):
                            v = [v]
                        if isinstance(v, list):
                            all_urls.extend(v)
                    loras = m.get("loras", [])
                    if isinstance(loras, str):
                        loras = [loras]
                    all_urls.extend(loras)
                    results = _validate_urls(all_urls)
                    return _build_url_validation_html(results)

                fin_validate_urls_btn.click(
                    fn=_editor_validate,
                    inputs=ALL_INPUTS,
                    outputs=[fin_validate_out])

                def _editor_download(*vals):
                    data = _build(*vals[1:])
                    return _download_finetune_files(data)

                fin_download_btn.click(
                    fn=_editor_download,
                    inputs=ALL_INPUTS,
                    outputs=[fin_download_out])

            # ── LOCAL ──
            with gr.TabItem("Local", id="local"):
                l_refresh = gr.Button("Refresh", variant="primary")
                l_sel_id = gr.Textbox(
                    visible=False, elem_id="l-selected", value="")
                l_cards = gr.HTML(
                    "<div style='color:#9ca3af;text-align:center;"
                    "padding:32px'>Click Refresh to list local finetunes</div>")
                l_detail = gr.JSON(label="Content")
                with gr.Row():
                    l_load = gr.Button(
                        "Load & Switch", variant="primary")
                    l_del = gr.Button("Delete")
                    l_export = gr.Button("Export JSON")
                    l_up = gr.Button("Upload to Registry")
                with gr.Row():
                    l_imp_file = gr.File(
                        label="Import .json", file_types=[".json"])
                    l_imp_btn = gr.Button("Import & Switch")
                l_status = gr.Textbox(label="Status")

            # ── UPLOAD ──
            with gr.TabItem("Upload", id="upload"):
                gr.Markdown(
                    "Upload a local finetune JSON to "
                    "the community registry")
                with gr.Row():
                    u_refresh = gr.Button("Refresh")
                u_list = gr.Dropdown(
                    label="Local Finetune", choices=[],
                    interactive=True, allow_custom_value=True)
                u_preview = gr.JSON(label="Preview")
                u_btn = gr.Button("Upload", variant="primary")
                u_status = gr.Textbox(label="Status")

            # ── CIVITAI ──
            with gr.TabItem("CivitAI", id="civitai"):
                gr.Markdown(
                    "Search CivitAI for models and "
                    "auto-fill into Create/Edit.")
                with gr.Row():
                    ci_search = gr.Textbox(
                        label="Search", scale=3, container=False,
                        placeholder="e.g. fantasy landscape, "
                                    "character style...")
                    ci_type = gr.Dropdown(
                        label="Type",
                        choices=["All", "LORA", "Checkpoint",
                                 "TextualInversion", "Hypernetwork",
                                 "AestheticGradient", "ControlNet",
                                 "Poses", "Wildcards", "MotionModule",
                                 "VAE"],
                        value="All", scale=1)
                    ci_base = gr.Dropdown(
                        label="Base Model",
                        choices=["All", "SDXL 1.0", "SDXL 0.9",
                                 "SD 1.5", "SD 2.0", "SD 2.1", "Pony",
                                 "Flux.1 D", "Flux.1 S",
                                 "SD 3", "SD 3.5"],
                        value="All", scale=1)
                    ci_search_btn = gr.Button(
                        "Search", variant="primary", scale=1)
                with gr.Row():
                    ci_load_btn = gr.Button(
                        "Fill Selected into Create/Edit",
                        variant="primary", scale=1)
                    ci_clear_btn = gr.Button("Clear", scale=1)
                ci_status = gr.Markdown(
                    "Enter a search term to find models on CivitAI")
                ci_results = gr.HTML(
                    "<div style='color:#9ca3af;text-align:center;"
                    "padding:32px'>Click Search</div>")
                ci_selected = gr.Textbox(
                    visible=False, elem_id="civitai-selected", value="{}")

                def _ci_do_search(query, mtype, base):
                    if not query or not query.strip():
                        return ("<div style='color:#9ca3af;"
                                "text-align:center;padding:32px'>"
                                "Enter a search term</div>", "")
                    data = _civitai_search(
                        query.strip(), model_type=mtype,
                        base_model=base)
                    html_out = _civitai_render_results(data)
                    count = (len(data.get("items", []))
                             if "error" not in data else 0)
                    status = (
                        f"**{count} results** for "
                        f"'{html.escape(query.strip())}'"
                        if count else "No results found")
                    return html_out, status

                ci_search_btn.click(
                    fn=_ci_do_search,
                    inputs=[ci_search, ci_type, ci_base],
                    outputs=[ci_results, ci_status])
                ci_search.submit(
                    fn=_ci_do_search,
                    inputs=[ci_search, ci_type, ci_base],
                    outputs=[ci_results, ci_status])

                def _ci_fill(civitai_json_str):
                    return _civitai_extract_fill_data(civitai_json_str)

                ci_load_btn.click(
                    fn=_ci_fill,
                    inputs=[ci_selected],
                    outputs=ALL_INPUTS + [ci_status])

                def _ci_clear():
                    return ("<div style='color:#9ca3af;text-align:center;"
                            "padding:32px'>Click Search</div>",
                            "{}",
                            "Enter a search term to find models "
                            "on CivitAI")

                ci_clear_btn.click(
                    fn=_ci_clear,
                    outputs=[ci_results, ci_selected, ci_status])

            # ── SHARED LOCAL HANDLERS ──
            def _loc_list():
                p = Path(FINETUNES_DIR)
                if not p.exists():
                    return [], "*No finetunes directory*"
                fs = sorted(p.glob("*.json"))
                plural = "" if len(fs) == 1 else "s"
                return ([(f.stem, f.stem) for f in fs],
                        f"**{len(fs)} local finetune{plural}**")

            def _loc_detail(fid):
                if not fid:
                    return {}
                p = Path(FINETUNES_DIR) / f"{fid}.json"
                if not p.exists():
                    return {"error": "not found"}
                return json.loads(p.read_text(encoding="utf-8"))

            def _fmt_local_cards():
                p = Path(FINETUNES_DIR)
                if not p.exists():
                    return ("<div style='color:#9ca3af;text-align:center;"
                            "padding:32px'>No finetunes directory</div>",
                            "*No finetunes directory*")
                fs = sorted(p.glob("*.json"))
                if not fs:
                    return ("<div style='color:#9ca3af;text-align:center;"
                            "padding:32px'>No local finetunes</div>",
                            "**0 local finetunes**")
                _h = ['<div class="fm-cards-container">']
                for fpath in fs:
                    fid = fpath.stem
                    try:
                        data = json.loads(
                            fpath.read_text(encoding="utf-8"))
                    except Exception:
                        data = {}
                    m = data.get("model", {})
                    name = html.escape(m.get("name", fid) or fid)
                    arch = html.escape(
                        m.get("architecture", "") or "")
                    author = html.escape(
                        m.get("author", "") or "local")
                    desc = html.escape(
                        (m.get("description", "") or "")[:300])
                    ftags = m.get("tags", [])
                    if isinstance(ftags, str):
                        ftags = [t.strip() for t in ftags.split(",")
                                 if t.strip()]
                    tags_badges = ""
                    if ftags:
                        tags_badges = '<div style="margin-top:2px">'
                        for t in ftags[:4]:
                            et = html.escape(t.strip())
                            tags_badges += (
                                f'<span class="fm-badge">{et}</span> ')
                        if len(ftags) > 4:
                            tags_badges += (
                                f'<span style="color:#9ca3af;'
                                f'font-size:10px">'
                                f'+{len(ftags)-4}</span>')
                        tags_badges += '</div>'
                    fid_safe = (fid.replace("\\", "\\\\")
                                   .replace("'", "\\'")
                                   .replace('"', '\\"'))
                    onclick = (
                        "document.querySelector('#l-selected textarea')"
                        f".value='{fid_safe}';"
                        "document.querySelector('#l-selected textarea')"
                        ".dispatchEvent(new Event('input',"
                        "{bubbles:true}))")
                    urls = m.get("URLs", [])
                    if isinstance(urls, str):
                        urls = [urls]
                    files_html = ""
                    if urls:
                        link_parts = []
                        for u in urls[:3]:
                            eu = html.escape(u)
                            short = (u.rstrip("/").split("/")[-1]
                                     if "/" in u else u)
                            if len(short) > 50:
                                short = short[:47] + "..."
                            link_parts.append(
                                f'<a href="{eu}" target="_blank" '
                                f'rel="noopener">'
                                f'{html.escape(short)}</a>')
                        files_html = ('<div class="fm-card-files">' +
                                      " ".join(link_parts))
                        if len(urls) > 3:
                            files_html += (
                                f' <span style="color:#9ca3af">'
                                f'+{len(urls)-3} more</span>')
                        files_html += '</div>'
                    loras = m.get("loras", [])
                    if isinstance(loras, str):
                        loras = [loras]
                    loras_html = ""
                    if loras:
                        loras_html = (
                            '<div class="fm-card-loras">'
                            '<b>LoRAs:</b> ')
                        loras_html += " ".join(
                            html.escape(l) for l in loras[:3])
                        if len(loras) > 3:
                            loras_html += (
                                f' <span style="color:#9ca3af">'
                                f'+{len(loras)-3} more</span>')
                        loras_html += '</div>'
                    _h.append(
                        f'<div class="fm-card" onclick="{onclick}">'
                        f"<div class='fm-card-title'>{name}</div>"
                        f"<div class='fm-card-meta'>"
                        f"<span class='fm-badge'>{html.escape(arch) or 'finetune'}</span> "
                        f"{author}</div>"
                        f"<div class='fm-card-desc'>{desc}</div>"
                        f"{tags_badges}{files_html}{loras_html}"
                        f"</div>")
                _h.append('</div>')
                plural = "" if len(fs) == 1 else "s"
                return ("".join(_h),
                        f"**{len(fs)} local finetune{plural}**")

            l_refresh.click(
                fn=_fmt_local_cards,
                outputs=[l_cards, l_status])
            l_sel_id.input(
                fn=_loc_detail, inputs=[l_sel_id],
                outputs=[l_detail])
            u_refresh.click(
                fn=_loc_list, outputs=[u_list, u_status])
            u_list.change(
                fn=_loc_detail, inputs=[u_list],
                outputs=[u_preview])

            def _loc_load(fid):
                if not fid:
                    return "Select one", gr.update(), gr.update()
                p = Path(FINETUNES_DIR) / f"{fid}.json"
                if not p.exists():
                    return "Not found", gr.update(), gr.update()
                if (hasattr(self, "refresh_model_defs")
                        and self.refresh_model_defs):
                    self.refresh_model_defs()
                t, tab = (
                    self.switch_to_model(fid, False)
                    if hasattr(self, "switch_to_model")
                    else (gr.update(), gr.update()))
                return f"Switched to '{fid}'", t, tab

            l_load.click(
                fn=_loc_load, inputs=[l_sel_id],
                outputs=[l_status, self.model_choice_target,
                         self.main_tabs])

            def _loc_del(fid):
                if not fid:
                    return "Select one"
                p = Path(FINETUNES_DIR) / f"{fid}.json"
                if p.exists():
                    p.unlink()
                else:
                    return "Not found"
                if (hasattr(self, "refresh_model_defs")
                        and self.refresh_model_defs):
                    self.refresh_model_defs()
                return f"Deleted {fid}"

            l_del.click(
                fn=_loc_del, inputs=[l_sel_id],
                outputs=[l_status])

            l_export_btn = gr.DownloadButton("Export JSON", value=None)

            def _loc_export_detail(fid):
                if not fid:
                    return {"error": "not found"}, None
                p = Path(FINETUNES_DIR) / f"{fid}.json"
                if not p.exists():
                    return {"error": "not found"}, None
                return json.loads(p.read_text(encoding="utf-8")), str(p)

            l_export_btn.click(
                fn=_loc_export_detail, inputs=[l_sel_id],
                outputs=[l_detail, l_export_btn])
            l_export.click(
                fn=_loc_export_detail, inputs=[l_sel_id],
                outputs=[l_detail, l_export_btn])

            def _loc_import(file):
                if file is None:
                    return "Select a .json", gr.update(), gr.update()
                s = (file if isinstance(file, str)
                     else file.get("name") or file.get("path"))
                if not s or not Path(s).exists():
                    return "Not found", gr.update(), gr.update()
                try:
                    data = json.loads(
                        Path(s).read_text(encoding="utf-8"))
                except Exception as e:
                    return f"Invalid: {e}", gr.update(), gr.update()
                if "model" not in data:
                    return ("Missing 'model'",
                            gr.update(), gr.update())
                fid = Path(s).stem
                _write_finetune(fid, data)
                if (hasattr(self, "refresh_model_defs")
                        and self.refresh_model_defs):
                    self.refresh_model_defs()
                t, tab = (
                    self.switch_to_model(fid, False)
                    if hasattr(self, "switch_to_model")
                    else (gr.update(), gr.update()))
                return f"Imported '{fid}'", t, tab

            l_imp_btn.click(
                fn=_loc_import, inputs=[l_imp_file],
                outputs=[l_status, self.model_choice_target,
                         self.main_tabs])

            def _up(fid):
                if not fid:
                    return "Select one"
                if not REGISTRY_TOKEN:
                    return "Missing config.json"
                p = Path(FINETUNES_DIR) / f"{fid}.json"
                if not p.exists():
                    return "Not found"
                _hf_upload(
                    fid,
                    json.loads(p.read_text(encoding="utf-8")))
                return f"Uploaded '{fid}'"

            l_up.click(
                fn=_up, inputs=[l_sel_id], outputs=[l_status])
            u_btn.click(
                fn=_up, inputs=[u_list], outputs=[u_status])
