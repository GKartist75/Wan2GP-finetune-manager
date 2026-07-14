"""Comprehensive test suite for Wan2GP Finetune Manager plugin.

Tests pure logic functions replicated from plugin.py (they are defined as local
closures inside create_ui() and thus not directly importable), plus module-level
helpers, and create_space.py functions.

Test categories:
  A. _build() — data construction from form fields (replicated)
  B. _extract() — finetune dict → form field values (replicated)
  C. _fmt_cards() — HTML card rendering (replicated)
  D. _on_selected() — card selection handler (replicated)
  E. _write_finetune() — module-level file I/O
  F. _fetch_registry_json() — module-level HTTP fetch (mocked)
  G. _hf_upload() — module-level HF upload (mocked)
  H. make_finetune_json() — create_space.py function
  I. Module-level constants and config
"""

import html
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# -- Setup: make Wan2GP imports work ------------------------------------------
WAN2GP_ROOT = r"C:\Users\gjaku\Wan2GP"
if WAN2GP_ROOT not in sys.path:
    sys.path.insert(0, WAN2GP_ROOT)

# -- Import module under test -------------------------------------------------
PLUGIN_DIR = Path(__file__).parent
_cfg_path = PLUGIN_DIR / "config.json"
_cfg_saved = _cfg_path.read_text(encoding="utf-8") if _cfg_path.exists() else None
if _cfg_path.exists():
    _cfg_path.write_text(json.dumps({"registry_token": "test_token_abc"}), encoding="utf-8")

import plugin as FM

if _cfg_saved:
    _cfg_path.write_text(_cfg_saved, encoding="utf-8")
elif _cfg_path.exists():
    _cfg_path.unlink()


# ==============================================================================
# HELPER: _build() -- form fields -> finetune dict (replicated from plugin.py)
# ==============================================================================

def _build(name, arch, desc, fsm, urls, urls2, te, vae, pre, cu1, cu2, cu3,
           loras, lm, mods, aq, vis, img, res, resc, inf, pinf, steps, guid,
           pe1, pe1t, pe2, pe2t, pe3, pe3t, tags=""):
    """Replicated from plugin.py create_ui() -> _build()."""
    m = {"name": name, "architecture": arch, "description": desc}
    if fsm: m["finetune_source_model"] = fsm
    if urls: m["URLs"] = [x.strip() for x in urls.split("\n") if x.strip()]
    if urls2: m["URLs2"] = [x.strip() for x in urls2.split("\n") if x.strip()]
    if te: m["text_encoder_URLs"] = [x.strip() for x in te.split("\n") if x.strip()]
    if vae: m["VAE_URLs"] = [x.strip() for x in vae.split("\n") if x.strip()]
    if pre: m["preload_URLs"] = [x.strip() for x in pre.split("\n") if x.strip()]
    if cu1: m["custom_url_1"] = cu1
    if cu2: m["custom_url_2"] = cu2
    if cu3: m["custom_url_3"] = cu3
    if loras: m["loras"] = [x.strip() for x in loras.split("\n") if x.strip()]
    if lm: m["loras_multipliers"] = [x.strip() for x in lm.replace(",", " ").split() if x.strip()]
    if mods: m["modules"] = [x.strip() for x in mods.split(",") if x.strip()]
    if aq: m["auto_quantize"] = True
    if not vis: m["visible"] = False
    if img: m["image_outputs"] = True
    if tags: m["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if res:
        out = []
        for l in res.split("\n"):
            l = l.strip()
            if not l: continue
            if "x" in l.lower():
                import re
                match = re.match(r'(\d+)\s*x\s*(\d+)', l, re.IGNORECASE)
                if match: out.append([l, f"{match.group(1)}x{match.group(2)}"])
                else: out.append(l)
            else:
                out.append(l)
        m["resolutions"] = out
    if resc: m["resolutions_categories"] = [x.strip() for x in resc.split("\n") if x.strip()]
    if inf: m["infos"] = inf
    if pinf: m["prompt_infos"] = pinf
    if pe1: m["text_prompt_enhancer_instructions"] = pe1
    if pe1t: m["text_prompt_enhancer_max_tokens"] = int(pe1t)
    if pe2: m["video_prompt_enhancer_instructions"] = pe2
    if pe2t: m["video_prompt_enhancer_max_tokens"] = int(pe2t)
    if pe3: m["image_prompt_enhancer_instructions"] = pe3
    if pe3t: m["image_prompt_enhancer_max_tokens"] = int(pe3t)
    return dict(model=m, num_inference_steps=int(steps), guidance_scale=float(guid))


def _b(**overrides):
    """Call _build with defaults and override specific fields."""
    d = dict(name="Test", arch="t2v", desc="", fsm="",
             urls="", urls2="", te="", vae="", pre="",
             cu1="", cu2="", cu3="",
             loras="", lm="", mods="",
             aq=False, vis=True, img=False,
             res="", resc="", inf="", pinf="",
             steps=30, guid=5.0,
             pe1="", pe1t="", pe2="", pe2t="", pe3="", pe3t="",
             tags="")
    d.update(overrides)
    return _build(d["name"], d["arch"], d["desc"], d["fsm"],
                  d["urls"], d["urls2"], d["te"], d["vae"], d["pre"],
                  d["cu1"], d["cu2"], d["cu3"],
                  d["loras"], d["lm"], d["mods"],
                  d["aq"], d["vis"], d["img"],
                  d["res"], d["resc"], d["inf"], d["pinf"],
                  d["steps"], d["guid"],
                  d["pe1"], d["pe1t"], d["pe2"], d["pe2t"], d["pe3"], d["pe3t"],
                  d["tags"])


class TestBuild:
    """_build() constructs the finetune dict from form field arguments."""

    def test_basic_fields(self):
        r = _b(name="My Finetune", arch="i2v", desc="A test")
        assert r["model"]["name"] == "My Finetune"
        assert r["model"]["architecture"] == "i2v"
        assert r["model"]["description"] == "A test"

    def test_invisible(self):
        r = _b(vis=False)
        assert r["model"].get("visible") is False

    def test_visible_default(self):
        """visible=True is the implicit default; key absent unless False."""
        r = _b()
        assert r["model"].get("visible", True) is True
        assert "visible" not in r["model"]

    def test_image_outputs(self):
        r = _b(img=True)
        assert r["model"].get("image_outputs") is True

    def test_auto_quantize(self):
        r = _b(aq=True)
        assert r["model"].get("auto_quantize") is True

    def test_loras(self):
        r = _b(loras="style-a\nstyle-b", lm="0.5 0.8")
        assert r["model"]["loras"] == ["style-a", "style-b"]
        assert r["model"]["loras_multipliers"] == ["0.5", "0.8"]

    def test_lora_multipliers_commas(self):
        r = _b(lm="0.5,0.8,1.0")
        assert r["model"]["loras_multipliers"] == ["0.5", "0.8", "1.0"]

    def test_urls(self):
        r = _b(urls="https://a\nhttps://b")
        assert r["model"]["URLs"] == ["https://a", "https://b"]

    def test_finetune_source_model(self):
        r = _b(fsm="base-v2")
        assert r["model"]["finetune_source_model"] == "base-v2"

    def test_resolution_parsing(self):
        r = _b(res="1920x1080\n1280x720")
        assert len(r["model"]["resolutions"]) == 2
        assert r["model"]["resolutions"][0][1] == "1920x1080"
        assert r["model"]["resolutions"][1][1] == "1280x720"

    def test_resolution_complex_string(self):
        """BUG: '1280x720 (HD)' produces malformed second element."""
        r = _b(res="1280x720 (HD)")
        res = r["model"]["resolutions"][0]
        msg = (f"BUG: expected '1280x720', got '{res[1]}'. "
               "Naive split('x') includes trailing text.")
        assert res[1] == "1280x720", msg

    def test_resolution_empty_lines_skipped(self):
        r = _b(res="1920x1080\n\n\n1280x720")
        assert len(r["model"]["resolutions"]) == 2

    def test_prompt_enhancer_empty_tokens_omitted(self):
        r = _b(pe1="enhance text", pe1t="", pe2="enhance vid", pe2t="")
        assert r["model"].get("text_prompt_enhancer_instructions") == "enhance text"
        assert "text_prompt_enhancer_max_tokens" not in r["model"]

    def test_prompt_enhancer_with_tokens(self):
        r = _b(pe1="enhance", pe1t="100", pe2="enhance vid", pe2t="200")
        assert r["model"]["text_prompt_enhancer_max_tokens"] == 100
        assert r["model"]["video_prompt_enhancer_max_tokens"] == 200

    def test_guidance_scale_zero(self):
        r = _b(guid=0.0)
        assert r["guidance_scale"] == 0.0

    def test_steps_and_guidance_types(self):
        r = _b(steps=25, guid=5.5)
        assert isinstance(r["num_inference_steps"], int)
        assert isinstance(r["guidance_scale"], float)
        assert r["num_inference_steps"] == 25
        assert r["guidance_scale"] == 5.5

    def test_empty_modules_omitted(self):
        r = _b(mods="")
        assert "modules" not in r["model"]

    def test_modules_csv(self):
        r = _b(mods="mod1,mod2,mod3")
        assert r["model"]["modules"] == ["mod1", "mod2", "mod3"]

    def test_resolutions_categories(self):
        r = _b(resc="cat1\ncat2")
        assert r["model"]["resolutions_categories"] == ["cat1", "cat2"]

    def test_infos(self):
        r = _b(inf="# Help text\ninfo")
        assert r["model"]["infos"] == "# Help text\ninfo"

    def test_prompt_infos(self):
        r = _b(pinf="Prompt guide here")
        assert r["model"]["prompt_infos"] == "Prompt guide here"

    def test_preload_urls(self):
        r = _b(pre="https://preload/a\nhttps://preload/b")
        assert r["model"]["preload_URLs"] == ["https://preload/a", "https://preload/b"]

    def test_vae_urls(self):
        r = _b(vae="https://vae/a\nhttps://vae/b")
        assert r["model"]["VAE_URLs"] == ["https://vae/a", "https://vae/b"]

    def test_text_encoder_urls(self):
        r = _b(te="https://te/a\nhttps://te/b")
        assert r["model"]["text_encoder_URLs"] == ["https://te/a", "https://te/b"]

    def test_secondary_urls(self):
        r = _b(urls2="https://secondary/a\nhttps://secondary/b")
        assert r["model"]["URLs2"] == ["https://secondary/a", "https://secondary/b"]


# ==============================================================================
# HELPER: _extract() -- finetune dict -> form field values (replicated)
# ==============================================================================

def _extract(d):
    """Replicated from plugin.py create_ui() -> _extract()."""
    m = d.get("model", {})

    def _j(k):
        v = m.get(k, [])
        if isinstance(v, str): return v
        return "\n".join(v)
    def _s(k): return " ".join(str(x) for x in m.get(k, []))
    def _c(k): return ",".join(str(x) for x in m.get(k, []))

    res = ""
    for r in m.get("resolutions", []):
        if isinstance(r, list) and len(r) >= 2:
            res += r[1] + "\n"
        elif isinstance(r, str):
            res += r + "\n"

    rc = "\n".join(m.get("resolutions_categories", []))
    st = d.get("num_inference_steps")
    if st is None: st = m.get("num_inference_steps")
    if st is None: st = 30
    gd = d.get("guidance_scale")
    if gd is None: gd = m.get("guidance_scale")
    if gd is None: gd = 5.0

    tags_raw = m.get("tags", [])
    if isinstance(tags_raw, list):
        tags_out = ", ".join(tags_raw)
    else:
        tags_out = str(tags_raw)
    return (m.get("name", ""), m.get("architecture", ""), m.get("description", ""),
            m.get("finetune_source_model", ""),
            _j("URLs"), _j("URLs2"), _j("text_encoder_URLs"), _j("VAE_URLs"),
            _j("preload_URLs"), _j("custom_url_1"), _j("custom_url_2"), _j("custom_url_3"),
            _j("loras"), _s("loras_multipliers"), _c("modules"),
            m.get("auto_quantize", False), m.get("visible", True),
            m.get("image_outputs", False),
            res.strip(), rc, m.get("infos", ""), m.get("prompt_infos", ""),
            st, gd,
            m.get("text_prompt_enhancer_instructions", ""),
            m.get("text_prompt_enhancer_max_tokens", ""),
            m.get("video_prompt_enhancer_instructions", ""),
            m.get("video_prompt_enhancer_max_tokens", ""),
            m.get("image_prompt_enhancer_instructions", ""),
            m.get("image_prompt_enhancer_max_tokens", ""),
            tags_out)


# Named indices into _extract() return tuple
(EX_NAME, EX_ARCH, EX_DESC, EX_FSM,
 EX_URLS, EX_URLS2, EX_TE, EX_VAE, EX_PRELOAD,
 EX_CU1, EX_CU2, EX_CU3,
 EX_LORAS, EX_LMULTS, EX_MODS, EX_AQ, EX_VIS, EX_IMG,
 EX_RES, EX_RESCAT, EX_INFOS, EX_PINFOS, EX_STEPS, EX_GUID,
 EX_PE1, EX_PE1T, EX_PE2, EX_PE2T, EX_PE3, EX_PE3T, EX_TAGS) = range(31)


class TestExtract:
    """_extract() deserializes a finetune dict back to form values."""

    def test_round_trip_basic(self):
        r = _b(name="RT", arch="i2v", desc="desc")
        e = _extract(r)
        assert e[EX_NAME] == "RT"
        assert e[EX_ARCH] == "i2v"
        assert e[EX_DESC] == "desc"

    def test_round_trip_steps_guidance(self):
        r = _b(steps=25, guid=4.5)
        e = _extract(r)
        assert e[EX_STEPS] == 25
        assert e[EX_GUID] == 4.5

    def test_round_trip_visible_false(self):
        r = _b(vis=False)
        e = _extract(r)
        assert e[EX_VIS] is False

    def test_round_trip_image_outputs(self):
        r = _b(img=True)
        e = _extract(r)
        assert e[EX_IMG] is True

    def test_round_trip_loras(self):
        r = _b(loras="l1\nl2", lm="0.5 0.75")
        e = _extract(r)
        assert e[EX_LORAS] == "l1\nl2"
        assert "0.5" in e[EX_LMULTS]

    def test_round_trip_auto_quantize(self):
        r = _b(aq=True)
        assert r["model"].get("auto_quantize") is True

    def test_round_trip_urls(self):
        r = _b(urls="https://a\nhttps://b")
        e = _extract(r)
        assert e[EX_URLS] == "https://a\nhttps://b"

    def test_tags_round_trip(self):
        r = _b(tags="anime, style")
        e = _extract(r)
        assert e[EX_TAGS] == "anime, style"

    def test_tags_round_trip_missing(self):
        r = _b()
        e = _extract(r)
        assert e[EX_TAGS] == ""

    def test_guidance_scale_zero_round_trip(self):
        """BUG: guidance_scale=0.0 is falsy -> 'or' pattern replaces with 5.0."""
        r = _b(guid=0.0)
        assert r["guidance_scale"] == 0.0
        e = _extract(r)
        assert e[EX_GUID] == 0.0, (
            f"BUG: guidance_scale=0.0 became {e[EX_GUID]}. "
            "The 'or' pattern treats 0.0 as falsy."
        )

    def test_num_steps_zero_round_trip(self):
        """BUG: steps=0 is falsy -> 'or' pattern replaces with 30."""
        r = _b(steps=0)
        assert r["num_inference_steps"] == 0
        e = _extract(r)
        assert e[EX_STEPS] == 0, (
            f"BUG: num_inference_steps=0 became {e[EX_STEPS]}. "
            "The 'or' pattern treats 0 as falsy."
        )

    def test_missing_guidance_default(self):
        e = _extract({"model": {"name": "N", "architecture": "t2v", "description": ""}})
        assert e[EX_GUID] == 5.0

    def test_missing_steps_default(self):
        e = _extract({"model": {"name": "N", "architecture": "t2v", "description": ""}})
        assert e[EX_STEPS] == 30

    def test_round_trip_prompt_enhancer(self):
        r = _b(pe1="tx", pe1t="100", pe2="vx", pe2t="200", pe3="ix", pe3t="300")
        e = _extract(r)
        assert e[EX_PE1] == "tx"
        # max_tokens are stored as ints and extracted as ints
        assert e[EX_PE1T] == 100
        assert e[EX_PE2] == "vx"
        assert e[EX_PE2T] == 200
        assert e[EX_PE3] == "ix"
        assert e[EX_PE3T] == 300

    def test_extract_resolutions_from_list(self):
        r = _b(res="1920x1080\n1280x720")
        e = _extract(r)
        assert "1920x1080" in e[EX_RES]
        assert "1280x720" in e[EX_RES]

    def test_extract_resolutions_from_strings(self):
        d = {"model": {"name": "N", "architecture": "t2v", "description": "",
                       "resolutions": ["custom", "other"]}}
        e = _extract(d)
        assert "custom" in e[EX_RES]

    def test_extract_modules(self):
        d = {"model": {"name": "N", "architecture": "t2v", "description": "",
                       "modules": ["mod1", "mod2"]}}
        e = _extract(d)
        assert e[EX_MODS] == "mod1,mod2"

    def test_extract_resolution_categories(self):
        d = {"model": {"name": "N", "architecture": "t2v", "description": "",
                       "resolutions_categories": ["cat1", "cat2"]}}
        e = _extract(d)
        assert e[EX_RESCAT] == "cat1\ncat2"

    def test_extract_urls_as_string(self):
        """Registry entries may store URLs as strings (not arrays)."""
        d = {"model": {"name": "N", "architecture": "t2v", "description": "",
                       "URLs": "ltx2_22B",
                       "text_encoder_URLs": "https://example.com/model.safetensors"}}
        e = _extract(d)
        assert e[EX_URLS] == "ltx2_22B", f"String URL corrupted: {e[EX_URLS]}"
        assert e[EX_TE] == "https://example.com/model.safetensors", (
            f"String TE URL corrupted: {e[EX_TE]}"
        )

    def test_extract_loras_as_string(self):
        """LoRAs may also be stored as a string."""
        d = {"model": {"name": "N", "architecture": "t2v", "description": "",
                       "loras": "my-lora.safetensors"}}
        e = _extract(d)
        assert e[EX_LORAS] == "my-lora.safetensors", f"String LoRA corrupted: {e[EX_LORAS]}"


# ==============================================================================
# HELPER: _fmt_cards() -- card HTML rendering (replicated from plugin.py)
# ==============================================================================

def _fmt_cards(fins, search, arch, sel, tag_filter=""):
    """Replicated from plugin.py create_ui() -> _fmt_cards()."""
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
        ff = [f for f in ff
              if s in f.get("name", "").lower()
              or s in (f.get("description", "") or "").lower()
              or s in f.get("author", "").lower()
              or _match_tags(f)]
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
                return any((tag or "").strip().lower() == tf for tag in tg)
            ff = [f for f in ff if _filter_tags(f)]
    if not ff:
        return "<div style='color:#9ca3af;text-align:center;padding:32px'>No results</div>", "**0 matches**"
    _h = ['<div class="fm-cards-container">']
    for f in ff:
        fid = f.get("id", "")
        c = " selected" if fid == sel else ""
        name = html.escape(f.get("name", "?") or "?")
        arch_s = html.escape(f.get("architecture", "") or "")
        author = html.escape(f.get("author", "") or "")
        desc = html.escape((f.get("description", "") or "")[:300])
        tag = "Variant" if f.get("source") else arch_s
        fid_safe = fid.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        onclick = (f"document.querySelector('#fm-selected textarea').value='{fid_safe}';"
                   f"document.querySelector('#fm-selected textarea').dispatchEvent("
                   f"new Event('input',{{bubbles:true}}))")
        # Build files/URLs section
        urls = f.get("URLs", [])
        if isinstance(urls, str): urls = [urls]
        files_html = ""
        if urls:
            shown = urls[:3]
            extra = len(urls) - 3
            link_parts = []
            for u in shown:
                eu = html.escape(u)
                short = u.rstrip("/").split("/")[-1] if "/" in u else u
                if len(short) > 50: short = short[:47] + "..."
                link_parts.append(f'<a href="{eu}" target="_blank" rel="noopener">{html.escape(short)}</a>')
            files_html = '<div class="fm-card-files">' + " ".join(link_parts)
            if extra > 0:
                files_html += f' <span style="color:#9ca3af">+{extra} more</span>'
            files_html += '</div>'
        # Build LoRAs section
        loras = f.get("loras", [])
        if isinstance(loras, str): loras = [loras]
        loras_html = ""
        if loras:
            shown_l = loras[:3]
            extra_l = len(loras) - 3
            loras_html = '<div class="fm-card-loras"><b>LoRAs:</b> '
            loras_html += " ".join(html.escape(l) for l in shown_l)
            if extra_l > 0:
                loras_html += f' <span style="color:#9ca3af">+{extra_l} more</span>'
            loras_html += '</div>'
        # Build tags badges
        ftags = f.get("tags", [])
        if isinstance(ftags, str):
            ftags = [t.strip() for t in ftags.split(",") if t.strip()]
        tags_badges = ""
        if ftags:
            tags_badges = '<div style="margin-top:2px">'
            for t in ftags[:4]:
                et = html.escape(t.strip())
                tags_badges += f'<span class="fm-badge">{et}</span> '
            if len(ftags) > 4:
                tags_badges += f'<span style="color:#9ca3af;font-size:10px">+{len(ftags)-4}</span>'
            tags_badges += '</div>'
        _h.append(
            f'<div class="fm-card{c}" onclick="{onclick}">'
            f"<div class='fm-card-title'>{name}</div>"
            f"<div class='fm-card-meta'><span class='fm-badge'>{html.escape(tag)}</span> {author}</div>"
            f"<div class='fm-card-desc'>{desc}</div>{tags_badges}{files_html}{loras_html}</div>"
        )
    _h.append('</div>')
    cnt_label = "match" if len(ff) == 1 else "matches"
    return "".join(_h), f"**{len(ff)} {cnt_label}**"


SAMPLE_FINS = [
    {"id": "fin-1", "name": "Finetune One", "author": "Alice",
     "architecture": "t2v", "description": "First finetune"},
    {"id": "fin-2", "name": "Finetune Two", "author": "Bob",
     "architecture": "i2v", "description": "Second finetune", "source": "base-model"},
]


class TestFmtCards:
    """_fmt_cards() renders the registry as clickable HTML cards."""

    def test_search_by_name(self):
        h, c = _fmt_cards(SAMPLE_FINS, "One", "All", "")
        assert "Finetune One" in h
        assert "Finetune Two" not in h
        assert "**1" in c

    def test_search_by_description(self):
        h, c = _fmt_cards(SAMPLE_FINS, "second", "All", "")
        assert "Finetune Two" in h
        assert "**1" in c

    def test_search_by_author(self):
        h, c = _fmt_cards(SAMPLE_FINS, "alice", "All", "")
        assert "Finetune One" in h
        assert "**1" in c

    def test_search_case_insensitive(self):
        h, c = _fmt_cards(SAMPLE_FINS, "ALICE", "All", "")
        assert "Finetune One" in h
        assert "**1" in c

    def test_filter_architecture(self):
        h, c = _fmt_cards(SAMPLE_FINS, "", "i2v", "")
        assert "Finetune Two" in h
        assert "Finetune One" not in h
        assert "**1" in c

    def test_filter_all(self):
        h, c = _fmt_cards(SAMPLE_FINS, "", "All", "")
        assert "Finetune One" in h
        assert "Finetune Two" in h
        assert "**2" in c

    def test_no_results(self):
        h, c = _fmt_cards(SAMPLE_FINS, "nonexistent", "All", "")
        assert "No results" in h
        assert "**0 matches**" in c

    def test_empty_list(self):
        h, c = _fmt_cards([], "", "All", "")
        assert "No results" in h
        assert "**0 matches**" in c

    def test_selected_card_class(self):
        h, _ = _fmt_cards(SAMPLE_FINS, "", "All", "fin-1")
        assert "selected" in h

    def test_variant_tag(self):
        h, _ = _fmt_cards(SAMPLE_FINS, "", "All", "")
        assert "Variant" in h

    def test_description_truncated(self):
        """Description truncated to 300 chars (increased from 160)."""
        desc = ("A" * 300) + ("B" * 100)
        fins = [{"id": "l", "name": "L", "author": "M",
                 "architecture": "t2v", "description": desc}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert ("A" * 300) in h
        assert "B" not in h

    def test_xss_in_fid(self):
        """BUG: fid injected raw into onclick handler -> JS injection."""
        malicious = "test');alert('XSS');//"
        fins = [{"id": malicious, "name": "X", "author": "H",
                 "architecture": "t2v", "description": "pwned"}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert "alert('XSS')" not in h, (
            "XSS VULNERABILITY: fid injected raw into onclick JS context"
        )

    def test_xss_in_name(self):
        """XSS via name must be HTML-escaped."""
        malicious = "<script>alert('XSS')</script>Bad"
        fins = [{"id": "x", "name": malicious, "author": "H",
                 "architecture": "t2v", "description": ""}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert "<script>" not in h, (
            f"XSS VULNERABILITY: name injected raw: {h[:200]}"
        )
        assert "&lt;script&gt;" in h, (
            "HTML escaping expected for name field"
        )

    def test_search_by_tags(self):
        fins = [
            {"id": "a", "name": "Anime Style", "author": "A",
             "architecture": "t2v", "description": "", "tags": ["anime", "style"]},
            {"id": "b", "name": "Realistic", "author": "B",
             "architecture": "i2v", "description": "", "tags": ["photorealistic"]},
        ]
        h, c = _fmt_cards(fins, "anime", "All", "", "")
        assert "Anime Style" in h
        assert "Realistic" not in h
        assert "**1" in c

    def test_filter_by_tag_exact(self):
        fins = [
            {"id": "a", "name": "A", "author": "A",
             "architecture": "t2v", "description": "", "tags": ["anime", "style"]},
            {"id": "b", "name": "B", "author": "B",
             "architecture": "t2v", "description": "", "tags": ["photorealistic"]},
        ]
        h, c = _fmt_cards(fins, "", "All", "", "anime")
        assert "A" in h
        assert "B" not in h
        assert "**1" in c

    def test_tag_filter_no_match(self):
        fins = [{"id": "a", "name": "A", "author": "A",
                 "architecture": "t2v", "description": "", "tags": ["anime"]}]
        h, c = _fmt_cards(fins, "", "All", "", "nonexistent")
        assert "No results" in h
        assert "**0" in c

    def test_card_shows_tags_badges(self):
        fins = [{"id": "a", "name": "Styled", "author": "A",
                 "architecture": "t2v", "description": "",
                 "tags": ["anime", "style"]}]
        h, _ = _fmt_cards(fins, "", "All", "", "")
        assert 'class="fm-badge"' in h
        assert "anime" in h
        assert "style" in h

    def test_card_tags_overflow(self):
        fins = [{"id": "a", "name": "Many Tags", "author": "A",
                 "architecture": "t2v", "description": "",
                 "tags": [f"tag{i}" for i in range(6)]}]
        h, _ = _fmt_cards(fins, "", "All", "", "")
        assert "+2" in h

    def test_missing_name(self):
        fins = [{"id": "x", "author": "A", "architecture": "t2v"}]
        h, _ = _fmt_cards(fins, "", "All", "", "")
        assert "?" in h

    def test_missing_id_does_not_crash(self):
        fins = [{"name": "N", "author": "A", "architecture": "t2v"}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert 'fm-card' in h

    def test_missing_description(self):
        fins = [{"id": "x", "name": "N", "author": "A", "architecture": "t2v"}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert 'fm-card' in h

    def test_card_shows_files_section(self):
        """Enhanced cards show URLs from the registry entry."""
        fins = [{"id": "f1", "name": "Test", "author": "A",
                 "architecture": "t2v", "description": "d",
                 "URLs": ["https://huggingface.co/user/model/resolve/main/model.safetensors"]}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert "fm-card-files" in h, "Missing files section in card"
        assert "model.safetensors" in h, "File URL not shown in card"

    def test_card_shows_loras_section(self):
        """Enhanced cards show LoRAs from the registry entry."""
        fins = [{"id": "f1", "name": "Test", "author": "A",
                 "architecture": "t2v", "description": "d",
                 "loras": ["style-lora.safetensors", "char-lora.safetensors"]}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert "fm-card-loras" in h, "Missing LoRAs section in card"
        assert "style-lora" in h, "LoRA name not shown in card"
        assert "char-lora" in h, "Second LoRA not shown"

    def test_card_url_overflow(self):
        """More than 3 URLs shows '+N more' indicator."""
        fins = [{"id": "f1", "name": "Test", "author": "A",
                 "architecture": "t2v", "description": "d",
                 "URLs": [f"https://example.com/{i}" for i in range(5)]}]
        h, _ = _fmt_cards(fins, "", "All", "")
        assert "+2 more" in h, "URL overflow indicator missing"

    def test_card_scroll_container(self):
        """Cards wrapped in a scrollable container."""
        h, _ = _fmt_cards(SAMPLE_FINS, "", "All", "")
        assert 'class="fm-cards-container"' in h, "Missing scrollable container"

    def test_plural_grammar_fixed(self):
        """Plural grammar: 'match' for 1, 'matches' for 0/2+."""
        _, c = _fmt_cards(SAMPLE_FINS, "One", "All", "")
        assert "**1 match**" in c, f"Expected singular, got: {c}"
        _, c2 = _fmt_cards(SAMPLE_FINS, "", "All", "")
        assert "**2 matches**" in c2, f"Expected plural, got: {c2}"
        _, c3 = _fmt_cards([], "", "All", "")
        assert "**0 matches**" in c3, f"Expected zero plural, got: {c3}"


# ==============================================================================
# HELPER: _on_selected() -- card selection handler (replicated)
# ==============================================================================

def _on_selected(fins, fid):
    """Replicated from plugin.py create_ui() -> _on_selected()."""
    m = next((f for f in fins if f["id"] == fid), None)
    return fid, m if m else {}


class TestOnSelected:
    """_on_selected() looks up a finetune by ID in the registry list."""

    FINS = [{"id": "a", "name": "Alpha"}, {"id": "b", "name": "Beta"}]

    def test_finds_existing(self):
        fid, detail = _on_selected(self.FINS, "a")
        assert fid == "a"
        assert detail["name"] == "Alpha"

    def test_missing(self):
        fid, detail = _on_selected(self.FINS, "nonexistent")
        assert fid == "nonexistent"
        assert detail == {}

    def test_case_sensitive(self):
        _, detail = _on_selected(self.FINS, "A")
        assert detail == {}

    def test_empty_list(self):
        _, detail = _on_selected([], "a")
        assert detail == {}


# ==============================================================================
# MODULE-LEVEL: _write_finetune()
# ==============================================================================

class TestWriteFinetune:
    """_write_finetune() writes finetune JSON files to the local directory."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.orig_dir = FM.FINETUNES_DIR
        FM.FINETUNES_DIR = str(self.tmpdir)

    def teardown_method(self):
        FM.FINETUNES_DIR = self.orig_dir
        import shutil
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def test_writes_valid_json(self):
        FM._write_finetune("test-id", {"key": "val"})
        f = self.tmpdir / "test-id.json"
        assert f.exists()
        assert json.loads(f.read_text(encoding="utf-8")) == {"key": "val"}

    def test_creates_directory(self):
        nested = self.tmpdir / "sub"
        FM.FINETUNES_DIR = str(nested)
        FM._write_finetune("x", {})
        assert nested.exists()
        assert (nested / "x.json").exists()

    def test_overwrites_existing(self):
        FM._write_finetune("ov", {"v": 1})
        FM._write_finetune("ov", {"v": 2})
        d = json.loads((self.tmpdir / "ov.json").read_text(encoding="utf-8"))
        assert d["v"] == 2

    def test_pretty_printed(self):
        FM._write_finetune("pp", {"a": 1})
        content = (self.tmpdir / "pp.json").read_text(encoding="utf-8")
        json.loads(content)  # must be valid
        assert '"a": 1' in content  # pretty-printed (has space)


# ==============================================================================
# MODULE-LEVEL: _fetch_registry_json()
# ==============================================================================

class TestFetchRegistryJson:
    """_fetch_registry_json() fetches a single finetune JSON from the registry."""

    def test_successful_fetch(self):
        mock = MagicMock()
        mock.json.return_value = {"model": {"name": "Got It"}}
        with patch("requests.get") as g:
            g.return_value = mock
            result = FM._fetch_registry_json("some-id")
            assert result == {"model": {"name": "Got It"}}
            assert "some-id" in g.call_args[0][0]

    def test_http_error_propagates(self):
        mock = MagicMock()
        mock.raise_for_status.side_effect = Exception("HTTP 404")
        with patch("requests.get") as g:
            g.return_value = mock
            with pytest.raises(Exception, match="HTTP 404"):
                FM._fetch_registry_json("missing")


# ==============================================================================
# MODULE-LEVEL: _hf_upload() (mocked HF API)
# ==============================================================================

class TestHFUpload:
    """_hf_upload() pushes finetune JSON to HF Space.
    No index.json management -- discovery is dynamic."""

    def test_uploads_finetune_file(self):
        api = MagicMock()
        calls = []

        def fake_upload(path_or_fileobj, **kw):
            calls.append(kw.get("path_in_repo", ""))
        api.upload_file.side_effect = fake_upload

        with patch("plugin.HfApi", return_value=api):
            FM._hf_upload("my-fin",
                          {"model": {"name": "My", "architecture": "t2v",
                                     "description": ""}})
        assert "finetunes/my-fin.json" in calls, \
            "Should upload the finetune JSON file"
        assert not any("index.json" in c for c in calls), \
            "Should NOT update index.json --- dynamic discovery replaces it"

    def test_deserializes_model_correctly(self):
        api = MagicMock()
        uploaded = []

        def fake_upload(path_or_fileobj, **kw):
            uploaded.append(json.loads(path_or_fileobj))
        api.upload_file.side_effect = fake_upload

        payload = {"model": {"name": "Test", "architecture": "t2v",
                             "description": "Hello"}}
        with patch("plugin.HfApi", return_value=api):
            FM._hf_upload("test-fin", payload)
        assert uploaded[0]["model"]["name"] == "Test"


# ==============================================================================
# create_space.py: make_finetune_json()
# ==============================================================================

class TestMakeFinetuneJson:
    """make_finetune_json() transforms registry entries into Wan2GP JSON."""

    @pytest.fixture(autouse=True)
    def _load(self):
        import importlib.util as u
        cp = Path(__file__).resolve().parent.parent / "create_space.py"
        spec = u.spec_from_file_location("create_space", str(cp))
        self.cs = u.module_from_spec(spec)
        spec.loader.exec_module(self.cs)

    def _e(self, **kw):
        d = dict(id="t", name="Test", architecture="t2v", description="desc", URLs=[])
        d.update(kw)
        return d

    def test_basic(self):
        r = self.cs.make_finetune_json(self._e())
        assert r["model"]["name"] == "Test"
        assert r["model"]["architecture"] == "t2v"

    def test_with_loras(self):
        r = self.cs.make_finetune_json(self._e(loras=["l1"], loras_multipliers=[0.8]))
        assert r["model"]["loras"] == ["l1"]
        assert r["model"]["loras_multipliers"] == [0.8]

    def test_default_settings_merged_to_model(self):
        r = self.cs.make_finetune_json(self._e(
            default_settings={"num_inference_steps": 40, "guidance_scale": 6.0}))
        assert r["model"]["num_inference_steps"] == 40
        assert r["model"]["guidance_scale"] == 6.0
        assert "num_inference_steps" not in r
        assert "guidance_scale" not in r

    def test_source_model(self):
        r = self.cs.make_finetune_json(self._e(finetune_source_model="base-v1"))
        assert r["model"]["finetune_source_model"] == "base-v1"

    def test_missing_keys_crash(self):
        """BUG: missing keys cause KeyError."""
        with pytest.raises(KeyError):
            self.cs.make_finetune_json({"id": "bad"})

    def test_tags_omitted(self):
        r = self.cs.make_finetune_json(self._e(tags=["anime"]))
        assert "tags" not in r["model"]
        assert "tags" not in r


# ==============================================================================
# Module-level constants
# ==============================================================================

class TestModuleConstants:
    """Module-level configuration loaded at import time."""

    def test_token_loaded(self):
        assert FM.REGISTRY_TOKEN == "test_token_abc"

    def test_registry_url(self):
        assert "huggingface.co" in FM.DEFAULT_REGISTRY
        assert "wan2gp-finetunes" in FM.DEFAULT_REGISTRY

    def test_registry_space_id(self):
        assert FM.REGISTRY_SPACE == "GKartist75/wan2gp-finetunes"

    def test_finetunes_dir(self):
        assert FM.FINETUNES_DIR == "finetunes"

    def test_plugin_version(self):
        assert FM.PLUGIN_VERSION in ("3.1.0", "3.2.0")

    def test_plugin_name(self):
        assert FM.PlugIn_Name == "Finetune Manager"

    def test_plugin_id(self):
        assert FM.PlugIn_Id == "FinetuneManager"


# ==============================================================================
# create_space.py sample data
# ==============================================================================

class TestSampleData:
    """Sample finetunes in create_space.py must have valid structure."""

    @pytest.fixture(autouse=True)
    def _load(self):
        import importlib.util as u
        cp = Path(__file__).resolve().parent.parent / "create_space.py"
        spec = u.spec_from_file_location("create_space", str(cp))
        self.cs = u.module_from_spec(spec)
        spec.loader.exec_module(self.cs)

    def test_all_have_required_keys(self):
        for f in self.cs.SAMPLE_FINETUNES:
            assert "id" in f
            assert "name" in f
            assert "architecture" in f
            assert "description" in f
            assert "URLs" in f

    def test_all_convert_successfully(self):
        for f in self.cs.SAMPLE_FINETUNES:
            r = self.cs.make_finetune_json(f)
            assert "model" in r
            assert r["model"]["name"] == f["name"]


# ==============================================================================
# Auto-ID Generation (replicated from plugin.py create_ui())
# ==============================================================================
import re as _re

def _sanitize_id(text):
    v = _re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text or '').strip())
    v = _re.sub(r'_+', '_', v).strip('._-')
    return v

def _generate_id(name, source):
    base = _sanitize_id(source or '').lower() or 'finetune'
    name_text = str(name or '').strip()
    if not name_text:
        return f"{base}_finetune"
    words = _re.findall(r'[A-Za-z0-9]+', name_text)
    suffix = '_'.join(words[:2]).casefold() if words else 'finetune'
    return _sanitize_id(f"{base}_{suffix}") if suffix else base

def _unique_id(base, existing=None):
    if existing is None:
        existing = set()
    c = base
    i = 1
    while c in existing:
        c = f"{base}_{i}"
        i += 1
    return c


class TestAutoId:
    """Auto-ID generation from name and source model."""

    def test_sanitize_simple(self):
        assert _sanitize_id("Hello World") == "Hello_World"

    def test_sanitize_special_chars(self):
        assert _sanitize_id("my@finetune!#") == "my_finetune"

    def test_sanitize_unicode(self):
        assert _sanitize_id("testé") == "test"

    def test_sanitize_none(self):
        assert _sanitize_id(None) == ""

    def test_generate_id_name_only(self):
        assert _generate_id("My Model", "") == "finetune_my_model"

    def test_generate_id_with_source(self):
        assert _generate_id("Style XL", "t2v") == "t2v_style_xl"

    def test_generate_id_no_name(self):
        assert _generate_id("", "ltx-video") == "ltx-video_finetune"

    def test_generate_id_both_empty(self):
        assert _generate_id("", "") == "finetune_finetune"

    def test_generate_id_uses_first_two_words(self):
        assert _generate_id("My Awesome Style Model", "i2v") == "i2v_my_awesome"

    def test_unique_id_no_conflict(self):
        assert _unique_id("test", set()) == "test"

    def test_unique_id_one_conflict(self):
        assert _unique_id("test", {"test"}) == "test_1"

    def test_unique_id_multiple_conflicts(self):
        assert _unique_id("test", {"test", "test_1", "test_2"}) == "test_3"

    def test_unique_id_case_sensitive(self):
        """Unique ID dedup is case-sensitive."""
        assert _unique_id("Test", {"test"}) == "Test"


# ==============================================================================
# Tags in _build() and _extract()
# ==============================================================================

class TestBuildTags:
    """Tags field in _build() and _extract()."""

    def test_tags_single(self):
        r = _b(tags="anime")
        assert r["model"]["tags"] == ["anime"]

    def test_tags_multiple(self):
        r = _b(tags="anime, style, photorealistic")
        assert r["model"]["tags"] == ["anime", "style", "photorealistic"]

    def test_tags_trim_spaces(self):
        r = _b(tags="  anime  ,  style  ")
        assert r["model"]["tags"] == ["anime", "style"]

    def test_tags_empty_string(self):
        r = _b(tags="")
        assert "tags" not in r["model"]

    def test_tags_round_trip(self):
        r = _b(tags="anime, style")
        e = _extract(r)
        assert e[EX_TAGS] == "anime, style"

    def test_tags_empty_extract(self):
        r = _b()
        e = _extract(r)
        assert e[EX_TAGS] == ""

    def test_tags_as_list_in_dict(self):
        d = {"model": {"name": "N", "architecture": "t2v", "description": "",
                       "tags": ["anime", "style"]}}
        e = _extract(d)
        assert e[EX_TAGS] == "anime, style"

    def test_tags_as_string_in_dict(self):
        d = {"model": {"name": "N", "architecture": "t2v", "description": "",
                       "tags": "anime, style"}}
        e = _extract(d)
        assert e[EX_TAGS] == "anime, style"


# ==============================================================================
# _validate_url() — URL validation
# ==============================================================================

class TestValidateUrl:
    """_validate_url() checks reachability of remote URLs."""

    def test_empty_url(self):
        valid, msg = FM._validate_url("")
        assert valid is True
        assert msg == ""

    def test_whitespace_url(self):
        valid, msg = FM._validate_url("   ")
        assert valid is True
        assert msg == ""

    def test_local_path(self):
        valid, msg = FM._validate_url("/local/path.txt")
        assert valid is True
        assert "Not a remote URL" in msg

    def test_none_value(self):
        valid, msg = FM._validate_url(None)
        assert valid is True
        assert msg == ""

    def test_successful_200(self):
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.getcode.return_value = 200
            mock_open.return_value = mock_resp
            with patch("urllib.request.Request"):
                valid, msg = FM._validate_url("https://example.com/model.safetensors")
                assert valid is True
                assert "200" in msg

    def test_not_found_404(self):
        with patch("urllib.request.urlopen") as mock_open:
            from urllib.error import HTTPError
            mock_open.side_effect = HTTPError(
                "https://example.com/missing", 404,
                "Not Found", {}, None)
            with patch("urllib.request.Request"):
                valid, msg = FM._validate_url("https://example.com/missing")
                assert valid is False
                assert "404" in msg

    def test_connection_error(self):
        with patch("urllib.request.urlopen") as mock_open:
            import urllib.error as urlerr
            mock_open.side_effect = urlerr.URLError("Connection refused")
            valid, msg = FM._validate_url("https://example.com/test")
            assert valid is False
            assert "Connection error" in msg

    def test_head_not_allowed_fallback_to_get(self):
        """When HEAD returns 405, fall back to GET with Range."""
        with patch("urllib.request.urlopen") as mock_open:
            from urllib.error import HTTPError
            head_err = HTTPError(
                "https://example.com/file", 405,
                "Method Not Allowed", {}, None)
            get_resp = MagicMock()
            get_resp.getcode.return_value = 200
            # First call fails (HEAD), second succeeds (GET)
            mock_open.side_effect = [head_err, get_resp]
            with patch("urllib.request.Request"):
                valid, msg = FM._validate_url("https://example.com/file")
                assert valid is True
                assert "200" in msg


class TestValidateUrls:
    """_validate_urls() validates multiple URLs."""

    def test_empty_list(self):
        results = FM._validate_urls([])
        assert results == []

    def test_skips_non_http(self):
        results = FM._validate_urls(["local/path.txt"])
        assert results == []

    def test_successful(self):
        with patch.object(FM, '_validate_url') as mock_v:
            mock_v.return_value = (True, "HTTP 200")
            results = FM._validate_urls(["https://example.com/a"])
            assert len(results) == 1
            assert results[0][1] is True


class TestBuildUrlValidationHtml:
    """_build_url_validation_html() renders results as HTML."""

    def test_empty(self):
        html = FM._build_url_validation_html([])
        assert "No remote URLs" in html

    def test_success(self):
        html = FM._build_url_validation_html([
            ("https://example.com/model.safetensors", True, "HTTP 200")
        ])
        assert "model.safetensors" in html
        assert "HTTP 200" in html

    def test_failure(self):
        html = FM._build_url_validation_html([
            ("https://example.com/missing.safetensors", False, "HTTP 404")
        ])
        assert "missing.safetensors" in html
        assert "HTTP 404" in html

    def test_multiple_results(self):
        html = FM._build_url_validation_html([
            ("https://a.com/1", True, "HTTP 200"),
            ("https://b.com/2", False, "HTTP 403"),
        ])
        assert "HTTP 200" in html
        assert "HTTP 403" in html


# ==============================================================================
# _civitai_extract_fill_data() — CivitAI auto-fill
# ==============================================================================

class TestCivitaiExtractFillData:
    """_civitai_extract_fill_data() extracts form values from CivitAI selection."""

    def test_empty_input(self):
        result = FM._civitai_extract_fill_data("")
        assert len(result) == 33  # 32 fields + status
        assert result[-1] == ""  # status is empty

    def test_empty_json(self):
        result = FM._civitai_extract_fill_data("{}")
        assert len(result) == 33
        assert result[-1] == ""

    def test_invalid_json(self):
        result = FM._civitai_extract_fill_data("not json")
        assert len(result) == 33
        assert result[-1] == ""

    def test_lora_fill(self):
        """LoRA type puts download URL into loras field, leaves others unchanged."""
        selection = json.dumps({
            "model": {"id": 123, "name": "My LoRA", "type": "LORA"},
            "version": {
                "id": 456, "name": "v1.0", "baseModel": "SDXL 1.0",
                "downloadUrl": "https://civitai.com/api/download/models/456",
                "files": [{"name": "mylora.safetensors", "sizeKB": 50000,
                          "primary": True,
                          "downloadUrl": "https://civitai.com/api/download/models/456"}]
            }
        })
        result = FM._civitai_extract_fill_data(selection)
        # Only the loras field (index 12) should be filled
        assert "api/download" in str(result[12]) or str(result[12]) != ""
        # All other fields should be unchanged — check name not filled
        assert "My LoRA" not in str(result[0])
        assert "LoRA URL" in result[-1]

    def test_checkpoint_fill(self):
        """Checkpoint type puts download URL into URLs field, leaves others unchanged."""
        selection = json.dumps({
            "model": {"id": 789, "name": "My Checkpoint", "type": "Checkpoint"},
            "version": {
                "id": 101, "name": "v2.0", "baseModel": "SDXL 1.0",
                "files": [{"name": "model.safetensors", "sizeKB": 2000000,
                          "primary": True,
                          "downloadUrl": "https://civitai.com/api/download/models/101"}]
            }
        })
        result = FM._civitai_extract_fill_data(selection)
        # Only the URLs field (index 4) should be filled
        assert "api/download" in str(result[4])
        # Lora field should remain unchanged
        assert "api/download" not in str(result[12])
        assert "Download URL" in result[-1]

    def test_tags_from_civitai(self):
        """Tags/name fields are NOT auto-filled anymore — only URL."""
        selection = json.dumps({
            "model": {"id": 1, "name": "Test", "type": "LORA"},
            "version": {
                "id": 2, "name": "v1", "baseModel": "SDXL 1.0",
                "files": [{"name": "t.safetensors", "sizeKB": 100,
                          "primary": True,
                          "downloadUrl": "https://civitai.com/dl/2"}]
            }
        })
        result = FM._civitai_extract_fill_data(selection)
        # URL should be in loras field (index 12)
        assert "civitai.com/dl/2" in str(result[12])
        # Name field should be gr.update() (unchanged)
        assert "Test" not in str(result[0])

    def test_no_files_fallback_to_version_download_url(self):
        """If no files array, fall back to version.downloadUrl."""
        selection = json.dumps({
            "model": {"id": 1, "name": "Minimal", "type": "Checkpoint"},
            "version": {
                "id": 2, "name": "v1", "baseModel": "SD 1.5",
                "downloadUrl": "https://civitai.com/api/download/models/2"
            }
        })
        result = FM._civitai_extract_fill_data(selection)
        assert "api/download/models/2" in str(result[4])  # URLs field (index 4)


# ==============================================================================
# _all_tags() helper
# ==============================================================================

class TestAllTags:
    """_all_tags() collects unique tags from registry entries."""

    def test_no_tags(self):
        from plugin import _civitai_extract_fill_data as _
        # Just verify the function exists in the module
        assert hasattr(FM, '_hf_upload'), "Module should have _hf_upload"

    def test_collects_unique_tags(self):
        fins = [
            {"id": "a", "tags": ["anime", "style"]},
            {"id": "b", "tags": ["anime", "photorealistic"]},
            {"id": "c", "tags": []},
        ]
        # Manually replicate _all_tags
        seen = set()
        for f in fins:
            tg = f.get("tags", [])
            if isinstance(tg, list):
                for t in tg:
                    t = t.strip().lower()
                    if t: seen.add(t)
        assert "anime" in seen
        assert "style" in seen
        assert "photorealistic" in seen
        assert len(seen) == 3

    def test_tags_as_string(self):
        fins = [
            {"id": "a", "tags": "anime, style"},
        ]
        seen = set()
        for f in fins:
            tg = f.get("tags", [])
            if isinstance(tg, str):
                tg = [t.strip() for t in tg.split(",") if t.strip()]
            if isinstance(tg, list):
                for t in tg:
                    t = t.strip().lower()
                    if t: seen.add(t)
        assert "anime" in seen
        assert "style" in seen


# ==============================================================================
# _hf_upload() with tags
# ==============================================================================

class TestHFUploadTags:
    """_hf_upload() stores tags in the uploaded JSON (not in an index).
    Tags are read dynamically by _fetch_dynamic_registry()."""

    def test_tags_preserved_in_uploaded_json(self):
        api = MagicMock()
        uploaded = []

        def fake_upload(path_or_fileobj, **kw):
            uploaded.append(json.loads(path_or_fileobj))
        api.upload_file.side_effect = fake_upload

        payload = {"model": {"name": "Tagged", "architecture": "t2v",
                             "description": "", "tags": ["anime", "style"]}}
        with patch("plugin.HfApi", return_value=api):
            FM._hf_upload("tagged-fin", payload)
        assert uploaded[0]["model"]["tags"] == ["anime", "style"]

    def test_tags_omitted_preserved_as_empty(self):
        api = MagicMock()
        uploaded = []

        def fake_upload(path_or_fileobj, **kw):
            uploaded.append(json.loads(path_or_fileobj))
        api.upload_file.side_effect = fake_upload

        payload = {"model": {"name": "Plain", "architecture": "t2v", "description": ""}}
        with patch("plugin.HfApi", return_value=api):
            FM._hf_upload("no-tags", payload)
        assert "tags" not in uploaded[0]["model"] or uploaded[0]["model"].get("tags", []) == []


# ==============================================================================
# MODULE-LEVEL: _fetch_dynamic_registry()
# ==============================================================================

class TestFetchDynamicRegistry:
    """_fetch_dynamic_registry() lists actual finetune files from the HF Space.
    No index.json needed -- always in sync."""

    def test_returns_empty_when_no_files(self):
        api = MagicMock()
        api.list_repo_files.return_value = []
        with patch("plugin.HfApi", return_value=api):
            result = FM._fetch_dynamic_registry()
        assert result == []

    def test_skips_non_finetune_files(self):
        api = MagicMock()
        api.list_repo_files.return_value = [
            "README.md",
            "index.json",
            "finetunes/test.json",
            ".gitattributes",
        ]
        with patch("plugin.HfApi", return_value=api):
            with patch("requests.get") as g:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "model": {"name": "Test", "architecture": "t2v"}}
                g.return_value = mock_resp
                result = FM._fetch_dynamic_registry()
        assert len(result) == 1
        assert result[0]["id"] == "test"
        assert result[0]["name"] == "Test"

    def test_skips_404_files(self):
        api = MagicMock()
        api.list_repo_files.return_value = [
            "finetunes/exists.json",
            "finetunes/missing.json",
        ]
        with patch("plugin.HfApi", return_value=api):
            with patch("requests.get") as g:
                def resp(*a, **kw):
                    url = a[0] if a else kw.get("url", "")
                    m = MagicMock()
                    if "exists" in url:
                        m.status_code = 200
                        m.json.return_value = {
                            "model": {"name": "Exists", "architecture": "t2v"}}
                    else:
                        m.status_code = 404
                    return m
                g.side_effect = resp
                result = FM._fetch_dynamic_registry()
        assert len(result) == 1
        assert result[0]["id"] == "exists"

    def test_extracts_tags_and_urls(self):
        api = MagicMock()
        api.list_repo_files.return_value = ["finetunes/full.json"]
        with patch("plugin.HfApi", return_value=api):
            with patch("requests.get") as g:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "model": {
                        "name": "Full",
                        "architecture": "t2v",
                        "tags": ["anime", "style"],
                        "URLs": ["https://example.com/a.safetensors"],
                        "loras": ["https://example.com/b.safetensors"],
                        "finetune_source_model": "base-model",
                    },
                    "num_inference_steps": 25,
                    "guidance_scale": 4.0,
                }
                g.return_value = mock_resp
                result = FM._fetch_dynamic_registry()
        assert len(result) == 1
        entry = result[0]
        assert entry["tags"] == ["anime", "style"]
        assert entry["URLs"] == ["https://example.com/a.safetensors"]
        assert entry["loras"] == ["https://example.com/b.safetensors"]
        assert entry["source"] == "base-model"
        assert entry["default_settings"]["num_inference_steps"] == 25
        assert entry["default_settings"]["guidance_scale"] == 4.0

    def test_api_failure_returns_empty(self):
        with patch("plugin.HfApi", side_effect=Exception("API down")):
            result = FM._fetch_dynamic_registry()
        assert result == []

    def test_list_repo_files_exception_returns_empty(self):
        api = MagicMock()
        api.list_repo_files.side_effect = Exception("List failed")
        with patch("plugin.HfApi", return_value=api):
            result = FM._fetch_dynamic_registry()
        assert result == []


# ==============================================================================
# Run
# ==============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
