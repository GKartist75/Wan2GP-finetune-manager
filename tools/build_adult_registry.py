#!/usr/bin/env python
"""Convert adult_pack_manifest.json finetunes into adult-registry Space files.

Usage (from repo root):
    python tools/build_adult_registry.py

Output: tools/adult_registry/
    finetunes/<id>.json        Wan2GP finetune defs (URLs pinned, sha-tracked)
    pack/adult_pack_manifest.json  verbatim copy (verify/prefetch layer)
    README.md                  Space card

Only the manifest's ``finetunes`` array is published here; every entry already
carries a pinned, token-free HTTPS URL and SHA-256. The manifest's ``civitai``
resolver catalog stays pack-side (Splashecream LoRAs / unverified civitai
queries need the live resolver at install time).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "adult_pack_manifest.json"
OUT = ROOT / "tools" / "adult_registry"

REPO = "GKartist75/wan2gp-finetunes-adult"

README = """---
title: Wan2GP Adult Finetunes
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: static
pinned: false
license: other
---

# Wan2GP adult finetune registry

Companion registry of the [community registry](https://huggingface.co/spaces/GKartist75/wan2gp-finetunes).
Generated from `adult_pack_manifest.json` by the Wan2GP Finetune Manager
(pack **{version}**, verified against Wan2GP **{verified}**).

- **No media / no explicit content is stored in this Space** — only
  Wan2GP finetune definitions (names, HTTPS download URLs, metadata).
- Applies mostly to H3 / LTX-2.5 finetuned models + Wan 2.2 releases.
- Wan2GP core consumers can read defs from this Space under `finetunes/`.
- Every URL is pinned with SHA-256 in `pack/adult_pack_manifest.json`
  (the verify/prefetch layer still lives pack-side).

## Uploading
Entries tagged `adult` are routed here automatically by the plugin
(v3.8.0+); everything else goes to the community registry. See
`tools/publish_adult_space.py`.

{total} finetunes · generated {generated}
"""


def _fin_id(entry: dict) -> str:
    fid = entry.get("id") or ""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", fid.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "unnamed"


def _model(entry: dict) -> dict:
    m: dict = {
        "name": entry.get("file") or entry.get("name", "?"),
        "architecture": entry.get("architecture", ""),
        "description": entry.get("description", ""),
        "URLs": list(entry.get("urls") or []),
        "author": entry.get("author") or (entry.get("repo") or "").split("/")[0]
        or "anonymous",
        "version": entry.get("version", "1.0.0"),
        "tags": ["adult", entry.get("family", "")],
    }
    note = entry.get("licenseNote")
    if note:
        m["description"] = (m["description"] + " — " + note).strip()
    loras = entry.get("preloadLoras")
    if isinstance(loras, list) and loras:
        m["loras"] = [
            (l.get("url") or l.get("file") or str(l)) if isinstance(l, dict) else str(l)
            for l in loras
        ]
        mult = entry.get("loraMultipliers")
        if isinstance(mult, list) and mult:
            m["loras_multipliers"] = mult
    return {k: v for k, v in m.items() if v != [] or k in ("URLs",)}


def build() -> None:
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST}")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "finetunes").mkdir(parents=True)
    (OUT / "pack").mkdir()

    shutil.copy2(MANIFEST, OUT / "pack" / "adult_pack_manifest.json")

    written = []
    for entry in man.get("finetunes", []):
        fid = _fin_id(entry)
        urls = entry.get("urls") or []
        if not urls:
            print(f"skip (no pinned URL): {fid}", file=sys.stderr)
            continue
        payload = {"model": _model(entry)}
        path = OUT / "finetunes" / f"{fid}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(fid)

    v = man.get("verifiedAgainst", {})
    (OUT / "README.md").write_text(
        README.format(
            version=man.get("manifestVersion"),
            verified=v.get("wan2gp", "?"),
            total=len(written),
            generated=man.get("generatedAt", ""),
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(written)} finetune defs + pack manifest under {OUT}")


if __name__ == "__main__":
    build()
