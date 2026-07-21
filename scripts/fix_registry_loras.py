"""Check and fix finetune JSONs in the HF Space registry.

Promotes model.loras / model.loras_multipliers to top-level
activated_loras / loras_multipliers so Wan2GP's load_settings_from_file
picks them up correctly.
"""
import json, sys, os, tempfile
from pathlib import Path
from huggingface_hub import HfApi

REGISTRY_URL = "https://huggingface.co/spaces/GKartist75/wan2gp-finetunes/raw/main"
SPACE = "GKartist75/wan2gp-finetunes"

# --- find config ---
cfg_paths = [
    Path("E:/DEVELOPMENT/WAN2GP/Finetune Manager/wan2gp-finetune-manager/config.json"),
    Path("C:/Users/gjaku/Wan2GP/plugins/Wan2GP-finetune-manager/config.json"),
]
token = None
for p in cfg_paths:
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
        token = cfg.get("registry_token", "")
        if token:
            break

if not token:
    print("ERROR: no registry_token found in config.json")
    sys.exit(1)

api = HfApi(token=token)

# --- list all finetune files ---
files = api.list_repo_files(repo_id=SPACE, repo_type="space")
fin_files = sorted([
    f for f in files if f.startswith("finetunes/") and f.endswith(".json")
])
print(f"Found {len(fin_files)} finetune JSONs in {SPACE}\n")

# --- process each file ---
import requests
updated = 0
for path_in_repo in fin_files:
    fid = path_in_repo[len("finetunes/"):-len(".json")]
    r = requests.get(f"{REGISTRY_URL}/{path_in_repo}", timeout=10)
    if r.status_code != 200:
        print(f"  {fid}: download failed (HTTP {r.status_code}), skipping")
        continue
    data = r.json()  # the raw JSON from HF (list of dict if it's a list, but usually a dict)

    # Handle if it's a list (some registry formats wrap in an array)
    if isinstance(data, list):
        # Could be a list with one element containing the actual data
        # This is an edge case — let's inspect
        print(f"  {fid}: is a list (len={len(data)}), trying first element")
        if len(data) >= 1 and isinstance(data[0], dict):
            data = data[0]
        else:
            print(f"  {fid}: unexpected list format, skipping")
            continue

    m = data.get("model", {})
    has_model_loras = "loras" in m
    has_top_activated = "activated_loras" in data
    has_top_lm = "loras_multipliers" in data

    changes = []

    if not has_top_activated and has_model_loras:
        loras = m["loras"]
        if isinstance(loras, list):
            data["activated_loras"] = loras
        elif isinstance(loras, str):
            data["activated_loras"] = [loras]
        changes.append("activated_loras (from model.loras)")

    if not has_top_lm and "loras_multipliers" in m:
        lms = m["loras_multipliers"]
        if isinstance(lms, list):
            data["loras_multipliers"] = " ".join(str(x) for x in lms)
        else:
            data["loras_multipliers"] = str(lms)
        changes.append("loras_multipliers (from model.loras_multipliers)")

    if not changes:
        print(f"  {fid}: no changes needed (activated_loras={has_top_activated}, model.loras={has_model_loras})")
        continue

    print(f"  {fid}: ADDING {', '.join(changes)}")
    print(f"       model.loras = {m.get('loras')}")
    print(f"       model.loras_multipliers = {m.get('loras_multipliers')}")

    # Upload back
    payload = json.dumps(data, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=path_in_repo,
        repo_id=SPACE,
        repo_type="space",
    )
    print(f"       ✅ Uploaded")
    updated += 1

print(f"\nDone. {updated} file(s) updated.")
