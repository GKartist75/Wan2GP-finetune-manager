# Wan2GP Finetune Manager Plugin v3.4.0

Community finetune registry plugin for Wan2GP. Browse, load, create, improve, and upload finetune JSONs with an integrated Create/Edit tab that matches the built-in Finetune Editor (Alt+F).

> **Latest version: v3.4.0** — [GitHub](https://github.com/GKartist75/Wan2GP-finetune-manager) | [HF Registry](https://huggingface.co/spaces/GKartist75/wan2gp-finetunes)

## Features

| Tab | Functionality |
|---|---|
| **Browse** | Refresh registry, search by name/description/author/ID, filter by architecture or tag, click cards for JSON details, validate URLs, download locally, improve/create variant |
| **Create / Edit** | Full tabbed form (URLs, LoRAs, Resolutions, Help, Prompt Enhancer, Settings) with Auto-ID, Markdown toolbar, Creator/Editor mode, Import/Export, URL validation, pre-download, two-step delete. All URL/LoRA fields include a **Browse 📂** button for native file selection. |
| **Local** | Visual card browser, load & switch, delete, export via DownloadButton, import .json, upload to registry |
| **Upload** | Pick a local finetune from dropdown, preview JSON, upload to community registry |
| **CivitAI** | Search CivitAI by query/type/base model, browse results with version details, one-click URL fill (routes to LoRA or main checkpoint automatically) |

## Install

### Easiest — Install from URL (in Wan2GP)
1. Open Wan2GP → **Plugins** tab
2. Click **Install from URL** (expand the section)
3. Paste the GitHub URL into the **GitHub URL** field:

   ```
   https://github.com/GKartist75/Wan2GP-finetune-manager
   ```

4. Click **Download and Install from URL**
5. Restart Wan2GP when prompted

### Alternative — Manual
Clone or download into `plugins/finetune_manager`, then restart Wan2GP.

## Upload to Registry

The plugin ships with a registry token. Anyone can upload finetune JSONs to the community registry at `GKartist75/wan2gp-finetunes`. No account or collaborator setup needed.

## Files

```
wan2gp-finetune-manager/
├── __init__.py
├── config.json          # Registry write token (renew at huggingface.co/settings/tokens)
├── plugin.py            # All plugin logic (3368 lines, 5 tabs)
├── plugin_info.json     # Plugin metadata v3.4.0
├── CHANGELOG.md         # Version history
└── .gitignore
```

## What's New in v3.4.0

- **Pure data functions** — `build_finetune_dict()` and `extract_finetune_fields()` separated from UI closures, making the data layer testable and importable
- **File Browser** — all URL and LoRA fields now have a **Browse 📂** button that opens the native file explorer with relevant extension filters
- **Performance** — model index and disk existence caches eliminate disk reads on every keystroke/dropdown change (no more slow "flashing" on load)
- **Type integrity** — `loras_multipliers`, `guidance_scale`, `preload_URLs` preserve their original types (int/float/string/list) during round-trips
- **Surrogate safety** — surrogate characters are stripped before JSON serialisation, preventing orjson crashes
- **CivitAI fill** — search results now auto-route to the correct field (LoRA or main checkpoint)

## 🔗 Links

- **GitHub**: https://github.com/GKartist75/Wan2GP-finetune-manager
- **HF Registry**: https://huggingface.co/spaces/GKartist75/wan2gp-finetunes — browse the finetune registry live with search/filter and click-to-expand JSON details
