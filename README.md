# Wan2GP Finetune Manager Plugin v3.5.0

Community finetune registry plugin for Wan2GP. Browse, load, create, improve, and upload finetune JSONs with an integrated Create/Edit tab that matches the built-in Finetune Editor (Alt+F).

> **Latest version: v3.5.0** — [GitHub](https://github.com/GKartist75/Wan2GP-finetune-manager) | [HF Registry](https://huggingface.co/spaces/GKartist75/wan2gp-finetunes) | [User Guide](./user-guide.html)

## Features

| Tab | Functionality |
|---|---|
| **Browse** | Refresh registry, search by name/description/author/ID, filter by architecture or tag, click cards for JSON details, validate URLs, download locally, improve/create variant |
| **Create / Edit** | Full tabbed form (URLs, LoRAs, Resolutions, Help, Prompt Enhancer, Settings) with Auto-ID, Markdown toolbar, Creator/Editor mode, Import/Export, URL validation, pre-download, two-step delete. All URL/LoRA fields include a **Browse 📂** button for native file selection. |
| **Local** | Visual card browser, load & switch, delete, export, import .json with preview, upload to registry |
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
├── plugin.py            # All plugin logic (4 tabs)
├── plugin_info.json     # Plugin metadata v3.5.0
├── CHANGELOG.md         # Version history
├── user-guide.html      # Friendly walkthrough of all features
└── .gitignore
```

## What's New in v3.5.0

- **User guide** — new `user-guide.html` with a friendly, visual walkthrough of all features and common workflows
- **Bug fixes** — textbox fields no longer get cleared when changing dropdown selections; `finetune_source_model` no longer gets overwritten on improve
- **Import preview** — selecting a .json file for import now shows its contents before you commit
- **Registry cache** — 30-second TTL cache prevents redundant fetches on rapid Refresh clicks
- **Dark mode** — card HTML uses CSS custom properties instead of hardcoded colors, adapting to Wan2GP's theme
- **Round-trip stability** — key ordering preserved when editing and re-uploading finetunes
- **Code deduplication** — shared URL/LoRA key lists and card HTML builder extracted to module-level, removing ~150 lines of duplicated code

## 🔗 Links

- **GitHub**: https://github.com/GKartist75/Wan2GP-finetune-manager
- **HF Registry**: https://huggingface.co/spaces/GKartist75/wan2gp-finetunes — browse the finetune registry live with search/filter and click-to-expand JSON details
- **User Guide**: [user-guide.html](./user-guide.html) — visual walkthrough of all features
