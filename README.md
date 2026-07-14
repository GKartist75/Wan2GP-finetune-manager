# Wan2GP Finetune Manager

[![Wan2GP](https://img.shields.io/badge/Wan2GP-v12.3-blue)]()
[![Version](https://img.shields.io/badge/version-3.2.0-brightgreen)]()
[![Status](https://img.shields.io/badge/status-active-success)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

A **Wan2GP plugin** for browsing, creating, editing, and sharing community finetune configurations.

---

## Features

### Browse & Discover
- **Registry Browser** — Browse community finetunes from the HF Registry
- **Search & Filter** — Search by name, description, author, or tags; filter by architecture
- **Card UI** — Scrollable card view with descriptions, URLs, LoRAs, and tag badges
- **URL Validation** — Check URL reachability with per-URL green/red status indicators

### Load & Switch
- **Download & Switch** — Download a finetune JSON and immediately activate it in Wan2GP
- **Download** — Download locally without switching the active model
- **Pre-Download** — Download checkpoint/LoRA files before loading

### Create, Edit & Improve
- **Full Finetune Editor** — Tabbed interface: URLs, LoRAs, Resolutions, Help, Prompt Enhancer, Settings
- **Markdown Toolbar** — Bold, Italic, Heading, List, Link, Code buttons
- **Auto-ID** — Automatic ID generation with deduplication
- **Creator/Editor Mode** — Context-sensitive action buttons
- **Improve / Create Variant** — Create variant from registry, pre-fill editor
- **Tags & Categories** — Comma-separated tags with badge display and filtering

### CivitAI Integration
- **CivitAI Browser** — Search CivitAI models directly from Wan2GP
- **Filter & Browse** — Filter by type (Checkpoint, LoRA) and base model
- **One-Click Fill** — Select a version, fill the download URL into the editor

### Upload & Share
- **Upload to HF Registry** — Contribute finetune JSONs to the community registry
- **Index Integration** — Automatic visibility in the Browse tab
- **Tags included** — Full metadata in registry entries

---

## Installation

1. Open Wan2GP → **Plugins** tab → **Available Plugins**
2. Find "Finetune Manager" and click **Enable**

Or manually copy the `wan2gp-finetune-manager` folder into your Wan2GP `plugins/` directory.

---

## Usage

1. Click the **Finetune Manager** tab
2. Click **Refresh Registry** to load finetunes
3. Click a card to select it
4. Choose: **Download**, **Download & Switch**, or **Improve / Create Variant**

---

## Development

```bash
pytest test_plugin.py -v    # 147 unit tests
```

---

## License

MIT License — see [LICENSE](./LICENSE)

---

Made for the Wan2GP community by GKartist75
