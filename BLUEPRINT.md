# Wan2GP Finetune Manager — Architectural Blueprint

> **Status:** Active · **Version:** 3.1.0 · **Date:** 2026-07-14
> **Author:** GKartist75
> **Host:** Wan2GP v12.3

---

## 1. System Context & High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Wan2GP Application                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                Plugin Manager (shared/utils/plugins.py)    │  │
│  │  - Loads plugins from plugins/ directory                    │  │
│  │  - Injects host component refs (model_choice_target, etc)  │  │
│  │  - Provides WAN2GPPlugin base class                        │  │
│  └───────────────────┬───────────────────────────────────────┘  │
│                      │ extends                                  │
│  ┌───────────────────▼───────────────────────────────────────┐  │
│  │            FinetuneManagerPlugin (plugin.py)               │  │
│  │  Gradio-based plugin with 4 tabs: Browse | Create/Edit    │  │
│  │  | Local | Upload                                          │  │
│  │  - Reads/writes local finetunes/ directory                 │  │
│  │  - Fetches/pushes to HuggingFace registry Space            │  │
│  │  - Calls refresh_model_defs()/switch_to_model() on host    │  │
│  └───────────────────┬───────────────────────────────────────┘  │
│                      │                                          │
│  ┌───────────────────▼───────────────────────────────────────┐  │
│  │  finetunes/ (local FS directory)                           │  │
│  │  - Wan2GP's model finetune JSON files                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │ HTTP / HF Hub API
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│          HuggingFace Space: GKartist75/wan2gp-finetunes         │
│  Static site served via HF Spaces (SDK: static)                 │
│  Contents:                                                      │
│  ├── index.html      → Browser-based finetune browser           │
│  ├── index.json      → Machine-readable registry index          │
│  ├── finetunes/      → Per-finetune JSON files                  │
│  └── README.md       → Space card                               │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Plugin, not fork** — the manager lives as a Wan2GP plugin, not a modification to the host. This means zero host-side changes for installation and updates.
- **Gradio-native UI** — uses the same Gradio framework as the host's own UI, ensuring visual consistency and access to host-provided Gradio components.
- **Registry as a static HF Space** — the community registry is a static site (no backend), making it free to host, trivially scalable, and easy to mirror.
- **Token-based write access** — uploads use an HF access token stored in `config.json`. This avoids requiring contributors to set up their own HF auth.

---

## 2. Component Map

### 2.1 Plugin (`wan2gp-finetune-manager/plugin.py`)

| Component | Type | Role |
|-----------|------|------|
| `FinetuneManagerPlugin` | Class (extends `WAN2GPPlugin`) | Main plugin entry point |
| `_hf_upload()` | Module-level function | Uploads a finetune JSON + updates index.json on HF Space |
| `_fetch_registry_json()` | Module-level function | Downloads a single finetune JSON from the registry |
| `_write_finetune()` | Module-level function | Writes a finetune JSON to the local `finetunes/` directory |
| `_build()` | Module-level function | Constructs the finetune model dict from form field values |
| `_extract()` | Module-level function | Reads a finetune dict and fills form fields |
| `_fmt_cards()` | Module-level function | Renders browse tab card HTML from finetune list |
| `_loc_list()` / `_loc_detail()` | Module-level functions | List / read local finetune files |
| `REGISTRY_TOKEN` | Module-level variable | Cached write token from `config.json` |
| `DEFAULT_REGISTRY` | Constant | Base URL for the HF Space raw content |
| `REGISTRY_SPACE` | Constant | HF Space repo identifier |
| `FINETUNES_DIR` | Constant | Local directory name: `finetunes` |

### 2.2 Registry Creator (`create_space.py`)

| Component | Role |
|-----------|------|
| `main()` | Orchestrates: create Space → upload README → upload index.html → upload index.json → upload sample finetune JSONs |
| `make_finetune_json()` | Transforms a registry index entry into a full Wan2GP-compatible finetune JSON |
| `SAMPLE_FINETUNES` | Example data: Anime I2V, Cinematic T2V, Pixel Art T2V |
| `INDEX_HTML` | Self-contained static HTML+JS for browsing finetunes in a browser |

### 2.3 Host API Surface (Wan2GP `shared/utils/plugins.py`)

The plugin depends on the following host-provided interfaces:

| API | Type | Usage |
|-----|------|-------|
| `WAN2GPPlugin` | Base class | `setup_ui()`, `add_tab()`, `request_component()`, `request_global()` |
| `self.refresh_model_defs` | Callable | Injected by host during setup; refreshes model definition cache |
| `self.switch_to_model(id, load)` | Callable | Injected by host; switches the active model to a finetune |
| `self.model_choice_target` | Gradio component | Injected; triggers model switching via `gr.update()` |
| `self.main_tabs` | Gradio component | Injected; used to switch to the main model tab |
| `self.state` | Gradio State | Injected by `request_component("state")` |
| `LocalFilePickerTextbox` | Gradio component | Custom file picker from `shared/gradio/local_file_picker.py` |
| `CHECKPOINT_FILE_EXTENSIONS` | Set | Allowed extensions for checkpoint file selection |

---

## 3. Data Flow

### 3.1 Browsing & Loading a Finetune

```
User clicks "Refresh Registry"
  → HTTP GET {DEFAULT_REGISTRY}/index.json
  → Parse finetune[] array
  → Render card HTML in browse tab
  → Store full list in gr.State(registry)

User clicks "Load to Wan2GP" on a card
  → _fetch_registry_json(fid)
      → HTTP GET {DEFAULT_REGISTRY}/finetunes/{fid}.json
  → _write_finetune(fid, data)
      → Save to finetunes/{fid}.json on local FS
  → self.refresh_model_defs()
      → Host rescans the finetunes/ directory
  → self.switch_to_model(fid, False)
      → Host loads the model
  → gr.update() on model_choice_target & main_tabs
```

### 3.2 Creating & Uploading a Finetune

```
User fills form → live JSON preview updates via _build()
User clicks "Save & Upload"
  → _build() creates the full finetune dict from form fields
  → _write_finetune(id, data) saves locally
  → _hf_upload(id, data):
      1. HfApi.upload_file(finetunes/{id}.json)
      2. Download current index.json from HF Space
      3. Upsert entry into finetunes[] list
      4. HfApi.upload_file(index.json)
  → refresh_model_defs()
```

### 3.3 Improve / Create Variant Flow

```
User clicks "Improve" on a registry finetune
  → Download full finetune JSON from registry
  → Set finetune_source_model to the original architecture
  → Generate variant ID: {original_id}_variant
  → Save locally
  → Switch to the variant (opens editor in host)
  → User edits via host's Finetune Editor toolbar
```

---

## 4. Data Schema

### 4.1 Registry Index (`index.json`)

```json
{
  "registry": "wan2gp-finetune-registry",
  "version": 1,
  "finetunes": [
    {
      "id": "anime-i2v",
      "name": "Anime Style I2V",
      "author": "GKartist75",
      "version": "1.0.0",
      "architecture": "i2v",
      "description": "...",
      "source": null,
      "URLs": ["https://.../anime-i2v.safetensors"],
      "loras": [],
      "loras_multipliers": [],
      "default_settings": { "num_inference_steps": 30, "guidance_scale": 5.0 }
    }
  ]
}
```

### 4.2 Finetune JSON (Wan2GP schema — stored in `finetunes/{id}.json`)

```json
{
  "model": {
    "name": "Anime Style I2V",
    "architecture": "i2v",
    "description": "...",
    "URLs": ["https://.../model.safetensors"],
    "URLs2": [],
    "text_encoder_URLs": [],
    "VAE_URLs": [],
    "preload_URLs": [],
    "loras": ["anime-lighting.safetensors"],
    "loras_multipliers": [0.8],
    "modules": [],
    "auto_quantize": false,
    "visible": true,
    "image_outputs": false,
    "finetune_source_model": "",
    "resolutions": [],
    "resolutions_categories": [],
    "infos": "",
    "prompt_infos": "",
    "text_prompt_enhancer_instructions": "",
    "text_prompt_enhancer_max_tokens": null,
    "video_prompt_enhancer_instructions": "",
    "video_prompt_enhancer_max_tokens": null,
    "image_prompt_enhancer_instructions": "",
    "image_prompt_enhancer_max_tokens": null
  },
  "num_inference_steps": 30,
  "guidance_scale": 5.0
}
```

---

## 5. UI Tab Architecture

```
Plugin Tab: "Finetune Manager"
├── Tab: Browse
│   ├── Search textbox + Architecture dropdown + Refresh button
│   ├── Card list (HTML, client-side click via JS dispatch)
│   ├── Detail JSON viewer
│   ├── "Load to Wan2GP" → triggers model switch
│   └── "Improve" → creates variant, opens editor
│
├── Tab: Create / Edit
│   ├── Import JSON (file upload + fill button)
│   ├── Form fields (ID, Name, Architecture, Description)
│   ├── Accordion: URLs (Main, Secondary, Text Encoder, Preload)
│   ├── Accordion: LoRAs
│   ├── Accordion: Advanced (VAE, Modules, Auto-Quantize, etc.)
│   ├── Accordion: Help Text
│   ├── Accordion: Prompt Enhancer
│   ├── Accordion: Settings (Steps, Guidance)
│   ├── Live JSON preview
│   ├── Save Locally / Export JSON / Save & Upload
│   └── Status textbox
│
├── Tab: Local
│   ├── Refresh button + dropdown list
│   ├── Detail JSON viewer
│   ├── Load & Switch / Delete / Upload to Registry
│   ├── Import .json + Import & Switch
│   └── Status textbox
│
└── Tab: Upload
    ├── Dropdown of local finetunes
    ├── Preview JSON
    ├── Upload button
    └── Status textbox
```

---

## 6. External Dependencies

| Dependency | Version (approx.) | Purpose | Source |
|------------|-------------------|---------|--------|
| `gradio` | ≥4.x | UI framework — tabs, buttons, forms, state | Host environment |
| `requests` | any | HTTP calls to registry for download/search | Host environment |
| `huggingface_hub` | ≥0.20 | HfApi for uploads to HF Space | Host environment |
| `pathlib` | stdlib | Cross-platform path handling | stdlib |
| `json` | stdlib | Serialization/deserialization of finetune data | stdlib |

All dependencies are provided by the Wan2GP host environment — the plugin has no `requirements.txt` or `pyproject.toml`.

---

## 7. Security Model

| Concern | Mechanism |
|---------|-----------|
| **Registry write access** | HF token in `config.json` (excluded from version control via `.gitignore`). Token is scoped to the specific Space. |
| **Read access** | Public HF Space — no auth needed for reads |
| **Local file safety** | The plugin only reads/writes to `finetunes/` directory. File operations use `pathlib` with basic existence checks. |
| **User-supplied data** | No sanitization of registry data beyond basic JSON parsing. The host's model loader is responsible for validating finetune JSON structure. |
| **Registry token leakage** | `config.json` in `.gitignore`. The `sync.ps1` deploy script skips `__pycache__` and `.gitignore` but notably does not skip `config.json` — the token is synced to the Wan2GP plugins folder. This is intentional (token is needed at runtime) but means the Wan2GP install directory contains a write-capable token. |

---

## 8. Error Handling Patterns

| Scenario | Behavior |
|----------|----------|
| Network failure fetching registry | Returns empty list, shows "error" message in status |
| Network failure uploading | Upload fails with exception shown in status; local save still succeeds |
| Registry index missing field | Tolerated — field defaults to empty string/list via `.get()` |
| Missing `config.json` | REGISTRY_TOKEN stays empty; upload buttons show "Missing config.json" |
| File not found locally | Returns `{"error": "not found"}` or status message |
| Invalid JSON import | Shows parse error in status, form not filled |
| Missing `model` key in imported JSON | Status shows error, form not filled |

---

## 9. Extension Points

1. **Adding a new architecture** — add to the dropdown choices in the Browse filter and Create/Edit form. No code change beyond the dropdown values list.
2. **Adding new fields to the finetune schema** — add a form field in `Create/Edit` tab, update `_build()` and `_extract()` to pass the field through.
3. **Custom registry** — change `DEFAULT_REGISTRY` and `REGISTRY_SPACE` constants. Extend `config.json` to allow user-configured registry.
4. **Auth for uploads** — the current token-in-config approach can be extended to OAuth or per-user tokens without changing the upload flow.

---

## 10. Known Limitations

1. **No deduplication check** — the registry allows overwriting an existing finetune ID. There's no conflict detection.
2. **No validation on upload** — any valid JSON with a `model` key is accepted. Structure validation is left to the client.
3. **No user attribution enforcement** — `author` is a free-text field. No identity verification.
4. **Token in config.json** — the write token is stored in plaintext on disk. Not suitable for multi-user deployment.
5. **No pagination** — the browse tab loads the entire registry into memory. Fine for dozens of entries; would need pagination for hundreds+.
6. **Sync.ps1 includes token** — the `sync.ps1` deploy script copies `config.json` to the Wan2GP plugins folder, meaning the token is present in two locations.

---

## 11. Development Workflow

```
Edit plugin.py → run sync.ps1 → restart Wan2GP → test in UI

create_space.py: standalone script, run once to bootstrap the registry
  → python create_space.py
  → Uploads sample finetunes + index + UI to HF Space
```

The `sync.ps1` PowerShell script copies all files from `wan2gp-finetune-manager/` to `C:\Users\gjaku\Wan2GP\plugins\wan2gp-finetune-manager\`, excluding `__pycache__` and `.gitignore`.

---

## 12. File Inventory

```
E:\DEVELOPMENT\WAN2GP\Finetune Manager\
├── BLUEPRINT.md                          ← This file
├── IDEA.md                               ← Brief project idea
├── README.md                             ← User-facing documentation
├── create_space.py                       ← Registry bootstrapper script
├── wan2gp-finetune-manager.zip           ← Distribution package
└── wan2gp-finetune-manager/
    ├── __init__.py                       ← Empty package marker
    ├── config.json                       ← HF registry token (gitignored)
    ├── plugin.py                         ← All plugin logic (~530 lines)
    ├── plugin_info.json                  ← Plugin metadata
    ├── sync.ps1                          ← Dev deployment script
    └── .gitignore                        ← Excludes tokens and cache
```
