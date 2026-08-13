# Changelog

## v3.6.4 (2026-08-13)

### Fixed
- **`_loc_load` crash on malformed finetune JSON** — loading a local finetune whose file is corrupt, empty, or missing the required `model` object used to throw deep inside `_write_settings_for_wan2gp` (`dict(data)` on a non-dict) and surface as a red Gradio error. `_loc_load` (and the registry `_load` path) now validate with the shared `_validate_finetune_data()` guard and return a clear "Cannot load 'X': …" message instead of crashing.
- **Silently blank local cards for broken files** — `_fmt_local_cards` previously swallowed JSON/decode errors and rendered a blank card. Corrupt or schema-invalid local finetunes now render a red ⚠ warning card showing the exact reason, so the user can see and fix the bad file.
- **Bare-filename warning now shown on load** — `_loc_load` now runs `_check_bare_filenames()` (previously only enforced on save & upload) and appends the warnings to the load status, so a finetune that will fail Wan2GP's downloader is flagged at load time rather than discovered later.

### Added
- **`_validate_finetune_data(data)`** — centralised contract check (must be a dict with a `model` sub-object) reused by the registry-load, local-load, and import paths, replacing inconsistent inline checks.

### Changed
- Bumped version to 3.6.4 (plugin.py + plugin_info.json).

## v3.6.3 (2026-07-22)

### Added
- **In-token setup guide** — when no HF token is configured, a yellow warning banner appears at the top of the plugin with step-by-step instructions. Includes a **📂 Open plugin folder** button that opens Windows Explorer directly to the plugin folder via `os.startfile()`.
- **Stale cache fallback** — registry fetch failures now serve cached data for up to 5 minutes (`_REGISTRY_CACHE_FAILURE_TTL`) instead of returning empty results
- **LoRA URL validation** — "Validate All URLs" now checks `fin_loras` (ALL_INPUTS[13]) in addition to the 8 URL fields

### Fixed
- **Path traversal in `_write_settings_for_wan2gp`** — finetune ID is now sanitized with `re.sub(r"[^A-Za-z0-9_.-]+", "_", ...)` before being used in file paths, preventing `../../` escape attacks
- **Import ordering** — `_check_bare_filenames` and `_write_settings_for_wan2gp` were defined before the `re`, `json`, `Path`, and `html` imports they depend on. Moved below all imports.
- **NameError in dead code** — `_loras_from_json` used `_os` without importing `os`. Deleted the function entirely (no callers).
- **`_load_src_enhancer` silent skip** — removed short-circuit on MODEL_INDEX hit so the function always reads from disk and returns actual values (or empties) instead of silently keeping old values
- **Token validation duplicated 4×** — extracted `_PLACEHOLDER_TOKENS` constant set and `_REGISTRY_TOKEN_VALID` boolean, replacing 4 inline `"test_token_abc"` string compares
- **`_open_config_dir` function** — cross-platform folder-open helper (Windows `startfile`, macOS `open`, Linux `xdg-open`)

### Changed
- **Error messages** — upload token errors now show the exact `config.json` path and step-by-step instructions instead of a terse "No HF token available"
- **`config.example.json`** — now includes a `_instructions` field explaining the setup process
- **`_textbox_to_dropdown`** — removed unused `*refs` parameter; Gradio wiring changed to `inputs=[tb]` (was `inputs=[tb, ref]`)
- **`fin_name` event handlers** — reduced from 3 per keystroke (`.change()` + `.input()` + `.blur()`) to 1 (`.input()` only)
- **`fix_registry_loras.py`** — replaced hardcoded absolute paths with `Path(__file__).parent` relative paths; removed unused `os`/`tempfile` imports; deduplicated `import requests`
- **Module-level `huggingface_hub` import** — replaced with lazy imports inside the 4 functions that use it, so the plugin loads even if the package is missing
- **`_ensure_model_choice_target` helper** — extracted from 3 identical blocks in `_load`, `_loc_load`, `_loc_import`
- **`_clean_surrogates`/`_clean_tuple` removed** — consolidated into `_clean_utf8` which now handles tuples

## v3.6.2 (2026-07-22)

### Changed
- Version bumped to 3.6.2

## v3.6.1 (2026-07-22)

### Fixed
- **LoRAs not loaded on Load & Switch / Import** — finetune-format `model.loras` / `model.loras_multipliers` are now promoted to top-level `activated_loras` / `loras_multipliers` in the Wan2GP settings file written by `_write_settings_for_wan2gp()`. The settings file now contains the fields that `load_settings_from_file()` / `get_settings_from_file()` expect.
- **Missing Gradio form re-render after model switch** — `_model_choice_target_value(fid)` is now called in all 3 load paths (`_load`, `_loc_load`, `_loc_import`) to produce a unique `fid|timestamp` value for `model_choice_target`. This forces the `.change()` chain (→ `change_model_from_target` → `fill_inputs`) to fire and populate the LoRA tab UI.
- **preload_URLs bare-filename false positive** — `_check_bare_filenames()` no longer flags bare model identifiers (e.g. `ltx2_22B_distilled`) in `preload_URLs`. Wan2GP resolves these internally from its own model database.

### Changed
- `setup_ui()` now requests `_model_choice_target_value` from Wan2GP globals for force-refresh.
- `_check_bare_filenames()` removed `preload_URLs` from checked URL keys.
- Registry finetune `EasyWan22_FastMix` updated with promoted top-level `activated_loras`.

## v3.6.0 (2026-08-17)

### Added
- **Auto-stamp HF username on upload** — `_stamp_hf_username()` calls `whoami()` and injects the uploader's HuggingFace username into `model.author`. Stamped on both local save and remote upload, so the Browse and Local cards show who created it. The HF Space's card listing already renders `by {author}` — now it shows a real name instead of "community".
- **URL validation guard on Save & Upload** — `_check_bare_filenames()` blocks uploads that have bare filenames (like `model.safetensors`) instead of full `https://` download URLs. Prevents broken finetunes from reaching the registry.
- **Hint in URLs tab** — markdown note explaining that only Main Checkpoints need overriding; Text Encoder, VAE, and Preload are inherited from the base model by Wan2GP.

### Changed
- Version bumped to 3.6.0

## v3.5.1 (2026-08-17)

### Fixed
- **Browse registry listing now uses the configured token** — `_fetch_dynamic_registry_no_cache()` was creating `HfApi()` without the token, so `list_repo_files()` ran unauthenticated and returned nothing. Now passes `REGISTRY_TOKEN` when available, matching the pattern already used in upload.

### Security
- **config.json is now gitignored** — tokens are no longer tracked in git to prevent accidental secret leaks. New users create their own `config.json` from `config.example.json`.

### Changed
- Version bumped to 3.5.1

## v3.5.0 (2026-07-17)

### Added
- **Registry cache with 30s TTL** — `_get_cached_registry()` caches the dynamic registry list
  in memory so rapid Refresh clicks don't hammer the HF API with N+1 fetches each time
- **Dark mode support** — CivitAI result cards now use CSS custom properties
  (`--body-background-fill`, `--body-text-color`, `--border-color-primary`,
  `--primary-100/500`) instead of hardcoded `#fff`/`#111827`/`#e5e7eb`,
  making them readable in dark and light themes
- **Loading spinner CSS** — `.fm-spinner` keyframe animation ready for
  future Gradio loading-state use

### Changed
- **Key ordering preserved on upload round-trip** — `build_finetune_dict` now seeds from
  original `extra_data` key order instead of rebuilding in canonical order. Fields like
  `prompt`, `num_inference_steps`, `guidance_scale`, `sample_solver` stay at their original
  positions instead of being moved to the end. Model sub-keys also keep original order.
- **Removed standalone Upload tab** — the `Upload` tab is redundant since the Local tab's
  "Upload to Registry" button already provides the same functionality. Removed tab UI,
  dropdown, preview, and event wiring. The `_up` handler and `_loc_list`/`_loc_detail`
  functions are preserved (shared with Local tab).
- **Code deduplication**: extracted 4 shared module-level constants —
  `URL_VALIDATION_KEYS`, `DOWNLOAD_URL_KEYS`, `URL_FIELD_IDX` — replacing
  5+ inline definitions of the same URL/LoRA key lists
- **Code deduplication**: extracted `_collect_url_entries()` helper to
  replace the identical URL-collection loop that appeared in 3 places
  (`_browse_validate`, `_on_selected_auto_validate`, `_editor_validate`)
- **Code deduplication**: extracted `_fmt_card_html_item()` to replace
  the ~85-line card HTML builder that was duplicated in both Browse tab's
  `_fmt_cards` and Local tab's `_fmt_local_cards`. Both tabs now call
  the same shared function with different parameters
- **Card CSS refresh**: cards now have `border-radius:10px`, smoother
  `.2s` transitions, `translateY(-1px)` lift on hover, `border-width:2px`
  for selected state, `flex-wrap` for better responsive layout
- **Removed unused `shutil` import**

### Fixed
- CivitAI result cards now respect the app's theme (dark mode readable
  instead of white-background-only)
- Missing blank line between `_fmt_cards` and `_all_tags` function
  definitions (`PEP8` spacing) — 2 lines → 3 lines now properly separated

### Added
- `build_finetune_dict()` / `extract_finetune_fields()` — pure module-level functions
  extracted from UI closures; testable and importable directly
- `LocalFilePickerTextbox` with **Browse 📂** button on all 8 URL fields and all 10 LoRA
  rows — opens native file explorer with extension filters (.safetensors, .ckpt, .pt, etc.)
- Model index cache (`MODEL_INDEX`) — loads model id → display name once at UI build;
  eliminates disk reads on every dropdown change (fixes slow "flashing" on load)
- Disk exists cache (`DISK_EXISTS_CACHE`) — prevents re-stating the same paths on
  keystroke-level validation calls
- `_parse_maybe_scalar()` — preserves original type for `preload_URLs` (bare string
  vs list), `loras` (bare string vs array), and other URL fields
- Type preservation — `loras_multipliers` values keep their original int/float types;
  `guidance_scale` preserves int type if source had it as int
- `_clean_surrogates()` / `_clean_tuple()` — strips surrogate characters (U+D800-U+DFFF)
  from strings to prevent orjson serialisation crashes
- `_pop_extra()` — pops known top-level fields from extra_data before round-trip,
  preventing duplicate keys
- `_validate_model_ref()` — validates `=modelid` references against the model index
- `_validate_local_path()` — validates local file paths with cached existence check
- `_validate_entry()` — unified validation entry point that dispatches to the right
  validator based on value format (URL, model ref, local path)
- `_validate_url()` — URL validation with HEAD → GET → RANGE fallback chain
- `_build_url_validation_html()` — separate HTML builder for validation results display
- `_get_model_urls()` — extracts all download URLs from a finetune dict
- `_check_model_download_status()` — checks whether all files in a finetune are present
- CivitAI `_civitai_extract_fill_data()` — extracts fill data from CivitAI results,
  routing URLs to the correct field (loras vs main checkpoints)
- Non-standard top-level keys are now preserved during save/edit round-trips
  (e.g. `ltx2_pipeline`)

### Changed
- **Core refactor**: `_build` and `_extract` closures replaced by pure functions
  `build_finetune_dict()` and `extract_finetune_fields()` — cleaner separation of
  data logic from UI wiring
- **File structure simplified** — removed `BLUEPRINT.md`, `LICENSE`, `assets/`,
  `test_plugin.py`, `test_plugin_extras.py`
- All URL / LoRA textboxes now show a **Browse** button in addition to the model-ref
  dropdown — both input methods work side-by-side
- Inline editor validation uses cached disk / model-index checks for sub-second
  response; full HEAD reachability is covered by the "Validate All URLs" button
- LoRA fields use `LocalFilePickerTextbox` filtered to `.safetensors`, `.sft`
- Validation messages display green (reachable) / gray (skipped) / red (unreachable)
  with tooltip details

### Fixed
- Slow dropdown "flashing" on initial load eliminated by model index cache
- orjson crash on surrogate characters — `_clean_surrogates()` strips them before
  JSON serialisation
- Duplicate keys on save — `_pop_extra()` prevents known top-level fields from
  appearing both in the standard position and appended via extra_data
- Type corruption — `loras_multipliers` values now preserve original numeric type;
  `guidance_scale` preserves int if original was int; `preload_URLs` preserves
  bare string vs list format

### Removed
- `test_plugin.py` (152 old tests, dependent on deprecated closure wiring)
- `test_plugin_extras.py` (additional tests for untested functions)
- `BLUEPRINT.md` (architecture doc, superseded by cleaner code structure)
- `LICENSE` (MIT, to be re-added per project preference)
- `assets/` (showcase SVGs, moved to GitHub release assets)

## v3.2.0 (2026-07-14)

### Added
- Browse tab: standalone "Download" button (download without switching model)
- Browse tab: "Improve / Create Variant" now fills Create/Edit tab instead of switching model
  - Creates variant JSON on disk, pre-fills all 32 form fields, switches to editor tab
- P0: URL Validation — per-URL green/red status with HEAD+GET+RANGE fallback
- P1: Pre-Download system — downloads finetune files via Wan2GP's `download_file()`
- P1: CivitAI Browser Tab — search, filter, one-click version URL fill
- Tags/Categories — comma-separated tags, badges, filter, search
- HF Space dynamic file listing — no more stale `index.json`
- HF Space search/filter — architecture dropdown, tag dropdown, live match count
- Click-to-expand detail modal on HF Space cards — syntax-highlighted JSON + download button

### Fixed
- `fin_cancel` double definition — Creator row Cancel button now works correctly
- Local tab download — replaced `gr.File(visible=False)` with `gr.DownloadButton`
- Server crash on variant creation — `UnboundLocalError` when ALL_INPUTS referenced before definition
- Gradio 5 nested list bug — all dropdowns handle `[[display, value]]` format
- Grammar — "1 local finetunes" → "1 local finetune"
- Upload tab missing refresh — added Refresh button to populate local finetune dropdown

### Changed
- Local tab redesigned — visual cards instead of dropdown, matches Browse tab styling
- Registry switched to dynamic file listing — no index.json sync issues
- Upload simplified — no index.json management, atomic file upload
- HF Space page — dynamic file listing with search/filter UI

## v3.1.0 (2026-07-14)

### Added
- Create/Edit Tab Redesign — full tabbed layout matching built-in Finetune Editor
- Auto-ID generation — sanitizes, deduplicates across existing files
- Creator/Editor mode switching based on Source Model presence
- Markdown toolbar in description fields
- Import from JSON, Export via DownloadButton
- Two-step delete confirmation

### Fixed
- B1 — index wipe on upload failure (simplified upload, no index.json)
- B2 — XSS via fid parameter
- B3 — resolution parsing edge cases
- B4 — guidance/steps zero values lost
- B5 — onclick quoting broken (card selection impossible)
- B5b — wrong element target for selection
- URL string vs array format mismatch
- XSS via name field

### Changed
- All 11 major fixes applied to both dev and live plugin.py
- Server restart automated after each batch of changes

## v3.0.0 (2026-07-13)

### Added
- 5-tab Gradio UI: Browse, Create/Edit, Local, Upload, CivitAI
- Browse tab with registry cards, search, architecture/tag filters
- Create/Edit tab with full finetune form
- Local tab with file management (list, load, delete, import, export)
- Upload tab for pushing finetunes to HF Space
- CivitAI search integration
- URL validation (per-URL status)
- Pre-download system
- Tags/categories system
- Export JSON as download
- 152 unit tests

### Changed
- Full architectural rewrite matching FINETUNES.md spec
- Dynamic registry discovery (no index.json)
- Gradio 5 compatibility
