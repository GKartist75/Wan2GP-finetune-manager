# Changelog

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
