# Changelog

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
