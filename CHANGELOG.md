# Changelog

## [3.2.0] — 2026-07-14
### Added
- **Download button** — saves finetune JSON locally without switching model
- **Improve / Create Variant fills editor** — creates variant and pre-fills Create/Edit tab
### Changed
- **CivitAI fill simplified** — only fills download URL, leaves other fields unchanged
### Fixed
- Server crash: UnboundLocalError when ALL_INPUTS referenced before definition

## [3.1.0] — 2026-07-14
### Added
- URL Validation (P0) — HEAD requests with GET+RANGE fallback, per-URL status
- Pre-Download system (P1) — download_file() for HF Hub and generic URLs
- CivitAI Browser tab (P1) — search, filter, browse, one-click fill
- Tags/Categories system — comma-separated tags, badges, filter
### Fixed
- Python 3.11 f-string backslash issue in CivitAI render
### Tests
- 147 unit tests (up from 105)

## [3.0.0] — 2026-07-13
### Added
- Complete Create/Edit tab redesign matching built-in Finetune Editor (Alt+F)
- Tabbed layout: URLs/LoRAs/Resolutions/Help/Prompt Enhancer/Settings
- Markdown editor toolbar, auto-ID generation, Source Model field
- Creator/Editor mode switching, Create & New, two-step delete
### Fixed
- B1: Index wipe on upload failure
- B2: XSS via fid
- B3: Resolution parsing
- B4: Guidance/steps zero lost
- B5: onclick quoting broken
- URL string vs array format mismatch
### Tests
- 105 unit tests (up from 92)

## [2.0.0] — 2026-07-13
### Added
- Initial 4-tab release: Browse, Create/Edit, Local, Upload
- Registry browsing with card UI
- Search, filter, Download & Switch
- Basic finetune editor, Local management
- HF Registry upload with token auth
- 92 unit tests
