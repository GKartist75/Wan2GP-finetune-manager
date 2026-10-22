---
title: Wan2GP Finetune Registry
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# Wan2GP Finetune Registry

Community finetune definitions for Wan2GP.

- Browse finetunes at the [web UI](/)
- Machine-readable index: [`/index.json`](/index.json)
- Each finetune has its own JSON file under `/finetunes/<id>.json`

## Security and trust

**These recipes are community-contributed.** The registry and the
Finetune Manager check structure and URL policy (HTTPS only, no credentials, no
local/private hosts, no bare filenames or embedded HTML), but that is a
guardrail, not a safety guarantee. It does not prove the uploader is honest or
that the file behind a URL is safe.

Before downloading:

- Verify every URL starts with `https://` and uses a host you expect, normally
  `huggingface.co`, `civitai.com` or `civitai.red`.
- Check the repository, path and filename for lookalike accounts or re-uploads.
- Registry uploads made through the Finetune Manager are restricted to the
  usual Hugging Face and CivitAI hosts; other hosts can be saved locally but
  cannot be published to this Space.
- Prefer entries with a published SHA-256 and verify it where possible. A
  matching hash proves the bytes match; it does not prove they are safe.
- Treat `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf` and `.safetensors` files from
  strangers with caution. Do not load a model just because it is listed here.
- Never put a Hugging Face or CivitAI token in a finetune JSON or a shared
  recipe.

Report malicious or broken entries through
[GitHub Issues](https://github.com/GKartist75/Wan2GP-finetune-manager/issues).

## For Wan2GP Plugin Developers

```python
import requests
index = requests.get("https://gkartist75-wan2gp-finetunes.hf.space/index.json").json()
for f in index["finetunes"]:
    print(f["name"], f["architecture"])
```
