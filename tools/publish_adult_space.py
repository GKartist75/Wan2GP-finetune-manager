#!/usr/bin/env python
"""Publish a registry Space (e.g. the adult registry) from staged files.

Usage:
    python tools/publish_adult_space.py            # dry run (default)
    python tools/publish_adult_space.py --push     # actually upload
    python tools/publish_adult_space.py --push --token hf_xxx

- Token: `--token` arg or HF_TOKEN environment variable. The token needs
  write access to the Space (owner or a Space withHF token).
- Creates the Space as a static SDK Space if missing.
- Uploads: README.md, finetunes/*.json, pack/adult_pack_manifest.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

STAGING = Path(__file__).resolve().parent / "adult_registry"
DEFAULT_REPO = "GKartist75/wan2gp-finetunes-adult"
README = """---
title: Wan2GP Adult Finetunes
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: static
pinned: false
license: other
---
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--push", action="store_true", help="actually upload")
    ap.add_argument(
        "--pr",
        action="store_true",
        help="publish as a single Pull Request (works without write scope)",
    )
    args = ap.parse_args()

    if not STAGING.exists():
        sys.exit("staging missing — run tools/build_adult_registry.py first")
    files = sorted(p for p in STAGING.rglob("*") if p.is_file())
    if not files:
        sys.exit("staging empty")
    print(f"{len(files)} files staged for {args.repo}:")
    for p in files:
        print(f"  {p.relative_to(STAGING)}  ({p.stat().st_size} B)")

    if not args.push:
        print("\ndry run — add --push to upload")
        return
    if not args.token:
        sys.exit("no HF token — pass --token or set HF_TOKEN")

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import CommitOperationAdd
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=args.token)
    try:
        api.space_info(repo_id=args.repo)
        print(f"space exists: {args.repo}")
    except RepositoryNotFoundError:
        print(f"creating static space {args.repo} ...")
        api.create_repo(
            repo_id=args.repo, repo_type="space", exist_ok=True, space_sdk="static"
        )
        api.upload_file(
            path_or_fileobj=README.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="space",
        )

    ops = [
        CommitOperationAdd(
            path_in_repo=str(p.relative_to(STAGING)).replace("\\", "/"),
            path_or_fileobj=p.read_bytes(),
        )
        for p in files
    ]
    if args.pr:
        res = api.create_commit(
            repo_id=args.repo,
            repo_type="space",
            operations=ops,
            commit_message="Publish adult pack registry (finetunes + pack manifest)",
            create_pr=True,
        )
        print(f"PR created: {getattr(res, 'pr_url', res)}")
    else:
        for p in files:
            rel = str(p.relative_to(STAGING)).replace("\\", "/")
            api.upload_file(
                path_or_fileobj=p.read_bytes(),
                path_in_repo=rel,
                repo_id=args.repo,
                repo_type="space",
            )
            print(f"uploaded {rel}")
    print("done")


if __name__ == "__main__":
    main()
