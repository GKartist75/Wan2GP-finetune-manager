"""Pure tests for the finetune-manager download/recipe helpers.

Run with::

    python -m unittest discover -s tests -v

Covers the functions that survived the v4.0 slim-down of ``fm_pack``
(download + CivitAI recipe builders). Pack-catalog behavior moved to the
adult Space registry and is exercised by ``tools/build_adult_registry.py``.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import fm_pack


class DownloadTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.bin"
            p.write_bytes(b"abc")
            self.assertEqual(
                fm_pack.sha256_file(p),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_download_verified_exists(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.bin"
            p.write_bytes(b"abc")
            digest = fm_pack.sha256_file(p)
            res = fm_pack.download_verified(
                "https://example.test/a.bin",
                p,
                sha256=digest,
                expected_bytes=3,
            )
            self.assertEqual(res["status"], "exists")
            self.assertEqual(res["bytes"], 3)


class CivitaiTests(unittest.TestCase):
    def test_kind_mapping(self):
        self.assertEqual(fm_pack.civitai_kind("LORA"), "lora")
        self.assertEqual(fm_pack.civitai_kind("LoCon"), "lora")
        self.assertEqual(fm_pack.civitai_kind("DoRA"), "lora")
        self.assertEqual(fm_pack.civitai_kind("Checkpoint"), "checkpoint")
        self.assertIsNone(fm_pack.civitai_kind("Workflows"))
        self.assertIsNone(fm_pack.civitai_kind(None))

    @staticmethod
    def _version():
        return {
            "id": 99,
            "name": "Q8 High",
            "baseModel": "Wan Video 2.2 I2V-A14B",
            "files": [
                {
                    "name": "model_high.gguf",
                    "type": "Model",
                    "primary": True,
                    "sizeKB": 1024,
                    "downloadUrl": "https://civitai.com/api/download/models/99",
                    "hashes": {"SHA256": "a" * 64},
                },
                {
                    "name": "model_low.gguf",
                    "type": "Model",
                    "primary": False,
                    "sizeKB": 1024,
                    "downloadUrl": "https://civitai.com/api/download/models/100",
                },
                {
                    "name": "preview.png",
                    "type": "Image",
                    "primary": False,
                    "downloadUrl": "https://civitai.com/api/download/models/101",
                },
            ],
        }

    def test_primary_only_skips_secondary_and_non_weights(self):
        files = fm_pack.civitai_weight_files(
            self._version()["files"], primary_only=True
        )
        self.assertEqual([f["name"] for f in files], ["model_high.gguf"])
        all_files = fm_pack.civitai_weight_files(self._version()["files"])
        self.assertEqual(
            [f["name"] for f in all_files],
            ["model_high.gguf", "model_low.gguf"],
        )

    def test_recipe_url_keeps_the_real_filename(self):
        url = fm_pack.civitai_recipe_url(
            {
                "name": "model high.gguf",
                "downloadUrl": "https://civitai.com/api/download/models/99",
            }
        )
        self.assertEqual(os.path.basename(url), "model high.gguf")
        self.assertEqual(
            url.split("#", 1)[0], "https://civitai.com/api/download/models/99"
        )

    def test_recipe_url_leaves_named_urls_alone(self):
        url = "https://example.test/files/model.safetensors"
        self.assertEqual(
            fm_pack.civitai_recipe_url(
                {"name": "model.safetensors", "downloadUrl": url}
            ),
            url,
        )

    def test_checkpoint_recipe_splits_high_low(self):
        model = {
            "id": 7,
            "name": "Tasty Model",
            "type": "Checkpoint",
            "description": "<p>Hello <b>world</b></p>",
            "tags": ["wan", "wan"],
        }
        recipe = fm_pack.civitai_recipe(
            model, self._version(), architecture="i2v_2_2"
        )["model"]
        self.assertEqual(recipe["architecture"], "i2v_2_2")
        self.assertEqual(
            [os.path.basename(u) for u in recipe["URLs"]], ["model_high.gguf"]
        )
        self.assertEqual(
            [os.path.basename(u) for u in recipe["URLs2"]], ["model_low.gguf"]
        )
        self.assertIn("Hello world", recipe["description"])
        self.assertEqual(recipe["tags"], ["civitai", "wan"])

    def test_lora_recipe_goes_to_loras(self):
        model = {"id": 8, "name": "Style LoRA", "type": "LORA"}
        version = {
            "id": 9,
            "name": "v1",
            "files": [
                {
                    "name": "style.safetensors",
                    "type": "Model",
                    "primary": True,
                    "downloadUrl": "https://civitai.com/api/download/models/9",
                }
            ],
        }
        recipe = fm_pack.civitai_recipe(
            model, version, architecture="i2v_2_2", primary_only=True
        )["model"]
        self.assertNotIn("URLs", recipe)
        self.assertEqual(
            [os.path.basename(u) for u in recipe["loras"]], ["style.safetensors"]
        )
        self.assertEqual(recipe["loras"][0].split("#", 1)[0],
                         "https://civitai.com/api/download/models/9")
        self.assertEqual(recipe["loras_multipliers"], [1])
