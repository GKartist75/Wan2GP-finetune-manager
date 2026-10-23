"""Pure tests for the recipe asset resolver.

Run with::

    python -m unittest discover -s tests -v

No Wan2GP, Gradio, or network access is required.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fm_assets


class SourceParsingTests(unittest.TestCase):
    def test_url_filename_strips_query(self):
        self.assertEqual(
            fm_assets.url_file_name("https://example.test/a/model.safetensors?token=x"),
            "model.safetensors",
        )

    def test_url_filename_override_fragment(self):
        self.assertEqual(
            fm_assets.url_file_name(
                "https://civitai.com/api/download/models/99#/My Model.gguf"
            ),
            "My Model.gguf",
        )
        remote, url, sub, name = fm_assets.parse_source(
            "https://civitai.com/api/download/models/99#/My Model.gguf|wan"
        )
        self.assertEqual(url, "https://civitai.com/api/download/models/99#/My Model.gguf")
        self.assertEqual(sub, "wan")
        self.assertEqual(name, "My Model.gguf")

    def test_url_suffix_is_split(self):
        remote, url, sub, name = fm_assets.parse_source(
            "https://example.test/model.safetensors?download=1|adult-pack/wan"
        )
        self.assertTrue(remote)
        self.assertEqual(url, "https://example.test/model.safetensors?download=1")
        self.assertEqual(sub, "adult-pack/wan")
        self.assertEqual(name, "model.safetensors")

    def test_local_path_has_no_suffix(self):
        remote, url, sub, name = fm_assets.parse_source(r"D:\models\my.safetensors")
        self.assertFalse(remote)
        self.assertIsNone(url)
        self.assertIsNone(sub)
        self.assertEqual(name, "my.safetensors")

    def test_bad_file_names_are_rejected(self):
        for name in (
            "..",
            "a/b.safetensors",
            r"a\b.safetensors",
            "a:b",
            "x.",
            "CON.safetensors",
            "nul",
            "COM1.ckpt",
        ):
            with self.subTest(name=name):
                self.assertIsNone(fm_assets.safe_file_name(name))
        self.assertEqual(fm_assets.safe_file_name("good.safetensors"), "good.safetensors")

    def test_reserved_subfolder_is_rejected(self):
        for sub in ("adult-pack/CON", "a/nul/", "LPT1"):
            with self.subTest(sub=sub):
                with self.assertRaises(ValueError):
                    fm_assets.safe_subdir(sub)


class DestinationTests(unittest.TestCase):
    def test_model_suffix_goes_under_ckpts(self):
        dest = fm_assets.asset_path(
            "https://example.test/model.safetensors|adult-pack/wan",
            "model",
            "/ckpts",
            "/loras/wan",
        )
        self.assertEqual(dest, Path("/ckpts/adult-pack/wan/model.safetensors"))

    def test_lora_defaults_to_lora_collection(self):
        dest = fm_assets.asset_path(
            "https://example.test/lora.safetensors",
            "lora",
            "/ckpts",
            "/loras/wan",
        )
        self.assertEqual(dest, Path("/loras/wan/lora.safetensors"))

    def test_lora_percent_lora_dir(self):
        dest = fm_assets.asset_path(
            "https://example.test/lora.safetensors|%lora_dir",
            "lora",
            "/ckpts",
            "/loras/wan",
        )
        self.assertEqual(dest, Path("/loras/wan/lora.safetensors"))

    def test_nested_subfolder_is_preserved(self):
        dest = fm_assets.asset_path(
            "https://example.test/vae.safetensors|encoders/vae",
            "vae",
            "/ckpts",
            "/loras/wan",
        )
        self.assertEqual(dest, Path("/ckpts/encoders/vae/vae.safetensors"))

    def test_local_absolute_path_is_kept(self):
        dest = fm_assets.asset_path(r"D:\models\local.safetensors", "model", "/ckpts", "/loras")
        self.assertEqual(dest, Path(r"D:\models\local.safetensors"))

    def test_traversal_is_rejected(self):
        for sub in ("../outside", "a/../../outside", "/absolute", r"C:\absolute"):
            with self.subTest(sub=sub):
                with self.assertRaises(ValueError):
                    fm_assets.asset_path(
                        f"https://example.test/model.safetensors|{sub}",
                        "model",
                        "/ckpts",
                        "/loras/wan",
                    )

    def test_lora_without_a_collection_is_an_error(self):
        with self.assertRaises(ValueError):
            fm_assets.asset_path(
                "https://example.test/lora.safetensors",
                "lora",
                "/ckpts",
                None,
            )


class RecipeInventoryTests(unittest.TestCase):
    def test_reference_and_preload_names_are_not_files(self):
        model = {
            "URLs": "=other_recipe",
            "URLs2": ["https://example.test/second.safetensors"],
            "preload_URLs": "some_model_type",
            "loras": ["=another_recipe"],
        }
        sources = list(fm_assets.iter_asset_sources(model))
        self.assertEqual(
            [s.role for s in sources if not s.reference],
            ["second"],
        )
        assets = fm_assets.build_inventory(model, "/ckpts", "/loras/wan")
        self.assertEqual([asset.role for asset in assets], ["second"])

    def test_custom_urls_are_included(self):
        model = {"custom_url_1": "https://example.test/custom.bin"}
        sources = list(fm_assets.iter_asset_sources(model))
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].role, "custom1")

    def test_present_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpts = root / "ckpts"
            loras = root / "loras" / "wan"
            (ckpts / "adult-pack").mkdir(parents=True)
            loras.mkdir(parents=True)
            (ckpts / "adult-pack" / "model.safetensors").write_bytes(b"model")
            model = {
                "architecture": "wan",
                "URLs": "https://example.test/model.safetensors|adult-pack",
                "text_encoder_URLs": ["https://example.test/te.safetensors"],
                "loras": ["https://example.test/style.safetensors"],
            }
            assets = fm_assets.build_inventory(model, ckpts, loras)
            by_role = {asset.role: asset for asset in assets}
            self.assertTrue(by_role["model"].present)
            self.assertEqual(by_role["model"].bytes, 5)
            self.assertFalse(by_role["text_encoder"].present)
            self.assertEqual(
                Path(by_role["text_encoder"].destination),
                ckpts / "te.safetensors",
            )
            self.assertEqual(
                Path(by_role["lora"].destination),
                loras / "style.safetensors",
            )
            self.assertTrue(by_role["lora"].downloadable)
            summary = fm_assets.inventory_summary(assets)
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["present"], 1)
            self.assertEqual(summary["missing"], 2)
            self.assertEqual(summary["downloadable"], 2)

    def test_civitai_token_hint(self):
        self.assertTrue(fm_assets.is_civitai_url("https://civitai.com/api/download/models/1"))
        self.assertTrue(fm_assets.is_civitai_url("https://image.civitai.com/x/y.safetensors"))
        self.assertFalse(fm_assets.is_civitai_url("https://example.test/model.safetensors"))


class PackIdTests(unittest.TestCase):
    def test_pack_aliases(self):
        self.assertEqual(
            fm_assets.finetune_candidates("adultpack_foo"),
            ["adultpack_foo", "foo"],
        )
        self.assertEqual(
            fm_assets.finetune_candidates("foo"),
            ["foo", "adultpack_foo"],
        )

    def test_pack_display_id(self):
        self.assertEqual(fm_assets.pack_display_id("adultpack_foo"), "foo")
        self.assertEqual(fm_assets.pack_display_id("foo"), "foo")


if __name__ == "__main__":
    unittest.main()
