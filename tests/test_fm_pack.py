"""Pure tests for the adult pack catalog helpers.

Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fm_pack

MANIFEST = Path(__file__).resolve().parent.parent / "adult_pack_manifest.json"


class ManifestTests(unittest.TestCase):
    def test_manifest_loads(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        self.assertGreaterEqual(len(manifest["loras"]), 1)
        self.assertGreaterEqual(len(manifest["finetunes"]), 1)
        self.assertGreaterEqual(len(manifest["civitai"]), 1)

    def test_default_state_has_selections(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        state = fm_pack.default_state(manifest)
        self.assertIsInstance(state["selections"]["loras"], list)
        self.assertGreater(len(state["selections"]["loras"]), 0)

    def test_pair_selection_keeps_both_halves(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        first = manifest["loras"][0]["id"]
        selected = fm_pack.selected_ids_with_pairs(manifest, [first])
        self.assertIn(first, selected)
        self.assertGreaterEqual(len(selected), 2)


class RecipeTests(unittest.TestCase):
    def test_recipe_appends_pack_suffix(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        entry = manifest["finetunes"][0]
        doc = fm_pack.build_recipe(entry)
        model = doc["model"]
        self.assertTrue(all("|adult-pack" in u for u in model["URLs"]))
        self.assertEqual(model["architecture"], entry["architecture"])

    def test_recipe_handles_second_phase(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        entry = next(x for x in manifest["finetunes"] if x.get("urls2"))
        doc = fm_pack.build_recipe(entry)
        self.assertTrue(all("|adult-pack" in u for u in doc["model"]["URLs2"]))

    def test_write_recipe_round_trip(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        entry = manifest["finetunes"][0]
        with tempfile.TemporaryDirectory() as td:
            path = fm_pack.write_recipe(entry, td)
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["model"]["name"], entry["name"])


class DownloadTests(unittest.TestCase):
    def test_pack_weight_url(self):
        self.assertEqual(
            fm_pack.pack_weight_url("https://example.test/a.safetensors"),
            "https://example.test/a.safetensors|adult-pack",
        )
        self.assertEqual(
            fm_pack.pack_weight_url("https://example.test/a.safetensors|adult-pack"),
            "https://example.test/a.safetensors|adult-pack",
        )
        self.assertEqual(
            fm_pack.pack_weight_url("C:\\models\\a.safetensors"),
            "C:\\models\\a.safetensors",
        )

    def test_checkpoint_assets(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        entry = manifest["finetunes"][0]
        assets = fm_pack.checkpoint_assets(entry, "/ckpts")
        self.assertGreaterEqual(len(assets), 1)
        self.assertTrue(
            all(
                a["destination"] and "adult-pack" in str(a["destination"])
                for a in assets
            )
        )

    def test_lora_destination(self):
        manifest = fm_pack.load_manifest(MANIFEST)
        entry = manifest["loras"][0]

        def resolver(arch):
            return f"/loras/{arch}"

        dest = fm_pack.lora_destination(entry, resolver)
        self.assertIn("adult-pack", str(dest))
        self.assertTrue(str(dest).endswith(entry["file"]))

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
    def test_detect_pair(self):
        self.assertEqual(fm_pack.detect_pair("foo_high.safetensors"), "high")
        self.assertEqual(fm_pack.detect_pair("foo_LOW_noise.safetensors"), "low")
        self.assertIsNone(fm_pack.detect_pair("foo.safetensors"))

    def test_split_halves(self):
        high, low = fm_pack.split_halves(
            ["a_high.safetensors", "b_low.safetensors"],
            ["/x/a_high.safetensors", "/x/b_low.safetensors"],
        )
        self.assertEqual(high, ["/x/a_high.safetensors"])
        self.assertEqual(low, ["/x/b_low.safetensors"])

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

    def test_install_writes_into_requested_folder(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                fm_pack,
                "download_verified",
                return_value={"status": "downloaded", "bytes": 10},
            ) as download:
                result = fm_pack.install_civitai_version(
                    self._version(), td, token="key"
                )
            self.assertEqual(
                result["files"], ["model_high.gguf", "model_low.gguf"]
            )
            self.assertEqual(len(result["paths"]), 2)
            self.assertEqual(result["errors"], [])
            self.assertEqual(
                [Path(call.args[1]).parent for call in download.call_args_list],
                [Path(td), Path(td)],
            )

    def test_install_reports_per_file_errors(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                fm_pack,
                "download_verified",
                side_effect=[ValueError("server returned text/html"), {"bytes": 5}],
            ):
                result = fm_pack.install_civitai_version(self._version(), td)
            self.assertEqual(result["files"], ["model_low.gguf"])
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("model_high.gguf", result["errors"][0])


class InventoryTests(unittest.TestCase):
    def test_lora_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sub").mkdir()
            (root / "sub" / "x.safetensors").write_bytes(b"x")
            (root / "note.txt").write_text("x")
            items = fm_pack.lora_inventory(root)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["rel"], "sub/x.safetensors")


class SourceParsingTests(unittest.TestCase):
    def test_parse_source_url(self):
        self.assertEqual(
            fm_pack.parse_source_url("https://huggingface.co/Owner/Repo/tree/main"),
            ("hf", "Owner/Repo"),
        )
        self.assertEqual(fm_pack.parse_source_url("Owner/Repo"), ("hf", "Owner/Repo"))
        self.assertEqual(
            fm_pack.parse_source_url("https://civitai.red/models/123456?modelVersionId=9"),
            ("civitai", "123456"),
        )
        self.assertEqual(fm_pack.parse_source_url("98765"), ("civitai", "98765"))
        self.assertIsNone(fm_pack.parse_source_url("https://example.com/x/y"))
        self.assertIsNone(fm_pack.parse_source_url("../../etc/passwd"))

    def test_match_arch_prefers_specific_keyword(self):
        manifest = {
            "archMap": [
                {"id": "wan", "architecture": "i2v_2_2", "match": ["wan2.2"]},
                {"id": "h3", "architecture": "minimax_h3", "match": ["minimax h3"]},
            ]
        }
        hit = fm_pack.match_arch(manifest, "some MINIMAX H3 model for wan2.2")
        self.assertEqual(hit["id"], "h3")

    def test_classify_and_precision(self):
        self.assertEqual(fm_pack.classify_file("x_lora.safetensors", 10 * 1024**3), "lora")
        self.assertEqual(fm_pack.classify_file("big.safetensors", 8 * 1024**3), "checkpoint")
        self.assertEqual(fm_pack.precision_of("model-Q4_K_M.gguf"), "gguf_q4_k_m")
        self.assertEqual(fm_pack.precision_of("model-fp8.safetensors"), "fp8")

    def test_repo_path_from_url(self):
        self.assertEqual(
            fm_pack._repo_path_from_url(
                "https://huggingface.co/o/r/resolve/main/I2V/High/a%20b.safetensors"
            ),
            "I2V/High/a b.safetensors",
        )


class CustomEntryTests(unittest.TestCase):
    def test_add_and_merge_custom_lora(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "custom.json"
            entry = fm_pack.build_custom_entry(
                source="hf",
                title="My Model",
                file="my_lora.safetensors",
                url="https://huggingface.co/o/r/resolve/main/my_lora.safetensors",
                architecture="i2v_2_2",
                kind="lora",
                bytes=123,
                sha256="a" * 64,
                repo="o/r",
                family="wan",
            )
            self.assertTrue(entry["id"].startswith("custom_hf_"))
            self.assertEqual(entry["group"], "custom")
            fm_pack.add_custom(entry, path)
            custom = fm_pack.load_custom(path)
            self.assertEqual(len(custom["loras"]), 1)
            merged = fm_pack.merged_catalog({"loras": [], "finetunes": []}, custom)
            self.assertEqual(merged["loras"][0]["id"], entry["id"])
            fm_pack.drop_custom(entry["id"], path)
            self.assertEqual(fm_pack.load_custom(path)["loras"], [])

    def test_checkpoint_entry_builds_recipe(self):
        entry = fm_pack.build_custom_entry(
            source="civitai",
            title="Tasty",
            file="tasty_high.safetensors",
            url="https://civitai.red/api/download/x",
            architecture="i2v_2_2",
            kind="checkpoint",
            bytes=500,
            sha256=None,
            family="wan",
        )
        self.assertIn("Tasty", entry["name"])
        self.assertTrue(entry["needsToken"])
        doc = fm_pack.build_recipe(entry)
        self.assertEqual(doc["model"]["architecture"], "i2v_2_2")
        self.assertTrue(doc["model"]["URLs"][0].endswith("|adult-pack"))

    def test_rejects_unsafe_or_missing_data(self):
        with self.assertRaises(ValueError):
            fm_pack.build_custom_entry(
                source="hf", title="x", file="",
                url="https://x/y", architecture="i2v_2_2",
            )
        with self.assertRaises(ValueError):
            fm_pack.build_custom_entry(
                source="hf", title="x", file="ok.safetensors",
                url="https://x/y", architecture="",
            )
        entry = fm_pack.build_custom_entry(
            source="hf", title="x", file="../evil.bin",
            url="https://x/y", architecture="i2v_2_2",
        )
        self.assertEqual(entry["file"], "evil.bin")


class UpdatePinningTests(unittest.TestCase):
    def test_resolve_civitai_version_prefers_pin(self):
        with mock.patch.object(fm_pack, "_civitai_api") as api:
            self.assertEqual(
                fm_pack.resolve_civitai_version({"id": "1"}, pinned=7), 7
            )
            self.assertEqual(
                fm_pack.resolve_civitai_version(
                    {"id": "1", "civitaiVersionId": 9}, pinned=7
                ),
                7,
            )
            api.assert_not_called()
        with mock.patch.object(
            fm_pack,
            "_civitai_api",
            return_value={"items": [{"baseModel": "Wan", "modelVersions": [{"id": 5}]}]},
        ):
            self.assertEqual(
                fm_pack.resolve_civitai_version({"id": "1", "query": "some model"}), 5
            )

    def test_fetch_civitai_honours_pinned_version(self):
        seen: list[str] = []

        def api(_site, path, _token, _params=None):
            seen.append(path)
            if path == "model-versions/222":
                return {
                    "id": 222,
                    "files": [
                        {
                            "name": "pinned.safetensors",
                            "sizeKB": 1024,
                            "downloadUrl": "/api/download/1",
                            "type": "Model",
                            "hashes": {"SHA256": "a" * 64},
                        }
                    ],
                    "images": [],
                }
            raise AssertionError(f"unexpected API path {path}")

        def fake_download(url, dest, **kwargs):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"x" * 10)
            return {"status": "downloaded", "bytes": 10, "path": str(dest)}

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with mock.patch.object(fm_pack, "_civitai_api", side_effect=api), \
                    mock.patch.object(fm_pack, "download_verified", side_effect=fake_download):
                res = fm_pack.fetch_civitai(
                    {"id": "civ1", "name": "Pinned"},
                    ckpts_dir=base / "ckpts",
                    lora_dir_for=lambda arch: base / "loras",
                    finetunes_dir=base / "finetunes",
                    pinned=222,
                )
        self.assertEqual(res["versionId"], 222)
        self.assertEqual(seen, ["model-versions/222"])
        self.assertEqual(res["paths"], [str(base / "ckpts" / "adult-pack" / "pinned.safetensors")])

    def test_apply_updates_pins_civitai_version(self):
        state = {
            "updates": {
                "items": [
                    {"id": "c1", "kind": "civitai", "liveVersion": 111},
                    {"id": "c2", "kind": "civitai"},
                ]
            }
        }
        res = fm_pack.apply_updates(state, ["c1", "c2"])
        self.assertEqual(state["civitaiPinned"]["c1"], 111)
        self.assertEqual(res["civitaiPinned"], {"c1": 111})
        self.assertEqual(res["pins"], {})
        self.assertEqual(res["skipped"], [{"id": "c2", "reason": "no live version"}])

    def _manifest(self, *, loras=None, finetunes=None):
        return {
            "archMap": [{"id": "wan", "architecture": "i2v_2_2", "match": ["wan2.2"]}],
            "finetunes": finetunes or [],
            "loras": loras or [],
            "civitai": [],
            "groups": [],
        }

    def _lora(self, name, size=10):
        return {
            "id": name,
            "name": name,
            "family": "wan",
            "wan2gpIds": ["i2v_2_2"],
            "file": f"{name}.safetensors",
            "url": f"https://huggingface.co/o/r/resolve/main/{name}.safetensors",
            "repo": "o/r",
            "bytes": size,
        }

    def test_apply_updates_pins_hash_and_size(self):
        state = {
            "updates": {
                "items": [
                    {"id": "a", "kind": "lora", "liveSha": "b" * 64, "liveBytes": 42},
                    {"id": "b", "kind": "lora"},
                ]
            }
        }
        res = fm_pack.apply_updates(state, ["a", "b"])
        self.assertEqual(len(res["applied"]), 1)
        self.assertEqual(res["skipped"][0]["id"], "b")
        entry = fm_pack.pinned_entry({"id": "a", "sha256": "a" * 64, "bytes": 1}, state["pins"])
        self.assertEqual(entry["sha256"], "b" * 64)
        self.assertEqual(entry["bytes"], 42)

    def test_live_report_statuses_and_pins(self):
        manifest = self._manifest(
            loras=[self._lora("keep"), self._lora("gone"), self._lora("move", 5)]
        )
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ckpts = base / "ckpts"
            fins = base / "finetunes"
            loras = base / "loras" / "wan"
            (loras / "adult-pack").mkdir(parents=True)
            (loras / "adult-pack" / "keep.safetensors").write_bytes(b"x" * 10)
            (loras / "adult-pack" / "move.safetensors").write_bytes(b"x" * 5)
            tree = [
                {"type": "file", "path": "keep.safetensors", "size": 10,
                 "lfs": {"oid": "1" * 64}},
                {"type": "file", "path": "move.safetensors", "size": 6,
                 "lfs": {"oid": "2" * 64}},
                {"type": "file", "path": "gone.safetensors", "size": 11,
                 "lfs": {"oid": "3" * 64}},
            ]
            state = {}
            with mock.patch.object(fm_pack, "hf_tree", return_value=tree):
                report = fm_pack.live_update_report(
                    manifest, ckpts, lambda arch: loras, fins, state
                )
            by_id = {x["id"]: x for x in report["items"]}
            self.assertEqual(by_id["keep"]["status"], "current")
            self.assertEqual(by_id["gone"]["status"], "missing")
            self.assertEqual(by_id["move"]["status"], "changed")
            self.assertEqual(state["updates"]["changed"], 2)
            fm_pack.apply_updates(state, ["move"])
            pinned = fm_pack.pinned_entry(
                next(x for x in manifest["loras"] if x["id"] == "move"), state["pins"]
            )
            self.assertEqual(pinned["bytes"], 6)
            self.assertEqual(pinned["sha256"], "2" * 64)

    def test_live_report_covers_both_finetune_halves(self):
        entry = {
            "id": "ft",
            "name": "FT",
            "repo": "o/r",
            "urls": ["https://huggingface.co/o/r/resolve/main/high/a.safetensors"],
            "urls2": ["https://huggingface.co/o/r/resolve/main/low/a.safetensors"],
            "sha256": "1" * 64,
            "sha256Low": "2" * 64,
            "bytes": 10,
            "bytesLow": 20,
        }
        manifest = self._manifest(finetunes=[entry])
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            state = {}
            tree = [
                {"type": "file", "path": "high/a.safetensors", "size": 10,
                 "lfs": {"oid": "1" * 64}},
                {"type": "file", "path": "low/a.safetensors", "size": 20,
                 "lfs": {"oid": "2" * 64}},
            ]
            with mock.patch.object(fm_pack, "hf_tree", return_value=tree):
                report = fm_pack.live_update_report(
                    manifest, base / "ckpts", lambda arch: base / "loras",
                    base / "finetunes", state,
                )
            ids = {x["id"] for x in report["items"]}
            self.assertEqual(ids, {"ft", "ft#low"})



