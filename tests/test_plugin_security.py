"""Regression tests for plugin-side hardening that needs no Wan2GP host."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_plugin():
    gradio = types.ModuleType("gradio")
    shared = types.ModuleType("shared")
    utils = types.ModuleType("shared.utils")
    plugins = types.ModuleType("shared.utils.plugins")
    gradio_local = types.ModuleType("shared.gradio")
    picker = types.ModuleType("shared.gradio.local_file_picker")

    class WAN2GPPlugin:
        pass

    class LocalFilePickerTextbox:
        pass

    plugins.WAN2GPPlugin = WAN2GPPlugin
    picker.LocalFilePickerTextbox = LocalFilePickerTextbox
    stubs = {
        "gradio": gradio,
        "shared": shared,
        "shared.utils": utils,
        "shared.utils.plugins": plugins,
        "shared.gradio": gradio_local,
        "shared.gradio.local_file_picker": picker,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        import plugin

        return plugin
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


plugin = _load_plugin()


class RecipePathTests(unittest.TestCase):
    def test_windows_device_names_are_not_file_stems(self):
        self.assertEqual(plugin._sanitize_fin_id("CON"), "CON_finetune")
        self.assertEqual(plugin._sanitize_fin_id("nul"), "nul_finetune")

    def test_resolver_rejects_traversal_and_devices(self):
        with tempfile.TemporaryDirectory() as td:
            old = Path.cwd()
            try:
                import os

                os.chdir(td)
                (Path(td) / "finetunes").mkdir()
                self.assertIsNone(plugin._resolve_finetune_path("../outside"))
                self.assertIsNone(plugin._resolve_finetune_path("CON"))
            finally:
                os.chdir(old)


class RegistrySafetyTests(unittest.TestCase):
    def test_strict_validation_rejects_bare_filenames(self):
        data = {"model": {"architecture": "ltx2_22B", "URLs": ["model.safetensors"]}}
        problem = plugin._validate_finetune_data(data, registry=True)
        self.assertIn("incomplete download URL", problem or "")

    def test_strict_validation_accepts_https_weights(self):
        data = {
            "model": {
                "architecture": "ltx2_22B",
                "URLs": ["https://huggingface.co/acme/model.safetensors"],
            }
        }
        self.assertIsNone(plugin._validate_finetune_data(data, registry=True))


class HostWarningTests(unittest.TestCase):
    def test_expected_hosts_and_subdomains_are_not_flagged(self):
        data = {
            "model": {
                "URLs": ["https://huggingface.co/acme/model.safetensors"],
                "loras": [
                    "https://cdn-lfs.huggingface.co/acme/lora.safetensors",
                    "https://civitai.com/api/download/models/1",
                ],
                "preload_URLs": ["ltx2_22B"],
            }
        }
        self.assertEqual(plugin._unexpected_url_hosts(data), [])
        self.assertEqual(plugin._unexpected_host_warning(data), "")

    def test_unexpected_public_host_is_flagged(self):
        data = {
            "model": {
                "URLs": [
                    "https://evil.example/model.safetensors|sub/dir",
                    "https://evil.example/other.safetensors",
                ],
                "loras": ["https://huggingface.co/acme/lora.safetensors"],
            }
        }
        self.assertEqual(plugin._unexpected_url_hosts(data), ["evil.example"])
        self.assertEqual(plugin._unexpected_host_warning(data), "evil.example")

    def test_card_renders_escaped_host_badge(self):
        html = plugin._fmt_card_html_item(
            "id", "name", "arch", "author", "desc", [], [], [],
            False, host_warning="evil.example",
        )
        self.assertIn("check host: evil.example", html)
        self.assertIn("verify it yourself", html)

    def test_format_validation_reports_unexpected_host(self):
        ok, msg = plugin._validate_entry(
            "https://evil.example/model.safetensors", network=False
        )
        self.assertTrue(ok)
        self.assertIn("unexpected host", msg)
        self.assertIn("evil.example", msg)


class UploadHostGateTests(unittest.TestCase):
    def test_upload_gate_rejects_unexpected_host(self):
        data = {
            "model": {
                "architecture": "ltx2_22B",
                "URLs": ["https://evil.example/model.safetensors"],
            }
        }
        problem = plugin._upload_host_problem(data)
        self.assertIn("Registry uploads are limited", problem or "")
        self.assertIn("evil.example", problem or "")

    def test_upload_gate_accepts_expected_hosts(self):
        data = {
            "model": {
                "architecture": "ltx2_22B",
                "URLs": ["https://huggingface.co/acme/model.safetensors"],
                "loras": ["https://civitai.com/api/download/models/1"],
            }
        }
        self.assertIsNone(plugin._upload_host_problem(data))

    def test_download_validation_warns_but_does_not_block(self):
        data = {
            "model": {
                "architecture": "ltx2_22B",
                "URLs": ["https://evil.example/model.safetensors"],
            }
        }
        self.assertIsNone(plugin._validate_finetune_data(data, registry=True))
        self.assertEqual(plugin._unexpected_url_hosts(data), ["evil.example"])

    def test_hf_upload_rejects_before_network(self):
        with mock.patch("huggingface_hub.HfApi") as api:
            with self.assertRaises(ValueError):
                plugin._hf_upload(
                    "entry",
                    {
                        "model": {
                            "architecture": "ltx2_22B",
                            "URLs": ["https://evil.example/model.safetensors"],
                        }
                    },
                )
            api.assert_not_called()

    def test_hf_upload_rejects_http_before_network(self):
        with mock.patch("huggingface_hub.HfApi") as api:
            with self.assertRaises(ValueError):
                plugin._hf_upload(
                    "entry",
                    {
                        "model": {
                            "architecture": "ltx2_22B",
                            "URLs": ["http://huggingface.co/acme/model.safetensors"],
                        }
                    },
                )
            api.assert_not_called()

    def test_hf_upload_rejects_embedded_token_before_network(self):
        with mock.patch("huggingface_hub.HfApi") as api:
            with self.assertRaises(ValueError):
                plugin._hf_upload(
                    "entry",
                    {
                        "model": {
                            "architecture": "ltx2_22B",
                            "URLs": [
                                "https://civitai.com/api/download/models/1?token=secret"
                            ],
                        }
                    },
                )
            api.assert_not_called()

    def test_token_scan_covers_source_model_field(self):
        data = {
            "model": {
                "finetune_source_model": (
                    "https://civitai.com/api/download/models/1?token=secret"
                )
            }
        }
        found = plugin._find_token_urls(data)
        self.assertEqual(len(found), 1)
        self.assertIn("finetune_source_model", found[0])

    def test_hf_upload_allows_expected_host(self):
        data = {
            "model": {
                "architecture": "ltx2_22B",
                "URLs": ["https://huggingface.co/acme/model.safetensors"],
            }
        }
        with mock.patch("huggingface_hub.HfApi") as api_cls, mock.patch.object(
            plugin, "_publish_family_index"
        ) as publish:
            plugin._hf_upload("entry", data)
        api_cls.return_value.upload_file.assert_called_once()
        publish.assert_called_once()


class CardSafetyTests(unittest.TestCase):
    def test_non_string_registry_fields_do_not_crash_or_render_raw(self):
        html = plugin._fmt_card_html_item(
            {"bad": "id"},
            {"bad": "name"},
            {"bad": "arch"},
            123,
            {"bad": "desc"},
            [{"bad": "url"}],
            [{"bad": "lora"}],
            [{"bad": "tag"}],
            False,
        )
        self.assertIn("&#x27;bad&#x27;", html)
        self.assertNotIn("<img", html)


if __name__ == "__main__":
    unittest.main()
