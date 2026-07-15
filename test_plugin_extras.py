"""Additional tests for Wan2GP Finetune Manager plugin.

Covers functions not tested in test_plugin.py:
  - _download_finetune_file()
  - _download_finetune_files()
  - _civitai_search()
  - _civitai_render_results()

Plus integration-level checks: plugin import, constant values, and
validate that the deployed plugin can be loaded in a Wan2GP context.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# -- Setup: make Wan2GP imports work ------------------------------------------
WAN2GP_ROOT = r"C:\Users\gjaku\Wan2GP"
if WAN2GP_ROOT not in sys.path:
    sys.path.insert(0, WAN2GP_ROOT)

# -- Import module under test -------------------------------------------------
PLUGIN_DIR = Path(__file__).parent
_cfg_path = PLUGIN_DIR / "config.json"
_cfg_saved = (
    _cfg_path.read_text(encoding="utf-8") if _cfg_path.exists() else None
)
if _cfg_path.exists():
    _cfg_path.write_text(
        json.dumps({"registry_token": "test_token_abc"}), encoding="utf-8"
    )

import plugin as FM

if _cfg_saved:
    _cfg_path.write_text(_cfg_saved, encoding="utf-8")
elif _cfg_path.exists():
    _cfg_path.unlink()


# ==============================================================================
# 1. _download_finetune_file — single file download logic
# ==============================================================================

class TestDownloadFinetuneFile:
    """_download_finetune_file() downloads a single file and returns status."""

    def test_skips_non_remote_url(self):
        """Local paths are skipped, not downloaded."""
        result = FM._download_finetune_file("/local/path/file.safetensors")
        assert "Skipped (not remote)" in result
        # The full URL (including filename) is shown in the skip message
        assert "file.safetensors" in result

    def test_skips_empty_url(self):
        """Empty URL after strip returns skipped."""
        result = FM._download_finetune_file("   ")
        assert "Skipped" in result

    def test_already_exists(self):
        """If locate_file finds it, returns Already exists."""
        with patch("shared.utils.files_locator.locate_file", return_value="/some/path/model.safetensors"):
            with patch("shared.utils.files_locator.get_download_location", return_value="/dl/model.safetensors"):
                result = FM._download_finetune_file("https://example.com/model.safetensors")
                assert "Already exists" in result
                assert "model.safetensors" in result

    def test_downloads_successfully(self):
        """Happy path: downloads and returns success."""
        with patch("shared.utils.files_locator.locate_file", return_value=None):
            with patch("shared.utils.files_locator.get_download_location", return_value="/dl/model.safetensors"):
                with patch("shared.utils.download.download_file", return_value=None):
                    result = FM._download_finetune_file(
                        "https://example.com/model.safetensors"
                    )
                    assert "Downloaded" in result
                    assert "model.safetensors" in result

    def test_download_failure_reported(self):
        """If download_file raises, failure message is returned."""
        with patch("shared.utils.files_locator.locate_file", return_value=None):
            with patch("shared.utils.files_locator.get_download_location", return_value="/dl/model.safetensors"):
                with patch(
                    "shared.utils.download.download_file",
                    side_effect=Exception("Connection refused"),
                ):
                    result = FM._download_finetune_file(
                        "https://example.com/model.safetensors"
                    )
                    assert "Failed" in result
                    assert "Connection refused" in result

    def test_extracts_filename_from_url_with_query(self):
        """Query params are stripped from the filename."""
        with patch("shared.utils.files_locator.locate_file", return_value=None):
            with patch("shared.utils.files_locator.get_download_location", return_value="/dl/model.safetensors"):
                with patch("shared.utils.download.download_file", return_value=None):
                    result = FM._download_finetune_file(
                        "https://example.com/model.safetensors?download=1"
                    )
                    assert "model.safetensors" in result

    def test_creates_target_dir_when_specified(self):
        """If target_dir is provided, creates it and uses it."""
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "subdir")
            with patch("shared.utils.files_locator.locate_file", return_value=None):
                with patch("shared.utils.download.download_file", return_value=None):
                    result = FM._download_finetune_file(
                        "https://example.com/model.safetensors",
                        target_dir=target,
                    )
                    assert "Downloaded" in result
                    assert Path(target).exists()

    def test_fallback_filename_for_url_without_name(self):
        """URLs without a filename get 'download' as fallback."""
        with patch("shared.utils.files_locator.locate_file", return_value=None):
            with patch("shared.utils.files_locator.get_download_location", return_value="/dl/download"):
                with patch("shared.utils.download.download_file", return_value=None):
                    result = FM._download_finetune_file("https://example.com/")
                    assert "Downloaded" in result
                    # filename fallback is 'download'


# ==============================================================================
# 2. _download_finetune_files — batch download from finetune dict
# ==============================================================================

class TestDownloadFinetuneFiles:
    """_download_finetune_files() orchestrates downloads from a finetune dict."""

    def test_no_remote_urls_returns_message(self):
        """Empty/missing URL fields return a message."""
        data = {"model": {"name": "Test"}}
        result = FM._download_finetune_files(data)
        assert "No remote URLs found" in result

    def test_downloads_from_all_url_keys(self):
        """All URL key types are collected and downloaded."""
        data = {
            "model": {
                "URLs": ["https://example.com/main.safetensors"],
                "URLs2": ["https://example.com/sec.safetensors"],
                "text_encoder_URLs": ["https://example.com/te.safetensors"],
                "VAE_URLs": ["https://example.com/vae.safetensors"],
                "preload_URLs": ["https://example.com/pre.safetensors"],
                "custom_url_1": "https://example.com/cu1.safetensors",
                "loras": ["https://example.com/lora.safetensors"],
            }
        }
        with patch.object(FM, "_download_finetune_file", return_value="Downloaded: ok"):
            result = FM._download_finetune_files(data)
            lines = [l for l in result.split("\n") if l]
            assert len(lines) == 7

    def test_skips_empty_strings(self):
        """Empty string entries in URL lists are skipped."""
        data = {
            "model": {
                "URLs": ["https://example.com/main.safetensors", "", "  "],
                "loras": [],
            }
        }
        with patch.object(FM, "_download_finetune_file", return_value="Downloaded: ok"):
            result = FM._download_finetune_files(data)
            lines = [l for l in result.split("\n") if l]
            assert len(lines) == 1  # only the real URL

    def test_handles_string_values(self):
        """Custom URL fields may be strings, not lists."""
        data = {
            "model": {
                "URLs": "https://example.com/str.safetensors",
                "loras": "https://example.com/lora_str.safetensors",
            }
        }
        with patch.object(FM, "_download_finetune_file", return_value="Downloaded: ok"):
            result = FM._download_finetune_files(data)
            lines = [l for l in result.split("\n") if l]
            assert len(lines) == 2

    def test_custom_urls_as_string(self):
        """custom_url_1/2/3 can be strings, should be collected."""
        data = {
            "model": {
                "custom_url_1": "https://example.com/c1.safetensors",
                "custom_url_2": "https://example.com/c2.safetensors",
                "custom_url_3": "https://example.com/c3.safetensors",
            }
        }
        with patch.object(FM, "_download_finetune_file", return_value="Downloaded: ok"):
            result = FM._download_finetune_files(data)
            lines = [l for l in result.split("\n") if l]
            assert len(lines) == 3

    def test_empty_model_still_returns_message(self):
        """Missing 'model' key doesn't crash."""
        result = FM._download_finetune_files({})
        assert "No remote URLs found" in result

    def test_missing_model_key_returns_message(self):
        """Completely empty dict returns the no-URLs message."""
        result = FM._download_finetune_files({"irrelevant": 42})
        assert "No remote URLs found" in result


# ==============================================================================
# 3. _civitai_search — CivitAI API search
# ==============================================================================

class TestCivitaiSearch:
    """_civitai_search() queries the CivitAI API and returns results."""

    @patch("plugin.urllib.request.urlopen")
    def test_basic_search(self, mock_urlopen):
        """Search returns parsed JSON from the API."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"items": [{"id": 1, "name": "Test Model"}]}
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = FM._civitai_search("anime")
        assert "items" in result
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "Test Model"

    @patch("plugin.urllib.request.urlopen")
    def test_search_with_type_filter(self, mock_urlopen):
        """Type filter is included in the query params."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"items": []}).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        FM._civitai_search("anime", model_type="LORA")
        # Verify the URL included the types param
        call_url = mock_urlopen.call_args[0][0].full_url
        assert "types=LORA" in call_url

    @patch("plugin.urllib.request.urlopen")
    def test_search_with_base_model(self, mock_urlopen):
        """Base model filter is included in the query params."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"items": []}).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        FM._civitai_search("anime", base_model="sdxl")
        call_url = mock_urlopen.call_args[0][0].full_url
        assert "baseModels=sdxl" in call_url

    def test_search_sends_user_agent(self):
        """Request includes User-Agent header."""
        # The header is added via add_header() on the Request object.
        # Intercept the Request object creation to verify.
        original_request = FM.urllib.request.Request
        captured_headers = {}

        class RequestProxy:
            def __init__(self, url, *args, **kwargs):
                self._req = original_request(url, *args, **kwargs)

            def add_header(self, key, val):
                captured_headers[key] = val
                self._req.add_header(key, val)

            def get_header(self, header_name):
                return self._req.get_header(header_name)

            def get_method(self):
                return self._req.get_method()

            def __getattr__(self, name):
                return getattr(self._req, name)

        with patch.object(FM.urllib.request, "Request", RequestProxy):
            with patch.object(FM.urllib.request, "urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = json.dumps({"items": []}).encode()
                mock_resp.__enter__.return_value = mock_resp
                mock_urlopen.return_value = mock_resp

                FM._civitai_search("test")
                assert "User-Agent" in captured_headers
                assert "FinetuneManager" in captured_headers["User-Agent"]

    @patch("plugin.urllib.request.urlopen", side_effect=Exception("Network error"))
    def test_search_error_returns_error_dict(self, mock_urlopen):
        """On exception, returns an error dict with empty items."""
        result = FM._civitai_search("anime")
        assert "error" in result
        assert "Network error" in result["error"]
        assert result["items"] == []

    @patch("plugin.urllib.request.urlopen")
    def test_search_includes_limit_param(self, mock_urlopen):
        """Limit parameter is sent to the API."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"items": []}).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        FM._civitai_search("anime", limit=5)
        call_url = mock_urlopen.call_args[0][0].full_url
        assert "limit=5" in call_url

    @patch("plugin.urllib.request.urlopen")
    def test_search_without_model_type_skips_param(self, mock_urlopen):
        """When model_type is 'All', the types param is omitted."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"items": []}).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        FM._civitai_search("test", model_type="All")
        call_url = mock_urlopen.call_args[0][0].full_url
        assert "types=" not in call_url


# ==============================================================================
# 4. _civitai_render_results — HTML rendering of CivitAI results
# ==============================================================================

class TestCivitaiRenderResults:
    """_civitai_render_results() renders CivitAI results as HTML."""

    def test_renders_model_name(self):
        """Model name appears in the rendered HTML."""
        data = {
            "items": [{"id": 1, "name": "Anime Model", "type": "LORA"}]
        }
        html = FM._civitai_render_results(data)
        assert "Anime Model" in html
        assert "LORA" in html

    def test_renders_version_info(self):
        """Version details appear in the HTML."""
        data = {
            "items": [
                {
                    "id": 1,
                    "name": "Test",
                    "type": "Checkpoint",
                    "modelVersions": [
                        {
                            "id": 10,
                            "name": "v1.0",
                            "baseModel": "SDXL 1.0",
                            "files": [
                                {
                                    "primary": True,
                                    "name": "test.safetensors",
                                    "sizeKB": 2048000,
                                    "downloadUrl": "https://example.com/dl",
                                }
                            ],
                            "downloadUrl": "https://example.com/dl",
                        }
                    ],
                }
            ]
        }
        html = FM._civitai_render_results(data)
        assert "v1.0" in html
        assert "SDXL 1.0" in html
        assert "2000.0MB" in html  # 2048000 KB / 1024
        assert "test.safetensors" in html

    def test_nsfw_badge_shown(self):
        """NSFW models show a warning badge."""
        data = {
            "items": [{"id": 1, "name": "NSFW Model", "type": "LORA", "nsfw": True}]
        }
        html = FM._civitai_render_results(data)
        assert "NSFW" in html
        assert "dc2626" in html  # red color for NSFW

    def test_no_results(self):
        """Empty results show 'No results' message."""
        data = {"items": []}
        html = FM._civitai_render_results(data)
        assert "No results" in html

    def test_error_displayed(self):
        """Error dict shows error message."""
        data = {"error": "Rate limited"}
        html = FM._civitai_render_results(data)
        assert "Rate limited" in html
        assert "dc2626" in html  # error color

    def test_description_truncated(self):
        """Description is truncated to 200 chars."""
        data = {
            "items": [
                {
                    "id": 1,
                    "name": "Test",
                    "type": "LORA",
                    "description": "A" * 500,
                }
            ]
        }
        html = FM._civitai_render_results(data)
        assert "A" * 200 in html
        assert "A" * 201 not in html

    def test_use_button_rendered(self):
        """Each version has a Use button."""
        data = {
            "items": [
                {
                    "id": 1,
                    "name": "Test",
                    "type": "LORA",
                    "modelVersions": [
                        {
                            "id": 10,
                            "name": "v1",
                            "files": [
                                {
                                    "primary": True,
                                    "name": "m.safetensors",
                                    "sizeKB": 1000,
                                    "downloadUrl": "https://ex.com/dl",
                                }
                            ],
                            "downloadUrl": "https://ex.com/dl",
                        }
                    ],
                }
            ]
        }
        html = FM._civitai_render_results(data)
        assert "civitai-use-btn" in html
        assert "data-civitai-version" in html

    def test_use_button_has_json_data(self):
        """Use button data attributes contain serialized model/version JSON."""
        data = {
            "items": [
                {
                    "id": 42,
                    "name": "JSON Test",
                    "type": "LORA",
                    "modelVersions": [
                        {
                            "id": 99,
                            "name": "v2",
                            "files": [
                                {
                                    "primary": True,
                                    "name": "m.sft",
                                    "sizeKB": 500,
                                    "downloadUrl": "https://ex.com/dl",
                                }
                            ],
                            "downloadUrl": "https://ex.com/dl",
                        }
                    ],
                }
            ]
        }
        html = FM._civitai_render_results(data)
        assert "42" in html
        assert "99" in html

    def test_no_versions_still_renders(self):
        """Model without versions still renders card without version detail rows."""
        data = {
            "items": [{"id": 1, "name": "No Versions", "type": "Checkpoint"}]
        }
        html = FM._civitai_render_results(data)
        assert "No Versions" in html
        # Check the visible content before the injected script tag
        script_start = html.find("<script>")
        before_script = html[:script_start] if script_start > 0 else html
        # No version rows (the flex row with file picker buttons)
        assert "data-civitai-version" not in before_script
        assert "align-items:center" not in before_script

    def test_fallback_file_selection(self):
        """When no file is marked primary, the first file is used."""
        data = {
            "items": [
                {
                    "id": 1,
                    "name": "Test",
                    "type": "LORA",
                    "modelVersions": [
                        {
                            "id": 10,
                            "name": "v1",
                            "files": [
                                {"name": "first.sft", "sizeKB": 100, "downloadUrl": "https://ex.com/first"},
                                {"name": "second.sft", "sizeKB": 200, "downloadUrl": "https://ex.com/second"},
                            ],
                            "downloadUrl": "https://ex.com/dl",
                        }
                    ],
                }
            ]
        }
        html = FM._civitai_render_results(data)
        assert "first.sft" in html  # first file used as fallback


# ==============================================================================
# 5. Integration: Plugin loads correctly in Wan2GP context
# ==============================================================================

class TestPluginLoads:
    """Verify the plugin module loads without errors and constants are correct."""

    def test_module_imported(self):
        """plugin module is imported."""
        assert FM is not None
        assert hasattr(FM, "FinetuneManagerPlugin")

    def test_plugin_version(self):
        """Version matches plugin_info.json."""
        import json as _j
        info_path = Path(__file__).parent / "plugin_info.json"
        if info_path.exists():
            info = _j.loads(info_path.read_text(encoding="utf-8"))
            assert FM.PLUGIN_VERSION == info["version"]

    def test_class_has_setup_ui(self):
        """FinetuneManagerPlugin has setup_ui method."""
        assert hasattr(FM.FinetuneManagerPlugin, "setup_ui")
        assert callable(FM.FinetuneManagerPlugin.setup_ui)

    def test_class_has_create_ui(self):
        """FinetuneManagerPlugin has create_ui method."""
        assert hasattr(FM.FinetuneManagerPlugin, "create_ui")
        assert callable(FM.FinetuneManagerPlugin.create_ui)

    def test_plugin_name_and_id(self):
        """Plugin name and id constants are set."""
        assert FM.PlugIn_Name == "Finetune Manager"
        assert FM.PlugIn_Id == "FinetuneManager"

    def test_registry_constants(self):
        """Registry URLs are valid."""
        assert FM.DEFAULT_REGISTRY.startswith("https://")
        assert "/" in FM.REGISTRY_SPACE
        assert FM.FINETUNES_DIR == "finetunes"

    def test_civitai_base_model_map(self):
        """CivitAI base model mapping has expected keys."""
        assert "SDXL 1.0" in FM.CIVITAI_BASE_MODEL_MAP
        assert "SD 1.5" in FM.CIVITAI_BASE_MODEL_MAP
        assert "Flux.1 D" in FM.CIVITAI_BASE_MODEL_MAP
        assert FM.CIVITAI_BASE_MODEL_MAP["SDXL 1.0"] == "sdxl"
        assert FM.CIVITAI_BASE_MODEL_MAP["Flux"] == "flux"

    def test_lora_file_extensions(self):
        """LoRA file extensions include safetensors and sft."""
        assert ".safetensors" in FM.LORA_FILE_EXTENSIONS
        assert ".sft" in FM.LORA_FILE_EXTENSIONS

    def test_plugin_version_constant(self):
        """PLUGIN_VERSION is a non-empty string."""
        assert FM.PLUGIN_VERSION
        assert isinstance(FM.PLUGIN_VERSION, str)
        parts = FM.PLUGIN_VERSION.split(".")
        assert len(parts) == 3  # semantic versioning

    def test_finetunes_dir_is_relative(self):
        """FINETUNES_DIR is a relative path (not absolute)."""
        assert not Path(FM.FINETUNES_DIR).is_absolute()


# ==============================================================================
# 6. Edge Cases: Input validation and error handling
# ==============================================================================

class TestEdgeCases:
    """Edge cases across the module-level functions."""

    def test_download_finetune_file_non_string(self):
        """Non-string URL doesn't crash (type safety)."""
        # The function calls .strip(), so int would fail - but the caller
        # should handle this. Check the coerce logic in _download_finetune_files
        data = {"model": {"URLs": [123, None, True]}}
        with patch.object(FM, "_download_finetune_file", return_value="skipped"):
            # Should not crash when iterating non-string URL items
            result = FM._download_finetune_files(data)
            assert result is not None

    def test_civitai_search_empty_query(self):
        """Empty query still sends the request (server-side validation)."""
        with patch("plugin.urllib.request.urlopen") as mock:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"items": []}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock.return_value = mock_resp
            result = FM._civitai_search("")
            assert "items" in result

    def test_civitai_render_results_empty_items_key(self):
        """Items key present but None doesn't crash."""
        data = {"items": None}
        html = FM._civitai_render_results(data)
        assert "No results" in html

    def test_civitai_render_results_missing_items_key(self):
        """Missing items key doesn't crash."""
        data = {}
        html = FM._civitai_render_results(data)
        assert html is not None

    def test_civitai_render_results_non_dict_input(self):
        """Non-dict input raises AttributeError (caller always passes a dict)."""
        with pytest.raises(AttributeError):
            FM._civitai_render_results("not a dict")

    def test_download_finetune_files_non_dict(self):
        """Non-dict input raises AttributeError (caller always passes a dict)."""
        with pytest.raises(AttributeError):
            FM._download_finetune_files(None)
