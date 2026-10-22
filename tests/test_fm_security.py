"""Tests for the plugin's URL policy and download guards.

Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import fm_pack
import fm_security


class _Response(io.BytesIO):
    def __init__(self, body: bytes, content_type: str = "application/octet-stream"):
        super().__init__(body)
        self.status = 200
        self.headers = {
            "Content-Length": str(len(body)),
            "Content-Type": content_type,
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class UrlPolicyTests(unittest.TestCase):
    def test_accepts_public_https(self):
        self.assertIsNone(fm_security.url_problem("https://huggingface.co/a/b.json"))

    def test_rejects_plain_http(self):
        problem = fm_security.url_problem("http://huggingface.co/a/b.json")
        self.assertIn("HTTP", problem or "")

    def test_rejects_credentials(self):
        problem = fm_security.url_problem("https://user:pass@example.com/file")
        self.assertIn("credentials", problem or "")

    def test_rejects_non_http_schemes(self):
        for url in (
            "javascript:alert(1)",
            "file:///C:/Windows/win.ini",
            "data:text/plain,hello",
            "ftp://example.com/a",
        ):
            with self.subTest(url=url):
                self.assertIsNotNone(fm_security.url_problem(url))

    def test_rejects_loopback_and_private_literals_without_dns(self):
        for url in (
            "https://127.0.0.1/a",
            "https://10.0.0.1/a",
            "https://[::1]/a",
            "https://localhost/a",
        ):
            with self.subTest(url=url):
                self.assertIsNotNone(fm_security.url_problem(url))

    def test_allowlist_rejects_unlisted_host(self):
        problem = fm_security.url_problem(
            "https://evil.example/a", allowed_hosts=("huggingface.co",)
        )
        self.assertIn("allowlisted", problem or "")

    def test_allowlist_accepts_subdomain(self):
        self.assertIsNone(
            fm_security.url_problem(
                "https://cdn-lfs.hf.co/a", allowed_hosts=("hf.co",)
            )
        )

    def test_rejects_control_characters(self):
        self.assertIsNotNone(
            fm_security.url_problem("https://example.com/a\nb")
        )

    def test_content_problem(self):
        self.assertIsNotNone(fm_security.content_problem("text/html; charset=utf-8"))
        self.assertIsNotNone(fm_security.content_problem("application/json"))
        self.assertIsNone(fm_security.content_problem("application/octet-stream"))

    def test_probe_reports_blocked_redirect_as_error(self):
        error = urllib.error.HTTPError(
            "https://evil.test/", 302, "blocked redirect: blocked local host", None, None
        )
        with mock.patch("fm_security.open_url", side_effect=error):
            code, message = fm_security.probe_url("https://huggingface.co/a")
        self.assertIsNone(code)
        self.assertIn("blocked redirect", message or "")


class DownloadGuardTests(unittest.TestCase):
    def test_unsafe_url_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model.safetensors"
            with self.assertRaises(fm_security.UnsafeUrl):
                fm_pack.download_verified(
                    "http://127.0.0.1/model.safetensors",
                    dest,
                    allowed_hosts=None,
                )
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_repo_id_traversal_is_rejected(self):
        self.assertFalse(fm_pack.repo_id_ok("../evil"))
        self.assertFalse(fm_pack.repo_id_ok("owner/.."))
        self.assertTrue(fm_pack.repo_id_ok("owner/repo"))

    def test_download_writes_verified_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model.safetensors"
            with mock.patch(
                "fm_security.open_url", return_value=_Response(b"weights")
            ):
                result = fm_pack.download_verified(
                    "https://huggingface.co/a/model.safetensors", dest
                )
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(dest.read_bytes(), b"weights")

    def test_download_rejects_html_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model.safetensors"
            with mock.patch(
                "fm_security.open_url",
                return_value=_Response(b"<html>", "text/html"),
            ):
                with self.assertRaises(ValueError):
                    fm_pack.download_verified(
                        "https://huggingface.co/a/model.safetensors", dest
                    )
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_download_rejects_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model.safetensors"
            with mock.patch(
                "fm_security.open_url", return_value=_Response(b"short")
            ):
                with self.assertRaises(ValueError):
                    fm_pack.download_verified(
                        "https://huggingface.co/a/model.safetensors",
                        dest,
                        expected_bytes=99,
                    )
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())


class DeleteContainmentTests(unittest.TestCase):
    def test_civitai_delete_ignores_paths_outside_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpts = root / "ckpts"
            loras = root / "loras"
            fins = root / "finetunes"
            for folder in (ckpts, loras, fins):
                folder.mkdir()
            outside = root / "outside.txt"
            outside.write_text("keep", encoding="utf-8")
            inside = ckpts / "adult-pack" / "ok.safetensors"
            inside.parent.mkdir()
            inside.write_text("remove", encoding="utf-8")
            state = {
                "civitaiInstalled": {
                    "42": {"files": [str(outside), str(inside)]}
                }
            }
            result = fm_pack.delete_civitai(
                {"id": "42", "architecture": "wan"},
                state,
                ckpts,
                lambda _arch: loras,
                fins,
            )
            self.assertTrue(outside.exists())
            self.assertFalse(inside.exists())
            self.assertEqual(result["removedFiles"], [str(inside)])

    def test_finetune_delete_ignores_absolute_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpts = root / "ckpts"
            fins = root / "finetunes"
            ckpts.mkdir()
            fins.mkdir()
            outside = root / "outside.safetensors"
            outside.write_bytes(b"keep")
            entry = {"id": "x", "urls": [str(outside)]}
            result = fm_pack.delete_finetune(entry, ckpts, fins)
            self.assertEqual(result["removedWeights"], [])
            self.assertTrue(outside.is_file())
