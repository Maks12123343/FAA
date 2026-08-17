import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import download_ready_from_site as downloader
from backend.gemini_image import _normalize_png, is_valid_thumbnail


class ThumbnailDeliveryTests(unittest.TestCase):
    def test_normalize_png_is_exact_youtube_size(self):
        source = BytesIO()
        Image.new("RGB", (900, 1200), "red").save(source, format="JPEG")

        normalized = _normalize_png(source.getvalue())

        with Image.open(BytesIO(normalized)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1920, 1080))

    def test_cached_thumbnail_validation_rejects_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thumbnail.png"
            Image.new("RGB", (1280, 720), "blue").save(path, format="PNG")
            self.assertFalse(is_valid_thumbnail(str(path)))

            Image.new("RGB", (1920, 1080), "blue").save(path, format="PNG")
            self.assertTrue(is_valid_thumbnail(str(path)))

    def test_failed_thumbnail_does_not_publish_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                out_dir=tmp,
                state_file="",
                project_ids=[],
                base_url="http://127.0.0.1:5050",
                retries=1,
                timeout=5,
                download_timeout=60,
                languages=["pl"],
                latest_per_language=False,
                force=False,
                dry_run=False,
            )
            item = {
                "project_id": "russia_ukraine_war_pl_123",
                "language": "pl",
                "language_name": "Polish",
                "thumbnail_image_url": "/thumbnail",
                "title": "Title",
            }

            def fake_download(url, dest, *unused_args, **unused_kwargs):
                if "thumbnail" in str(dest):
                    raise RuntimeError("simulated thumbnail failure")
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(b"video")

            with (
                patch.object(downloader, "_select_projects", return_value=[item]),
                patch.object(downloader, "_project_metadata", return_value=item),
                patch.object(downloader, "_download_file", side_effect=fake_download),
            ):
                self.assertEqual(downloader._run_once(args), 0)

            self.assertEqual(list(Path(tmp).rglob("video*.mp4")), [])
            self.assertEqual(list(Path(tmp).rglob("*.stage")), [])


if __name__ == "__main__":
    unittest.main()
