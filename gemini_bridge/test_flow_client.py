from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from gemini_bridge.flow_client import FlowCliClient, resolve_flow_model


class FlowClientTests(unittest.TestCase):
    def test_model_aliases_keep_old_faa_settings_compatible(self):
        self.assertEqual(resolve_flow_model("flow-nano-pro"), "nano-pro")
        self.assertEqual(resolve_flow_model("gemini-3.1-flash-image"), "nano-pro")
        self.assertEqual(resolve_flow_model("flow-nano2"), "nano2")
        self.assertEqual(resolve_flow_model("flow-image4"), "image4")

    def test_unknown_model_falls_back_to_configured_default(self):
        self.assertEqual(resolve_flow_model("unknown", "image4"), "image4")

    def test_output_discovery_accepts_flow_generated_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "nested" / "flow-result.jpeg"
            generated.parent.mkdir()
            generated.write_bytes(b"image-data")
            client = FlowCliClient(
                profile="faa",
                home=str(root / "home"),
                default_model="flow-nano-pro",
                timeout=600,
                retries=3,
                max_image_bytes=50 * 1024 * 1024,
            )
            self.assertEqual(client._find_output(root / "missing.jpg", root), generated)

    def test_generation_forces_single_16_by_9_flow_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "home" / "profile_faa"
            profile.mkdir(parents=True)
            (profile / ".gflow_account").write_text("account@example.test", encoding="utf-8")
            client = FlowCliClient(
                profile="faa",
                home=str(root / "home"),
                default_model="flow-nano-pro",
                timeout=600,
                retries=3,
                max_image_bytes=50 * 1024 * 1024,
            )
            captured = []

            def fake_run(command, timeout):
                captured.append((command, timeout))
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(b"x" * 25_000)
                return subprocess.CompletedProcess(command, 0, "ok", "")

            with patch.object(FlowCliClient, "installed", new=True):
                with patch.object(client, "_run", side_effect=fake_run):
                    data, mime = client.generate("test prompt", "gemini-3.1-flash-image")

            command, timeout = captured[0]
            self.assertEqual(len(data), 25_000)
            self.assertEqual(mime, "image/jpeg")
            self.assertEqual(timeout, 600)
            self.assertEqual(command[command.index("--model") + 1], "nano-pro")
            self.assertEqual(command[command.index("--aspect") + 1], "16:9")
            self.assertEqual(command[command.index("--count") + 1], "1")
            self.assertEqual(command[command.index("--ui-mode") + 1], "classic")


if __name__ == "__main__":
    unittest.main()
