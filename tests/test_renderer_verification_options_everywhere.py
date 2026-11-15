"""Every speech verification gets the passage's own options, so chosen pronunciations are always checked."""

import ast
from pathlib import Path
import unittest

RENDERER = Path(__file__).resolve().parents[1] / "epub_tts_narrator" / "renderer"


class SpeechVerificationOptionsTests(unittest.TestCase):
    def test_every_speech_validate_audio_call_passes_verification_options(self):
        missing = []
        for path in sorted(RENDERER.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None))
                        == "validate_audio"):
                    continue
                purpose = next((keyword.value for keyword in node.keywords if keyword.arg == "purpose"), None)
                speech = isinstance(purpose, ast.Constant) and purpose.value == "speech"
                spread = any(keyword.arg is None for keyword in node.keywords)
                if speech and not spread:
                    missing.append(f"{path.name}:{node.lineno}")
        # A speech take verified without them never has its chosen readings
        # checked (name_shape), so a misread name such as Ce'Nedra passes.
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
