"""Ctrl-C drains admitted Gemini casting work before cleanup."""

from concurrent.futures import ThreadPoolExecutor
import unittest

import test_renderer_casting_concurrency as concurrency
import test_renderer_gemini_fresh_regeneration as fixtures


@unittest.skipUnless(fixtures.FFMPEG, "FFmpeg is required for casting integration")
class CastingInterruptionTests(unittest.TestCase):
    setUp = fixtures.GeminiFreshRegenerationTests.setUp
    session = concurrency.CastingConcurrencyTests.session

    def test_keyboard_interrupt_drains_started_sibling_before_cleanup(self):
        remote = concurrency.GeminiOverlapTransport(
            self, blocked=(1,), release_on_overlap=False)
        remote.errors[2] = KeyboardInterrupt("fixture casting interrupt")
        session = self.session(remote)

        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(session.ensure_ready_many, ["voice_mara", "voice_lena"])
            try:
                self.assertTrue(remote.first_interaction.wait(concurrency.WAIT_SECONDS))
                self.assertTrue(remote.first_cleanup.wait(concurrency.WAIT_SECONDS))
                self.assertFalse(pending.done())
                self.assertTrue((session.store.directory / ".lock").exists())
            finally:
                remote.release.set()
            with self.assertRaises(KeyboardInterrupt):
                pending.result(2 * concurrency.WAIT_SECONDS)

        self.assertEqual(sum(session.store.load(voice_id) is not None
                             for voice_id in ("voice_mara", "voice_lena")), 1)
        self.assertEqual(remote.hosted_voice_ids, set())
        self.assertFalse((session.store.directory / ".lock").exists())


if __name__ == "__main__":
    unittest.main()
