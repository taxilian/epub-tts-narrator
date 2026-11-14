"""The shared 1x/2x listening speed: one setting, honored by every AudioPlayer, pitch unchanged."""

import asyncio
import hashlib
import math
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer import playback_speed
from epub_tts_narrator.renderer.audition import AudioPlayer, run_casting
from epub_tts_narrator.renderer.playback_speed import PlaybackSpeed

POPEN = "epub_tts_narrator.renderer.audition.subprocess.Popen"
WHICH = "epub_tts_narrator.renderer.audition.shutil.which"


def running_process():
    process = Mock()
    process.poll.return_value = None
    process.returncode = 0
    process.wait.return_value = 0
    return process


class PlaybackSpeedTests(unittest.TestCase):
    def test_the_speed_switches_between_1x_and_2x_and_says_which(self):
        speed = PlaybackSpeed()
        self.assertEqual(speed.label, "1x")
        self.assertEqual(speed.toggle(), 2.0)
        self.assertEqual(speed.label, "2x")
        self.assertEqual(speed.toggle(), 1.0)
        with self.assertRaisesRegex(ValueError, "1x or 2x"):
            PlaybackSpeed(1.5)

    def test_a_player_without_its_own_speed_uses_the_shared_one(self):
        self.assertIs(playback_speed.speed_of(object()), playback_speed.SHARED)
        self.assertIs(AudioPlayer(["afplay"]).speed, playback_speed.SHARED)
        own = PlaybackSpeed()
        self.assertIs(playback_speed.speed_of(AudioPlayer(["afplay"], speed=own)), own)

    def test_toggling_restarts_a_player_that_can_restart(self):
        player = Mock(speed=PlaybackSpeed())
        self.assertEqual(playback_speed.toggle(player), "2x")
        player.restart.assert_called_once_with()
        self.assertEqual(playback_speed.help_text(player), "> speed (2x)")
        self.assertIn("now 2x", playback_speed.toggled_notice(player))

    def test_line_command_is_only_the_speed_key(self):
        player = Mock(speed=PlaybackSpeed())
        self.assertIsNone(playback_speed.line_command("q", player))
        self.assertIsNone(playback_speed.line_command(">>", player))
        self.assertEqual(player.speed.label, "1x")
        self.assertIn("now 2x", playback_speed.line_command(" > ", player))
        self.assertEqual(player.speed.label, "2x")


class AudioPlayerSpeedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.sample = self.root / "narrator-take-attempt-001.wav"
        self.sample.write_bytes(b"fixture")

    def test_afplay_gets_its_time_pitch_rate_only_at_2x(self):
        speed = PlaybackSpeed()
        player = AudioPlayer(["afplay", "-v", "0.5"], speed=speed)
        with patch(WHICH, return_value="/usr/bin/afplay"), patch(POPEN, return_value=running_process()) as popen:
            player.start(self.sample)
            self.assertEqual(popen.call_args[0][0], ["/usr/bin/afplay", "-v", "0.5", str(self.sample)])
            speed.toggle()
            player.start(self.sample)
            self.assertEqual(popen.call_args[0][0],
                             ["/usr/bin/afplay", "-v", "0.5", "-r", "2", "-q", "1", str(self.sample)])
        player.stop()

    def test_toggling_the_shared_speed_changes_every_player_next_play(self):
        self.addCleanup(setattr, playback_speed.SHARED, "rate", playback_speed.SHARED.rate)
        playback_speed.SHARED.rate = 1.0
        first, second = AudioPlayer(["afplay"]), AudioPlayer(["afplay"])
        playback_speed.toggle(first)  # Not playing: nothing restarts.
        with patch(WHICH, return_value="/usr/bin/afplay"), patch(POPEN, return_value=running_process()) as popen:
            second.start(self.sample)
        self.assertIn("-r", popen.call_args[0][0])
        second.stop()

    def test_restart_plays_the_same_file_again_and_poll_never_sees_a_gap(self):
        speed = PlaybackSpeed()
        old, new = running_process(), running_process()
        player = AudioPlayer(["afplay"], speed=speed)
        with patch(WHICH, return_value="/usr/bin/afplay"), patch(POPEN, side_effect=[old, new]) as popen:
            self.assertFalse(player.restart())  # Nothing plays yet.
            player.start(self.sample)
            self.assertEqual(playback_speed.toggle(player), "2x")
        old.terminate.assert_called_once_with()
        self.assertIs(player.process, new)
        self.assertTrue(player.poll())
        self.assertEqual(popen.call_args[0][0][-1], str(self.sample))
        self.assertIn("-r", popen.call_args[0][0])
        new.poll.return_value = 0
        self.assertFalse(player.restart())  # Finished audio is reported as finished, not replayed.
        self.assertFalse(player.poll())

    def test_another_player_without_ffmpeg_says_how_to_get_back_to_1x(self):
        player = AudioPlayer(["fixture-player"], speed=PlaybackSpeed(2.0))
        with patch(WHICH, side_effect=lambda name: None if name == "ffmpeg" else "/bin/fixture-player"), \
                patch(POPEN) as popen:
            with self.assertRaisesRegex(ValueError, "afplay or FFmpeg.*> to return to 1x"):
                player.start(self.sample)
        popen.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg/ffprobe required")
    def test_another_player_hears_a_named_tempo_copy_at_the_same_pitch_and_the_take_is_untouched(self):
        rate, seconds, tone = 24000, 1.0, 440
        source = self.root / "mara-chapter-3-take-attempt-002.wav"
        with wave.open(str(source), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * tone * index / rate)))
                                        for index in range(int(rate * seconds))))
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        real_popen, real_which, played = subprocess.Popen, shutil.which, []

        def launch(argv, **kwargs):
            if argv[0] == "/bin/fixture-player":
                played.append(argv)
                return running_process()
            return real_popen(argv, **kwargs)  # FFmpeg itself runs for real.

        player = AudioPlayer(["fixture-player"], speed=PlaybackSpeed(2.0))
        with patch(WHICH, side_effect=lambda name: "/bin/fixture-player" if name == "fixture-player"
                   else real_which(name)), patch(POPEN, side_effect=launch):
            player.start(source)
        copy = Path(played[0][-1])
        self.assertEqual(copy.name, "mara-chapter-3-take-attempt-002-2x-speed-listening-copy.wav")
        self.assertNotEqual(copy.parent, source.parent)
        with wave.open(str(copy)) as heard:
            frames, heard_rate = heard.getnframes(), heard.getframerate()
            samples = struct.unpack(f"<{frames}h", heard.readframes(frames))
        duration = frames / heard_rate
        crossings = sum(1 for left, right in zip(samples, samples[1:]) if left < 0 <= right)
        self.assertAlmostEqual(duration, seconds / 2, delta=0.05)
        self.assertAlmostEqual(crossings / duration, tone, delta=tone * 0.03)  # Pitch unchanged.
        player.stop()
        self.assertFalse(copy.exists())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


class LineMenuSpeedTests(unittest.TestCase):
    def test_casting_menu_shows_the_speed_and_greater_than_switches_it(self):
        from test_renderer_audition import SessionDouble
        player = Mock(speed=PlaybackSpeed())
        answers = iter([">", "q"])
        lines = []
        run_casting(SessionDouble(), player, input_fn=lambda prompt: next(answers), output=lines.append,
                    playback_fn=lambda *_: True)
        text = "\n".join(lines)
        self.assertIn(">: playback speed (now 1x)", text)
        self.assertIn("Playback speed is now 2x", text)
        self.assertIn(">: playback speed (now 2x)", text)
        self.assertEqual(player.speed.label, "2x")


class SpeedPlayer:
    """AudioPlayer's start/poll/stop/restart with its own speed; plays until stopped."""

    def __init__(self):
        self.speed, self.started, self.restarts, self.playing = PlaybackSpeed(), [], 0, False

    def start(self, path):
        self.started.append((path, self.speed.label))
        self.playing = True

    def poll(self):
        return self.playing

    def stop(self):
        self.playing = False

    def restart(self):
        if self.playing:
            self.restarts += 1
            self.started.append((self.started[-1][0], self.speed.label))
        return self.playing


class TakesScreenSpeedTests(unittest.IsolatedAsyncioTestCase):
    """The run shell's (and review-ui's) Takes dialog: > switches the speed and restarts the take playing."""

    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    async def test_greater_than_switches_speed_restarts_the_playing_take_and_the_help_says_so(self):
        from textual.app import App
        from epub_tts_narrator.run_shell_screens import TakesScreen
        from test_render_repairs import FakeRepair
        player = SpeedPlayer()
        screen = TakesScreen("Chapter 4: every take of this passage", FakeRepair(), player)

        class Host(App):
            def on_mount(self):
                self.push_screen(screen)

        async with Host().run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            help_line = lambda: str(screen.query_one("#takes-help").render())  # noqa: E731
            self.assertIn("> speed (1x)", help_line())
            await pilot.press("greater_than_sign")  # Nothing playing: only the next play changes.
            self.assertEqual((player.speed.label, player.restarts), ("2x", 0))
            self.assertIn("> speed (2x)", help_line())
            await pilot.press("enter")
            await pilot.pause(0.1)
            self.assertEqual(player.started, [("/clips/take-1.wav", "2x")])
            await pilot.press("greater_than_sign")
            self.assertEqual(player.restarts, 1)
            self.assertEqual(player.started[-1], ("/clips/take-1.wav", "1x"))
            self.assertIn("> speed (1x)", help_line())


class PronunciationQueueSpeedTests(unittest.IsolatedAsyncioTestCase):
    """In the listening queue any key interrupts a walk, except > which only changes the speed."""

    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    async def test_greater_than_restarts_the_word_at_2x_without_interrupting_listening(self):
        from epub_tts_narrator.renderer.pronounce_tui import PronunciationQueueApp
        player, heard = SpeedPlayer(), []

        def command(input_fn, output, playback_fn, observer, wait_fn, redraw):
            heard.append(playback_fn(player, "/previews/qarm-pronunciation-preview.wav"))
            return 0

        app = PronunciationQueueApp(command, title="pronunciations — book")
        async with app.run_test(size=(150, 42)) as pilot:
            deadline = asyncio.get_running_loop().time() + 5
            while not player.started:
                self.assertLess(asyncio.get_running_loop().time(), deadline)
                await pilot.pause(0.05)
            await pilot.press("greater_than_sign")
            await pilot.pause(0.1)
            self.assertEqual((player.restarts, player.speed.label, heard), (1, "2x", []))
            self.assertIn("> speed (2x)", str(app.query_one("#pron-help").render()))
            await pilot.press("x")  # Any other key still interrupts.
            while app.is_running:
                self.assertLess(asyncio.get_running_loop().time(), deadline)
                await pilot.pause(0.05)
        self.assertEqual(heard, [False])


if __name__ == "__main__":
    unittest.main()
