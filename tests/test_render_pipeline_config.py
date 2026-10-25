"""Validated pipeline scheduler settings project without touching render identity."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.chapter_protocol import SAFE_RENDER_KEYS, validate_render_settings
from epub_tts_narrator.renderer.config import DEFAULTS, RENDER_CONTRACT, validate_distributed_settings
from epub_tts_narrator.renderer.render_pipeline_config import (
    PIPELINE_RENDER_POOL_DEFAULTS, SCHEDULERS, chapter_share_for, effective_scheduling,
)
from epub_tts_narrator.renderer.runpod_config import build_remote_config

SCHEDULER_KEYS = frozenset(PIPELINE_RENDER_POOL_DEFAULTS)


def configuration(**pool):
    config = deepcopy(DEFAULTS)
    config["render_pool"].update(pool)
    if pool.get("workers") is None:
        config["render_pool"]["workers"] = [{
            "name": "fixture", "type": "fish_http", "url": "http://fixture.invalid/v1/tts",
            "concurrency": 8}]
    config["chapter_server"].update(runpod_api_key="file-runpod-key")
    config["chapter_storage"].update(bucket="books", endpoint_url="https://storage.example.test",
                                     region="default", access_key_id="ceph-key",
                                     secret_access_key="ceph-secret")
    return config


class PipelineConfigDefaultsTests(unittest.TestCase):
    def test_scheduler_default_is_legacy_and_automatic_knob_defaults_are_zero(self):
        self.assertEqual(PIPELINE_RENDER_POOL_DEFAULTS["scheduler"], "legacy")
        self.assertEqual(DEFAULTS["render_pool"]["scheduler"], "legacy")
        self.assertEqual(DEFAULTS["render_pool"]["chapter_concurrency"], 0)
        self.assertEqual(DEFAULTS["render_pool"]["synthesis_backlog"], 0)
        self.assertEqual(DEFAULTS["render_pool"]["pending_audio_mb"], 256)
        self.assertEqual(DEFAULTS["render_pool"]["review_concurrency"], 2)

    def test_effective_values_match_the_plan_table(self):
        legacy = effective_scheduling(configuration(scheduler="legacy"))
        self.assertEqual(legacy["chapter_concurrency"], 2)
        pipeline = effective_scheduling(configuration(scheduler="pipeline"))
        self.assertEqual(pipeline["chapter_concurrency"], 4)
        # Managed 8-slot remote: logical 128, backlog 32, per-chapter share 32.
        self.assertEqual(pipeline["max_inflight"], 128)
        self.assertEqual(pipeline["synthesis_backlog"], 32)
        self.assertEqual(chapter_share_for(pipeline), 32)
        self.assertEqual(pipeline["review_concurrency"], 2)
        self.assertEqual(pipeline["pending_audio_mb"], 256)

    def test_automatic_limits_follow_worker_capacity_and_stay_above_it(self):
        two_slots = effective_scheduling(configuration(
            workers=[{"name": "w", "type": "fish_http", "url": "http://w.invalid/v1/tts",
                      "concurrency": 2}]))
        self.assertEqual(two_slots["max_inflight"], 64)
        self.assertEqual(two_slots["synthesis_backlog"], 8)
        local_only = effective_scheduling(configuration(
            workers=[{"name": "w", "type": "local", "concurrency": 1}]))
        self.assertEqual(local_only["max_inflight"], 16)
        eight = effective_scheduling(configuration())
        self.assertEqual(eight["max_inflight"], 128)
        self.assertGreaterEqual(eight["max_inflight"], 8)
        self.assertGreaterEqual(two_slots["max_inflight"], 2)

    def test_explicit_values_are_preserved_over_automatic_ones(self):
        explicit = effective_scheduling(configuration(
            scheduler="pipeline", chapter_concurrency=6, max_inflight=40,
            synthesis_backlog=16, review_concurrency=3, pending_audio_mb=512))
        self.assertEqual(explicit["chapter_concurrency"], 6)
        self.assertEqual(explicit["max_inflight"], 40)
        self.assertEqual(explicit["synthesis_backlog"], 16)
        self.assertEqual(explicit["review_concurrency"], 3)
        self.assertEqual(explicit["pending_audio_mb"], 512)

    def test_increasing_chapters_never_multiplies_shared_physical_caps(self):
        four = effective_scheduling(configuration(scheduler="pipeline", chapter_concurrency=4))
        sixteen = effective_scheduling(configuration(scheduler="pipeline", chapter_concurrency=16))
        for key in ("max_inflight", "synthesis_backlog", "review_concurrency", "pending_audio_mb"):
            self.assertEqual(four[key], sixteen[key], key)
        self.assertLess(chapter_share_for(sixteen), chapter_share_for(four))


class PipelineConfigValidationTests(unittest.TestCase):
    def test_malformed_scheduler_settings_are_rejected(self):
        for key, values in (
                ("scheduler", ("turbo", True, [], 0, "PIPELINE")),
                ("chapter_concurrency", (True, -1, 17, "4", 1.5, 256)),
                ("synthesis_backlog", (True, -1, 257, "8", 2.5)),
                ("pending_audio_mb", (True, 0, 63, 4097, "256", 256.0)),
                ("review_concurrency", (True, 0, 9, "2", 2.5))):
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, f"render_pool.{key}"):
                        validate_distributed_settings(configuration(**{key: value}))

    def test_valid_scheduler_setting_ranges_are_accepted(self):
        for key, values in (("scheduler", SCHEDULERS), ("chapter_concurrency", (0, 1, 16)),
                            ("synthesis_backlog", (0, 8, 256)), ("pending_audio_mb", (64, 256, 4096)),
                            ("review_concurrency", (1, 2, 8))):
            for value in values:
                with self.subTest(key=key, value=value):
                    validate_distributed_settings(configuration(**{key: value}))

    def test_explicit_synthesis_backlog_below_tts_capacity_is_rejected(self):
        config = configuration(synthesis_backlog=4, workers=[
            {"name": "w", "type": "fish_http", "url": "http://w.invalid/v1/tts", "concurrency": 8}])
        with self.assertRaisesRegex(ValueError, "synthesis_backlog"):
            validate_distributed_settings(config)

    def test_explicit_max_inflight_still_must_cover_total_tts_capacity(self):
        config = configuration(max_inflight=4, workers=[
            {"name": "w", "type": "fish_http", "url": "http://w.invalid/v1/tts", "concurrency": 8}])
        with self.assertRaisesRegex(ValueError, "max_inflight"):
            validate_distributed_settings(config)


class PipelineProjectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def legacy_remote(self, local):
        deployment = {"tts_backend": "sglang", "tts_concurrency": 8,
                      "state_dir": str(self.directory / "state")}
        return build_remote_config(local, {}, "pod", "app-key", deployment=deployment)

    def test_scheduler_knobs_never_enter_the_wire_contract_or_chapter_settings(self):
        self.assertTrue(SCHEDULER_KEYS.isdisjoint(SAFE_RENDER_KEYS))
        job = {key: value["default"] if isinstance(value, dict) and "default" in value else None
               for key, value in RENDER_CONTRACT.items() if value.get("remote_safe")}
        with self.assertRaisesRegex(ValueError, "unsupported settings"):
            validate_render_settings({**job, "scheduler": "pipeline"})
        with self.assertRaisesRegex(ValueError, "unsupported settings"):
            validate_render_settings({**job, "chapter_concurrency": 4})

    def test_legacy_projection_is_unchanged_and_carries_no_scheduler_controls(self):
        local = configuration(scheduler="legacy", chapter_concurrency=4, max_inflight=40)
        remote = self.legacy_remote(local)
        self.assertNotIn("scheduler", remote["render_pool"])
        self.assertNotIn("chapter_concurrency", remote["render_pool"])
        self.assertNotIn("synthesis_backlog", remote["render_pool"])
        self.assertNotIn("review_concurrency", remote["render_pool"])
        self.assertEqual(remote["render_pool"]["max_inflight"], 32)

    def test_pipeline_projection_keeps_worker_config_and_honors_automatic_inflight(self):
        local = configuration(scheduler="pipeline", chapter_concurrency=4, max_inflight=0,
                              synthesis_backlog=0, review_concurrency=2, pending_audio_mb=256)
        remote = self.legacy_remote(local)
        worker, = remote["render_pool"]["workers"]
        self.assertEqual(worker["concurrency"], 8)
        self.assertEqual(worker["protocol"], "fish_sglang_drama3")
        self.assertEqual(worker["url"], "http://127.0.0.1:8080/v1/audio/speech")
        self.assertEqual(remote["render_pool"]["scheduler"], "pipeline")
        self.assertEqual(remote["render_pool"]["chapter_concurrency"], 4)
        self.assertEqual(remote["render_pool"]["max_inflight"], 0)
        self.assertEqual(remote["render_pool"]["synthesis_backlog"], 0)
        self.assertEqual(remote["render_pool"]["review_concurrency"], 2)
        self.assertEqual(remote["render_pool"]["pending_audio_mb"], 256)
        effective = effective_scheduling(remote)
        self.assertEqual(effective["max_inflight"], 128)

    def test_projections_do_not_alter_fingerprint_input_keys(self):
        legacy = configuration(scheduler="legacy")
        pipeline = configuration(scheduler="pipeline", chapter_concurrency=4,
                                 synthesis_backlog=16, review_concurrency=3)
        legacy_remote = self.legacy_remote(legacy)
        pipeline_remote = self.legacy_remote(pipeline)
        self.assertEqual(pipeline_remote["render"], legacy_remote["render"])
        self.assertEqual(pipeline_remote["engine"], legacy_remote["engine"])
        self.assertEqual(pipeline_remote["verification"], legacy_remote["verification"])

    def test_unknown_scheduler_mode_fails_projection_before_any_worker_use(self):
        local = configuration()
        local["render_pool"]["scheduler"] = "turbo"
        with self.assertRaisesRegex(ValueError, "scheduler"):
            self.legacy_remote(local)

    def test_conflicting_projected_backlog_fails_before_paid_work(self):
        local = configuration(scheduler="pipeline", synthesis_backlog=4,
                              workers=[{"name": "w", "type": "fish_http",
                                        "url": "http://w.invalid/v1/tts", "concurrency": 1}])
        deployment = {"tts_backend": "sglang", "tts_concurrency": 8,
                      "state_dir": str(self.directory / "state")}
        with self.assertRaisesRegex(ValueError, "synthesis_backlog"):
            build_remote_config(local, {}, "pod", "app-key", deployment=deployment)


if __name__ == "__main__":
    unittest.main()
