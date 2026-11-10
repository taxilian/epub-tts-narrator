"""GPU selection: any NVIDIA GPU with enough memory, measured cost first, fail closed."""

import unittest

from epub_tts_narrator.renderer import runpod_gpu
from epub_tts_narrator.renderer.runpod_gpu import (
    MEASURED_REALTIME, MIN_GPU_MEMORY_GB, catalog_memory, catalog_price, eligible, stock_in,
)

MEASURED = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
DATACENTER = "US-CO-1"


def catalog_row(gpu_id, price=2.0, *, memory=96, manufacturer="NVIDIA", availability="HIGH",
                datacenters=(DATACENTER,), community=None, name=None):
    prices = {"secure": price}
    if community is not None:
        prices["community"] = community
    return {"id": gpu_id, "name": name or gpu_id, "manufacturer": manufacturer, "memory": memory,
            "availability": availability, "price": prices,
            "dataCenters": [{"id": center, "availability": "HIGH"} for center in datacenters]}


def choose(rows, **overrides):
    options = dict(max_hourly_price=2.60)
    options.update(overrides)
    return runpod_gpu.select_placement(rows, **options)


class EligibilityTests(unittest.TestCase):
    def test_any_nvidia_gpu_with_enough_memory_is_eligible(self):
        for gpu_id, memory in (("NVIDIA A100 80GB PCIe", 80), ("NVIDIA H200 NVL", 143), (MEASURED, 96)):
            with self.subTest(gpu=gpu_id):
                self.assertTrue(eligible(catalog_row(gpu_id, memory=memory)))

    def test_sxm_gpus_are_never_eligible(self):
        """Runpod serves SXM Pods without a host-local Pod disk."""
        cases = (
            ("id marked", catalog_row("NVIDIA A100-SXM4-80GB", memory=80)),
            ("name marked", catalog_row("NVIDIA H100 80GB HBM3", memory=80, name="H100 SXM")),
            ("sxm6 id", catalog_row("NVIDIA B300 SXM6 AC", memory=288)),
        )
        for label, row in cases:
            with self.subTest(case=label):
                self.assertFalse(eligible(row))

    def test_a_pinned_sxm_gpu_reports_the_host_local_disk_reason(self):
        rows = [catalog_row("NVIDIA A100-SXM4-80GB", memory=80)]
        with self.assertRaises(ValueError) as raised:
            choose(rows, gpu_type_id="NVIDIA A100-SXM4-80GB")
        self.assertIn("host-local Pod disk", str(raised.exception))

    def test_too_little_memory_other_vendors_and_unknown_data_are_never_eligible(self):
        cases = (
            ("48 GB card", catalog_row("NVIDIA L40S", memory=48)),
            ("48 GB MIG slice", catalog_row("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 48GB", memory=48)),
            ("AMD", catalog_row("AMD Instinct MI300X OAM", memory=192, manufacturer="AMD")),
            ("unknown vendor", catalog_row("unknown", memory=192, manufacturer="UNKNOWN")),
            ("no memory", {**catalog_row("NVIDIA H200"), "memory": None}),
            ("memory text", catalog_row("NVIDIA H200", memory="141")),
            ("memory bool", catalog_row("NVIDIA H200", memory=True)),
            ("memory NaN", catalog_row("NVIDIA H200", memory=float("nan"))),
            ("no id", {**catalog_row("NVIDIA H200"), "id": None}),
            ("not a row", "NVIDIA H200"),
        )
        for label, row in cases:
            with self.subTest(case=label):
                self.assertFalse(eligible(row))

    def test_the_memory_floor_is_configurable_and_inclusive(self):
        row = catalog_row("NVIDIA A100 80GB PCIe", memory=80)
        self.assertTrue(eligible(row, min_memory_gb=80))
        self.assertFalse(eligible(row, min_memory_gb=81))
        self.assertEqual(MIN_GPU_MEMORY_GB, 56)

    def test_a_pinned_gpu_admits_only_that_exact_id(self):
        self.assertTrue(eligible(catalog_row(MEASURED), gpu_type_id=MEASURED))
        self.assertFalse(eligible(catalog_row("NVIDIA H200", memory=141), gpu_type_id=MEASURED))
        # A pin never bypasses the memory floor.
        self.assertFalse(eligible(catalog_row(MEASURED, memory=48), gpu_type_id=MEASURED))

    def test_catalog_helpers_reject_malformed_data(self):
        self.assertEqual(catalog_price(catalog_row("gpu", 2)), 2)
        for row in ({"price": {"secure": True}}, {"price": {"secure": "2.0"}},
                    {"price": {"secure": float("inf")}}, {"price": []}, {}, None):
            self.assertIsNone(catalog_price(row))
        self.assertEqual(catalog_memory(catalog_row("gpu", memory=80)), 80)
        for row in ({"memory": 0}, {"memory": -1}, {"memory": float("inf")}, {}, None):
            self.assertIsNone(catalog_memory(row))
        self.assertEqual(stock_in(catalog_row("gpu"), DATACENTER), 0)
        for row in ({"availability": "HIGH"}, catalog_row("gpu", datacenters=("US-OTHER-1",)),
                    {"availability": "HIGH", "dataCenters": ["broken"]},
                    catalog_row("gpu", availability="NONE"), catalog_row("gpu", availability=None)):
            self.assertIsNone(stock_in(row, DATACENTER))


class RankingTests(unittest.TestCase):
    def test_a_measured_gpu_beats_cheaper_unmeasured_ones(self):
        # The catalog says nothing about speed, so a measured cost wins over a guess.
        rows = [catalog_row("NVIDIA A100 80GB PCIe", 1.39, memory=80), catalog_row(MEASURED, 2.49)]
        self.assertEqual(choose(rows).gpu_type_id, MEASURED)

    def test_measured_gpus_rank_by_cost_per_generated_audio_hour(self):
        faster = "NVIDIA FIXTURE FAST"
        original = dict(MEASURED_REALTIME)
        self.addCleanup(lambda: (MEASURED_REALTIME.clear(), MEASURED_REALTIME.update(original)))
        MEASURED_REALTIME[faster] = 4.2  # twice as fast at under twice the price
        rows = [catalog_row(MEASURED, 1.20), catalog_row(faster, 2.30)]
        self.assertEqual(choose(rows).gpu_type_id, faster)
        rows = [catalog_row(MEASURED, 1.00), catalog_row(faster, 2.30)]
        self.assertEqual(choose(rows).gpu_type_id, MEASURED)

    def test_without_a_measured_gpu_the_cheapest_eligible_one_wins(self):
        rows = [catalog_row("NVIDIA H200", 2.40, memory=141), catalog_row("NVIDIA A100 80GB PCIe", 1.39, memory=80),
                catalog_row("NVIDIA L40S", 0.79, memory=48)]
        result = choose(rows)
        self.assertEqual((result.gpu_type_id, result.hourly_price, result.memory_gb),
                         ("NVIDIA A100 80GB PCIe", 1.39, 80.0))

    def test_price_cap_applies_to_every_gpu(self):
        rows = [catalog_row(MEASURED, 2.61), catalog_row("NVIDIA H200", 2.70, memory=141)]
        with self.assertRaisesRegex(ValueError, "max_hourly_price"):
            choose(rows)
        self.assertEqual(choose(rows, max_hourly_price=2.65).gpu_type_id, MEASURED)

    def test_unusable_rows_are_skipped_without_failing_the_choice(self):
        bad_price = catalog_row(MEASURED, float("nan"))
        no_stock = catalog_row("NVIDIA H200", 1.00, memory=141, availability="NONE")
        elsewhere = catalog_row("NVIDIA H100 PCIe", 1.00, memory=80, datacenters=("CA-MTL-3",))
        fine = catalog_row("NVIDIA A100 80GB PCIe", 1.79, memory=80)
        self.assertEqual(choose([bad_price, no_stock, elsewhere, fine]).gpu_type_id, "NVIDIA A100 80GB PCIe")

    def test_pinned_gpu_is_the_only_candidate(self):
        rows = [catalog_row(MEASURED, 2.49), catalog_row("NVIDIA A100 80GB PCIe", 1.39, memory=80)]
        self.assertEqual(choose(rows, gpu_type_id="NVIDIA A100 80GB PCIe").gpu_type_id, "NVIDIA A100 80GB PCIe")
        with self.assertRaisesRegex(ValueError, "GPU NVIDIA H200 with at least 56 GB"):
            choose(rows, gpu_type_id="NVIDIA H200")


class USPlacementTests(unittest.TestCase):
    def test_configured_location_is_only_a_hint(self):
        row = catalog_row(MEASURED, datacenters=("US-NC-2", "US-CO-1"))
        self.assertEqual(choose([row], preferred_data_center="US-NC-2").data_center_id, "US-NC-2")
        row["dataCenters"][0]["availability"] = "LOW"
        self.assertEqual(choose([row], preferred_data_center="US-NC-2").data_center_id, "US-CO-1")
        self.assertEqual(choose([row]).data_center_id, "US-CO-1")

    def test_cost_outranks_stock_and_location(self):
        rows = [catalog_row("NVIDIA A100 80GB PCIe", 1.79, memory=80, datacenters=("US-NC-2",)),
                catalog_row("NVIDIA H100 PCIe", 1.99, memory=80, datacenters=("US-CO-1",))]
        rows[0]["dataCenters"][0]["availability"] = "LOW"
        result = choose(rows, preferred_data_center="US-CO-1")
        self.assertEqual((result.gpu_type_id, result.data_center_id), ("NVIDIA A100 80GB PCIe", "US-NC-2"))

    def test_non_us_locations_are_never_selected(self):
        rows = [catalog_row("NVIDIA H200", 1.00, memory=141, datacenters=("CA-MTL-3", "EU-RO-1")),
                catalog_row(MEASURED, 2.49, datacenters=("US-CO-1",))]
        self.assertEqual(choose(rows, preferred_data_center="CA-MTL-3").data_center_id, "US-CO-1")

    def test_no_eligible_us_placement_fails_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "US datacenter on Runpod Secure Cloud"):
            choose([catalog_row(MEASURED, 2.49, datacenters=("CA-MTL-3",))])
        with self.assertRaisesRegex(ValueError, "invalid GPU catalog"):
            choose({"gpus": []})

    def test_invalid_stock_entries_do_not_mask_valid_stock_for_the_same_location(self):
        row = catalog_row(MEASURED, 2.19)
        row["dataCenters"].insert(0, {"id": DATACENTER, "availability": []})
        self.assertEqual(choose([row]).gpu_type_id, MEASURED)


class CloudPriceTests(unittest.TestCase):
    def test_each_cloud_is_priced_from_its_own_catalog_key(self):
        row = catalog_row(MEASURED, 2.0, community=1.25)
        self.assertEqual(catalog_price(row), 2.0)
        self.assertEqual(catalog_price(row, "COMMUNITY"), 1.25)
        result = choose([row], cloud="COMMUNITY")
        self.assertEqual((result.cloud, result.hourly_price), ("COMMUNITY", 1.25))

    def test_community_price_decides_the_ranking_and_cap(self):
        rows = [catalog_row("NVIDIA A100 80GB PCIe", 1.79, memory=80, community=1.39),
                catalog_row("NVIDIA H100 PCIe", 2.89, memory=80, community=1.19)]
        self.assertEqual(choose(rows).gpu_type_id, "NVIDIA A100 80GB PCIe")
        self.assertEqual(choose(rows, cloud="COMMUNITY").gpu_type_id, "NVIDIA H100 PCIe")
        with self.assertRaisesRegex(ValueError, "Community Cloud"):
            choose([catalog_row(MEASURED, 2.0, community=3.0)], cloud="COMMUNITY")

    def test_missing_community_price_is_never_selected(self):
        with self.assertRaises(ValueError):
            choose([catalog_row(MEASURED)], cloud="COMMUNITY")

    def test_unknown_cloud_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown Runpod cloud"):
            choose([], cloud="SPOT")


if __name__ == "__main__":
    unittest.main()
