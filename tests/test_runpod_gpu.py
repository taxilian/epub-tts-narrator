"""GPU selection policy prefers affordable H100s with fail-closed fallback."""

import math
import unittest

from epub_tts_narrator.renderer.runpod_gpu import (
    DEFAULT_GPU_TYPE_ID, H100_CANDIDATE_IDS, PREFERRED_H100_HOURLY_LIMIT,
    available_in, catalog_price, select_gpu,
)

FALLBACK = DEFAULT_GPU_TYPE_ID
DATACENTER = "US-FIXTURE-1"


def catalog_row(gpu_id, price=2.0, availability="HIGH", datacenters=(DATACENTER,)):
    return {"id": gpu_id, "name": gpu_id, "availability": availability,
            "price": {"secure": price},
            "dataCenters": [{"id": center, "availability": "HIGH"} for center in datacenters]}


def select(rows, **overrides):
    options = dict(fallback_id=FALLBACK, prefer_h100=True, data_center_id=DATACENTER,
                   max_hourly_price=2.60)
    options.update(overrides)
    return select_gpu(rows, **options)


class RunpodGpuPolicyTests(unittest.TestCase):
    def test_affordable_available_h100_is_preferred_over_cheaper_fallback(self):
        chosen, price = select([catalog_row(FALLBACK, 1.50), catalog_row("NVIDIA H100 PCIe", 2.19)])
        self.assertEqual(chosen, "NVIDIA H100 PCIe")
        self.assertEqual(price, 2.19)

    def test_strict_limit_cuts_h100_selection_to_the_fallback(self):
        for price, expected in ((2.59, "NVIDIA H100 80GB HBM3"), (2.60, FALLBACK), (2.61, FALLBACK)):
            with self.subTest(price=price):
                chosen, _ = select([catalog_row(FALLBACK, 2.00), catalog_row("NVIDIA H100 80GB HBM3", price)])
                self.assertEqual(chosen, expected)

    def test_configured_cap_bounds_h100_selection_below_the_preference_limit(self):
        chosen, _ = select([catalog_row(FALLBACK, 2.20), catalog_row("NVIDIA H100 NVL", 2.40)],
                           max_hourly_price=2.30)
        self.assertEqual(chosen, FALLBACK)
        chosen, _ = select([catalog_row(FALLBACK, 2.40), catalog_row("NVIDIA H100 NVL", 2.29)],
                           max_hourly_price=2.30)
        self.assertEqual(chosen, "NVIDIA H100 NVL")

    def test_cheapest_eligible_h100_wins_and_ties_break_by_catalog_id(self):
        rows = [catalog_row("NVIDIA H100 NVL", 2.10), catalog_row("NVIDIA H100 80GB HBM3", 2.10),
                catalog_row("NVIDIA H100 PCIe", 2.05)]
        self.assertEqual(select(rows)[0], "NVIDIA H100 PCIe")
        tie = [catalog_row("NVIDIA H100 NVL", 2.05), catalog_row("NVIDIA H100 80GB HBM3", 2.05)]
        self.assertEqual(select(tie)[0], "NVIDIA H100 80GB HBM3")

    def test_invalid_h100_pricing_is_skipped_without_failing(self):
        for price in (None, "2.19", True, float("nan"), float("inf"), 0, -1):
            with self.subTest(price=price):
                rows = [catalog_row("NVIDIA H100 PCIe", price), catalog_row("NVIDIA H100 NVL", 2.19)]
                self.assertEqual(select(rows)[0], "NVIDIA H100 NVL")
                self.assertEqual(select([rows[0], catalog_row(FALLBACK, 2.00)])[0], FALLBACK)

    def test_h100_without_known_stock_in_the_configured_datacenter_falls_back(self):
        cases = (
            ("overall missing", catalog_row("NVIDIA H100 PCIe", 2.19, availability=None)),
            ("overall NONE", catalog_row("NVIDIA H100 PCIe", 2.19, availability="NONE")),
            ("overall unknown", catalog_row("NVIDIA H100 PCIe", 2.19, availability="SOLD_OUT")),
            ("datacenter absent", catalog_row("NVIDIA H100 PCIe", 2.19, datacenters=("US-OTHER-1",))),
        )
        for label, row in cases:
            with self.subTest(case=label):
                self.assertEqual(select([row, catalog_row(FALLBACK, 2.00)])[0], FALLBACK)
        no_stock_center = catalog_row("NVIDIA H100 PCIe", 2.19)
        no_stock_center["dataCenters"] = [{"id": DATACENTER, "availability": "NONE"}]
        self.assertEqual(select([no_stock_center, catalog_row(FALLBACK, 2.00)])[0], FALLBACK)

    def test_unknown_gpu_ids_are_never_selected_even_when_cheap(self):
        impostor = catalog_row("NVIDIA H100", 0.75)
        rows = [catalog_row("NVIDIA A100 80GB PCIe", 0.50), catalog_row("NVIDIA H200", 1.00),
                impostor, catalog_row(FALLBACK, 2.00)]
        self.assertEqual(select(rows)[0], FALLBACK)

    def test_prefer_h100_false_pins_the_configured_fallback(self):
        rows = [catalog_row("NVIDIA H100 PCIe", 1.00), catalog_row(FALLBACK, 2.00)]
        self.assertEqual(select(rows, prefer_h100=False)[0], FALLBACK)

    def test_h100_fallback_cannot_bypass_the_preference_price_limit(self):
        with self.assertRaises(ValueError):
            select([catalog_row("NVIDIA H100 PCIe", 3.0)], fallback_id="NVIDIA H100 PCIe",
                   max_hourly_price=4.10)

    def test_explicitly_pinned_h100_uses_the_configured_price_cap(self):
        self.assertEqual(select([catalog_row("NVIDIA H100 PCIe", 3.0)], fallback_id="NVIDIA H100 PCIe",
                                prefer_h100=False, max_hourly_price=4.10), ("NVIDIA H100 PCIe", 3.0))

    def test_fallback_selection_requires_known_stock_datacenter_and_readable_price(self):
        cases = (
            ("absent row", []),
            ("datacenter missing", [catalog_row(FALLBACK, 2.00, datacenters=("US-OTHER-1",))]),
            ("overall NONE", [catalog_row(FALLBACK, 2.00, availability="NONE")]),
            ("unknown availability", [catalog_row(FALLBACK, 2.00, availability="SOLD_OUT")]),
            ("price missing", [{**catalog_row(FALLBACK), "price": {}}]),
            ("price bool", [catalog_row(FALLBACK, True)]),
            ("price NaN", [catalog_row(FALLBACK, float("nan"))]),
            ("over cap", [catalog_row(FALLBACK, 2.61)]),
        )
        for label, rows in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    select(rows)

    def test_over_cap_fallback_failure_mentions_price_and_creates_nothing(self):
        with self.assertRaisesRegex(ValueError, "price"):
            select([catalog_row(FALLBACK, 100)])

    def test_missing_datacenter_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "data_center_id"):
            select([catalog_row(FALLBACK)], data_center_id="")

    def test_price_and_availability_helpers_reject_malformed_catalog_data(self):
        self.assertEqual(catalog_price(catalog_row("gpu", 2)), 2)
        for row in ({"price": {"secure": True}}, {"price": {"secure": "2.0"}},
                    {"price": {"secure": float("inf")}}, {"price": []}, {}):
            self.assertIsNone(catalog_price(row))
        self.assertTrue(available_in(catalog_row("gpu"), DATACENTER))
        for row in ({"availability": "HIGH"}, catalog_row("gpu", datacenters=("US-OTHER-1",)),
                    {"availability": "HIGH", "dataCenters": ["broken"]}):
            self.assertFalse(available_in(row, DATACENTER))

    def test_policy_constants_match_the_public_runpod_catalog_and_split_roles(self):
        self.assertEqual(H100_CANDIDATE_IDS,
                         {"NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"})
        self.assertNotIn(DEFAULT_GPU_TYPE_ID, H100_CANDIDATE_IDS)
        self.assertNotEqual(DEFAULT_GPU_TYPE_ID, "NVIDIA RTX 6000 Ada Generation")
        self.assertEqual(PREFERRED_H100_HOURLY_LIMIT, 2.60)


if __name__ == "__main__":
    unittest.main()
