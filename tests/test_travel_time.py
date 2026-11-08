"""Tests for Task 3, travel-time estimation and the road network (owner: Jonathan Yu)."""

import os
import tempfile
import unittest

from tests.helpers import config, prepared
from tbrgs.network import RoadNetwork, angle_difference, normalise_road
from tbrgs.travel_time import TravelTimeModel, hourly_flow


class TestTravelTime(unittest.TestCase):
    def setUp(self):
        self.model = TravelTimeModel()

    def test_coefficients_match_the_conversion_note(self):
        self.assertAlmostEqual(self.model.a, -1.46484375)
        self.assertAlmostEqual(self.model.b, 93.75)

    def test_key_points_of_the_curve(self):
        self.assertEqual(self.model.speed_from_flow(0), 60)
        self.assertAlmostEqual(self.model.speed_from_flow(351.5625), 60, places=3)   # meets the speed limit
        self.assertAlmostEqual(self.model.speed_from_flow(1500), 32, places=6)        # capacity
        self.assertEqual(self.model.speed_from_flow(2500), 32)                        # over capacity: capped

    def test_speed_falls_as_flow_rises(self):
        speeds = [self.model.speed_from_flow(f) for f in range(352, 1500, 50)]
        self.assertTrue(all(a > b for a, b in zip(speeds, speeds[1:])))

    def test_speed_and_flow_are_inverse(self):
        for flow in (400, 800, 1200, 1499):
            self.assertAlmostEqual(self.model.flow_from_speed(self.model.speed_from_flow(flow)), flow, places=6)

    def test_link_time(self):
        self.assertAlmostEqual(self.model.link_time_s(1.0, 0), 60 + 30)          # 1 km at 60 km/h + delay
        self.assertAlmostEqual(self.model.link_time_s(1.0, 1500), 3600 / 32 + 30)
        self.assertAlmostEqual(self.model.link_time_s(0.0, 800), 30)

    def test_bad_values(self):
        with self.assertRaises(ValueError):
            TravelTimeModel(capacity_speed_kmh=70)
        with self.assertRaises(ValueError):
            self.model.link_time_s(-1, 100)
        self.assertEqual(self.model.speed_from_flow(float("nan")), 60)

    def test_hourly_flow(self):
        self.assertEqual(hourly_flow(100), 400)


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepared().data
        cls.network = RoadNetwork(cls.data, config().path_of("data", "manual_links_file"))

    def test_every_site_reaches_every_other(self):
        self.assertEqual(len(self.network.sites), 40)
        self.assertTrue(self.network.is_strongly_connected())

    def test_edges_are_sensible(self):
        for (start, end), edge in self.network.edges.items():
            self.assertIn((end, start), self.network.edges)            # two-way
            self.assertGreater(edge.distance_km, 0.1)
            self.assertLess(edge.distance_km, 4.0)
            self.assertTrue(edge.detectors)
            for detector in edge.detectors:
                self.assertIn(detector, self.data.flows.columns)

    def test_known_links(self):
        for a, b in ((2000, 3685), (3001, 3002), (4034, 4035), (3001, 4821), (4263, 4812)):
            self.assertIn((a, b), self.network.edges)
        self.assertNotIn((2000, 3002), self.network.edges)

    def test_flow_comes_from_the_approach_facing_the_previous_site(self):
        edge = self.network.edges[(2000, 3685)]   # driving south on Warrigal Rd into site 3685
        self.assertEqual(edge.rule, "approach")
        self.assertEqual([self.data.detectors.loc[d, "location"] for d in edge.detectors],
                         ["WARRIGAL_RD N of HIGHBURY_RD"])

    def test_two_detectors_on_one_approach_are_added(self):
        edge = self.network.edges[(4321, 4335)]
        self.assertEqual(sorted(edge.detectors), ["4335-2", "4335-6"])
        self.assertEqual(edge.combine, "sum")

    def test_origin_rule_also_builds_a_connected_network(self):
        network = RoadNetwork(self.data, config().path_of("data", "manual_links_file"), flow_site="origin")
        self.assertTrue(network.is_strongly_connected())
        self.assertEqual(set(network.edges), set(self.network.edges))

    def test_road_aliases(self):
        self.assertEqual(normalise_road("MAROONDAH_HWY"), "WHITEHORSE_RD")
        self.assertEqual(normalise_road("HIGH STREET_RD"), "HIGH_STREET_RD")
        self.assertEqual(angle_difference(350, 10), 20)

    def test_manual_link_with_unknown_site(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "links.csv")
            with open(path, "w", encoding="utf-8") as file:
                file.write("site_a,site_b,via,approach_at_a,approach_at_b,reason\n1,2000,X,,,test\n")
            with self.assertRaises(ValueError):
                RoadNetwork(self.data, path)


if __name__ == "__main__":
    unittest.main()
