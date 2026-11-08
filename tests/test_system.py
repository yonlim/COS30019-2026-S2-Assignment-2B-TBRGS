"""End-to-end tests: CLI, configuration, the full pipeline and the GUI (owner: Lau Ka Meng)."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from tests.helpers import PROJECT_DIR, system, trained
from tbrgs.config import Config


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = system()

    def check_result(self, result, origin, destination, k):
        self.assertTrue(1 <= len(result.routes) <= k)
        times = [r.time_s for r in result.routes]
        self.assertEqual(times, sorted(times))
        for route in result.routes:
            self.assertEqual((route.path[0], route.path[-1]), (origin, destination))
            self.assertEqual(len(route.path), len(set(route.path)))
            self.assertAlmostEqual(route.time_s, sum(leg.time_s for leg in route.edges), places=6)
            self.assertAlmostEqual(route.distance_km, sum(leg.distance_km for leg in route.edges), places=6)
            for leg in route.edges:
                self.assertGreaterEqual(leg.speed_kmh, 32 - 1e-9)
                self.assertLessEqual(leg.speed_kmh, 60 + 1e-9)

    def test_brief_example_with_every_source(self):
        for source in self.system.available_sources():
            result = self.system.find_routes(2000, 3002, "2006-10-25 08:00", model=source, method="AS", k=5)
            self.check_result(result, 2000, 3002, 5)
            self.assertEqual(len(result.routes), 5, source)

    def test_night_is_faster_than_the_morning_peak(self):
        night = self.system.find_routes(2000, 3002, "2006-10-25 03:00", model="observed", k=1)
        peak = self.system.find_routes(2000, 3002, "2006-10-25 08:00", model="observed", k=1)
        self.assertLess(night.routes[0].time_s, peak.routes[0].time_s)

    def test_origin_is_destination(self):
        result = self.system.find_routes(2000, 2000, "2006-10-25 08:00", model="historical", k=5)
        self.assertEqual([r.path for r in result.routes], [[2000]])
        self.assertEqual(result.routes[0].time_s, 0)

    def test_dates_outside_the_data_are_mapped(self):
        result = self.system.find_routes(2000, 3002, "2026-11-09 08:07", model="historical", k=1)
        self.assertEqual(pd.Timestamp(result.time_used), pd.Timestamp("2006-10-30 08:00"))   # a Monday
        self.assertIn("outside the data", result.note)

    def test_first_hours_fall_back_to_the_historical_average(self):
        result = self.system.find_routes(2000, 3002, "2006-10-01 01:00", model="xgboost" if trained("xgboost")
                                         else "historical", k=1)
        if result.model == "xgboost":
            self.assertEqual(result.fallback_detectors, 140)

    def test_bad_requests(self):
        with self.assertRaises(ValueError):
            self.system.find_routes(9999, 3002, "2006-10-25 08:00", model="historical")
        with self.assertRaises(ValueError):
            self.system.find_routes(2000, 3002, "2006-10-25 08:00", model="historical", k=6)
        with self.assertRaises(ValueError):
            self.system.find_routes(2000, 3002, "2006-10-25 08:00", model="historical", method="ZZZ")

    def test_travel_settings_change_the_answer(self):
        normal = self.system.find_routes(2000, 3002, "2006-10-25 08:00", model="historical", k=1)
        no_delay = self.system.find_routes(2000, 3002, "2006-10-25 08:00", model="historical", k=1,
                                           travel_overrides={"intersection_delay_s": 0})
        legs = len(normal.routes[0].path) - 1
        self.assertLessEqual(no_delay.routes[0].time_s, normal.routes[0].time_s - 30 * legs + 1e-6)


class TestConfig(unittest.TestCase):
    def test_save_values_keeps_comments(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "config.ini")
            shutil.copy(os.path.join(PROJECT_DIR, "config.ini"), path)
            config = Config(path)
            config.save_values("routing", {"k": 3, "method": "CUS1"})
            again = Config(path)
            self.assertEqual(again.getint("routing", "k"), 3)
            self.assertEqual(again.get("routing", "method"), "CUS1")
            with open(path, encoding="utf-8") as file:
                self.assertIn("; One of AS, CUS1", file.read())


class TestCommandLine(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "main.py", *args], cwd=PROJECT_DIR, capture_output=True,
                              text=True, timeout=300)

    def test_route_query(self):
        out = self.run_cli("--origin", "2000", "--destination", "3002", "--model", "historical", "-k", "3")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Route 3:", out.stdout)
        self.assertIn("2000 ->", out.stdout)

    def test_unknown_site(self):
        out = self.run_cli("--origin", "1", "--destination", "3002", "--model", "historical")
        self.assertEqual(out.returncode, 1)
        self.assertIn("not a SCATS site", out.stderr)


def display_available():
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(display_available(), "no display / tkinter")
class TestGUI(unittest.TestCase):
    def test_find_routes_through_the_gui(self):
        import time
        import tkinter as tk
        from gui import TBRGSApp
        root = tk.Tk()
        try:
            app = TBRGSApp(root, system())
            app.model_label.set("Historical average")
            app.origin.set(app.site_labels[2000])
            app.destination.set(app.site_labels[3002])
            app.find_routes()
            deadline = time.time() + 60
            while app.result is None and time.time() < deadline:
                root.update()
                time.sleep(0.05)
            self.assertIsNotNone(app.result)
            self.assertEqual(len(app.route_table.get_children()), 5)
            self.assertTrue(app.leg_table.get_children())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
