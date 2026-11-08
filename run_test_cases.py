"""Run the documented system test cases and record the results.

Usage (from this folder):
    python run_test_cases.py

Each case is a route query (or a deliberately bad one) through the full
system: flow prediction -> travel times -> top-k routes. Its result is
checked automatically and written to results/test_cases_results.csv and
tests/TEST_CASES.txt (which is the table quoted in the report). The unit
tests (python -m unittest discover -s tests -v) check the modules one by one.
"""

from __future__ import annotations

import os
import sys
import textwrap

import pandas as pd

from tbrgs.config import load_config
from tbrgs.routing import OPTIMAL_METHODS
from tbrgs.system import TBRGS

WEDNESDAY_PEAK = "2006-10-25 08:00"


def best_model(system):
    available = system.available_sources()
    return "lstm" if "lstm" in available else available[0]


def routes_text(result, limit=5):
    return "; ".join(f"R{r.rank} {r.minutes:.1f} min via {'-'.join(map(str, r.path))}"
                     for r in result.routes[:limit])


def well_formed(result, origin, destination):
    times = [r.time_s for r in result.routes]
    return (bool(result.routes) and times == sorted(times)
            and all(r.path[0] == origin and r.path[-1] == destination and len(set(r.path)) == len(r.path)
                    for r in result.routes))


def cases(system):
    """Yield (id, owner, scenario, inputs, expected, function returning (passed, actual))."""
    lstm = best_model(system)
    find = system.find_routes

    def brief(model):
        def run():
            r = find(2000, 3002, WEDNESDAY_PEAK, model=model, method="AS", k=5)
            return well_formed(r, 2000, 3002) and len(r.routes) == 5, routes_text(r, 2)
        return run

    def night_vs_peak():
        night = find(2000, 3002, "2006-10-25 03:00", model=lstm, k=1)
        peak = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1)
        return (night.routes[0].time_s < peak.routes[0].time_s,
                f"03:00 {night.routes[0].minutes:.1f} min vs 08:00 {peak.routes[0].minutes:.1f} min")

    def reverse():
        r = find(3002, 2000, "2006-10-25 17:30", model=lstm, k=5)
        return well_formed(r, 3002, 2000) and len(r.routes) == 5, routes_text(r, 2)

    def same_site():
        r = find(2000, 2000, WEDNESDAY_PEAK, model=lstm, k=5)
        return [x.path for x in r.routes] == [[2000]] and r.routes[0].time_s == 0, routes_text(r)

    def adjacent():
        r = find(2000, 3685, WEDNESDAY_PEAK, model=lstm, k=3)
        leg = r.routes[0].edges[0]
        expected = leg.distance_km / leg.speed_kmh * 3600 + 30
        return (r.routes[0].path == [2000, 3685] and abs(r.routes[0].time_s - expected) < 1e-6,
                f"R1 {r.routes[0].path}, {leg.distance_km:.3f} km at {leg.speed_kmh:.1f} km/h + 30 s = "
                f"{r.routes[0].time_s:.1f} s; {len(r.routes)} routes")

    def manual_link_site():
        r = find(4812, 3002, WEDNESDAY_PEAK, model=lstm, k=3)
        return well_formed(r, 4812, 3002) and r.routes[0].path[1] in (4263, 4821), routes_text(r, 2)

    def long_trip():
        r = find(2820, 970, WEDNESDAY_PEAK, model=lstm, k=5)
        return well_formed(r, 2820, 970) and len(r.routes) == 5, routes_text(r, 2)

    def weekend():
        weekday = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1)
        saturday = find(2000, 3002, "2006-10-28 08:00", model=lstm, k=1)
        return (saturday.routes[0].time_s <= weekday.routes[0].time_s,
                f"Sat {saturday.routes[0].minutes:.1f} min vs Wed {weekday.routes[0].minutes:.1f} min")

    def outside():
        r = find(2000, 3002, "2026-11-11 08:00", model=lstm, k=1)
        return (pd.Timestamp(r.time_used) == pd.Timestamp(WEDNESDAY_PEAK) and "outside" in r.note,
                f"used {r.time_used}; note: {r.note[:60]}...")

    def k_one():
        r = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1)
        return len(r.routes) == 1, routes_text(r)

    def method_case(method):
        def run():
            best = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, method="AS", k=1).routes[0]
            r = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, method=method, k=1)
            route = r.routes[0]
            if method in OPTIMAL_METHODS:
                ok = abs(route.time_s - best.time_s) < 1e-6
            elif method in ("BFS", "CUS2"):
                ok = route.time_s >= best.time_s - 1e-6 and route.intersections <= best.intersections
            else:
                ok = route.time_s >= best.time_s - 1e-6
            return ok and well_formed(r, 2000, 3002), (
                f"{route.minutes:.1f} min, {route.intersections} intersections, {r.nodes_created} nodes "
                f"(A*: {best.minutes:.1f} min, {best.intersections} intersections)")
        return run

    def bad_site():
        try:
            find(9999, 3002, WEDNESDAY_PEAK, model=lstm)
        except ValueError as error:
            return "not a SCATS site" in str(error), f"ValueError: {str(error)[:50]}..."
        return False, "no error"

    def unaligned():
        r = find(2000, 3002, "2006-10-25 08:07", model=lstm, k=1)
        return pd.Timestamp(r.time_used) == pd.Timestamp(WEDNESDAY_PEAK), f"used {r.time_used}"

    def first_hours():
        r = find(2000, 3002, "2006-10-01 01:00", model=lstm, k=1)
        expected = 140 if lstm not in ("historical", "observed") else 0
        return r.fallback_detectors == expected and bool(r.routes), (
            f"{r.fallback_detectors} detectors used the historical average; {routes_text(r)}")

    def missing_data():
        # 3001-6 (CHURCH_ST SW of BARKERS_RD) has only 2 days of data.
        r = find(3001, 3002, "2006-10-26 08:00", model=lstm, k=1)
        return r.fallback_detectors >= 1 and bool(r.routes), (
            f"{r.fallback_detectors} detectors without recent history fell back; {routes_text(r)}")

    def origin_rule():
        a = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1, flow_site="destination").routes[0]
        b = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1, flow_site="origin").routes[0]
        return bool(b.path), f"destination rule {a.minutes:.1f} min, origin rule {b.minutes:.1f} min"

    def observed():
        r = find(2000, 3002, WEDNESDAY_PEAK, model="observed", k=5)
        return well_formed(r, 2000, 3002), routes_text(r, 2)

    def no_delay():
        normal = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1).routes[0]
        quick = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1,
                     travel_overrides={"intersection_delay_s": 0}).routes[0]
        return (quick.time_s <= normal.time_s - 30 * normal.intersections + 1e-6,
                f"{normal.minutes:.1f} min with 30 s delays, {quick.minutes:.1f} min without")

    def slower_capacity():
        normal = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1).routes[0]
        jam = find(2000, 3002, WEDNESDAY_PEAK, model=lstm, k=1,
                   travel_overrides={"capacity_flow_vph": 900, "capacity_speed_kmh": 25}).routes[0]
        return jam.time_s > normal.time_s, f"capacity 1500 veh/h: {normal.minutes:.1f} min; 900 veh/h at 25 km/h: {jam.minutes:.1f} min"

    yield "TC01", "Lim", "Brief example, LSTM flows", "2000 -> 3002, Wed 25 Oct 08:00, AS, k=5", \
        "5 loopless routes, fastest first", brief(lstm)
    yield "TC02", "Jonathan", "Brief example, GRU flows", "as TC01 with GRU", \
        "5 loopless routes, fastest first", brief("gru" if "gru" in system.available_sources() else lstm)
    yield "TC03", "Lau", "Brief example, XGBoost flows", "as TC01 with XGBoost", \
        "5 loopless routes, fastest first", brief("xgboost" if "xgboost" in system.available_sources() else lstm)
    yield "TC04", "Lim", "Night is faster than the morning peak", "2000 -> 3002 at 03:00 and 08:00", \
        "03:00 fastest time < 08:00 fastest time", night_vs_peak
    yield "TC05", "Lau", "Reverse trip in the evening peak", "3002 -> 2000, Wed 17:30", \
        "5 loopless routes", reverse
    yield "TC06", "Lau", "Origin equals destination", "2000 -> 2000", "one route [2000], 0 s", same_site
    yield "TC07", "Jonathan", "Neighbouring sites", "2000 -> 3685", \
        "route 1 is the direct link; time = distance/speed + 30 s", adjacent
    yield "TC08", "Jonathan", "Site joined by a manual link", "4812 -> 3002", \
        "routes leave 4812 through 4263 or 4821", manual_link_site
    yield "TC09", "Lau", "Long trip across the network", "2820 -> 970", "5 loopless routes", long_trip
    yield "TC10", "Lim", "Weekend morning", "2000 -> 3002, Sat 28 Oct 08:00 vs Wed", \
        "Saturday no slower than Wednesday", weekend
    yield "TC11", "Lau", "Date outside the dataset", "2026-11-11 08:00 (a Wednesday)", \
        "mapped to Wed 25 Oct 2006 08:00 with a note", outside
    yield "TC12", "Lau", "Only one route requested", "k=1", "exactly one route", k_one
    yield "TC13", "Lau", "Uniform-cost search (CUS1)", "method CUS1", "same time as A*", method_case("CUS1")
    yield "TC14", "Lau", "Breadth-first search", "method BFS", \
        "fewest intersections; not faster than A*", method_case("BFS")
    yield "TC15", "Lau", "Depth-first search", "method DFS", "a valid route; not faster than A*", method_case("DFS")
    yield "TC16", "Lau", "Greedy best-first search", "method GBFS", "a valid route; not faster than A*", \
        method_case("GBFS")
    yield "TC17", "Lau", "IDA* fewest moves (CUS2)", "method CUS2", \
        "fewest intersections; not faster than A*", method_case("CUS2")
    yield "TC18", "Lau", "Unknown site", "origin 9999", "clear error message", bad_site
    yield "TC19", "Lim", "Time not on a 15-minute boundary", "08:07", "rounded down to 08:00", unaligned
    yield "TC20", "Lim", "No history yet (start of the data)", "Sun 1 Oct 01:00", \
        "all 140 detectors use the historical average; routes still found", first_hours
    yield "TC21", "Lim", "Detector with missing days", "3001 -> 3002, Thu 26 Oct 08:00", \
        "detector 3001-6 falls back; routes still found", missing_data
    yield "TC22", "Jonathan", "Flow from the origin site instead", "flow_site=origin", \
        "routes found with the other flow rule", origin_rule
    yield "TC23", "Lim", "Observed flows instead of a model", "model observed", "5 loopless routes", observed
    yield "TC24", "Jonathan", "No intersection delay", "intersection_delay_s=0", \
        "at least 30 s per intersection faster", no_delay
    yield "TC25", "Jonathan", "Lower road capacity", "capacity 900 veh/h at 25 km/h", "slower than normal", \
        slower_capacity


def main() -> int:
    config = load_config()
    system = TBRGS(config)
    rows = []
    for case_id, owner, scenario, inputs, expected, run in cases(system):
        try:
            passed, actual = run()
        except Exception as error:  # a crash is a failed case, not a stopped run
            passed, actual = False, f"{type(error).__name__}: {error}"
        rows.append({"id": case_id, "owner": owner, "scenario": scenario, "input": inputs,
                     "expected": expected, "actual": actual, "result": "PASS" if passed else "FAIL"})
        print(f"{case_id} {'PASS' if passed else 'FAIL'}  {scenario}: {actual}")
    table = pd.DataFrame(rows)
    results_dir = config.path_of("paths", "results_dir")
    os.makedirs(results_dir, exist_ok=True)
    table.to_csv(os.path.join(results_dir, "test_cases_results.csv"), index=False)

    model = best_model(system)
    lines = ["TBRGS system test cases (generated by run_test_cases.py)", "=" * 56, "",
             f"Flows from: {model} unless the case says otherwise. Search: A* (AS) unless stated.",
             f"Passed {sum(r['result'] == 'PASS' for r in rows)} of {len(rows)}.", ""]
    for row in rows:
        lines.append(f"{row['id']}  [{row['result']}]  {row['scenario']}  (owner: {row['owner']})")
        for label in ("input", "expected", "actual"):
            wrapped = textwrap.wrap(str(row[label]), 92)
            lines.append(f"      {label:<9}{wrapped[0] if wrapped else ''}")
            lines.extend(f"               {line}" for line in wrapped[1:])
        lines.append("")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "TEST_CASES.txt"), "w",
              encoding="utf-8") as file:
        file.write("\n".join(lines))
    failed = [r["id"] for r in rows if r["result"] != "PASS"]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
