"""Tests for Task 4, top-k routing with the Part A methods (owner: Lau Ka Meng)."""

import itertools
import random
import unittest

from tests.helpers import system
from tbrgs.routing import METHODS, OPTIMAL_METHODS, build_search_problem, inconsistent_edges, path_cost, top_k_routes

# The directed example graph from Yen's algorithm on Wikipedia: C=1, D=2, E=3, F=4, G=5, H=6.
# Its three cheapest paths from C to H are C-E-F-H (5), C-E-G-H (7) and C-D-F-H (8).
YEN_EDGES = {(1, 2): 3, (1, 3): 2, (2, 4): 4, (3, 2): 1, (3, 4): 2, (3, 5): 3, (4, 5): 2, (4, 6): 1, (5, 6): 2}


def yen_problem():
    positions = {n: (0.0, 0.0) for n in range(1, 7)}   # heuristic 0: A* behaves like UCS
    return build_search_problem(positions, YEN_EDGES, 60.0)


class TestYen(unittest.TestCase):
    def test_textbook_example(self):
        nodes, edges = yen_problem()
        for method in ("AS", "CUS1"):
            found, stats = top_k_routes(nodes, edges, 1, 6, 3, method)
            self.assertEqual([(c, p) for c, p in found],
                             [(5, [1, 3, 4, 6]), (7, [1, 3, 5, 6]), (8, [1, 2, 4, 6])], method)
            self.assertGreater(stats.searches, 1)

    def test_asking_for_more_routes_than_exist(self):
        nodes, edges = yen_problem()
        found, _ = top_k_routes(nodes, edges, 1, 6, 50, "AS")
        # Every loopless C -> H path in this graph: 7 of them.
        self.assertEqual(len(found), 7)
        self.assertEqual(len({tuple(p) for _, p in found}), 7)

    def test_origin_equals_destination(self):
        nodes, edges = yen_problem()
        found, _ = top_k_routes(nodes, edges, 3, 3, 5, "AS")
        self.assertEqual(found, [(0, [3])])

    def test_no_route(self):
        nodes, edges = yen_problem()
        found, _ = top_k_routes(nodes, edges, 6, 1, 5, "AS")   # H has no outgoing edges
        self.assertEqual(found, [])

    def test_bad_input(self):
        nodes, edges = yen_problem()
        with self.assertRaises(ValueError):
            top_k_routes(nodes, edges, 1, 6, 3, "XYZ")
        with self.assertRaises(ValueError):
            top_k_routes(nodes, edges, 1, 99, 3, "AS")
        with self.assertRaises(ValueError):
            top_k_routes(nodes, edges, 1, 6, 0, "AS")
        with self.assertRaises(ValueError):
            build_search_problem({1: (0, 0), 2: (1, 0)}, {(1, 2): -5}, 60)

    def test_every_method_returns_loopless_sorted_routes(self):
        nodes, edges = yen_problem()
        for method in METHODS:
            found, _ = top_k_routes(nodes, edges, 1, 6, 4, method)
            self.assertTrue(found, method)
            costs = [c for c, _ in found]
            self.assertEqual(costs, sorted(costs), method)
            for cost, path in found:
                self.assertEqual(len(path), len(set(path)), method)           # no loops
                self.assertEqual((path[0], path[-1]), (1, 6))
                self.assertAlmostEqual(cost, path_cost(edges, path))


class TestBoroondara(unittest.TestCase):
    """The ported Part A methods on the real road network, with predicted travel times."""

    @classmethod
    def setUpClass(cls):
        cls.system = system()
        travel = cls.system.travel_time_model()
        estimates, _ = cls.system.edge_estimates("historical", "2006-10-25 08:00", travel)
        network = cls.system.network()
        positions = {s: network.position(s) for s in network.sites.index}
        cls.nodes, cls.edges = build_search_problem(positions, {k: e.time_s for k, e in estimates.items()},
                                                    travel.free_flow_seconds_per_km())
        cls.sites = list(network.sites.index)

    def test_heuristic_is_consistent(self):
        # Part A's own check: no edge is cheaper than the straight-line time at the speed limit.
        self.assertEqual(inconsistent_edges(self.nodes, self.edges, [3002]), [])
        self.assertEqual(inconsistent_edges(self.nodes, self.edges, [970]), [])

    def test_astar_matches_uniform_cost_search(self):
        random.seed(3)
        astar_nodes = ucs_nodes = 0
        for a, b in random.sample(list(itertools.permutations(self.sites, 2)), 80):
            (c1, _), = top_k_routes(self.nodes, self.edges, a, b, 1, "AS")[0]
            (c2, _), = top_k_routes(self.nodes, self.edges, a, b, 1, "CUS1")[0]
            self.assertAlmostEqual(c1, c2, places=6)
            astar_nodes += top_k_routes(self.nodes, self.edges, a, b, 1, "AS")[1].nodes_created
            ucs_nodes += top_k_routes(self.nodes, self.edges, a, b, 1, "CUS1")[1].nodes_created
        self.assertLess(astar_nodes, ucs_nodes)   # the heuristic saves work

    def test_other_methods_never_beat_the_optimum(self):
        for a, b in ((2000, 3002), (2820, 970), (4812, 2827)):
            best = top_k_routes(self.nodes, self.edges, a, b, 1, "CUS1")[0][0][0]
            for method in set(METHODS) - OPTIMAL_METHODS:
                cost = top_k_routes(self.nodes, self.edges, a, b, 1, method)[0][0][0]
                self.assertGreaterEqual(cost + 1e-6, best, method)

    def test_five_routes_between_far_apart_sites(self):
        found, _ = top_k_routes(self.nodes, self.edges, 2000, 3002, 5, "AS")
        self.assertEqual(len(found), 5)
        self.assertEqual(len({tuple(p) for _, p in found}), 5)


if __name__ == "__main__":
    unittest.main()
