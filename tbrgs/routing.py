"""Task 4: finding the top-k routes with the search methods from Part A.

Owner: Lau Ka Meng (105814507)

The six Part A methods are reused unchanged: the files in part_a/ are
copies of graph.py, uninformed.py, informed.py and custom.py from the team's
Assignment 2A repository. They are imported by adding part_a/ to Python's
search path, because they import each other by plain module name.

Porting the Boroondara network to Part A's format
------------------------------------------------
Part A methods take (nodes, edges, origin, destinations) where
    nodes = {node: (x, y)}  and  edges = {node: [(neighbour, cost), ...]}.
Here a node is a SCATS site and an edge cost is the predicted travel time in
seconds. A* and GBFS use the straight-line distance between node coordinates
as their heuristic, so the coordinates are given in "seconds at the speed
limit": x = x_km * (3600 / 60). The heuristic then equals the time to drive
the straight line at 60 km/h with no intersection delays. No link can be
driven faster than that, and link lengths are straight lines between the
same coordinates, so the heuristic never overestimates and is consistent,
and A* returns the fastest route (tests/test_routing.py checks this with
Part A's own inconsistent_edges()).

Top-k routes: Yen's algorithm
------------------------------
Yen (1971) finds the k fastest loopless routes. The fastest route comes from
one search. Each later route is found by taking every node of the previous
route in turn as a "spur" node, keeping the part of the route before it
(the root), blocking the next edge of every route already found with that
same root, blocking the root's nodes so the new route cannot loop back, and
searching again from the spur node. The cheapest new route found this way
is the next route. Every search is one call of a Part A method, so the
routes are only guaranteed to be the k fastest with an optimal method
(AS or CUS1); with DFS, BFS, GBFS or CUS2 the same procedure still returns
up to k different loopless routes, sorted by travel time.
"""

from __future__ import annotations

import heapq
import os
import sys
import time
from dataclasses import dataclass, field

_PART_A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part_a")
if _PART_A not in sys.path:
    sys.path.insert(0, _PART_A)

from custom import cus1, cus2                      # noqa: E402  (Part A, unchanged)
from informed import astar, gbfs, inconsistent_edges  # noqa: E402,F401
from uninformed import bfs, dfs                    # noqa: E402

METHODS = {"AS": astar, "CUS1": cus1, "BFS": bfs, "DFS": dfs, "GBFS": gbfs, "CUS2": cus2}
METHOD_LABELS = {
    "AS": "A* (AS)",
    "CUS1": "Uniform-cost search (CUS1)",
    "BFS": "Breadth-first search (BFS)",
    "DFS": "Depth-first search (DFS)",
    "GBFS": "Greedy best-first search (GBFS)",
    "CUS2": "IDA*, fewest moves (CUS2)",
}
OPTIMAL_METHODS = {"AS", "CUS1"}


def build_search_problem(positions_km: dict, edge_costs: dict, seconds_per_km: float):
    """Return (nodes, edges) in Part A's format.

    positions_km  {site: (x_km, y_km)}
    edge_costs    {(start, end): seconds}
    seconds_per_km  seconds to drive 1 km at the speed limit (the heuristic's unit)
    """
    nodes = {site: (x * seconds_per_km, y * seconds_per_km) for site, (x, y) in positions_km.items()}
    edges = {site: [] for site in nodes}
    for (start, end), cost in edge_costs.items():
        if cost < 0:
            raise ValueError(f"edge {start}->{end} has a negative cost")
        edges[start].append((end, float(cost)))
    # Part A expects every neighbour list sorted by node number (ascending child order).
    edges = {site: sorted(neighbours) for site, neighbours in edges.items()}
    return nodes, edges


def path_cost(edges: dict, path: list) -> float:
    costs = {(a, b): c for a in edges for b, c in edges[a]}
    return sum(costs[(a, b)] for a, b in zip(path, path[1:]))


@dataclass
class SearchStats:
    method: str
    searches: int = 0
    nodes_created: int = 0
    seconds: float = 0.0
    first_route_cost: float | None = None
    notes: list = field(default_factory=list)


def top_k_routes(nodes: dict, edges: dict, origin: int, destination: int, k: int = 5,
                 method: str = "AS"):
    """Return ([(cost, path), ...] sorted by cost, SearchStats) for up to k loopless routes."""
    method = method.upper()
    if method not in METHODS:
        raise ValueError(f"Unknown search method '{method}'. Choose from: {', '.join(METHODS)}")
    for site in (origin, destination):
        if site not in nodes:
            raise ValueError(f"Site {site} is not on the map")
    if k < 1:
        raise ValueError("k must be at least 1")

    search = METHODS[method]
    stats = SearchStats(method=method)
    costs = {(a, b): c for a in edges for b, c in edges[a]}
    started = time.perf_counter()

    def run(graph_edges, start):
        goal, created, path = search(nodes, graph_edges, start, [destination])
        stats.searches += 1
        stats.nodes_created += created
        return path if goal is not None else None

    def cost_of(path):
        return sum(costs[(a, b)] for a, b in zip(path, path[1:]))

    first = run(edges, origin)
    if first is None:
        stats.seconds = time.perf_counter() - started
        return [], stats
    found = [(cost_of(first), first)]
    stats.first_route_cost = found[0][0]
    seen = {tuple(first)}
    candidates = []  # heap of (cost, number of nodes, path tuple)

    while len(found) < k:
        previous = found[-1][1]
        for i in range(len(previous) - 1):
            spur, root = previous[i], previous[:i + 1]
            blocked_edges = {(p[i], p[i + 1]) for _, p in found if len(p) > i + 1 and p[:i + 1] == root}
            blocked_nodes = set(root[:-1])
            sub_edges = {u: [(v, c) for v, c in neighbours
                             if v not in blocked_nodes and (u, v) not in blocked_edges]
                         for u, neighbours in edges.items() if u not in blocked_nodes}
            spur_path = run(sub_edges, spur)
            if spur_path is None:
                continue
            route = tuple(root[:-1] + spur_path)
            if route not in seen:
                seen.add(route)
                heapq.heappush(candidates, (round(cost_of(route), 9), len(route), route))
        if not candidates:
            break  # fewer than k loopless routes exist
        cost, _, route = heapq.heappop(candidates)
        found.append((cost_of(route), list(route)))

    found.sort(key=lambda item: (round(item[0], 9), len(item[1]), item[1]))
    stats.seconds = time.perf_counter() - started
    if method not in OPTIMAL_METHODS:
        stats.notes.append(f"{METHOD_LABELS[method]} does not guarantee the fastest route; "
                           "the routes are sorted by their travel time.")
    return found, stats
