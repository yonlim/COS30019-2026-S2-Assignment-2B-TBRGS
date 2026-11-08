"""Informed search methods: greedy best-first search (GBFS) and A* (AS).

Both methods use the heuristic

    h(n) = straight-line distance from n to the NEAREST destination.

The agent only has to reach one destination, so the cheapest cost from n is
the cheapest of its costs to each destination. Taking the smallest
straight-line distance keeps h(n) at or below that cost whenever no edge is
cheaper than its own straight-line length.

Consistency. A* with an explored set (rule 3 in graph.py) is only
guaranteed to return a cheapest path if h is consistent:

    h(a) <= cost(a, b) + h(b)   for every edge a -> b, and h(goal) = 0.

Straight-line distance obeys the triangle inequality, so this can only fail
on an edge that costs less than its straight-line length.
inconsistent_edges() lists any such edges in a problem, so test results can
be explained (see tests/cheap_edge.txt).

Both methods create their nodes through SearchTree and keep them in a
PriorityFrontier, so they follow the team's shared rules at the top of
graph.py: every node counted, no child for an expanded state, ascending
child order, ties by insertion order, goal test on expansion.
"""

from graph import PriorityFrontier, SearchTree, straight_line_distance


def make_heuristic(nodes, destinations):
    """Return h(state): straight-line distance from state to its nearest destination.

    Values are cached because the same state can be scored many times.
    h is 0 at every destination.
    """
    cache = {}

    def h(state):
        if state not in cache:
            cache[state] = min(straight_line_distance(nodes, state, goal)
                               for goal in destinations)
        return cache[state]

    return h


def inconsistent_edges(nodes, edges, destinations):
    """Return the edges where h(a) > cost(a, b) + h(b).

    Each item is (a, b, cost, h(a), h(b)). An empty list means the heuristic
    is consistent on this problem, so A* is guaranteed to find a cheapest
    path. A small tolerance stops floating-point rounding being reported.
    """
    h = make_heuristic(nodes, destinations)
    return [
        (a, b, cost, h(a), h(b))
        for a in sorted(edges)
        for b, cost in edges[a]
        if h(a) > cost + h(b) + 1e-9
    ]


def _best_first_search(edges, origin, destinations, priority):
    """Expand the frontier node with the lowest priority(node) until a goal is expanded.

    Returns (goal, nodes_created, path), or (None, nodes_created, []) if no
    destination can be reached.
    """
    goals = set(destinations)
    tree = SearchTree(edges)
    frontier = PriorityFrontier()
    start = tree.root(origin)
    frontier.push(start, priority(start))
    explored = set()  # graph nodes that have already been expanded

    while frontier:
        node = frontier.pop()
        if node.state in explored:
            continue  # reached again by another route; already expanded
        if node.state in goals:
            return node.state, tree.nodes_created, node.path()
        explored.add(node.state)
        for child in tree.expand(node, explored):  # ascending node order
            frontier.push(child, priority(child))

    return None, tree.nodes_created, []


def gbfs(nodes, edges, origin, destinations):
    """Greedy best-first search: always expand the node that looks closest to a destination.

    Priority f(n) = h(n). GBFS ignores the cost already paid, so it is often
    fast but the path it returns is not guaranteed to be the cheapest (see
    tests/greedy_misled.txt).

    Returns (goal, nodes_created, path), or (None, nodes_created, []) if no
    destination can be reached.
    """
    h = make_heuristic(nodes, destinations)
    return _best_first_search(edges, origin, destinations,
                              lambda node: h(node.state))


def astar(nodes, edges, origin, destinations):
    """A* search: always expand the node with the lowest f(n) = g(n) + h(n).

    g(n) is the cost from the origin to n. When the heuristic is consistent
    on the problem (inconsistent_edges() is empty), the first destination
    expanded is reached by a cheapest path.

    Returns (goal, nodes_created, path), or (None, nodes_created, []) if no
    destination can be reached.
    """
    h = make_heuristic(nodes, destinations)
    return _best_first_search(edges, origin, destinations,
                              lambda node: node.path_cost + h(node.state))
