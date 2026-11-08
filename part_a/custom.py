"""Custom search methods: CUS1 (uninformed) and CUS2 (informed).

CUS1 is uniform-cost search, an uninformed method that finds a path to a
destination (the cheapest one). CUS2 is IDA*, iterative-deepening A*, an
informed method that finds a path with the fewest moves.

CUS1 creates its nodes through SearchTree and keeps them in a
PriorityFrontier, so it follows every shared rule at the top of graph.py.

CUS2 follows rules 1, 2, 4 and 5. It has no priority queue: it searches
depth first and takes the smallest child first, so insertion order is not
needed for ties. Rule 3 is replaced inside each round, because a plain
explored set would make IDA* miss the fewest-moves path (a node first
reached by a long route would block the short route through it). Instead,
no child is created for a graph node already reached with the same or
fewer moves in the current round. Going round a cycle always adds moves,
so this still stops cycles.
"""

import math

from graph import PriorityFrontier, SearchTree, straight_line_distance

# Floating-point slack: 6.000000000001 / 6 must count as 1 move, not 2,
# otherwise rounding up could overestimate the moves left.
_EPSILON = 1e-9


def cus1(nodes, edges, origin, destinations):
    """Uniform-cost search: always expand the node with the lowest path cost g(n).

    g(n) is the total edge cost from the origin. UCS uses no information
    about where the destinations are, so it is uninformed. Edge costs are
    never negative, so the first destination it expands is reached by a
    cheapest path. Compared with BFS, it finds the lowest cost rather than
    the fewest moves (see tests/fewest_moves_vs_cheapest.txt).

    nodes is not used (UCS needs no coordinates) but is kept so every
    method has the same signature. Returns (goal, nodes_created, path), or
    (None, nodes_created, []) if no destination can be reached.
    """
    goals = set(destinations)
    tree = SearchTree(edges)
    frontier = PriorityFrontier()
    frontier.push(tree.root(origin), 0)
    explored = set()  # graph nodes that have already been expanded

    while frontier:
        node = frontier.pop()
        if node.state in explored:
            continue  # a cheaper copy was expanded already
        if node.state in goals:
            return node.state, tree.nodes_created, node.path()
        explored.add(node.state)
        for child in tree.expand(node, explored):  # ascending node order
            frontier.push(child, child.path_cost)

    return None, tree.nodes_created, []


def moves_heuristic(nodes, edges, destinations):
    """Return {state: h}, a lower bound on the moves needed to reach a destination.

    h(n) = ceil(straight-line distance from n to its nearest destination
                / straight-line length of the longest edge)

    One move covers one edge, and no edge spans more straight-line distance
    than the longest one (L). Any route to a destination is at least as long
    as the straight line to the nearest destination, so it needs at least
    distance / L moves, and moves are whole numbers. So h never
    overestimates the moves left, and h is 0 at every destination.
    """
    longest = max((straight_line_distance(nodes, a, b)
                   for a in edges for b, _ in edges[a]), default=0.0)
    h = {}
    for state in nodes:
        if longest == 0:  # no edges, so no useful estimate
            h[state] = 0
            continue
        nearest = min(straight_line_distance(nodes, state, goal) for goal in destinations)
        h[state] = max(0, math.ceil(nearest / longest - _EPSILON))
    return h


def cus2(nodes, edges, origin, destinations):
    """IDA*: depth-first rounds limited by f(n) = g(n) + h(n), where g counts moves.

    Round 1 uses the limit h(origin). Each round is a depth-first search that
    only expands nodes with f <= limit. If no destination is found, the
    limit becomes the smallest f that went over it and a new round starts.
    h never overestimates and every f is a whole number of moves, so the
    limit never passes the fewest-moves answer, and the first destination
    found is reached with the fewest moves. Edge costs are ignored (see
    tests/fewest_moves_vs_cheapest.txt).

    nodes_created counts the nodes created in every round, including the
    start node each time, because IDA* repeats earlier work in each round.

    Returns (goal, nodes_created, path), or (None, nodes_created, []) if no
    destination can be reached.
    """
    goals = set(destinations)
    h = moves_heuristic(nodes, edges, destinations)
    limit = h[origin]
    nodes_created = 0

    while True:
        goal_node, created, next_limit = _limited_search(edges, origin, goals, h, limit)
        nodes_created += created
        if goal_node is not None:
            return goal_node.state, nodes_created, goal_node.path()
        if next_limit == math.inf:  # nothing was cut off, so nothing is reachable
            return None, nodes_created, []
        limit = next_limit


def _limited_search(edges, origin, goals, h, limit):
    """One IDA* round: depth-first search expanding only nodes with f <= limit.

    Returns (goal_node, nodes_created, next_limit). goal_node is the first
    destination expanded, or None, and next_limit is the smallest f that was
    cut off (math.inf if nothing was).
    """
    tree = SearchTree(edges)
    frontier = [tree.root(origin)]  # a Python list used as a stack
    fewest = {origin: 0}  # fewest moves each graph node was reached with this round
    next_limit = math.inf

    while frontier:
        node = frontier.pop()
        if node.depth > fewest[node.state]:
            continue  # reached again with fewer moves since this copy was added
        if node.state in goals:
            return node, tree.nodes_created, next_limit

        # No child for a graph node already reached with as few moves.
        moves = node.depth + 1
        reached = {state for state, _ in edges[node.state]
                   if fewest.get(state, math.inf) <= moves}
        within_limit = []
        for child in tree.expand(node, reached):  # ascending node order
            fewest[child.state] = moves
            f = moves + h[child.state]
            if f > limit:
                next_limit = min(next_limit, f)  # created, but cut off
            else:
                within_limit.append(child)
        # Reverse, so the smallest child is pushed last and popped first.
        frontier.extend(reversed(within_limit))

    return None, tree.nodes_created, next_limit
