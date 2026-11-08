"""Uninformed search methods: depth-first search (DFS) and breadth-first search (BFS).

Neither method uses coordinates or edge costs; they only follow the order in
which nodes are reached. Both create their nodes through SearchTree, so they
follow the team's shared rules (see the top of graph.py).
"""

from collections import deque

from graph import SearchTree


def dfs(nodes, edges, origin, destinations):
    """Depth-first search: always expand the deepest node on the frontier.

    The frontier is a last-in, first-out stack, so the search follows one
    branch as far as it can before backing up to try the next one. Children
    are pushed in reverse order so the smallest-numbered child ends up on top
    and is expanded first. The path DFS finds is not necessarily the
    shortest or the cheapest.

    nodes is not used (DFS needs no coordinates) but is kept so every method
    has the same signature. Returns (goal, nodes_created, path), or
    (None, nodes_created, []) if no destination can be reached.
    """
    goals = set(destinations)
    tree = SearchTree(edges)
    frontier = [tree.root(origin)]  # a Python list used as a stack
    explored = set()  # graph nodes that have already been expanded

    while frontier:
        node = frontier.pop()
        if node.state in explored:
            continue  # reached again by another route; already expanded
        if node.state in goals:
            return node.state, tree.nodes_created, node.path()
        explored.add(node.state)
        # Reverse, so the smallest child is pushed last and popped first.
        frontier.extend(reversed(tree.expand(node, explored)))

    return None, tree.nodes_created, []


def bfs(nodes, edges, origin, destinations):
    """Breadth-first search: always expand the shallowest node on the frontier.

    The frontier is a first-in, first-out queue, so the search works through
    the tree one level at a time. Within a level, nodes come out in the
    order they were added, which keeps the brief's tie-breaking rule. BFS
    finds a path with the fewest moves, but not always the lowest cost.

    nodes is not used (BFS needs no coordinates) but is kept so every method
    has the same signature. Returns (goal, nodes_created, path), or
    (None, nodes_created, []) if no destination can be reached.
    """
    goals = set(destinations)
    tree = SearchTree(edges)
    frontier = deque([tree.root(origin)])
    explored = set()  # graph nodes that have already been expanded

    while frontier:
        node = frontier.popleft()
        if node.state in explored:
            continue  # reached again by another route; already expanded
        if node.state in goals:
            return node.state, tree.nodes_created, node.path()
        explored.add(node.state)
        frontier.extend(tree.expand(node, explored))

    return None, tree.nodes_created, []
