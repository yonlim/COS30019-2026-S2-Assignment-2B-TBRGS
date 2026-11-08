"""Shared code for the route-finding program (COS30019 Assignment 2, Part A).

This module reads problem files and provides the search-tree pieces that
every search method uses, so all six methods follow the same rules:

1. Every method is called as method(nodes, edges, origin, destinations)
   and returns (goal, nodes_created, path). If no destination can be
   reached it returns (None, nodes_created, []).
2. nodes_created counts every search-tree node created, including the
   start node. SearchTree does the counting, so no method counts by hand.
3. Repeated states: SearchTree.expand never creates a child for a graph
   node that has already been expanded, and each method skips any node it
   takes off the frontier whose graph node has already been expanded.
4. Ties: SearchTree.expand returns children in ascending node order, and
   PriorityFrontier gives back equal priorities in insertion order. This
   matches the brief: ascending order first, then chronological order for
   nodes on different branches.
5. The goal test runs when a node is taken off the frontier to be
   expanded, not when it is created.
"""

import heapq
import math
import re

# Section headings in a problem file (matched without regard to case).
SECTIONS = ("nodes", "edges", "origin", "destinations")

# A coordinate such as 4, -2 or 3.5.
_NUMBER = r"[-+]?\d+(?:\.\d+)?"

# "1: (4,1)" means node 1 is at x = 4, y = 1.
NODE_LINE = re.compile(rf"(\d+)\s*:\s*\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)")

# "(2,1): 4" means you can go from node 2 to node 1 at a cost of 4.
# Costs cannot be negative, so the cost part has no sign.
EDGE_LINE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\d+(?:\.\d+)?)")


def _to_number(text):
    """Turn "4" into 4 and "3.5" into 3.5."""
    return float(text) if "." in text else int(text)


def parse(filename):
    """Read a problem file and return (nodes, edges, origin, destinations).

    nodes         {node: (x, y)}
    edges         {node: [(neighbour, cost), ...]}, with an entry for every
                  node and each list sorted by neighbour. An edge is stored
                  only in the direction written, so "(2,1): 4" lets you go
                  from 2 to 1 but not from 1 to 2.
    origin        the start node
    destinations  the destination nodes, in the order they are listed

    Raises ValueError (naming the line) if the file is not in the expected
    format, and OSError if the file cannot be opened.
    """
    nodes = {}
    adjacent = {}  # {from_node: {to_node: cost}} while the file is read
    origin = None
    destinations = []
    section = None

    # "utf-8-sig" also accepts files saved with a byte-order mark,
    # which some Windows editors add.
    with open(filename, encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # A heading may have its content on the same line,
            # e.g. "Origin: 2" as well as "Origin:" followed by "2".
            heading, colon, rest = line.partition(":")
            if colon and heading.strip().lower() in SECTIONS:
                section = heading.strip().lower()
                line = rest.strip()
                if not line:
                    continue

            where = f"line {line_number}"

            if section == "nodes":
                match = NODE_LINE.fullmatch(line)
                if not match:
                    raise ValueError(f"{where}: expected a node like '1: (4,1)', got '{line}'")
                node = int(match.group(1))
                if node in nodes:
                    raise ValueError(f"{where}: node {node} is listed twice")
                nodes[node] = (_to_number(match.group(2)), _to_number(match.group(3)))

            elif section == "edges":
                match = EDGE_LINE.fullmatch(line)
                if not match:
                    raise ValueError(f"{where}: expected an edge like '(2,1): 4', got '{line}'")
                start, end = int(match.group(1)), int(match.group(2))
                cost = _to_number(match.group(3))
                # If the same edge is listed twice, keep the cheaper one.
                previous = adjacent.setdefault(start, {}).get(end)
                if previous is None or cost < previous:
                    adjacent[start][end] = cost

            elif section == "origin":
                if not line.isdecimal():
                    raise ValueError(f"{where}: expected the origin node number, got '{line}'")
                if origin is not None:
                    raise ValueError(f"{where}: only one origin is allowed")
                origin = int(line)

            elif section == "destinations":
                # Destinations are separated by semicolons, e.g. "5; 4".
                for part in re.split(r"[;,\s]+", line):
                    if not part:
                        continue
                    if not part.isdecimal():
                        raise ValueError(
                            f"{where}: expected destination node numbers like '5; 4', got '{line}'")
                    if int(part) not in destinations:
                        destinations.append(int(part))

            else:
                raise ValueError(f"{where}: expected a heading such as 'Nodes:', got '{line}'")

    _check(nodes, adjacent, origin, destinations)

    # Sorting each neighbour list here is what makes every method generate
    # children in ascending node order. Node numbers are integers, so 10
    # sorts after 2.
    edges = {node: sorted(adjacent.get(node, {}).items()) for node in nodes}
    return nodes, edges, origin, destinations


def _check(nodes, adjacent, origin, destinations):
    """Raise ValueError if a section is missing or uses a node that is not listed."""
    if not nodes:
        raise ValueError("no nodes are listed under 'Nodes:'")
    if origin is None:
        raise ValueError("no origin is given under 'Origin:'")
    if not destinations:
        raise ValueError("no destinations are given under 'Destinations:'")
    for node in [origin, *destinations]:
        if node not in nodes:
            raise ValueError(f"node {node} is used but not listed under 'Nodes:'")
    for start, ends in adjacent.items():
        for end in ends:
            for node in (start, end):
                if node not in nodes:
                    raise ValueError(
                        f"edge ({start},{end}) uses node {node}, which is not listed under 'Nodes:'")


def straight_line_distance(nodes, a, b):
    """Return the straight-line (Euclidean) distance between graph nodes a and b."""
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    return math.hypot(x2 - x1, y2 - y1)


class Node:
    """A node of the search tree: one graph node, reached by one particular path."""

    def __init__(self, state, parent=None, path_cost=0, depth=0):
        self.state = state          # the graph node number
        self.parent = parent        # the search-tree node this one was created from
        self.path_cost = path_cost  # g(n): total edge cost from the origin
        self.depth = depth          # number of moves (edges) from the origin

    def path(self):
        """Return the graph nodes from the origin to this node, in order."""
        states = []
        node = self
        while node is not None:
            states.append(node.state)
            node = node.parent
        states.reverse()
        return states

    def __repr__(self):
        return f"Node({self.state}, cost={self.path_cost}, depth={self.depth})"


class SearchTree:
    """Creates search-tree nodes for one search and counts how many it made.

    Every method creates its nodes through a SearchTree, so the node count,
    the child order and the repeated-state rule are the same in all of them.
    """

    def __init__(self, edges):
        self.edges = edges
        self.nodes_created = 0

    def root(self, origin):
        """Create (and count) the start node."""
        self.nodes_created += 1
        return Node(origin)

    def expand(self, node, explored):
        """Create (and count) the children of node, in ascending node order.

        explored is the set of graph nodes the method has already expanded.
        No child is created for those, which stops the search going round
        cycles such as 1 -> 3 -> 1.
        """
        children = [
            Node(neighbour, node, node.path_cost + cost, node.depth + 1)
            for neighbour, cost in self.edges[node.state]
            if neighbour not in explored
        ]
        self.nodes_created += len(children)
        return children


class PriorityFrontier:
    """A frontier that always gives back the node with the lowest priority.

    Equal priorities come out in the order they were added (first in, first
    out). Because SearchTree.expand adds siblings in ascending node order,
    this follows the brief's tie-breaking rule. For use by GBFS, A* and any
    other method that orders its frontier by a value.
    """

    def __init__(self):
        self._heap = []
        self._count = 0  # insertion counter, used to break ties

    def push(self, node, priority):
        """Add node with the given priority (lower comes out first)."""
        # Rounding stops tiny floating-point differences (e.g. 0.1 + 0.2
        # versus 0.3) from deciding a tie instead of the insertion order.
        heapq.heappush(self._heap, (round(priority, 9), self._count, node))
        self._count += 1

    def pop(self):
        """Remove and return the node with the lowest priority."""
        return heapq.heappop(self._heap)[2]

    def __len__(self):
        return len(self._heap)
