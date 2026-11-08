"""Task 3 (part 1): the Boroondara road network built from the SCATS sites.

Owner: Jonathan Yu Yung Jing (105815665)

The dataset has no map, only 40 intersections (SCATS sites) and the
location text of each detector, such as "WARRIGAL_RD N of TOORAK_RD". That
text says there is a road leg (Warrigal Rd) leaving the intersection to the
north. The network is built from these legs:

1. Each detector "ROAD DIR of CROSS" at site X looks for the nearest other
   site Y that also lies on ROAD, in direction DIR (within +/-60 degrees),
   no more than 4 km away. If one is found, X and Y are joined by a link.
2. Some roads change name along the way; ALIASES joins the names that the
   sites use for the same road (each one checked against a public source).
3. A few sites have no other monitored site on any of their roads.
   data/manual_links.csv joins them to the nearest site they really connect
   to, with the roads used and the reason.
4. Every link is two-way. The flow used for the directed edge A -> B comes
   from a detector chosen by the rules in _flow_detectors().

Positions: latitude and longitude are projected onto a flat x/y grid in
kilometres (equirectangular projection around the network's mean latitude;
over this 15 km area the error is well under 0.1%). Link length is the
straight-line distance between the two sites, as the brief allows.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

EARTH_RADIUS_KM = 6371.0088
MAX_LINK_KM = 4.0
ANGLE_TOLERANCE = 60.0
BEARINGS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}

# Different names for the same road (checked against Wikipedia's route articles):
#   Maroondah Hwy is Whitehorse Rd east of Union Rd, Surrey Hills;
#   Toorak Rd is signed Burwood Hwy east of Warrigal Rd;
#   Victoria St becomes Barkers Rd after crossing the Yarra on Victoria Bridge.
ALIASES = {
    "MAROONDAH_HWY": "WHITEHORSE_RD",
    "BURWOOD_HWY": "TOORAK_RD",
    "VICTORIA_ST": "BARKERS_RD",
}


def normalise_road(name) -> str:
    """'HIGH STREET_RD' -> 'HIGH_STREET_RD', then apply ALIASES."""
    if not name:
        return ""
    road = "_".join(str(name).upper().replace("_", " ").split())
    return ALIASES.get(road, road)


def angle_difference(a: float, b: float) -> float:
    difference = abs(a - b) % 360
    return min(difference, 360 - difference)


@dataclass(frozen=True)
class Link:
    """An undirected road link between two sites."""
    a: int
    b: int
    distance_km: float
    source: str   # "detector" (found from the location text) or "manual"
    road: str     # the road the link follows


@dataclass(frozen=True)
class Edge:
    """A directed edge start -> end, with the detectors whose flow sets its travel time."""
    start: int
    end: int
    distance_km: float
    detectors: tuple   # detector ids
    combine: str       # "sum" (one approach split over several detectors) or "mean"
    rule: str          # which rule chose the detectors (see _flow_detectors)


class RoadNetwork:
    def __init__(self, data, manual_links_path: str | None = None, flow_site: str = "destination",
                 max_link_km: float = MAX_LINK_KM, angle_tolerance: float = ANGLE_TOLERANCE):
        if flow_site not in ("destination", "origin"):
            raise ValueError("flow_site must be 'destination' or 'origin'")
        self.data = data
        self.flow_site = flow_site
        self.max_link_km = max_link_km
        self.angle_tolerance = angle_tolerance
        self.sites = data.sites.copy()
        self.detectors = data.detectors.copy()
        self.detectors["road_key"] = self.detectors["road"].map(normalise_road)
        self.detectors["cross_key"] = self.detectors["cross_road"].map(normalise_road)
        self._project()
        self.roads = {site: set(group["road_key"]) | set(group["cross_key"])
                      for site, group in self.detectors.groupby("site")}
        self.links = {}
        self.unmatched = []   # detectors whose road leaves the study area
        self._discover_links()
        self.manual_links = self._add_manual_links(manual_links_path)
        self.edges = self._build_edges()

    # ----- geometry ----------------------------------------------------------
    def _project(self) -> None:
        lat0 = math.radians(self.sites["latitude"].mean())
        lon0 = math.radians(self.sites["longitude"].mean())
        self.origin_latlon = (math.degrees(lat0), math.degrees(lon0))
        for frame in (self.sites, self.detectors):
            frame["x_km"] = [(math.radians(lon) - lon0) * math.cos(lat0) * EARTH_RADIUS_KM
                             for lon in frame["longitude"]]
            frame["y_km"] = [(math.radians(lat) - lat0) * EARTH_RADIUS_KM for lat in frame["latitude"]]

    def position(self, site: int) -> tuple:
        row = self.sites.loc[site]
        return float(row["x_km"]), float(row["y_km"])

    def distance(self, a: int, b: int) -> float:
        (x1, y1), (x2, y2) = self.position(a), self.position(b)
        return math.hypot(x2 - x1, y2 - y1)

    def bearing(self, a: int, b: int) -> float:
        """Compass bearing from site a to site b in degrees (0 = north, 90 = east)."""
        (x1, y1), (x2, y2) = self.position(a), self.position(b)
        return (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 360) % 360

    # ----- links ---------------------------------------------------------------
    def _discover_links(self) -> None:
        for detector, row in self.detectors.iterrows():
            if row["direction"] not in BEARINGS:
                continue
            site = int(row["site"])
            best = None
            for other in self.sites.index:
                if other == site or row["road_key"] not in self.roads.get(other, ()):
                    continue
                if angle_difference(self.bearing(site, other), BEARINGS[row["direction"]]) > self.angle_tolerance:
                    continue
                distance = self.distance(site, other)
                if distance <= self.max_link_km and (best is None or distance < best[0]):
                    best = (distance, int(other))
            if best is None:
                self.unmatched.append(detector)
                continue
            key = tuple(sorted((site, best[1])))
            if key not in self.links:
                self.links[key] = Link(key[0], key[1], round(best[0], 4), "detector", row["road_key"])

    def _add_manual_links(self, path) -> list:
        """Read data/manual_links.csv.

        Columns: site_a, site_b, via, approach_at_a, approach_at_b, reason.
        approach_at_b is the location text of the detector at site_b on the
        leg that traffic from site_a arrives on (and approach_at_a the other
        way round); it fixes the flow detector for that direction.
        """
        added = []
        self.manual_approaches = {}
        if not path or not os.path.exists(path):
            return added
        with open(path, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                a, b = int(row["site_a"]), int(row["site_b"])
                if a not in self.sites.index or b not in self.sites.index:
                    raise ValueError(f"manual link {a}-{b} uses a site that is not in the dataset")
                # In destination mode the edge a -> b uses the detector at b (approach_at_b);
                # in origin mode it uses the detector at a on the leg away from b, which the
                # file does not name, so the automatic rules decide.
                if self.flow_site == "destination":
                    if (row.get("approach_at_b") or "").strip():
                        self.manual_approaches[(a, b)] = row["approach_at_b"].strip()
                    if (row.get("approach_at_a") or "").strip():
                        self.manual_approaches[(b, a)] = row["approach_at_a"].strip()
                key = tuple(sorted((a, b)))
                if key not in self.links:
                    self.links[key] = Link(key[0], key[1], round(self.distance(a, b), 4), "manual",
                                           row.get("via", "").strip())
                    added.append(key)
        return added

    # ----- directed edges and their flow detectors ---------------------------------
    def _detectors_pointing(self, site: int, bearing: float, road: str | None) -> list:
        """Detectors on the one leg of site that points most nearly along bearing.

        Only legs within the angle tolerance count, and only legs on road if a
        road is given. Several detectors can share a leg (site 4335 has two on
        its High St leg); all of them are returned.
        """
        rows = self.detectors[self.detectors["site"] == site]
        candidates = []
        for detector, row in rows.iterrows():
            if row["direction"] not in BEARINGS:
                continue
            if road is not None and row["road_key"] != road:
                continue
            difference = angle_difference(BEARINGS[row["direction"]], bearing)
            if difference <= self.angle_tolerance:
                candidates.append((difference, row["road_key"], row["direction"], detector))
        if not candidates:
            return []
        best = min(candidates)
        return [c[3] for c in candidates if (c[0], c[1], c[2]) == best[:3]]

    def _manual_approach(self, start: int, end: int) -> list:
        """Detectors named in manual_links.csv for the edge start -> end, if any."""
        location = self.manual_approaches.get((start, end))
        if not location:
            return []
        site = end if self.flow_site == "destination" else start
        rows = self.detectors[(self.detectors["site"] == site)
                              & (self.detectors["location"].str.upper() == location.upper())]
        return list(rows.index)

    def _flow_detectors(self, start: int, end: int, road: str) -> tuple:
        """Choose the detectors whose counts stand for the traffic on start -> end.

        destination mode (default, the brief's "volume at SCATS site B"): the
        traffic on A -> B is the traffic arriving at B from A's side, which B's
        detector on the leg facing A counts. Rules, in order:
          1 "approach"       detectors at B on the link's road, on the leg facing A
          2 "approach (dir)" detectors at B on any road whose leg faces A
          3 "same road"      detectors at B on the link's road (other leg)
          4 "site average"   every detector at B
        origin mode (the conversion note's "flow from the starting site"): the
        traffic leaving A towards B is the traffic arriving at A on the leg
        opposite B, which then carries on to B. Same rules, applied at A with
        the bearing reversed.
        """
        if self.flow_site == "destination":
            site, bearing = end, self.bearing(end, start)
        else:
            site, bearing = start, (self.bearing(start, end) + 180) % 360
        manual = self._manual_approach(start, end)
        if manual:
            return tuple(manual), "sum", "approach (manual)"
        for rule, chosen, combine in (
                ("approach", self._detectors_pointing(site, bearing, road), "sum"),
                ("approach (dir)", self._detectors_pointing(site, bearing, None), "sum")):
            if chosen:
                return tuple(chosen), combine, rule
        same_road = list(self.detectors.index[(self.detectors["site"] == site)
                                              & (self.detectors["road_key"] == road)])
        if same_road:
            return tuple(same_road), "mean", "same road"
        return tuple(self.sites.loc[site, "detectors"]), "mean", "site average"

    def _build_edges(self) -> dict:
        edges = {}
        for (a, b), link in sorted(self.links.items()):
            for start, end in ((a, b), (b, a)):
                detectors, combine, rule = self._flow_detectors(start, end, link.road)
                edges[(start, end)] = Edge(start, end, link.distance_km, detectors, combine, rule)
        return edges

    # ----- queries -----------------------------------------------------------------
    def neighbours(self, site: int) -> list:
        return sorted(end for (start, end) in self.edges if start == site)

    def is_strongly_connected(self) -> bool:
        """True if every site can reach every other site."""
        sites = list(self.sites.index)
        if not sites:
            return True
        for direction in ("forward", "backward"):
            seen, stack = set(), [sites[0]]
            while stack:
                site = stack.pop()
                if site in seen:
                    continue
                seen.add(site)
                for (start, end) in self.edges:
                    nxt = end if direction == "forward" and start == site else \
                          start if direction == "backward" and end == site else None
                    if nxt is not None and nxt not in seen:
                        stack.append(nxt)
            if len(seen) != len(sites):
                return False
        return True

    def summary(self) -> dict:
        rules = {}
        for edge in self.edges.values():
            rules[edge.rule] = rules.get(edge.rule, 0) + 1
        return {
            "sites": int(len(self.sites)),
            "links": len(self.links),
            "directed_edges": len(self.edges),
            "manual_links": len(self.manual_links),
            "strongly_connected": self.is_strongly_connected(),
            "edge_flow_rules": rules,
            "detectors_leaving_area": len(self.unmatched),
            "mean_link_km": round(sum(l.distance_km for l in self.links.values()) / max(len(self.links), 1), 3),
        }
