"""The end-to-end TBRGS: predicted flows -> travel times -> top-k routes.

Owner: Lau Ka Meng (105814507)

TBRGS is the one object the command line (main.py), the GUI (gui.py) and the
tests use. It loads the data and the road network once, loads trained models
when they are first needed, and answers route queries:

    system = TBRGS()
    result = system.find_routes(2000, 3002, "2006-10-25 08:00", model="lstm", method="AS", k=5)
    for route in result.routes:
        print(route.rank, route.minutes, route.path)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import pandas as pd

from .config import load_config
from .data import SeriesScaler, split_dates
from .models import DISPLAY_NAMES, load_model, trained_models
from .models.baselines import HistoricalAverageModel
from .network import RoadNetwork
from .predictor import HISTORICAL, OBSERVED, FlowPredictor
from .routing import METHOD_LABELS, OPTIMAL_METHODS, build_search_problem, top_k_routes
from .training import load_data
from .travel_time import TravelTimeModel, hourly_flow

SOURCE_LABELS = dict(DISPLAY_NAMES, observed="Observed flow (no prediction)")


@dataclass
class EdgeEstimate:
    start: int
    end: int
    distance_km: float
    flow_15min: float
    flow_vph: float
    speed_kmh: float
    time_s: float
    rule: str


@dataclass
class Route:
    rank: int
    path: list
    time_s: float
    distance_km: float
    edges: list  # EdgeEstimate for each step

    @property
    def minutes(self) -> float:
        return self.time_s / 60

    @property
    def intersections(self) -> int:
        """Controlled intersections passed after leaving the origin (each adds the delay)."""
        return len(self.path) - 1


@dataclass
class RouteResult:
    origin: int
    destination: int
    requested_time: str
    time_used: str
    model: str
    method: str
    k: int
    routes: list
    note: str = ""
    nodes_created: int = 0
    searches: int = 0
    search_ms: float = 0.0
    fallback_detectors: int = 0
    messages: list = field(default_factory=list)


class TBRGS:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.data = load_data(self.config)
        self.split = split_dates(self.data.flows, self.config.get("data", "train_end"),
                                 self.config.get("data", "validation_end"))
        self.model_dir = self.config.path_of("paths", "model_dir")
        scaler_path = os.path.join(self.model_dir, "scaler.json")
        # The scaler saved with the models is used if it exists; otherwise it is
        # refitted the same way (it only depends on the training period).
        self.scaler = (SeriesScaler.load(scaler_path) if os.path.exists(scaler_path)
                       else SeriesScaler().fit(self.data.flows, self.split.train[1]))
        self.historical = HistoricalAverageModel().fit(self.scaler.transform(self.data.flows),
                                                       self.split.train[1])
        self.predictor = FlowPredictor(self.data, self.scaler, self.historical)
        self._models = {}
        self._networks = {}

    # ----- settings ---------------------------------------------------------------
    def travel_time_model(self, overrides: dict | None = None) -> TravelTimeModel:
        section = {k: float(v) for k, v in self.config.section("travel_time").items() if k != "flow_site"}
        section.update({k: float(v) for k, v in (overrides or {}).items()
                        if k in TravelTimeModel.__dataclass_fields__ and v not in (None, "")})
        return TravelTimeModel(**{k: v for k, v in section.items() if k in TravelTimeModel.__dataclass_fields__})

    def network(self, flow_site: str | None = None) -> RoadNetwork:
        flow_site = flow_site or self.config.get("travel_time", "flow_site", "destination")
        if flow_site not in self._networks:
            self._networks[flow_site] = RoadNetwork(
                self.data, self.config.path_of("data", "manual_links_file"), flow_site=flow_site)
        return self._networks[flow_site]

    # ----- models ---------------------------------------------------------------------
    def available_sources(self) -> list:
        """Prediction sources that can be used now: trained models, then the reference sources."""
        return trained_models(self.model_dir) + [HISTORICAL, OBSERVED]

    def model(self, name: str):
        if name in (HISTORICAL, OBSERVED):
            return name
        if name not in self._models:
            self._models[name] = load_model(name, self.model_dir)
        return self._models[name]

    def forget_model(self, name: str) -> None:
        """Drop a cached model (after it has been retrained) and its cached predictions."""
        self._models.pop(name, None)
        self.predictor._cache = {k: v for k, v in self.predictor._cache.items() if k[0] != name}

    # ----- the pipeline -------------------------------------------------------------
    def predict_flows(self, model: str, when) -> tuple:
        return self.predictor.predict(self.model(model), when)

    def edge_estimates(self, model: str, when, travel: TravelTimeModel | None = None,
                       flow_site: str | None = None) -> tuple:
        """Predicted flow, speed and travel time on every directed edge at time `when`."""
        travel = travel or self.travel_time_model()
        network = self.network(flow_site)
        flows, info = self.predict_flows(model, when)
        estimates = {}
        for key, edge in network.edges.items():
            values = [float(flows[d]) for d in edge.detectors]
            count = sum(values) if edge.combine == "sum" else sum(values) / len(values)
            vph = hourly_flow(count)
            speed = travel.speed_from_flow(vph)
            estimates[key] = EdgeEstimate(edge.start, edge.end, edge.distance_km, count, vph, speed,
                                          travel.link_time_s(edge.distance_km, vph), edge.rule)
        return estimates, info

    def find_routes(self, origin: int, destination: int, when, model: str = "lstm",
                    method: str = "AS", k: int = 5, travel_overrides: dict | None = None,
                    flow_site: str | None = None) -> RouteResult:
        origin, destination, k = int(origin), int(destination), int(k)
        for site in (origin, destination):
            if site not in self.data.sites.index:
                raise ValueError(f"{site} is not a SCATS site in the dataset. "
                                 f"Valid sites: {', '.join(map(str, self.data.sites.index))}")
        if not 1 <= k <= 5:
            raise ValueError("The number of routes must be between 1 and 5")
        method = method.upper()
        travel = self.travel_time_model(travel_overrides)
        network = self.network(flow_site)
        estimates, info = self.edge_estimates(model, when, travel, flow_site)

        positions = {site: network.position(site) for site in network.sites.index}
        nodes, edges = build_search_problem(positions, {key: e.time_s for key, e in estimates.items()},
                                            travel.free_flow_seconds_per_km())
        started = time.perf_counter()
        found, stats = top_k_routes(nodes, edges, origin, destination, k, method)
        elapsed = (time.perf_counter() - started) * 1000

        routes = []
        for rank, (cost, path) in enumerate(found, start=1):
            steps = [estimates[(a, b)] for a, b in zip(path, path[1:])]
            routes.append(Route(rank, path, cost, sum(s.distance_km for s in steps), steps))
        messages = list(stats.notes)
        if info.get("fallback_detectors"):
            messages.append(f"{info['fallback_detectors']} detector(s) had no complete recent history; "
                            "their historical average was used.")
        if not routes:
            messages.append(f"No route from {origin} to {destination}.")
        return RouteResult(origin, destination, info["requested"], info["time_used"], model, method, k,
                           routes, info.get("note", ""), stats.nodes_created, stats.searches, elapsed,
                           info.get("fallback_detectors", 0), messages)

    # ----- text output ------------------------------------------------------------------
    def site_name(self, site: int) -> str:
        return str(self.data.sites.loc[site, "name"])

    def describe(self, result: RouteResult) -> str:
        """The result as plain text (used by main.py)."""
        lines = [
            f"Routes from {result.origin} ({self.site_name(result.origin)}) "
            f"to {result.destination} ({self.site_name(result.destination)})",
            f"Time: {pd.Timestamp(result.time_used):%a %d %b %Y %H:%M}   "
            f"Flows: {SOURCE_LABELS.get(result.model, result.model)}   "
            f"Search: {METHOD_LABELS.get(result.method, result.method)}",
        ]
        if result.note:
            lines.append(f"Note: {result.note}")
        for route in result.routes:
            lines.append("")
            lines.append(f"Route {route.rank}: {route.minutes:.1f} min, {route.distance_km:.2f} km, "
                         f"{route.intersections} intersections")
            lines.append("  " + " -> ".join(str(s) for s in route.path))
        for message in result.messages:
            lines.append(f"Note: {message}")
        optimal = "yes" if result.method in OPTIMAL_METHODS else "no"
        lines.append("")
        lines.append(f"Search work: {result.searches} searches, {result.nodes_created} nodes created, "
                     f"{result.search_ms:.1f} ms (fastest route guaranteed: {optimal})")
        return "\n".join(lines)
