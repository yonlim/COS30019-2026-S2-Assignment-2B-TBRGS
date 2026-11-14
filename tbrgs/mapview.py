"""OpenStreetMap view of the predicted traffic and the recommended routes (research).

Owner: Jonathan Yu Yung Jing (105815665)

Inspired by Google Maps' traffic layer: every road link is coloured by its
predicted speed and the routes are drawn on top, on OpenStreetMap tiles
(through the folium package). The result is a standalone HTML file that
opens in any browser (the map tiles need an internet connection).

Coordinate correction: the brief notes that the SCATS latitudes and
longitudes do not land on the real intersections. `python research.py
coordinates` compares them with VicRoads' traffic count locations for the
same intersections and estimates a constant shift; config.ini [map] holds
the shift used here.
"""

from __future__ import annotations

import os

import pandas as pd

ROUTE_COLOURS = ["#1565c0", "#7b1fa2", "#00838f", "#5d4037", "#c2185b"]


def _speed_colour(speed):
    return "#2e7d32" if speed >= 55 else "#f9a825" if speed >= 45 else "#ef6c00" if speed >= 38 else "#c62828"


def corrected_position(system, site) -> tuple:
    lat_offset = system.config.getfloat("map", "lat_offset", 0.0)
    lon_offset = system.config.getfloat("map", "lon_offset", 0.0)
    row = system.data.sites.loc[site]
    return float(row["latitude"]) + lat_offset, float(row["longitude"]) + lon_offset


def save_route_map(system, result, estimates=None, path: str | None = None) -> str:
    """Write an HTML map of the network and result's routes; return the file path."""
    import folium

    sites = system.data.sites
    points = {site: corrected_position(system, site) for site in sites.index}
    centre = (sum(p[0] for p in points.values()) / len(points), sum(p[1] for p in points.values()) / len(points))
    fmap = folium.Map(location=centre, zoom_start=13, tiles="Esri.WorldStreetMap", control_scale=True)

    network = system.network()
    roads = folium.FeatureGroup(name="Predicted speed on each link")
    for (a, b) in network.links:
        tooltip = f"{a} - {b}"
        colour = "#757575"
        if estimates:
            ab, ba = estimates[(a, b)], estimates[(b, a)]
            colour = _speed_colour(min(ab.speed_kmh, ba.speed_kmh))
            tooltip = (f"{a} -> {b}: {ab.flow_vph:.0f} veh/h, {ab.speed_kmh:.0f} km/h<br>"
                       f"{b} -> {a}: {ba.flow_vph:.0f} veh/h, {ba.speed_kmh:.0f} km/h")
        folium.PolyLine([points[a], points[b]], color=colour, weight=5, opacity=0.8, tooltip=tooltip).add_to(roads)
    roads.add_to(fmap)

    if result is not None:
        for i, route in reversed(list(enumerate(result.routes))):
            group = folium.FeatureGroup(name=f"Route {route.rank}: {route.minutes:.1f} min")
            folium.PolyLine([points[s] for s in route.path], color=ROUTE_COLOURS[i % len(ROUTE_COLOURS)],
                            weight=9 if i == 0 else 6, opacity=0.75,
                            tooltip=f"Route {route.rank}: {route.minutes:.1f} min, {route.distance_km:.2f} km").add_to(group)
            group.add_to(fmap)

    markers = folium.FeatureGroup(name="SCATS sites")
    for site, (lat, lon) in points.items():
        row = sites.loc[site]
        colour = "#2e7d32" if result is not None and site == result.origin else \
                 "#c62828" if result is not None and site == result.destination else "#263238"
        folium.CircleMarker((lat, lon), radius=6 if colour != "#263238" else 4, color=colour, fill=True,
                            fill_opacity=0.9,
                            popup=f"<b>{site}</b> {row['name']}<br>{row['description']}").add_to(markers)
    markers.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    if result is not None:
        when = pd.Timestamp(result.time_used)
        title = (f"TBRGS: {result.origin} to {result.destination}, {when:%a %d %b %Y %H:%M}, "
                 f"flows from {result.model.upper()}")
        fmap.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;padding:6px 10px;'
            f'border:1px solid #999;font:14px sans-serif">{title}</div>'))
    if path is None:
        results_dir = system.config.path_of("paths", "results_dir")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "route_map.html")
    fmap.save(path)
    return path
