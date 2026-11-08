"""Graphical user interface for the TBRGS (Tkinter + Matplotlib).

Owner: Lau Ka Meng (105814507); the map drawing is shared with Jonathan Yu.

Run from this folder:   python gui.py

Tabs
    Routes            choose a trip, see up to five routes on the map and in a table;
                      the map colours every road by its predicted speed
    Predictions       observed vs predicted flow for one detector over one day
    Model comparison  the evaluation results (results/metrics.csv) and chart
    Train             retrain a model with new settings
    Settings          travel-time parameters (speed limit, capacity, delay, flow rule)
On the map: left-click a site to make it the origin, right-click for the destination.
Defaults come from config.ini; the "Save as defaults" buttons write them back.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
import webbrowser

import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd

from tbrgs.config import load_config
from tbrgs.models import DISPLAY_NAMES, MODEL_CLASSES
from tbrgs.routing import METHOD_LABELS, OPTIMAL_METHODS
from tbrgs.system import SOURCE_LABELS, TBRGS

ROUTE_COLOURS = ["#1565c0", "#7b1fa2", "#00838f", "#5d4037", "#c2185b"]
# Speed bands for the "traffic" colouring of the map (km/h, lower bound, colour, label).
SPEED_BANDS = [(55, "#2e7d32", "55-60 km/h"), (45, "#f9a825", "45-55"),
               (38, "#ef6c00", "38-45"), (0, "#c62828", "32-38")]
TIMES = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
DATES = [f"2006-10-{d:02d}" for d in range(1, 32)]


def speed_colour(speed: float) -> str:
    for lower, colour, _ in SPEED_BANDS:
        if speed >= lower:
            return colour
    return SPEED_BANDS[-1][1]


class Worker:
    """Runs slow jobs (loading TensorFlow, training) off the GUI thread.

    Tkinter widgets may only be touched from the main thread, so results and
    log lines come back through a queue that the main thread polls.
    """

    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.busy = False
        root.after(100, self._poll)

    def run(self, job, on_done, on_error=None):
        self.busy = True

        def target():
            try:
                result = job()
                self.queue.put((on_done, result))
            except Exception as error:  # report any failure in the GUI, not the console only
                details = traceback.format_exc()
                self.queue.put((on_error or self._show_error, (error, details)))

        threading.Thread(target=target, daemon=True).start()

    def call_soon(self, func, *args):
        """Ask the main thread to call func(*args) (safe from any thread)."""
        self.queue.put((lambda a: func(*a), args))

    def _show_error(self, payload):
        error, details = payload
        messagebox.showerror("TBRGS", f"{error}\n\n{details[-1500:]}")

    def _poll(self):
        try:
            while True:
                callback, value = self.queue.get_nowait()
                self.busy = False if callback is not None else self.busy
                callback(value)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


class TBRGSApp:
    def __init__(self, root, system: TBRGS):
        self.root = root
        self.system = system
        self.config = system.config
        self.worker = Worker(root)
        self.result = None
        self.estimates = None
        self.selected_route = 0
        self.site_labels = {site: f"{site}  {row['name']}" for site, row in system.data.sites.iterrows()}
        self.label_to_site = {label: site for site, label in self.site_labels.items()}

        root.title("TBRGS - Traffic-Based Route Guidance System (Boroondara, October 2006)")
        root.geometry("1320x840")
        root.minsize(1100, 700)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))

        self._build_controls()
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(side="right", fill="both", expand=True, padx=(0, 8), pady=8)
        self._build_routes_tab()
        self._build_predictions_tab()
        self._build_comparison_tab()
        self._build_train_tab()
        self._build_settings_tab()
        self.status = tk.StringVar(value="Ready. Left-click a site on the map for the origin, "
                                         "right-click for the destination.")
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w").pack(
            side="bottom", fill="x", before=self.tabs)
        self.root.after(200, self.draw_map)

    # =========================================================================
    # Left panel: the trip
    # =========================================================================
    def _build_controls(self):
        panel = ttk.Frame(self.root, padding=8, width=310)
        panel.pack(side="left", fill="y")
        sites = list(self.site_labels.values())

        trip = ttk.LabelFrame(panel, text="Trip", padding=8)
        trip.pack(fill="x")
        self.origin = tk.StringVar(value=self.site_labels.get(self.config.getint("gui", "origin"), sites[0]))
        self.destination = tk.StringVar(value=self.site_labels.get(self.config.getint("gui", "destination"),
                                                                   sites[-1]))
        self.date = tk.StringVar(value=self.config.get("gui", "date"))
        self.time = tk.StringVar(value=self.config.get("gui", "time"))
        self._labelled(trip, "Origin (SCATS site)", ttk.Combobox(trip, textvariable=self.origin,
                                                                values=sites, width=36, state="readonly"))
        self._labelled(trip, "Destination (SCATS site)", ttk.Combobox(trip, textvariable=self.destination,
                                                                      values=sites, width=36, state="readonly"))
        ttk.Button(trip, text="Swap origin and destination", command=self.swap).pack(anchor="w", pady=(2, 6))
        row = ttk.Frame(trip)
        row.pack(fill="x")
        date_box = ttk.Frame(row)
        date_box.pack(side="left", padx=(0, 8))
        self._labelled(date_box, "Date (YYYY-MM-DD)", ttk.Combobox(date_box, textvariable=self.date,
                                                                   values=DATES, width=12))
        time_box = ttk.Frame(row)
        time_box.pack(side="left")
        self._labelled(time_box, "Time", ttk.Combobox(time_box, textvariable=self.time, values=TIMES, width=7))

        search = ttk.LabelFrame(panel, text="Prediction and search", padding=8)
        search.pack(fill="x", pady=8)
        self.model_label = tk.StringVar()
        self.model_box = ttk.Combobox(search, textvariable=self.model_label, width=36, state="readonly")
        self._labelled(search, "Flow prediction model", self.model_box)
        self.refresh_model_list(self.config.get("gui", "model"))
        self.method_label = tk.StringVar(value=METHOD_LABELS.get(self.config.get("routing", "method"),
                                                                 METHOD_LABELS["AS"]))
        self._labelled(search, "Search method (Part A)", ttk.Combobox(
            search, textvariable=self.method_label, values=list(METHOD_LABELS.values()),
            width=36, state="readonly"))
        self.k = tk.IntVar(value=self.config.getint("routing", "k"))
        self._labelled(search, "Number of routes", ttk.Spinbox(search, from_=1, to=5, textvariable=self.k,
                                                               width=5, state="readonly"))

        self.find_button = ttk.Button(panel, text="Find routes", style="Accent.TButton", command=self.find_routes)
        self.find_button.pack(fill="x", ipady=4)
        ttk.Button(panel, text="Open map in browser (OpenStreetMap)", command=self.open_browser_map).pack(
            fill="x", pady=(6, 0))
        ttk.Button(panel, text="Save trip as defaults", command=self.save_trip_defaults).pack(fill="x", pady=(6, 0))

        legend = ttk.LabelFrame(panel, text="Map legend", padding=8)
        legend.pack(fill="x", pady=8)
        for _, colour, label in SPEED_BANDS:
            line = ttk.Frame(legend)
            line.pack(anchor="w")
            swatch = tk.Canvas(line, width=26, height=10, highlightthickness=0)
            swatch.create_line(2, 5, 24, 5, fill=colour, width=4)
            swatch.pack(side="left")
            ttk.Label(line, text=f"predicted speed {label}").pack(side="left")
        ttk.Label(legend, text="Thick coloured lines: routes 1-5\nGreen dot: origin   Red dot: destination",
                  justify="left").pack(anchor="w", pady=(4, 0))

    @staticmethod
    def _labelled(parent, text, widget):
        ttk.Label(parent, text=text).pack(anchor="w")
        widget.pack(anchor="w", fill="x", pady=(0, 4))
        return widget

    def refresh_model_list(self, preferred: str | None = None):
        sources = self.system.available_sources()
        self.source_by_label = {SOURCE_LABELS.get(s, s): s for s in sources}
        self.model_box["values"] = list(self.source_by_label)
        current = preferred or self.source_by_label.get(self.model_label.get())
        if current not in sources:
            current = sources[0]
        self.model_label.set(SOURCE_LABELS.get(current, current))

    def selected_model(self) -> str:
        return self.source_by_label[self.model_label.get()]

    def selected_method(self) -> str:
        return {label: code for code, label in METHOD_LABELS.items()}[self.method_label.get()]

    def swap(self):
        a, b = self.origin.get(), self.destination.get()
        self.origin.set(b)
        self.destination.set(a)

    def trip_time(self):
        try:
            return pd.Timestamp(f"{self.date.get().strip()} {self.time.get().strip()}")
        except ValueError:
            raise ValueError("Enter the date as YYYY-MM-DD and the time as HH:MM") from None

    # =========================================================================
    # Routes tab: map + tables
    # =========================================================================
    def _build_routes_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Routes")
        panes = ttk.PanedWindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(panes, background="#fbfbf8", highlightthickness=0)
        self.canvas.bind("<Configure>", lambda e: self.draw_map())
        self.canvas.bind("<Button-1>", lambda e: self._map_click(e, "origin"))
        self.canvas.bind("<Button-3>", lambda e: self._map_click(e, "destination"))
        self.canvas.bind("<Button-2>", lambda e: self._map_click(e, "destination"))  # macOS right button
        panes.add(self.canvas, weight=5)

        lower = ttk.Frame(panes)
        panes.add(lower, weight=3)
        self.summary = tk.StringVar(value="No routes yet.")
        ttk.Label(lower, textvariable=self.summary, wraplength=900, justify="left").pack(anchor="w", padx=4, pady=2)
        tables = ttk.Frame(lower)
        tables.pack(fill="both", expand=True)
        columns = ("rank", "minutes", "km", "intersections", "path")
        self.route_table = ttk.Treeview(tables, columns=columns, show="headings", height=5)
        for column, heading, width in zip(columns, ("Route", "Travel time (min)", "Distance (km)",
                                                    "Intersections", "SCATS sites along the route"),
                                          (55, 110, 95, 90, 520)):
            self.route_table.heading(column, text=heading)
            self.route_table.column(column, width=width, anchor="center" if column != "path" else "w")
        self.route_table.pack(side="top", fill="x")
        self.route_table.bind("<<TreeviewSelect>>", self._route_selected)
        columns = ("leg", "from", "to", "km", "flow", "speed", "time", "rule")
        self.leg_table = ttk.Treeview(tables, columns=columns, show="headings", height=7)
        for column, heading, width in zip(columns, ("#", "From", "To", "km", "Flow (veh/h)", "Speed (km/h)",
                                                    "Time (s)", "Flow detector rule"),
                                          (35, 230, 230, 60, 90, 90, 70, 130)):
            self.leg_table.heading(column, text=heading)
            self.leg_table.column(column, width=width, anchor="center" if column not in ("from", "to") else "w")
        self.leg_table.pack(side="top", fill="both", expand=True, pady=(4, 0))

    def _map_transform(self):
        sites = self.system.network().sites
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 200)
        xs, ys = sites["x_km"], sites["y_km"]
        margin = 30
        scale = min((width - 2 * margin) / (xs.max() - xs.min()), (height - 2 * margin) / (ys.max() - ys.min()))
        x0 = margin + ((width - 2 * margin) - scale * (xs.max() - xs.min())) / 2
        y0 = margin + ((height - 2 * margin) - scale * (ys.max() - ys.min())) / 2

        def to_canvas(x_km, y_km):
            return x0 + (x_km - xs.min()) * scale, y0 + (ys.max() - y_km) * scale

        return to_canvas

    def draw_map(self):
        canvas = self.canvas
        canvas.delete("all")
        network = self.system.network()
        to_canvas = self._map_transform()
        points = {site: to_canvas(*network.position(site)) for site in network.sites.index}

        # Roads: one line per direction, offset to the left of travel (Australians drive on the left).
        for (start, end) in network.edges:
            (x1, y1), (x2, y2) = points[start], points[end]
            dx, dy = x2 - x1, y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
            ox, oy = -dy / length * 2.5, dx / length * 2.5
            colour = "#9e9e9e"
            if self.estimates is not None:
                colour = speed_colour(self.estimates[(start, end)].speed_kmh)
            canvas.create_line(x1 + ox, y1 + oy, x2 + ox, y2 + oy, fill=colour, width=3)

        # Routes, the selected one last (on top) and thicker.
        if self.result is not None and self.result.routes:
            order = [i for i in range(len(self.result.routes)) if i != self.selected_route] + [self.selected_route]
            for i in order:
                route = self.result.routes[i]
                coords = [c for site in route.path for c in points[site]]
                canvas.create_line(*coords, fill=ROUTE_COLOURS[i % len(ROUTE_COLOURS)],
                                   width=8 if i == self.selected_route else 4, capstyle="round",
                                   joinstyle="round")

        origin = self.label_to_site.get(self.origin.get())
        destination = self.label_to_site.get(self.destination.get())
        for site, (x, y) in points.items():
            fill = "#2e7d32" if site == origin else "#c62828" if site == destination else "white"
            r = 7 if site in (origin, destination) else 4
            canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="#263238")
            canvas.create_text(x + 6, y - 8, text=str(site), anchor="w", font=("TkDefaultFont", 8),
                               fill="#263238")
        canvas.create_text(10, 10, anchor="nw", fill="#455a64", font=("TkDefaultFont", 9),
                           text="Boroondara SCATS sites (north up). Left-click: origin, right-click: destination")
        self._points = points

    def _map_click(self, event, which):
        if not getattr(self, "_points", None):
            return
        site, distance = min(((s, (x - event.x) ** 2 + (y - event.y) ** 2) for s, (x, y) in self._points.items()),
                             key=lambda item: item[1])
        if distance > 15 ** 2:
            return
        (self.origin if which == "origin" else self.destination).set(self.site_labels[site])
        self.status.set(f"{which.capitalize()} set to {self.site_labels[site]}")
        self.draw_map()

    def find_routes(self):
        try:
            origin = self.label_to_site[self.origin.get()]
            destination = self.label_to_site[self.destination.get()]
            when = self.trip_time()
            k = int(self.k.get())
            overrides = self.travel_overrides()
        except (KeyError, ValueError) as error:
            messagebox.showerror("TBRGS", str(error))
            return
        model, method = self.selected_model(), self.selected_method()
        flow_site = self.flow_site.get()  # read Tk variables here: the job runs in another thread
        self.find_button.state(["disabled"])
        self.status.set(f"Finding routes with {SOURCE_LABELS.get(model, model)} "
                        f"(the first use of LSTM/GRU loads TensorFlow, which takes a few seconds)...")

        def job():
            result = self.system.find_routes(origin, destination, when, model=model, method=method, k=k,
                                             travel_overrides=overrides, flow_site=flow_site)
            estimates, _ = self.system.edge_estimates(model, when, self.system.travel_time_model(overrides),
                                                      flow_site)
            return result, estimates

        def done(value):
            self.find_button.state(["!disabled"])
            self.result, self.estimates = value
            self.selected_route = 0
            self.show_result()

        def failed(payload):
            self.find_button.state(["!disabled"])
            self.status.set("Route search failed.")
            self.worker._show_error(payload)

        self.worker.run(job, done, failed)

    def show_result(self):
        result = self.result
        self.route_table.delete(*self.route_table.get_children())
        for route in result.routes:
            self.route_table.insert("", "end", iid=str(route.rank - 1), values=(
                route.rank, f"{route.minutes:.1f}", f"{route.distance_km:.2f}", route.intersections,
                " -> ".join(str(s) for s in route.path)))
        when = pd.Timestamp(result.time_used)
        parts = [f"{len(result.routes)} route(s) from {self.site_labels[result.origin]} to "
                 f"{self.site_labels[result.destination]} at {when:%a %d %b %Y %H:%M}, flows from "
                 f"{SOURCE_LABELS.get(result.model, result.model)}, search {METHOD_LABELS[result.method]} "
                 f"({result.searches} searches, {result.nodes_created} nodes created, {result.search_ms:.1f} ms)."]
        if result.method not in OPTIMAL_METHODS:
            parts.append("This method does not guarantee the fastest route.")
        if result.note:
            parts.append(result.note)
        parts.extend(m for m in result.messages if "guarantee" not in m)
        self.summary.set(" ".join(parts))
        if result.routes:
            self.route_table.selection_set("0")
            self.status.set(f"Fastest route: {result.routes[0].minutes:.1f} min.")
        else:
            self.status.set("No route found.")
            self.leg_table.delete(*self.leg_table.get_children())
        self.draw_map()

    def _route_selected(self, _event=None):
        selection = self.route_table.selection()
        if not selection or self.result is None:
            return
        self.selected_route = int(selection[0])
        route = self.result.routes[self.selected_route]
        self.leg_table.delete(*self.leg_table.get_children())
        for i, leg in enumerate(route.edges, start=1):
            self.leg_table.insert("", "end", values=(
                i, self.site_labels[leg.start], self.site_labels[leg.end], f"{leg.distance_km:.2f}",
                f"{leg.flow_vph:.0f}", f"{leg.speed_kmh:.1f}", f"{leg.time_s:.0f}", leg.rule))
        self.draw_map()

    def open_browser_map(self):
        if self.result is None or not self.result.routes:
            messagebox.showinfo("TBRGS", "Find some routes first.")
            return
        try:
            from tbrgs.mapview import save_route_map
            path = save_route_map(self.system, self.result, self.estimates)
        except ImportError:
            messagebox.showerror("TBRGS", "The browser map needs the folium package: pip install folium")
            return
        webbrowser.open("file://" + os.path.abspath(path))
        self.status.set(f"Map saved to {path} and opened in the browser (map tiles need internet).")

    def save_trip_defaults(self):
        try:
            self.trip_time()
        except ValueError as error:
            messagebox.showerror("TBRGS", str(error))
            return
        self.config.save_values("gui", {
            "origin": self.label_to_site[self.origin.get()], "destination": self.label_to_site[self.destination.get()],
            "date": self.date.get().strip(), "time": self.time.get().strip(), "model": self.selected_model()})
        self.config.save_values("routing", {"k": int(self.k.get()), "method": self.selected_method()})
        self.status.set("Trip, model, method and k saved to config.ini.")

    # =========================================================================
    # Predictions tab
    # =========================================================================
    def _build_predictions_tab(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        tab = ttk.Frame(self.tabs, padding=6)
        self.tabs.add(tab, text="Predictions")
        controls = ttk.Frame(tab)
        controls.pack(fill="x")
        self.pred_site = tk.StringVar(value=self.origin.get())
        self.pred_detector = tk.StringVar()
        self.pred_date = tk.StringVar(value=self.config.get("gui", "date"))
        ttk.Label(controls, text="Site").pack(side="left")
        site_box = ttk.Combobox(controls, textvariable=self.pred_site, values=list(self.site_labels.values()),
                                width=34, state="readonly")
        site_box.pack(side="left", padx=4)
        site_box.bind("<<ComboboxSelected>>", lambda e: self._fill_detectors())
        ttk.Label(controls, text="Detector").pack(side="left", padx=(8, 0))
        self.detector_box = ttk.Combobox(controls, textvariable=self.pred_detector, width=36, state="readonly")
        self.detector_box.pack(side="left", padx=4)
        ttk.Label(controls, text="Day").pack(side="left", padx=(8, 0))
        ttk.Combobox(controls, textvariable=self.pred_date, values=DATES, width=11).pack(side="left", padx=4)
        ttk.Button(controls, text="Plot", command=self.plot_predictions).pack(side="left", padx=8)
        self._fill_detectors()
        self.pred_info = tk.StringVar(value="Plots the observed flow and the one-step-ahead prediction "
                                            "(15 minutes ahead) of every available model.")
        ttk.Label(tab, textvariable=self.pred_info, wraplength=950, justify="left").pack(anchor="w", pady=4)
        self.figure = Figure(figsize=(9, 4.6), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=tab)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _fill_detectors(self):
        site = self.label_to_site.get(self.pred_site.get())
        if site is None:
            return
        detectors = self.system.data.detectors
        rows = detectors[detectors["site"] == site]
        self.detector_labels = {f"{d}  {row['location']}": d for d, row in rows.iterrows()}
        self.detector_box["values"] = list(self.detector_labels)
        self.pred_detector.set(next(iter(self.detector_labels)))

    def plot_predictions(self):
        detector = self.detector_labels.get(self.pred_detector.get())
        day = self.pred_date.get().strip()
        try:
            pd.Timestamp(day)
        except ValueError:
            messagebox.showerror("TBRGS", "Enter the day as YYYY-MM-DD")
            return
        sources = [s for s in self.system.available_sources() if s != "observed"]
        self.status.set("Predicting a whole day for each model...")

        def job():
            series = {}
            for source in sources:
                series[source] = self.system.predictor.day_series(self.system.model(source), detector, day)
            return series

        def done(series):
            from tbrgs.metrics import mae
            self.figure.clear()
            axes = self.figure.add_subplot(111)
            first = next(iter(series.values()))
            axes.plot(first.index, first["observed"], color="black", linewidth=2, label="Observed")
            notes = []
            for source, frame in series.items():
                axes.plot(frame.index, frame["predicted"], linewidth=1.2, label=DISPLAY_NAMES.get(source, source))
                valid = frame.dropna()
                if len(valid):
                    notes.append(f"{DISPLAY_NAMES.get(source, source)} MAE {mae(valid['observed'], valid['predicted']):.1f}")
            axes.set_title(f"{self.pred_detector.get()} on {pd.Timestamp(first.index[0]):%a %d %b %Y}")
            axes.set_ylabel("Vehicles per 15 minutes")
            axes.set_xlabel("Time of day")
            axes.legend(loc="upper left", fontsize=8)
            axes.grid(alpha=0.3)
            import matplotlib.dates as mdates
            axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self.figure.tight_layout()
            self.figure_canvas.draw()
            when = pd.Timestamp(first.index[0])
            split = "test week" if when >= self.system.split.test[0] else \
                    "validation days" if when >= self.system.split.validation[0] else "training weeks"
            self.pred_info.set(f"This day is in the {split}. Mean absolute error over the day: " + ", ".join(notes))
            self.status.set("Prediction chart updated.")

        self.worker.run(job, done)

    # =========================================================================
    # Model comparison tab
    # =========================================================================
    def _build_comparison_tab(self):
        tab = ttk.Frame(self.tabs, padding=6)
        self.tabs.add(tab, text="Model comparison")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        # Buttons are packed first so they keep their full width; the label wraps instead.
        ttk.Button(top, text="Reload results", command=self.load_comparison).pack(side="right")
        ttk.Button(top, text="Run evaluation (about a minute)", command=self.run_evaluation).pack(side="right", padx=6)
        ttk.Label(top, text="Test week 25-31 Oct 2006 (never used in training). Lower MAE/RMSE/MAPE "
                            "and higher R2 are better.", wraplength=520).pack(side="left", fill="x", expand=True)
        columns = ("model", "MAE", "RMSE", "NRMSE", "MAPE", "R2", "train_s", "ms_per_1000")
        self.metric_table = ttk.Treeview(tab, columns=columns, show="headings", height=7)
        widths = {"model": 160, "train_s": 120, "ms_per_1000": 190}
        for column, heading in zip(columns, ("Model", "MAE", "RMSE", "NRMSE", "MAPE (%)", "R2",
                                              "Training time (s)", "Prediction (ms per 1,000)")):
            self.metric_table.heading(column, text=heading)
            self.metric_table.column(column, width=widths.get(column, 80),
                                     anchor="w" if column == "model" else "center")
        self.metric_table.pack(fill="x", pady=6)
        self.comparison_image_label = ttk.Label(tab)
        self.comparison_image_label.pack(fill="both", expand=True)
        self.load_comparison()

    def load_comparison(self):
        results_dir = self.config.path_of("paths", "results_dir")
        path = os.path.join(results_dir, "metrics.csv")
        self.metric_table.delete(*self.metric_table.get_children())
        if not os.path.exists(path):
            self.metric_table.insert("", "end", values=("No results yet: run python evaluate.py",) + ("",) * 7)
            return
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            self.metric_table.insert("", "end", values=(
                row["model"], f"{row['MAE']:.2f}", f"{row['RMSE']:.2f}", f"{row['NRMSE']:.3f}",
                f"{row['MAPE']:.1f}", f"{row['R2']:.3f}",
                "" if pd.isna(row.get("train_seconds")) else f"{row['train_seconds']:.0f}",
                "" if pd.isna(row.get("ms_per_1000")) else f"{row['ms_per_1000']:.1f}"))
        image_path = os.path.join(results_dir, "model_comparison.png")
        if os.path.exists(image_path):
            try:
                self._comparison_image = tk.PhotoImage(file=image_path)
                factor = max(1, int(self._comparison_image.width() / 1000) + 1)
                if factor > 1:
                    self._comparison_image = self._comparison_image.subsample(factor, factor)
                self.comparison_image_label.configure(image=self._comparison_image)
            except tk.TclError:
                pass

    def run_evaluation(self):
        self.status.set("Evaluating every model on the test week...")

        def job():
            import evaluate
            evaluate.run(self.config, figures=True, log=lambda line: None)

        def done(_):
            self.load_comparison()
            self.status.set("Evaluation finished; results/metrics.csv updated.")

        self.worker.run(job, done)

    # =========================================================================
    # Train tab
    # =========================================================================
    def _build_train_tab(self):
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="Train")
        ttk.Label(tab, text="Retrain a model with new settings. Training uses 1-21 Oct, early stopping uses "
                            "22-24 Oct, and the test week (25-31 Oct) is never seen. LSTM/GRU take about "
                            "10-20 minutes on a laptop CPU; XGBoost under a minute.", wraplength=950,
                  justify="left").pack(anchor="w")
        form = ttk.Frame(tab)
        form.pack(anchor="w", pady=8)
        self.train_model = tk.StringVar(value="xgboost")
        ttk.Label(form, text="Model").grid(row=0, column=0, sticky="w")
        box = ttk.Combobox(form, textvariable=self.train_model, values=list(MODEL_CLASSES), state="readonly", width=10)
        box.grid(row=0, column=1, sticky="w", padx=4)
        box.bind("<<ComboboxSelected>>", lambda e: self._fill_train_form())
        self.train_fields = ttk.Frame(form)
        self.train_fields.grid(row=1, column=0, columnspan=6, sticky="w", pady=6)
        self.train_vars = {}
        self._fill_train_form()
        buttons = ttk.Frame(tab)
        buttons.pack(anchor="w")
        self.train_button = ttk.Button(buttons, text="Train and save", command=self.start_training)
        self.train_button.pack(side="left")
        ttk.Button(buttons, text="Save settings as defaults", command=self.save_train_defaults).pack(side="left", padx=6)
        self.train_log = tk.Text(tab, height=18, width=120, font=("TkFixedFont", 9))
        self.train_log.pack(fill="both", expand=True, pady=6)

    def _fill_train_form(self):
        for child in self.train_fields.winfo_children():
            child.destroy()
        name = self.train_model.get()
        settings = self.config.section(name)
        settings = {"lags": self.config.get("features", "lags"), **settings}
        self.train_vars = {}
        for i, (key, value) in enumerate(settings.items()):
            ttk.Label(self.train_fields, text=key).grid(row=i // 4, column=(i % 4) * 2, sticky="e", padx=(8, 2))
            var = tk.StringVar(value=value)
            ttk.Entry(self.train_fields, textvariable=var, width=10).grid(row=i // 4, column=(i % 4) * 2 + 1,
                                                                         sticky="w")
            self.train_vars[key] = var

    def start_training(self):
        name = self.train_model.get()
        try:
            overrides = {key: (float(var.get()) if "." in var.get() else int(var.get()))
                         for key, var in self.train_vars.items()}
        except ValueError:
            messagebox.showerror("TBRGS", "Every setting must be a number.")
            return
        self.train_button.state(["disabled"])
        self.train_log.delete("1.0", "end")
        self.status.set(f"Training {name}...")

        def log(line):
            self.worker.call_soon(self._append_log, line)

        def job():
            from tbrgs.training import train_model
            from tbrgs.metrics import all_metrics
            model, prepared = train_model(self.config, name, None, overrides, log=log)
            test = prepared.windows["test"]
            actual = prepared.scaler.inverse(test.y, test.detectors)
            scores = all_metrics(actual, prepared.scaler.inverse(model.predict_windows(test), test.detectors))
            log(f"Test week: MAE {scores['MAE']:.2f}  RMSE {scores['RMSE']:.2f}  MAPE {scores['MAPE']:.1f}%  "
                f"R2 {scores['R2']:.3f}")
            return name

        def done(trained):
            self.train_button.state(["!disabled"])
            self.system.forget_model(trained)
            self.refresh_model_list(self.selected_model())
            self.status.set(f"{trained} retrained and saved. Run the evaluation to update the comparison.")

        def failed(payload):
            self.train_button.state(["!disabled"])
            self.worker._show_error(payload)

        self.worker.run(job, done, failed)

    def _append_log(self, line):
        self.train_log.insert("end", line + "\n")
        self.train_log.see("end")

    def save_train_defaults(self):
        name = self.train_model.get()
        values = {k: v.get() for k, v in self.train_vars.items() if k != "lags"}
        self.config.save_values(name, values)
        if "lags" in self.train_vars:
            self.config.save_values("features", {"lags": self.train_vars["lags"].get()})
        self.status.set(f"[{name}] settings saved to config.ini.")

    # =========================================================================
    # Settings tab
    # =========================================================================
    def _build_settings_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="Settings")
        ttk.Label(tab, text="Travel-time settings (Traffic Flow to Travel Time Conversion v1.0). Changes apply to "
                            "the next route search.", wraplength=950).pack(anchor="w")
        form = ttk.Frame(tab)
        form.pack(anchor="w", pady=8)
        self.travel_vars = {}
        labels = {"speed_limit_kmh": "Speed limit (km/h)", "capacity_flow_vph": "Flow at capacity (veh/h)",
                  "capacity_speed_kmh": "Speed at capacity (km/h)",
                  "intersection_delay_s": "Delay per controlled intersection (s)"}
        for i, (key, label) in enumerate(labels.items()):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=self.config.get("travel_time", key))
            ttk.Entry(form, textvariable=var, width=10).grid(row=i, column=1, sticky="w", padx=6)
            self.travel_vars[key] = var
        self.flow_site = tk.StringVar(value=self.config.get("travel_time", "flow_site"))
        ttk.Label(form, text="Flow for a link A -> B comes from").grid(row=len(labels), column=0, sticky="w", pady=(8, 2))
        ttk.Radiobutton(form, text="site B's detector on the approach from A (brief)", value="destination",
                        variable=self.flow_site).grid(row=len(labels), column=1, sticky="w")
        ttk.Radiobutton(form, text="site A's detector on the approach that leads to B (conversion note)",
                        value="origin", variable=self.flow_site).grid(row=len(labels) + 1, column=1, sticky="w")
        ttk.Button(tab, text="Save settings as defaults", command=self.save_travel_defaults).pack(anchor="w")
        net = self.system.network().summary()
        ttk.Label(tab, justify="left", wraplength=900, text=(
            f"\nRoad network: {net['sites']} SCATS sites, {net['links']} two-way links "
            f"({net['manual_links']} added by hand, see data/manual_links.csv), mean link {net['mean_link_km']} km. "
            f"Every site can reach every other site: {net['strongly_connected']}.")).pack(anchor="w")

    def travel_overrides(self) -> dict:
        values = {}
        for key, var in self.travel_vars.items():
            try:
                values[key] = float(var.get())
            except ValueError:
                raise ValueError(f"Settings: '{var.get()}' is not a number") from None
        self.system.travel_time_model(values)  # validates the combination
        return values

    def save_travel_defaults(self):
        try:
            values = self.travel_overrides()
        except ValueError as error:
            messagebox.showerror("TBRGS", str(error))
            return
        values["flow_site"] = self.flow_site.get()
        self.config.save_values("travel_time", values)
        self.status.set("Travel-time settings saved to config.ini.")


def main():
    root = tk.Tk()
    root.withdraw()
    splash = tk.Toplevel(root)
    splash.title("TBRGS")
    ttk.Label(splash, text="Loading the SCATS data and road network...", padding=20).pack()
    splash.update()
    try:
        system = TBRGS(load_config())
    except Exception as error:
        splash.destroy()
        messagebox.showerror("TBRGS", f"Could not start: {error}")
        root.destroy()
        return
    splash.destroy()
    root.deiconify()
    TBRGSApp(root, system)
    root.mainloop()


if __name__ == "__main__":
    main()
