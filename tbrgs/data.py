"""Task 1: data processing for the VicRoads SCATS dataset (Boroondara, October 2006).

Owner: Lim Yon (105811100)

The raw sheet has one row per detector per day and 96 columns, V00 to V95,
holding the number of vehicles counted in each 15-minute interval
(V00 = 00:00-00:15, V95 = 23:45-24:00). This module turns it into data
structures the models and the route finder can use:

    load_raw()        read the 'Data' sheet of "Scats Data October 2006.xls"
    TrafficData       a time-indexed table: one row per 15-minute interval,
                      one column per detector, NaN where a day is missing;
                      plus tables describing every detector and every site
    split_dates()     the chronological train / validation / test split
    SeriesScaler      per-detector scaling, fitted on the training period only
    make_windows()    sliding windows: the last `lags` intervals -> the next one

Data problems found while exploring the sheet, and how they are handled:

* Site 4335 has two detectors with the same Location text
  ("HIGH_ST NE of CHARLES_ST"); they differ in "VR Internal Loc" (2 and 6)
  and in their counts. A detector is therefore identified by
  (SCATS Number, VR Internal Loc), which gives 140 detectors, not 139.
* 31 rows (detector "AUBURN_RD N of BURWOOD_RD" at site 4266) have latitude
  and longitude 0. Their counts look normal, so the counts are kept and the
  detector takes the average position of the other detectors at its site.
* Some detectors miss whole days (34 of 140 have fewer than 31 days; detector
  "CHURCH_ST SW of BARKERS_RD" has only 2). Missing days stay NaN: they are
  never filled with zeros, and no training window may contain a NaN.
* The "Date" column carries a time of 00:15 on every row. Only the date part
  is used; interval i starts at date + 15 * i minutes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

VOLUME_COLUMNS = [f"V{i:02d}" for i in range(96)]
INTERVAL = pd.Timedelta(minutes=15)
INTERVALS_PER_DAY = 96
# Bump this when the processing changes, so old cache files are not reused.
PROCESSING_VERSION = 3

# "WARRIGAL_RD N of TOORAK_RD" -> road "WARRIGAL_RD", direction "N", cross road "TOORAK_RD"
LOCATION_PATTERN = re.compile(r"^\s*(.+?)\s+(N|S|E|W|NE|NW|SE|SW)\s+of\s+(.+?)\s*$", re.IGNORECASE)

REQUIRED_COLUMNS = ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE",
                    "VR Internal Loc", "Date", *VOLUME_COLUMNS]


# ---------------------------------------------------------------------------
# Reading the raw sheet
# ---------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    """Read the 'Data' sheet. The column names are on the sheet's second row."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: '{path}'")
    try:
        raw = pd.read_excel(path, sheet_name="Data", header=1)
    except ImportError as error:  # .xls files need the xlrd package
        raise ImportError("Reading the .xls dataset needs xlrd: pip install xlrd") from error
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"The 'Data' sheet is missing columns: {missing[:5]}")
    return raw


def parse_location(text: str):
    """Split "ROAD DIR of CROSS" into (road, direction, cross road); None if it does not match."""
    match = LOCATION_PATTERN.match(str(text))
    if not match:
        return None
    return match.group(1).strip().upper(), match.group(2).upper(), match.group(3).strip().upper()


def read_site_listing(path: str) -> dict:
    """Return {site number: VicRoads description} from the SCATS site listing workbook.

    The workbook contains a defined name that xlrd cannot evaluate, so name
    evaluation is switched off while it is opened. The descriptions are only
    shown to the user, so any failure returns an empty dict.
    """
    try:
        import xlrd
        import xlrd.book
        original = xlrd.book.Book.names_epilogue
        xlrd.book.Book.names_epilogue = lambda self: None
        try:
            with open(os.devnull, "w") as quiet:
                book = xlrd.open_workbook(path, logfile=quiet)
        finally:
            xlrd.book.Book.names_epilogue = original
        sheet = book.sheet_by_name("SCATS Site Numbers")
        listing = {}
        for row in range(sheet.nrows):
            number, description = sheet.cell_value(row, 0), sheet.cell_value(row, 1)
            if isinstance(number, float) and number == int(number):
                listing[int(number)] = str(description).strip()
        return listing
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# The processed dataset
# ---------------------------------------------------------------------------
@dataclass
class TrafficData:
    """The processed dataset.

    flows      DataFrame, index = start time of each 15-minute interval (every
               interval from the first to the last day), columns = detector
               ids such as "2000-1"; values are vehicles per 15 minutes, NaN
               where the detector has no data.
    detectors  DataFrame indexed by detector id: site, location, vr_loc,
               road, direction, cross_road, latitude, longitude, days.
    sites      DataFrame indexed by SCATS number: name, description,
               latitude, longitude, detectors (list of ids).
    report     dict of counts describing what the cleaning did.
    """

    flows: pd.DataFrame
    detectors: pd.DataFrame
    sites: pd.DataFrame
    report: dict

    @classmethod
    def from_raw(cls, raw: pd.DataFrame, site_listing: dict | None = None) -> "TrafficData":
        report = {"raw_rows": int(len(raw))}
        table = raw[REQUIRED_COLUMNS].copy()
        table["site"] = table["SCATS Number"].astype(int)
        table["vr_loc"] = table["VR Internal Loc"].astype(int)
        table["day"] = pd.to_datetime(table["Date"]).dt.normalize()
        table["detector"] = table["site"].astype(str) + "-" + table["vr_loc"].astype(str)

        duplicated = table.duplicated(["detector", "day"], keep="first")
        report["duplicate_rows_dropped"] = int(duplicated.sum())
        table = table.loc[~duplicated]

        # Coordinates of 0 mean "unknown".
        for column in ("NB_LATITUDE", "NB_LONGITUDE"):
            table[column] = table[column].where(table[column] != 0)
        report["rows_with_zero_coordinates"] = int(table["NB_LATITUDE"].isna().sum())

        detectors = cls._detector_table(table)
        sites = cls._site_table(table, detectors, site_listing or {})
        flows = cls._flow_table(table)
        report.update({
            "detectors": int(flows.shape[1]),
            "sites": int(len(sites)),
            "intervals": int(flows.shape[0]),
            "missing_values": int(flows.isna().sum().sum()),
            "first_interval": str(flows.index[0]),
            "last_interval": str(flows.index[-1]),
        })
        return cls(flows=flows, detectors=detectors, sites=sites, report=report)

    @staticmethod
    def _detector_table(table: pd.DataFrame) -> pd.DataFrame:
        grouped = table.groupby("detector", sort=True)
        detectors = pd.DataFrame({
            "site": grouped["site"].first(),
            "location": grouped["Location"].first(),
            "vr_loc": grouped["vr_loc"].first(),
            "latitude": grouped["NB_LATITUDE"].mean(),
            "longitude": grouped["NB_LONGITUDE"].mean(),
            "days": grouped["day"].nunique(),
        })
        parsed = detectors["location"].map(parse_location)
        detectors["road"] = parsed.map(lambda p: p[0] if p else None)
        detectors["direction"] = parsed.map(lambda p: p[1] if p else None)
        detectors["cross_road"] = parsed.map(lambda p: p[2] if p else None)
        # A detector without coordinates takes the mean position of its site.
        for column in ("latitude", "longitude"):
            site_mean = detectors.groupby("site")[column].transform("mean")
            detectors[column] = detectors[column].fillna(site_mean)
        return detectors

    @staticmethod
    def _site_table(table: pd.DataFrame, detectors: pd.DataFrame, listing: dict) -> pd.DataFrame:
        rows = []
        for site, group in detectors.groupby("site", sort=True):
            # Name the site like the brief does ("WARRIGAL_RD/TOORAK_RD"): the
            # road and cross road of the site's first row in the sheet.
            first_location = table.loc[table["site"] == site, "Location"].iloc[0]
            parsed = parse_location(first_location)
            name = f"{parsed[0]}/{parsed[2]}" if parsed else str(first_location)
            rows.append({
                "site": int(site),
                "name": name,
                "description": listing.get(int(site), ""),
                "latitude": float(group["latitude"].mean()),
                "longitude": float(group["longitude"].mean()),
                "detectors": list(group.index),
            })
        return pd.DataFrame(rows).set_index("site")

    @staticmethod
    def _flow_table(table: pd.DataFrame) -> pd.DataFrame:
        long = table.melt(id_vars=["detector", "day"], value_vars=VOLUME_COLUMNS,
                          var_name="interval", value_name="flow")
        long["time"] = long["day"] + INTERVAL * long["interval"].str[1:].astype(int)
        flows = long.pivot(index="time", columns="detector", values="flow").astype(float)
        full_index = pd.date_range(flows.index.min().normalize(),
                                   flows.index.max().normalize() + pd.Timedelta(days=1) - INTERVAL,
                                   freq=INTERVAL)
        flows = flows.reindex(full_index).sort_index(axis=1)
        flows.index.name = "time"
        flows.columns.name = None
        return flows

    # ----- convenience ------------------------------------------------------
    def detectors_at(self, site: int) -> list:
        return list(self.sites.loc[site, "detectors"])

    def site_label(self, site: int) -> str:
        return f"{site} {self.sites.loc[site, 'name']}"


def _file_signature(path: str) -> str:
    stat = os.stat(path)
    text = f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}|{PROCESSING_VERSION}"
    return hashlib.md5(text.encode()).hexdigest()


def load_traffic_data(path: str, cache_dir: str | None = None,
                      site_listing_path: str | None = None) -> TrafficData:
    """Load and process the dataset, reusing a cached copy when the file has not changed.

    Reading the .xls file takes several seconds, so the processed result is
    pickled in cache_dir. The cache is rebuilt when the dataset file changes.
    """
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"traffic_{_file_signature(path)}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as file:
                    return pickle.load(file)
            except Exception:
                pass  # a damaged or incompatible cache is simply rebuilt
    listing = read_site_listing(site_listing_path) if site_listing_path else {}
    data = TrafficData.from_raw(load_raw(path), listing)
    if cache_file:
        with open(cache_file, "wb") as file:
            pickle.dump(data, file)
    return data


# ---------------------------------------------------------------------------
# Splitting, scaling and windows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    """Inclusive date ranges (as Timestamps) for the three parts of the data."""
    train: tuple
    validation: tuple
    test: tuple

    def part(self, name: str) -> tuple:
        return getattr(self, name)


def split_dates(flows: pd.DataFrame, train_end: str, validation_end: str) -> Split:
    """Chronological split by the date of the predicted (target) interval.

    The first `lags` intervals of a later part use flows from the end of the
    earlier part as inputs, exactly as a deployed system would; the targets
    never overlap.
    """
    first, last = flows.index[0], flows.index[-1]
    train_stop = pd.Timestamp(train_end) + pd.Timedelta(days=1) - INTERVAL
    validation_stop = pd.Timestamp(validation_end) + pd.Timedelta(days=1) - INTERVAL
    if not first < train_stop < validation_stop < last:
        raise ValueError("train_end and validation_end must fall inside the dataset, in that order")
    return Split(train=(first, train_stop),
                 validation=(train_stop + INTERVAL, validation_stop),
                 test=(validation_stop + INTERVAL, last))


class SeriesScaler:
    """Scale each detector by the largest flow it recorded in the training period.

    Detectors carry very different volumes (a side street may peak at 30
    vehicles per 15 minutes and an arterial at 500), so one global model only
    works if each series is put on a common 0-1 scale. Fitting on training
    data only keeps test information out of the models.
    """

    def __init__(self, scales: dict | None = None):
        self.scales = dict(scales or {})

    def fit(self, flows: pd.DataFrame, end: pd.Timestamp) -> "SeriesScaler":
        peak = flows.loc[:end].max(axis=0, skipna=True)
        self.scales = {column: (float(value) if np.isfinite(value) and value > 0 else 1.0)
                       for column, value in peak.items()}
        return self

    def scale_array(self, columns) -> np.ndarray:
        return np.array([self.scales.get(column, 1.0) for column in columns], dtype=float)

    def transform(self, flows: pd.DataFrame) -> pd.DataFrame:
        return flows / self.scale_array(flows.columns)

    def inverse(self, values: np.ndarray, detectors) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale_array(detectors)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.scales, file, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "SeriesScaler":
        with open(path, encoding="utf-8") as file:
            return cls(json.load(file))


def time_features(times) -> np.ndarray:
    """Features describing when an interval starts: (sin, cos) of the time of day and a weekend flag."""
    times = pd.DatetimeIndex(times)
    minutes = times.hour * 60 + times.minute
    angle = 2 * np.pi * np.asarray(minutes, dtype=float) / (24 * 60)
    weekend = (np.asarray(times.day_of_week) >= 5).astype(float)
    return np.column_stack([np.sin(angle), np.cos(angle), weekend])


N_STEP_FEATURES = 4  # scaled flow + the three time features, for every lag step


@dataclass
class WindowSet:
    """Inputs and targets for one part of the data.

    X        float32 array (n, lags, 4): for each of the `lags` past intervals,
             the scaled flow and the three time features of that interval.
    y        float32 array (n,): the scaled flow of the next interval (target).
    detectors  array (n,) of detector ids.
    times    array (n,) of target interval start times.
    """

    X: np.ndarray
    y: np.ndarray
    detectors: np.ndarray
    times: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    @property
    def lags(self) -> int:
        return self.X.shape[1]

    def flat(self) -> np.ndarray:
        """Tabular features for tree models (see flat_features)."""
        return flat_features(self.X, self.times)

    def subset(self, mask) -> "WindowSet":
        return WindowSet(self.X[mask], self.y[mask], self.detectors[mask], self.times[mask])


def flat_features(X: np.ndarray, target_times) -> np.ndarray:
    """One row of tabular features per window, for models that cannot read sequences.

    Columns: flow(t-lags) ... flow(t-1), then sin and cos of the target's time
    of day and the target's weekend flag. This is the same information the
    recurrent models get (they see the time features of every input step,
    from which the target time follows), so the comparison stays fair.
    """
    return np.column_stack([np.asarray(X)[:, :, 0], time_features(target_times)]).astype(np.float32)


def flat_feature_names(lags: int) -> list:
    return [f"flow_t-{lags - i}" for i in range(lags)] + ["sin_time", "cos_time", "weekend"]


def make_windows(scaled: pd.DataFrame, lags: int, start=None, end=None,
                 detectors=None) -> WindowSet:
    """Build every complete window whose target interval lies in [start, end].

    scaled is a flow table already divided by SeriesScaler. A window is kept
    only if its `lags` inputs and its target are all present (no NaN), so
    missing days never become fake zeros.
    """
    if detectors is not None:
        scaled = scaled[list(detectors)]
    values = scaled.to_numpy(dtype=np.float32)          # (T, S)
    times = scaled.index
    n_times, n_series = values.shape
    if n_times <= lags:
        raise ValueError("Not enough intervals for the requested number of lags")

    # windows[t] holds values[t : t + lags + 1] for every series: lags inputs + 1 target.
    windows = np.lib.stride_tricks.sliding_window_view(values, lags + 1, axis=0)  # (T-lags, S, lags+1)
    target_times = times[lags:]
    keep_time = np.ones(len(target_times), dtype=bool)
    if start is not None:
        keep_time &= np.asarray(target_times >= pd.Timestamp(start))
    if end is not None:
        keep_time &= np.asarray(target_times <= pd.Timestamp(end))
    windows = windows[keep_time]                          # (n_t, S, lags+1)
    target_times = target_times[keep_time]

    complete = ~np.isnan(windows).any(axis=2)             # (n_t, S)
    t_index, s_index = np.nonzero(complete)

    step_features = time_features(times)                  # (T, 3)
    step_windows = np.lib.stride_tricks.sliding_window_view(step_features, lags, axis=0)  # (T-lags+1, 3, lags)
    step_windows = np.transpose(step_windows[:-1][keep_time], (0, 2, 1))  # (n_t, lags, 3)

    flows_in = windows[t_index, s_index, :lags]           # (n, lags)
    X = np.empty((len(t_index), lags, N_STEP_FEATURES), dtype=np.float32)
    X[:, :, 0] = flows_in
    X[:, :, 1:] = step_windows[t_index]
    y = windows[t_index, s_index, lags].astype(np.float32)
    columns = np.asarray(scaled.columns)
    return WindowSet(X=X, y=y, detectors=columns[s_index],
                     times=np.asarray(target_times)[t_index])
