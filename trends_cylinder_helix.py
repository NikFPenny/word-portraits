"""
trends_cylinder_helix.py

Takes a list of words, averages their Google Trends interest-over-time,
and maps that average onto a helix wrapped around a cylinder.

Pipeline:
    words + date range  -->  Google Trends (via pytrends)  -->  averaged
    0-100 interest series, sampled every 2 months  -->  helix coordinates
    (one revolution per year, radius driven by trend strength)  -->  JSON

Requires: pip install pytrends pandas --break-system-packages

NOTE ON NETWORK ACCESS:
This script must be run somewhere that can reach trends.google.com
(pytrends scrapes Google Trends directly, there's no official API key).
It will NOT work from a sandboxed environment with a restricted domain
allowlist. The geometry/math side (resampling, helix generation) is
fully unit-testable offline -- see `_demo_with_mock_data()` at the
bottom of this file.
"""

import json
import math
from datetime import datetime
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Fetching + averaging Google Trends data
# ---------------------------------------------------------------------------

def fetch_trends_series(
    words: List[str],
    start_date: str,
    end_date: str,
    geo: str = "",
) -> pd.Series:
    """
    Returns a single pandas Series (DatetimeIndex -> averaged 0-100 interest)
    representing the average global interest across all `words` over time.

    Google Trends only allows comparing up to 5 terms in one request. If more
    than 5 words are given, they're queried in batches of 4 plus a shared
    "anchor" word (the first word in the list, repeated in every batch), and
    each batch is rescaled so the anchor's values line up -- putting all
    words on one common relative scale before averaging.
    """
    from pytrends.request import TrendReq

    # retries/backoff help somewhat with Google's rate limiting, but a 429
    # can still happen -- see load_trends_from_csv() below for a fallback
    # that sidesteps live scraping entirely.
    pytrends = TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=1.5)
    timeframe = f"{start_date} {end_date}"

    if len(words) <= 5:
        pytrends.build_payload(kw_list=words, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if df.empty:
            raise ValueError("Google Trends returned no data for these words/timeframe.")
        df = df[words]  # drop 'isPartial' column
        return df.mean(axis=1)

    # --- more than 5 words: batch with a shared anchor term ---
    anchor = words[0]
    others = words[1:]
    batches = [others[i:i + 4] for i in range(0, len(others), 4)]

    import time

    combined: Optional[pd.DataFrame] = None
    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(5)  # space out requests to reduce 429 rate-limiting
        kw_list = [anchor] + batch
        pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if df.empty:
            continue
        df = df[kw_list]

        # rescale this batch so its anchor column matches the first batch's
        # anchor column (avoid divide-by-zero by falling back to 1.0)
        if combined is None:
            combined = df
        else:
            ref_anchor = combined[anchor]
            this_anchor = df[anchor]
            # align on shared index, compute a single scale factor
            shared_idx = ref_anchor.index.intersection(this_anchor.index)
            denom = this_anchor.loc[shared_idx].mean()
            scale = (ref_anchor.loc[shared_idx].mean() / denom) if denom else 1.0
            df = df * scale
            combined = combined.join(df.drop(columns=[anchor]), how="outer")

    if combined is None:
        raise ValueError("Google Trends returned no data for these words/timeframe.")

    return combined[words].mean(axis=1)


# ---------------------------------------------------------------------------
# 1b. Reliable fallback: read a CSV exported manually from Google Trends
# ---------------------------------------------------------------------------
#
# If fetch_trends_series() keeps hitting 429 (Google rate-limiting the
# scraper), this sidesteps the problem entirely:
#
#   1. Go to https://trends.google.com/trends/explore
#   2. Enter your words (up to 5 at once) and set the date range to match
#      START_DATE / END_DATE below.
#   3. Click the download icon (top-right of the "Interest over time" chart)
#      to get a CSV.
#   4. If you have more than 5 words, repeat with different groups of words,
#      including one repeated "anchor" word in every group, and pass all the
#      CSV paths to load_trends_from_csv() -- it rescales batches using the
#      anchor the same way fetch_trends_series() does.
#   5. Point CSV_PATHS at the file(s) and set MODE = "csv" in the config
#      block at the bottom of this file.

def load_trends_from_csv(csv_paths: List[str], words: List[str]) -> pd.Series:
    """
    Reads one or more CSVs exported from Google Trends' "Interest over time"
    chart and returns a single averaged 0-100 Series, same shape as
    fetch_trends_series().

    Google's export format varies depending on which page you download from.
    Two known formats are handled automatically:

      Format A (multi-term compare page, "explore" tool):
        Category: All categories

        Month,word1: (Worldwide),word2: (Worldwide)
        2020-01,45,32

      Format B (single-term page, "Time" column):
        "Time","word1"
        "2020-01-01",48

    If you exported multiple files because you have >5 words, include one
    shared "anchor" word (the first word in your `words` list) in every
    export, and this function will rescale each file against the first
    using that anchor column before averaging everything together.
    """
    anchor = words[0]
    combined: Optional[pd.DataFrame] = None

    for path in csv_paths:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        # auto-detect which line is the real header row (first column name
        # is Day/Week/Month/Time), since Google's export preamble varies
        header_idx = 0
        for i, line in enumerate(lines):
            first_field = line.split(",")[0].strip().strip('"')
            if first_field in ("Day", "Week", "Month", "Time"):
                header_idx = i
                break

        df = pd.read_csv(path, skiprows=header_idx)
        date_col = df.columns[0]  # "Month" / "Week" / "Day" / "Time"
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # column headers may be "word: (Worldwide)" or just "word" -- strip
        # any trailing ": (...)" annotation either way
        df.columns = [c.split(":")[0].strip() for c in df.columns]

        # Google marks low-volume points as "<1" -- coerce to numeric
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if combined is None:
            combined = df
        else:
            ref_anchor = combined[anchor]
            this_anchor = df[anchor]
            shared_idx = ref_anchor.index.intersection(this_anchor.index)
            denom = this_anchor.loc[shared_idx].mean()
            scale = (ref_anchor.loc[shared_idx].mean() / denom) if denom else 1.0
            df = df * scale
            combined = combined.join(
                df.drop(columns=[anchor], errors="ignore"), how="outer"
            )

    missing = [w for w in words if w not in combined.columns]
    if missing:
        raise ValueError(
            f"These words weren't found in the CSV column headers: {missing}. "
            f"Found columns: {list(combined.columns)}"
        )

    return combined[words].mean(axis=1)


# ---------------------------------------------------------------------------
# 2. Resampling to exact 2-month points
# ---------------------------------------------------------------------------

def resample_bimonthly(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    """
    Reindexes `series` onto exact 2-month-spaced timestamps from start_date
    to end_date (inclusive of start, extending to cover end), using linear
    interpolation for points that fall between the source series' native
    granularity (Trends returns daily/weekly/monthly depending on range).
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    target_dates = []
    d = start
    while d <= end:
        target_dates.append(d)
        d = d + pd.DateOffset(months=2)
    if target_dates[-1] < end:
        target_dates.append(end)

    target_index = pd.DatetimeIndex(target_dates)

    # combine source + target index so interpolation has real data to work from,
    # then pull out just the target points
    full_index = series.index.union(target_index)
    resampled = series.reindex(full_index).interpolate(method="time").reindex(target_index)

    # edge fill in case target points fall outside the source series' range
    resampled = resampled.ffill().bfill()
    return resampled


# ---------------------------------------------------------------------------
# 3. Mapping averaged values onto a cylindrical helix
# ---------------------------------------------------------------------------

def values_to_helix_coordinates(
    values: List[float],
    cylinder_height: float,
    cylinder_diameter: float,
    max_distance: float,
) -> List[List[float]]:
    """
    Converts a list of 0-100 interest values (one per 2-month point, evenly
    spaced in time) into [x, y, z] coordinates on a helix wrapped around a
    cylinder.

    - One full revolution (360 degrees) = 12 months = 6 points -> 60 degrees/point
    - Radius grows from the cylinder's surface (radius = diameter/2) outward
      by up to `max_distance`, proportional to the interest value (0-100)
    - z runs evenly from 0 to cylinder_height across all points
    """
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 points to build a helix.")

    base_radius = cylinder_diameter / 2.0
    angle_step = (2 * math.pi) / 6  # 60 degrees per 2-month point, 6 points/year

    coords = []
    for i, val in enumerate(values):
        theta = i * angle_step
        r = base_radius + (val / 100.0) * max_distance
        z = (i / (n - 1)) * cylinder_height

        x = r * math.cos(theta)
        y = r * math.sin(theta)
        coords.append([x, y, z])

    return coords


# ---------------------------------------------------------------------------
# 4. Top-level function
# ---------------------------------------------------------------------------

def trends_to_cylinder_helix(
    words: List[str],
    start_date: str,
    end_date: str,
    cylinder_height: float,
    cylinder_diameter: float,
    max_distance: float,
    geo: str = "",
) -> str:
    """
    Full pipeline: words + date range -> Google Trends -> averaged series ->
    resampled to 2-month points -> helix coordinates -> JSON string.

    Returns a JSON string: a list of [x, y, z] lists.
    """
    raw_series = fetch_trends_series(words, start_date, end_date, geo=geo)
    bimonthly = resample_bimonthly(raw_series, start_date, end_date)
    coords = values_to_helix_coordinates(
        bimonthly.tolist(), cylinder_height, cylinder_diameter, max_distance
    )
    return json.dumps(coords, indent=2)


# ---------------------------------------------------------------------------
# 5. Offline demo / self-test (no network required)
# ---------------------------------------------------------------------------

def _demo_with_mock_data():
    """
    Exercises resample_bimonthly() and values_to_helix_coordinates() with a
    synthetic trends-like series, so the geometry logic can be verified
    without hitting Google Trends. Useful for confirming your environment
    (pandas etc.) is set up correctly before trying a real fetch.
    """
    import numpy as np

    start_date, end_date = "2020-01-01", "2023-01-01"
    dates = pd.date_range(start_date, end_date, freq="W")
    # fake seasonal + growth trend, clipped to 0-100 like real Trends data
    t = np.linspace(0, 3, len(dates))
    values = 50 + 40 * np.sin(2 * math.pi * t) + 5 * t
    values = np.clip(values, 0, 100)
    mock_series = pd.Series(values, index=dates)

    bimonthly = resample_bimonthly(mock_series, start_date, end_date)
    print(f"[DEMO] Resampled to {len(bimonthly)} points, every 2 months:")
    print(bimonthly)

    coords = values_to_helix_coordinates(
        bimonthly.tolist(),
        cylinder_height=100.0,
        cylinder_diameter=20.0,
        max_distance=15.0,
    )
    print(f"\n[DEMO] Generated {len(coords)} coordinates:")
    for c in coords:
        print([round(v, 3) for v in c])

    return coords


# ---------------------------------------------------------------------------
# 6. RUN THIS FILE DIRECTLY -- edit the config below, then press Run in VS Code
# ---------------------------------------------------------------------------
#
# In VS Code:
#   1. Open this folder in VS Code.
#   2. Install the "Python" extension (Microsoft) if you haven't already.
#   3. Bottom-right corner / Ctrl+Shift+P -> "Python: Select Interpreter"
#      -> pick the Python you installed (the one `python --version` showed).
#   4. Open a terminal (Ctrl+`) and run:
#         python -m pip install pytrends pandas
#      (or just click the "Run Python File" play button in the top-right,
#      then install any missing packages it complains about the same way)
#   5. Edit MODE / WORDS / dates / cylinder dims below.
#   6. Click the "Run Python File" play button (top-right), or press
#      Ctrl+F5, or just run `python trends_cylinder_helix.py` in the terminal.
#
# Output gets written to helix_output.json next to this script.

MODE = "mock"   # one of: "mock", "live", "csv"
                # "mock" -- synthetic data, no network needed, just to sanity
                #           check your environment
                # "live" -- pytrends fetches directly from Google Trends.
                #           Can hit 429 rate-limit errors -- Google actively
                #           blocks this kind of scraping and there's no
                #           reliable fix, just retry later / less often.
                # "csv"  -- reads CSV(s) you download by hand from
                #           trends.google.com/trends/explore. Slower to set
                #           up but always works. See load_trends_from_csv()
                #           above for the export instructions.

WORDS = ["sculpture", "generative art", "parametric design"]
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

CSV_PATHS = ["trends_export.csv"]  # only used when MODE == "csv"

CYLINDER_HEIGHT = 200.0     # mm (or whatever unit your CAD pipeline uses)
CYLINDER_DIAMETER = 60.0
MAX_DISTANCE = 40.0         # max radial bulge beyond the cylinder surface

OUTPUT_FILE = "helix_output.json"


if __name__ == "__main__":
    if MODE == "mock":
        print("Running with synthetic mock data (MODE = 'mock').")
        print("Set MODE = 'live' or 'csv' to use real Google Trends data.\n")
        coords = _demo_with_mock_data()
    elif MODE == "live":
        print(f"Fetching Google Trends data live for: {WORDS}")
        print(f"Date range: {START_DATE} to {END_DATE}\n")
        raw_series = fetch_trends_series(WORDS, START_DATE, END_DATE)
        bimonthly = resample_bimonthly(raw_series, START_DATE, END_DATE)
        print(f"Resampled to {len(bimonthly)} points:")
        print(bimonthly)
        coords = values_to_helix_coordinates(
            bimonthly.tolist(), CYLINDER_HEIGHT, CYLINDER_DIAMETER, MAX_DISTANCE
        )
    elif MODE == "csv":
        print(f"Reading Google Trends data from CSV: {CSV_PATHS}")
        raw_series = load_trends_from_csv(CSV_PATHS, WORDS)
        bimonthly = resample_bimonthly(raw_series, START_DATE, END_DATE)
        print(f"Resampled to {len(bimonthly)} points:")
        print(bimonthly)
        coords = values_to_helix_coordinates(
            bimonthly.tolist(), CYLINDER_HEIGHT, CYLINDER_DIAMETER, MAX_DISTANCE
        )
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Use 'mock', 'live', or 'csv'.")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(coords, f, indent=2)

    print(f"\nWrote {len(coords)} coordinates to {OUTPUT_FILE}")