"""
trends_cylinder_helix.py

Takes a list of words, averages their Google Trends interest-over-time,
and maps that average onto a helix wrapped around a cylinder.

Pipeline:
    words + date range  -->  Google Trends (via pytrends)  -->  averaged
    0-100 interest series, sampled every 2 months  -->  helix coordinates
    (one revolution per year, radius driven by trend strength)  -->
    x/y values rescaled into a fixed 0-15 range  -->  JSON

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

def load_trends_from_csv(
    csv_paths: List[str],
    words: Optional[List[str]] = None,
    anchor_rescale: bool = False,
) -> pd.Series:
    """
    Reads one or more CSVs exported from Google Trends and returns a single
    averaged 0-100 Series, same shape as fetch_trends_series().

    Fully general: works whether you give it...
      - one file per word (e.g. separate single-term downloads), any number
        of files
      - one file with several words compared together (Google's "Explore"
        compare tool, up to 5 terms per file)
      - a mix of both

    All word-columns found across all files are gathered and averaged
    together at each timestamp.

    IMPORTANT CAVEAT: Google Trends normalizes each download's 0-100 scale
    independently. If you downloaded words as SEPARATE single-term files,
    each one's 100 means "that word's own peak" -- not the same absolute
    volume across words. Averaging them still gives a meaningful blended
    curve (each word's own rise-and-fall, combined), but it is NOT a true
    cross-word popularity comparison. For genuine cross-word comparability,
    use Google's compare tool (up to 5 terms in one download) instead.

    Args:
        csv_paths: list of CSV file paths to read.
        words: optional list of specific column names to use (must match
            the word/column headers found in the CSV(s)). If omitted, every
            word-column found across all files is used.
        anchor_rescale: if True, treat `words[0]` as a shared anchor term
            present in every file, and rescale each file against it before
            averaging -- preserves true relative scale when combining
            multiple >5-word "compare" exports. Only meaningful if `words`
            is given and each file actually contains that anchor column.

    Google's export format varies depending on which page you download from.
    Two known formats are handled automatically:

      Format A (multi-term compare page, "explore" tool):
        Category: All categories

        Month,word1: (Worldwide),word2: (Worldwide)
        2020-01,45,32

      Format B (single-term page, "Time" column):
        "Time","word1"
        "2020-01-01",48
    """
    def _read_one(path: str) -> pd.DataFrame:
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

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # "<1" -> NaN

        return df

    file_frames = [_read_one(p) for p in csv_paths]

    if anchor_rescale:
        if not words:
            raise ValueError("anchor_rescale=True requires `words` (words[0] is the anchor).")
        anchor = words[0]
        combined: Optional[pd.DataFrame] = None
        for df in file_frames:
            if anchor not in df.columns:
                raise ValueError(f"Anchor word '{anchor}' not found in one of the CSVs.")
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
    else:
        # simple general case: gather every word-column from every file,
        # aligned on date, no rescaling
        combined = file_frames[0]
        for df in file_frames[1:]:
            combined = combined.join(df, how="outer", rsuffix="_dup")

    use_cols = words if words else list(combined.columns)
    missing = [w for w in use_cols if w not in combined.columns]
    if missing:
        raise ValueError(
            f"These words weren't found in the CSV column headers: {missing}. "
            f"Found columns: {list(combined.columns)}"
        )

    return combined[use_cols].mean(axis=1)


# ---------------------------------------------------------------------------
# 2. Resampling to exact weekly points
# ---------------------------------------------------------------------------

def resample_weekly(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    """
    Reindexes `series` onto exact weekly-spaced timestamps from start_date
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
        d = d + pd.DateOffset(weeks=1)
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
    total_revolutions: float = 6.0,
) -> List[List[float]]:
    """
    Converts a list of interest values (one per week, evenly spaced in
    time) into [x, y, z] coordinates on a helix wrapped around a cylinder.

    - The whole point set is spread over exactly `total_revolutions` full
      turns around the cylinder, from the first point (angle 0) to the
      last point (angle = total_revolutions * 360 degrees) -- evenly
      divided by (n - 1) steps. This is independent of how much real
      calendar time the data covers or how many points there are, so the
      spiral always looks evenly wound regardless of dataset size; it's
      no longer tied to "one revolution per year."
    - Radius is scaled between the cylinder's surface (radius = diameter/2,
      at the LOWEST value in this dataset) and max_distance beyond that
      surface (at the HIGHEST value in this dataset) -- normalized to this
      dataset's own min/max range rather than assuming a fixed 0-100 scale
    - z runs evenly from 0 to cylinder_height across all points, regardless
      of how many points there are -- so the total height stays fixed even
      though weekly sampling produces far more points than the old
      bimonthly sampling did

    Note: x and y here can still be negative (they're raw cos/sin values
    around the cylinder's centerline). See scale_xy_about_origin() below,
    which is applied afterward to scale x and y about (0, 0) uniformly.
    """
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 points to build a helix.")

    base_radius = cylinder_diameter / 2.0
    angle_step = (2 * math.pi * total_revolutions) / (n - 1)

    value_min = min(values)
    value_max = max(values)
    value_range = value_max - value_min

    coords = []
    for i, val in enumerate(values):
        theta = i * angle_step

        # normalize this value to 0-1 across the dataset's own min/max,
        # then scale that 0-1 range onto [base_radius, base_radius + max_distance]
        normalized = (val - value_min) / value_range if value_range else 0.0
        r = base_radius + normalized * max_distance

        z = (i / (n - 1)) * cylinder_height

        x = r * math.cos(theta)
        y = r * math.sin(theta)
        coords.append([x, y, z])

    return coords


def smooth_radius_outliers(
    coords: List[List[float]],
    iqr_multiplier: float = 1.5,
) -> List[List[float]]:
    """
    Instead of dropping outlier points, replaces each outlier's x and y with
    the average of its immediate neighbors' x and y (the points directly
    before and after it in the list) -- smoothing it into the surrounding
    data instead of deleting it. z is always left untouched, so point count
    and even z-spacing along the helix are both preserved exactly.

    Outliers are detected using the standard IQR (interquartile range)
    method: any
    point whose distance from the origin (0, 0) in the x-y plane exceeds
    Q3 + iqr_multiplier * (Q3 - Q1) (the standard IQR method) is treated as
    an outlier. Averaging is always based on the ORIGINAL neighbor
    positions (not other already-smoothed values), so two adjacent outliers
    don't compound off each other.

    The first and last points only have one neighbor (there's nothing
    before the first point or after the last), so those use that single
    neighbor's x/y directly. If a point has no neighbors at all (a
    1-point dataset), it's left unchanged.
    """
    n = len(coords)
    radii = sorted(math.hypot(x, y) for x, y, z in coords)
    if n < 4:
        return coords  # not enough points for quartiles to be meaningful

    def _percentile(sorted_values, pct):
        k = (len(sorted_values) - 1) * pct
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    q1 = _percentile(radii, 0.25)
    q3 = _percentile(radii, 0.75)
    iqr = q3 - q1
    upper_bound = q3 + iqr_multiplier * iqr

    outlier_indices = [
        i for i, (x, y, z) in enumerate(coords) if math.hypot(x, y) > upper_bound
    ]

    smoothed = [list(c) for c in coords]
    for i in outlier_indices:
        neighbors = []
        if i - 1 >= 0:
            neighbors.append(coords[i - 1])
        if i + 1 < n:
            neighbors.append(coords[i + 1])

        if neighbors:
            avg_x = sum(nb[0] for nb in neighbors) / len(neighbors)
            avg_y = sum(nb[1] for nb in neighbors) / len(neighbors)
            smoothed[i][0] = avg_x
            smoothed[i][1] = avg_y
        # z (smoothed[i][2]) is never touched

    if outlier_indices:
        print(f"Smoothed {len(outlier_indices)} outlier point(s) (radius > {upper_bound:.3f})")

    return smoothed



def scale_xy_about_origin(
    coords: List[List[float]],
    max_abs: float = 15.0,
) -> List[List[float]]:
    """
    Scales x and y by ONE SHARED factor (not independently) so that the
    point farthest from the origin (0, 0) in the x-y plane lands exactly
    at distance max_abs from the origin. z is left untouched.

    Using a single shared scale factor -- instead of separately stretching
    x's range and y's range -- preserves the true relative proportions of
    the data exactly as they naturally are (whatever shape that happens to
    be, distorted or not) and just scales it uniformly, larger or smaller.
    It does NOT introduce any additional distortion of its own. Negative
    values and the (0, 0) center are preserved.

    Note: with a shared factor, only the single farthest point actually
    reaches max_abs -- every other point stays at its correct proportional
    distance from the origin, which may be much smaller if that one point
    is an outlier relative to the rest of the dataset.
    """
    max_radius = max(math.hypot(x, y) for x, y, z in coords)
    scale = max_abs / max_radius if max_radius else 1.0
    return [[x * scale, y * scale, z] for x, y, z in coords]


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
    xy_max_abs: float = 15.0,
    iqr_multiplier: float = 1.5,
    total_revolutions: float = 6.0,
) -> str:
    """
    Full pipeline: words + date range -> Google Trends -> averaged series ->
    resampled to weekly points -> helix coordinates -> x/y scaled about
    the origin -> JSON string.

    Returns a JSON string: a list of [x, y, z] lists.
    """
    raw_series = fetch_trends_series(words, start_date, end_date, geo=geo)
    weekly = resample_weekly(raw_series, start_date, end_date)
    coords = values_to_helix_coordinates(
        weekly.tolist(), cylinder_height, cylinder_diameter, max_distance,
        total_revolutions=total_revolutions
    )
    coords = smooth_radius_outliers(coords, iqr_multiplier=iqr_multiplier)
    coords = scale_xy_about_origin(coords, max_abs=xy_max_abs)
    return json.dumps(coords, indent=2)


# ---------------------------------------------------------------------------
# 5. Offline demo / self-test (no network required)
# ---------------------------------------------------------------------------

def _demo_with_mock_data():
    """
    Exercises resample_weekly() and values_to_helix_coordinates() with a
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

    weekly = resample_weekly(mock_series, start_date, end_date)
    print(f"[DEMO] Resampled to {len(weekly)} points, every week:")
    print(weekly)

    coords = values_to_helix_coordinates(
        weekly.tolist(),
        cylinder_height=100.0,
        cylinder_diameter=20.0,
        max_distance=15.0,
    )
    coords = scale_xy_about_origin(coords, max_abs=15.0)
    print(f"\n[DEMO] Generated {len(coords)} coordinates (x, y scaled about origin, max abs 15):")
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

MODE = "csv" # one of: "mock", "live", "csv"
                # "mock" -- synthetic data, no network needed, just to sanity
                #           check your environment
                # "live" -- pytrends fetches directly from Google Trends.
                #           Can hit 429 rate-limit errors -- Google actively
                #           blocks this kind of scraping and there's no
                #           reliable fix, just retry later / less often.
                # "csv"  -- reads CSV(s) you download by hand from
                #           trends.google.com/trends/explore. Slower to set
                #           up but always works. See load_trends_from_csv()
                #           above for details -- accepts any number of files,
                #           each with any number of word-columns, and
                #           averages everything together.

WORDS = ["sculpture", "generative art", "parametric design"]
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

# For MODE == "csv": list every CSV you downloaded. Can be one file per
# word, one file with several words compared together, or a mix -- all
# word-columns found get averaged. Leave CSV_WORDS = None to just use every
# column found; set it to a specific list to only use those columns (and
# ignore anything else in the files).
CSV_PATHS = ["god_06_to_26.csv"]
CSV_WORDS = None
CSV_ANCHOR_RESCALE = False  # True only if combining >5-word compare files
                            # that share one repeated anchor word

CYLINDER_HEIGHT = 82     # inches -- total height stays fixed regardless of point count
CYLINDER_DIAMETER = 0
MAX_DISTANCE = 15     # max radial bulge beyond the cylinder surface
XY_MAX_ABS = 15.0   # farthest remaining point from (0, 0) in x-y is scaled to this distance
IQR_MULTIPLIER = 1.5   # outlier cutoff before scaling; lower = more aggressive smoothing
TOTAL_REVOLUTIONS =30  # full turns spread evenly across ALL points, top to bottom --
                          # not tied to calendar time, just controls how tightly wound
                          # the spiral looks; raise for a tighter/denser wrap, lower for looser

OUTPUT_FILE = "god_06_to_26.json"


if __name__ == "__main__":
    if MODE == "mock":
        print("Running with synthetic mock data (MODE = 'mock').")
        print("Set MODE = 'live' or 'csv' to use real Google Trends data.\n")
        coords = _demo_with_mock_data()
    elif MODE == "live":
        print(f"Fetching Google Trends data live for: {WORDS}")
        print(f"Date range: {START_DATE} to {END_DATE}\n")
        raw_series = fetch_trends_series(WORDS, START_DATE, END_DATE)
        weekly = resample_weekly(raw_series, START_DATE, END_DATE)
        print(f"Resampled to {len(weekly)} points:")
        print(weekly)
        coords = values_to_helix_coordinates(
            weekly.tolist(), CYLINDER_HEIGHT, CYLINDER_DIAMETER, MAX_DISTANCE,
            total_revolutions=TOTAL_REVOLUTIONS
        )
        coords = smooth_radius_outliers(coords, iqr_multiplier=IQR_MULTIPLIER)
        coords = scale_xy_about_origin(coords, max_abs=XY_MAX_ABS)
    elif MODE == "csv":
        print(f"Reading Google Trends data from CSV(s): {CSV_PATHS}")
        raw_series = load_trends_from_csv(
            CSV_PATHS, words=CSV_WORDS, anchor_rescale=CSV_ANCHOR_RESCALE
        )
        weekly = resample_weekly(raw_series, START_DATE, END_DATE)
        print(f"Resampled to {len(weekly)} points:")
        print(weekly)
        coords = values_to_helix_coordinates(
            weekly.tolist(), CYLINDER_HEIGHT, CYLINDER_DIAMETER, MAX_DISTANCE,
            total_revolutions=TOTAL_REVOLUTIONS
        )
        coords = smooth_radius_outliers(coords, iqr_multiplier=IQR_MULTIPLIER)
        coords = scale_xy_about_origin(coords, max_abs=XY_MAX_ABS)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Use 'mock', 'live', or 'csv'.")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(coords, f, indent=2)

    print(f"\nWrote {len(coords)} coordinates to {OUTPUT_FILE}")