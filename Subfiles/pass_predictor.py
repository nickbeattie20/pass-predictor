from skyfield.api import load, wgs84, EarthSatellite
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

VANCOUVER_TZ = ZoneInfo("America/Vancouver")

# Ground station location (Fort Langley, rounded to ~1 km)
# UBC Vancouver: 49.265, -123.242, 60
GROUND_LAT = 49.17
GROUND_LON = -122.58
GROUND_ELEV_M = 10

MIN_ELEVATION_DEG = 30.0

# How many hours ahead to search for passes
SEARCH_WINDOW_HOURS = 48

# Local-time bounds for outings
EARLIEST_PASS_TIME = time(7, 0)
LATEST_PASS_TIME = time(23, 0)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_fallback_tle(norad_id, ts):
    """Load the last saved <norad_id>.tle from the project folder (offline fallback)."""
    path = PROJECT_DIR / f"{norad_id}.tle"
    sats = load.tle_file(str(path), reload=False)
    if not sats:
        raise RuntimeError(f"No fallback TLE found at {path}")
    return sats[0]

# METEOR M2-4: 59051
# METEOR M2-3: 57166

def load_satellite(name="METEOR M2-3", norad_id=57166, use_fallback=False):

    ts = load.timescale()

    if use_fallback:
        sat = load_fallback_tle(norad_id, ts)
        print(f"Using fallback TLE for {sat.name} (epoch: {sat.epoch.utc_jpl()})")
        print("NOTE: this may be stale. For real predictions, fetch a fresh TLE")
        print(f"from https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}")
        return sat, ts

    try:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
        sats = load.tle_file(url, filename=str(PROJECT_DIR / f"{norad_id}.tle"), reload=True)
        sat = sats[0]
        print(f"Loaded live TLE for {sat.name} (epoch: {sat.epoch.utc_jpl()})")
        return sat, ts
    except Exception as e:
        print(f"Could not fetch live TLE ({e}), using fallback.")
        sat = load_fallback_tle(norad_id, ts)
        return sat, ts


def find_passes(sat, ts, lat=GROUND_LAT, lon=GROUND_LON, elev_m=GROUND_ELEV_M,
                 hours_ahead=SEARCH_WINDOW_HOURS, min_elevation_deg=MIN_ELEVATION_DEG,
                 earliest_local_time=EARLIEST_PASS_TIME,
                 latest_local_time=LATEST_PASS_TIME):
   
    if earliest_local_time > latest_local_time:
        raise ValueError("earliest_local_time must be at or before latest_local_time")

    ground_station = wgs84.latlon(lat, lon, elev_m)

    t0 = ts.now()
    t1 = ts.tt_jd(t0.tt + hours_ahead / 24.0)

    times, events = sat.find_events(ground_station, t0, t1, altitude_degrees=min_elevation_deg)

    passes = []
    current_pass = None

    for t, event in zip(times, events):
        if event == 0:  # rise
            current_pass = {"rise": t, "culminate": None, "set": None}
        elif event == 1 and current_pass is not None:  # culminate
            current_pass["culminate"] = t
        elif event == 2 and current_pass is not None:  # set
            current_pass["set"] = t
            rise_local = current_pass["rise"].utc_datetime().astimezone(VANCOUVER_TZ)
            set_local = current_pass["set"].utc_datetime().astimezone(VANCOUVER_TZ)
            rise_time = rise_local.timetz().replace(tzinfo=None)
            set_time = set_local.timetz().replace(tzinfo=None)
            if (rise_local.date() == set_local.date()
                    and earliest_local_time <= rise_time
                    and set_time <= latest_local_time):
                passes.append(current_pass)
            current_pass = None

    # For each complete pass, sample elevation + distance every 10 seconds
    for p in passes:
        p["samples"] = sample_pass(sat, ground_station, ts, p["rise"], p["set"])

    return passes


def sample_pass(sat, ground_station, ts, rise_t, set_t, step_seconds=10):
   
    duration_s = (set_t.tt - rise_t.tt) * 86400.0
    n_samples = max(int(duration_s / step_seconds), 2)

    samples = []
    for i in range(n_samples + 1):
        t = ts.tt_jd(rise_t.tt + (i * step_seconds) / 86400.0)
        difference = sat - ground_station
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        samples.append({
            "time": t.utc_datetime(),
            "elevation_deg": alt.degrees,
            "azimuth_deg": az.degrees,
            "distance_km": distance.km,
        })

    # Central difference for range-rate (forward/backward diff at the ends)
    n = len(samples)
    for i in range(n):
        if n < 2:
            rate = 0.0
        elif i == 0:
            rate = (samples[1]["distance_km"] - samples[0]["distance_km"]) / step_seconds
        elif i == n - 1:
            rate = (samples[i]["distance_km"] - samples[i - 1]["distance_km"]) / step_seconds
        else:
            rate = (samples[i + 1]["distance_km"] - samples[i - 1]["distance_km"]) / (2 * step_seconds)
        samples[i]["range_rate_km_s"] = rate

    return samples


def print_pass_summary(passes):
    if not passes:
        print("No passes found above the minimum elevation within the time bounds "
              f"({EARLIEST_PASS_TIME:%H:%M}-{LATEST_PASS_TIME:%H:%M} "
              f"{VANCOUVER_TZ.key}).")
        return

    for i, p in enumerate(passes, 1):
        max_elev = max(s["elevation_deg"] for s in p["samples"])
        min_dist = min(s["distance_km"] for s in p["samples"])
        duration_min = (p["set"].tt - p["rise"].tt) * 24 * 60
        rise_time = p["rise"].utc_datetime().astimezone(VANCOUVER_TZ)
        culminate_time = p["culminate"].utc_datetime().astimezone(VANCOUVER_TZ)
        set_time = p["set"].utc_datetime().astimezone(VANCOUVER_TZ)
        print(f"\nPass {i}:")
        print(f"  Rise:      {rise_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Culminate: {culminate_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Set:       {set_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Duration:  {duration_min:.1f} minutes")
        print(f"  Max elevation: {max_elev:.1f} degrees")
        print(f"  Closest approach: {min_dist:.1f} km")


if __name__ == "__main__":
    # Pulls a live TLE from Celestrak by default (matches "live TLE data" on
    # the resume). Set use_fallback=True to force the offline hardcoded TLE
    # for testing without internet access.
    sat, ts = load_satellite(use_fallback=False)
    passes = find_passes(sat, ts)
    print_pass_summary(passes)
