"""
Satellite Pass Predictor - Streamlit UI

Run from the project folder (with the virtual environment active):
    streamlit run app.py

Reuses the existing modules in Subfiles/:
  - pass_predictor.sample_pass  (az/el/range/range-rate sampling)
  - link_budget.received_power_dbm / free_space_path_loss_db
  - doppler.doppler_shift_hz
"""

import io
import math
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.iokit import parse_tle_file

from Subfiles.pass_predictor import sample_pass
from Subfiles.link_budget import received_power_dbm
from Subfiles.doppler import doppler_shift_hz

LOCAL_TZ = ZoneInfo("America/Vancouver")
PROJECT_DIR = Path(__file__).parent

# Satellites you can pick from the sidebar: name -> (NORAD ID, downlink MHz)
PRESETS = {
    "METEOR M2-4": (59051, 137.900),
    "METEOR M2-3": (57166, 137.900),
    "ISS": (25544, 145.800),
}

LOCATIONS = {
    "Fort Langley": (49.17, -122.58, 10.0),   # rounded to ~1 km so the public app doesn't show an exact address
    "UBC Vancouver": (49.265, -123.242, 60.0),
    "Custom": None,
}

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(az_deg):
    return COMPASS[int((az_deg % 360) / 22.5 + 0.5) % 16]


# ---------------------------------------------------------------------------
# Weather forecast (Open-Meteo, free, no API key)
# ---------------------------------------------------------------------------

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_FIELDS = ["cloud_cover", "precipitation_probability", "precipitation",
                   "weather_code", "temperature_2m", "wind_speed_10m"]

# WMO weather codes -> (emoji, short description)
WMO_CODES = {
    0: ("☀️", "Clear"), 1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"),
    3: ("☁️", "Overcast"), 45: ("🌫️", "Fog"), 48: ("🌫️", "Freezing fog"),
    51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Drizzle"), 55: ("🌧️", "Heavy drizzle"),
    56: ("🌧️", "Freezing drizzle"), 57: ("🌧️", "Freezing drizzle"),
    61: ("🌦️", "Light rain"), 63: ("🌧️", "Rain"), 65: ("🌧️", "Heavy rain"),
    66: ("🌧️", "Freezing rain"), 67: ("🌧️", "Freezing rain"),
    71: ("🌨️", "Light snow"), 73: ("🌨️", "Snow"), 75: ("❄️", "Heavy snow"),
    77: ("🌨️", "Snow grains"), 80: ("🌦️", "Light showers"), 81: ("🌧️", "Showers"),
    82: ("⛈️", "Violent showers"), 85: ("🌨️", "Snow showers"), 86: ("❄️", "Heavy snow showers"),
    95: ("⛈️", "Thunderstorm"), 96: ("⛈️", "Thunderstorm, hail"), 99: ("⛈️", "Thunderstorm, hail"),
}


@st.cache_data(ttl=3600, show_spinner="Fetching weather forecast...")
def fetch_forecast(lat, lon):
    """Hourly forecast for the next 16 days, indexed by UTC hour.

    Returns (DataFrame or None, error message or None). Requests UTC so the
    match against pass times can't be thrown off by time-zone rules.
    """
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "hourly": ",".join(FORECAST_FIELDS),
        "forecast_days": 16,
        "timezone": "GMT",
        "wind_speed_unit": "kmh",
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time"), None
    except Exception as e:
        return None, str(e)


def weather_at(forecast, when):
    """Forecast row for the hour nearest `when` (an aware datetime), or None."""
    if forecast is None or forecast.empty:
        return None
    ts_utc = pd.Timestamp(when).tz_convert("UTC").round("h")
    if ts_utc not in forecast.index:
        return None
    row = forecast.loc[ts_utc]

    def val(key):
        v = row.get(key)
        return None if v is None or pd.isna(v) else float(v)

    code = val("weather_code")
    emoji, desc = WMO_CODES.get(int(code), ("❔", f"Code {int(code)}")) if code is not None else ("❔", "—")
    return {
        "emoji": emoji, "desc": desc,
        "cloud": val("cloud_cover"),
        "rain_prob": val("precipitation_probability"),
        "rain_mm": val("precipitation"),
        "temp": val("temperature_2m"),
        "wind": val("wind_speed_10m"),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

MAX_TLE_AGE_DAYS = 2  # a GitHub Action refreshes the saved .tle files daily


@st.cache_resource(ttl=6 * 3600, show_spinner="Loading orbit data (TLE)...")
def load_sat(norad_id):
    """Load the TLE for `norad_id`.

    The repo's <norad>.tle files are refreshed every day by a GitHub Action
    (.github/workflows/update-tles.yml), because Celestrak doesn't answer
    requests from Streamlit Cloud's servers. So: use the saved file if it's
    recent, and only try Celestrak directly (with a short timeout) when the
    file is missing or stale, e.g. for a custom NORAD ID or when run locally.

    Returns (satellite, timescale, source_description).
    """
    ts = load.timescale()
    cache_file = PROJECT_DIR / f"{norad_id}.tle"

    saved = None
    if cache_file.exists():
        sats = load.tle_file(str(cache_file), reload=False)
        if sats:
            saved = sats[0]
            age_days = ts.now() - saved.epoch
            if age_days <= MAX_TLE_AGE_DAYS:
                return saved, ts, f"saved TLE, {age_days * 24:.0f} h old (refreshed daily)"

    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    try:
        r = requests.get(url, timeout=5,
                         headers={"User-Agent": "pass-predictor (github.com/nickbeattie20/pass-predictor)"})
        r.raise_for_status()
        sats = list(parse_tle_file(io.BytesIO(r.content), ts))
        if not sats:  # Celestrak answers 200 with a text message when it has nothing to send
            raise ValueError(r.text.strip()[:80] or "empty response")
        try:
            cache_file.write_bytes(r.content)
        except OSError:
            pass
        return sats[0], ts, "live from Celestrak"
    except Exception as e:
        if saved is not None:
            age_days = ts.now() - saved.epoch
            return saved, ts, (f"saved TLE, {age_days:.1f} days old "
                               f"(live fetch failed: {type(e).__name__})")
        raise RuntimeError(f"Could not load a TLE for NORAD {norad_id} "
                           f"({type(e).__name__}). Celestrak may be unreachable from this server.")


def find_full_passes(sat, ts, station, hours_ahead, horizon_deg, min_max_el_deg,
                     time_window=None):
    """Find passes that rise above `horizon_deg`, keep those that peak above
    `min_max_el_deg`, and sample each one using pass_predictor.sample_pass."""
    t0 = ts.now()
    t1 = ts.tt_jd(t0.tt + hours_ahead / 24.0)
    times, events = sat.find_events(station, t0, t1, altitude_degrees=horizon_deg)

    passes, current = [], None
    for t, ev in zip(times, events):
        if ev == 0:
            current = {"rise": t, "culminate": None, "set": None}
        elif ev == 1 and current is not None and current["culminate"] is None:
            current["culminate"] = t
        elif ev == 2 and current is not None:
            current["set"] = t
            if current["culminate"] is not None:
                passes.append(current)
            current = None

    results = []
    for p in passes:
        samples = sample_pass(sat, station, ts, p["rise"], p["set"], step_seconds=5)
        max_el = max(s["elevation_deg"] for s in samples)
        if max_el < min_max_el_deg:
            continue
        rise_local = p["rise"].utc_datetime().astimezone(LOCAL_TZ)
        set_local = p["set"].utc_datetime().astimezone(LOCAL_TZ)
        if time_window is not None:
            start, end = time_window
            if not (start <= rise_local.time() and set_local.time() <= end
                    and rise_local.date() == set_local.date()):
                continue
        p["samples"] = samples
        p["max_el"] = max_el
        p["rise_local"] = rise_local
        p["set_local"] = set_local
        p["peak_local"] = p["culminate"].utc_datetime().astimezone(LOCAL_TZ)
        results.append(p)
    return results


def add_link_and_doppler(samples, freq_mhz, rf):
    """Annotate samples with received power, link margin and Doppler shift."""
    out = []
    for s in samples:
        p_rx = received_power_dbm(
            s["distance_km"],
            tx_power_dbm=rf["tx_power_dbm"],
            tx_gain_dbi=rf["tx_gain_dbi"],
            rx_gain_dbi=rf["rx_gain_dbi"],
            other_losses_db=rf["losses_db"],
            freq_mhz=freq_mhz,
        )
        out.append({
            **s,
            "rx_power_dbm": p_rx,
            "margin_db": p_rx - rf["noise_floor_dbm"],
            "doppler_khz": doppler_shift_hz(freq_mhz * 1e6, s["range_rate_km_s"]) / 1000.0,
        })
    return out


def footprint_circle(lat, lon, radius_km, n=180):
    """Points on a circle of `radius_km` (great-circle) around lat/lon."""
    R = 6371.0
    d = radius_km / R
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lats, lons = [], []
    for i in range(n + 1):
        brg = 2 * math.pi * i / n
        lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                         math.cos(lat1) * math.sin(d) * math.cos(brg))
        lon2 = lon1 + math.atan2(math.sin(brg) * math.sin(d) * math.cos(lat1),
                                 math.cos(d) - math.sin(lat1) * math.sin(lat2))
        lats.append(math.degrees(lat2))
        lons.append((math.degrees(lon2) + 540) % 360 - 180)
    return lats, lons


def coverage_radius_km(alt_km, min_el_deg):
    """Ground distance from the station to the sub-satellite point when the
    satellite is at `min_el_deg` elevation."""
    R = 6371.0
    eps = math.radians(min_el_deg)
    lam = math.acos(R * math.cos(eps) / (R + alt_km)) - eps
    return R * lam


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Satellite Pass Predictor", page_icon="🛰️", layout="wide")

with st.sidebar:
    st.header("Satellite")
    choice = st.selectbox("Target", list(PRESETS) + ["Custom NORAD ID"])
    if choice == "Custom NORAD ID":
        norad_id = int(st.number_input("NORAD ID", min_value=1, value=59051, step=1))
        freq_mhz = st.number_input("Downlink frequency (MHz)", value=137.900, format="%.3f")
    else:
        norad_id, default_freq = PRESETS[choice]
        freq_mhz = st.number_input("Downlink frequency (MHz)", value=default_freq, format="%.3f")

    st.header("Ground station")
    loc_choice = st.selectbox("Location", list(LOCATIONS))
    if LOCATIONS[loc_choice] is None:
        lat = st.number_input("Latitude", value=49.265, format="%.4f")
        lon = st.number_input("Longitude", value=-123.242, format="%.4f")
        elev_m = st.number_input("Elevation (m)", value=10.0)
    else:
        lat, lon, elev_m = LOCATIONS[loc_choice]

    st.header("Pass filter")
    hours_ahead = st.slider("Look ahead (hours)", 6, 96, 48, step=6)
    horizon_deg = st.slider("Horizon (rise/set at)", 0, 20, 10,
                            help="Elevation at which a pass starts and ends.")
    min_max_el = st.slider("Minimum peak elevation", 0, 80, 20,
                           help="Hide passes that never get higher than this.")
    use_window = st.checkbox("Only passes between set local times", value=False)
    time_window = None
    if use_window:
        c1, c2 = st.columns(2)
        start_t = c1.time_input("From", time(7, 0))
        end_t = c2.time_input("To", time(23, 0))
        time_window = (start_t, end_t)

    st.header("Weather")
    show_weather = st.checkbox("Show forecast for each pass", value=True)
    hide_rain = st.checkbox("Hide passes when rain is likely", value=False,
                            disabled=not show_weather)
    max_rain = st.slider("Max chance of rain (%)", 0, 100, 50, step=5,
                         disabled=not (show_weather and hide_rain))
    hide_cloud = st.checkbox("Hide passes when it's cloudy", value=False,
                             disabled=not show_weather,
                             help="Matters for visible-light images and optical tracking; "
                                  "the 137 MHz radio signal goes straight through cloud.")
    max_cloud = st.slider("Max cloud cover (%)", 0, 100, 70, step=5,
                          disabled=not (show_weather and hide_cloud))

    with st.expander("Link budget assumptions"):
        st.caption("Defaults: Meteor LRPT into the V-dipole + RTL-SDR. "
                   "Transmit power and noise floor are estimates - adjust as you learn more.")
        rf = {
            "tx_power_dbm": st.number_input("Satellite TX power (dBm)", value=37.0,
                                            help="37 dBm = 5 W (assumed)"),
            "tx_gain_dbi": st.number_input("Satellite antenna gain (dBi)", value=0.0),
            "rx_gain_dbi": st.number_input("Ground antenna gain (dBi)", value=2.0,
                                           help="V-dipole ~2 dBi"),
            "losses_db": st.number_input("Other losses (dB)", value=4.3,
                                         help="3 dB polarization + 0.3 dB coax + 1 dB misc"),
            "noise_floor_dbm": st.number_input("Receiver noise floor (dBm)", value=-116.0,
                                               help="kTB for 150 kHz (-122 dBm) + ~6 dB noise figure"),
        }

# Load satellite + passes
try:
    sat, ts, tle_source = load_sat(norad_id)
except RuntimeError as err:
    st.error(str(err))
    st.stop()

station = wgs84.latlon(lat, lon, elev_m)
passes = find_full_passes(sat, ts, station, hours_ahead, horizon_deg, min_max_el, time_window)

st.title(f"🛰️ {sat.name}")
epoch_local = sat.epoch.utc_datetime().astimezone(LOCAL_TZ)
st.caption(f"NORAD {norad_id} · {freq_mhz:.3f} MHz · TLE epoch "
           f"{epoch_local:%Y-%m-%d %H:%M %Z} · source: {tle_source} · "
           f"station {lat:.3f}, {lon:.3f}")

# Weather: attach the forecast for each pass's peak hour, then apply filters
forecast, weather_error = (fetch_forecast(lat, lon) if show_weather else (None, None))
if show_weather and weather_error:
    st.info(f"Weather forecast unavailable right now ({weather_error}). "
            "Showing passes without it.")
for p in passes:
    p["wx"] = weather_at(forecast, p["peak_local"]) if forecast is not None else None

hidden_by_weather = 0
if forecast is not None and (hide_rain or hide_cloud):
    kept = []
    for p in passes:
        wx = p["wx"]
        too_wet = hide_rain and wx and wx["rain_prob"] is not None and wx["rain_prob"] > max_rain
        too_cloudy = hide_cloud and wx and wx["cloud"] is not None and wx["cloud"] > max_cloud
        if too_wet or too_cloudy:
            hidden_by_weather += 1
        else:
            kept.append(p)
    passes = kept

if not passes:
    st.warning("No passes match these filters. Try lowering the minimum peak "
               "elevation, looking further ahead, or relaxing the weather filters.")
    st.stop()

# Annotate every pass once
for p in passes:
    p["annotated"] = add_link_and_doppler(p["samples"], freq_mhz, rf)
    p["peak_margin"] = max(s["margin_db"] for s in p["annotated"])
    p["min_range"] = min(s["distance_km"] for s in p["samples"])

# ---- Headline metrics ------------------------------------------------------
now_local = datetime.now(LOCAL_TZ)
next_p = next((p for p in passes if p["set_local"] > now_local), passes[0])
best_p = max(passes, key=lambda p: p["max_el"])

def until(dt):
    delta = dt - now_local
    if delta.total_seconds() <= 0:
        return "now"
    h, rem = divmod(int(delta.total_seconds()), 3600)
    return f"in {h} h {rem // 60} min" if h else f"in {rem // 60} min"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Next pass", next_p["rise_local"].strftime("%a %H:%M"), until(next_p["rise_local"]))
m2.metric("Next pass peak", f"{next_p['max_el']:.0f}°")
m3.metric("Best pass", best_p["peak_local"].strftime("%a %H:%M"), f"{best_p['max_el']:.0f}° peak")
m4.metric("Passes found", len(passes))

# ---- Pass table ------------------------------------------------------------
rows = []
for i, p in enumerate(passes, 1):
    first, last = p["samples"][0], p["samples"][-1]
    heading = "Northbound" if 90 < first["azimuth_deg"] < 270 else "Southbound"
    rows.append({
        "#": i,
        "Day": p["rise_local"].strftime("%a %b %d"),
        "Rise": p["rise_local"].strftime("%H:%M"),
        "Peak": p["peak_local"].strftime("%H:%M"),
        "Set": p["set_local"].strftime("%H:%M"),
        "Duration (min)": round((p["set_local"] - p["rise_local"]).total_seconds() / 60, 1),
        "Max elev (°)": round(p["max_el"], 1),
        "Rises in": compass(first["azimuth_deg"]),
        "Sets in": compass(last["azimuth_deg"]),
        "Direction": heading,
        "Closest (km)": round(p["min_range"]),
        "Peak margin (dB)": round(p["peak_margin"], 1),
        "Daylight": "☀️" if 7 <= p["peak_local"].hour < 17 else "🌙",
    })
    if show_weather:
        wx = p["wx"]
        rows[-1]["Forecast"] = f"{wx['emoji']} {wx['desc']}" if wx else "—"
        rows[-1]["Cloud (%)"] = None if not wx or wx["cloud"] is None else round(wx["cloud"])
        rows[-1]["Rain (%)"] = None if not wx or wx["rain_prob"] is None else round(wx["rain_prob"])
df = pd.DataFrame(rows)

st.subheader("Upcoming passes")
styled = df.style.background_gradient(subset=["Max elev (°)"], cmap="Greens", vmin=0, vmax=90)
styled = styled.format({"Duration (min)": "{:.1f}", "Max elev (°)": "{:.1f}",
                        "Closest (km)": "{:.0f}", "Peak margin (dB)": "{:.1f}"})
if show_weather and "Rain (%)" in df and df["Rain (%)"].notna().any():
    styled = styled.background_gradient(subset=["Rain (%)"], cmap="Blues", vmin=0, vmax=100)
    styled = styled.format({"Cloud (%)": "{:.0f}", "Rain (%)": "{:.0f}"}, na_rep="—")
st.dataframe(styled, hide_index=True, width="stretch")
notes = ["☀️ / 🌙 is a rough guide: Meteor's visible channels need daylight; "
         "channel 4 (infrared) works at night."]
if show_weather and forecast is not None:
    notes.append("Forecast is for the hour nearest each pass's peak; most reliable within "
                 "about 3 days. Weather data by Open-Meteo.com.")
if hidden_by_weather:
    notes.append(f"{hidden_by_weather} pass(es) hidden by your weather filters.")
st.caption(" ".join(notes))

# ---- Pass detail -----------------------------------------------------------
st.subheader("Pass detail")
labels = [f"#{r['#']}  {r['Day']} {r['Rise']}  ·  peak {r['Max elev (°)']}°" for r in rows]
default_idx = passes.index(best_p)
sel_label = st.selectbox("Choose a pass", labels, index=default_idx)
p = passes[labels.index(sel_label)]
samples = p["annotated"]
minutes = [(s["time"] - samples[0]["time"]).total_seconds() / 60 for s in samples]
clock = [s["time"].astimezone(LOCAL_TZ) for s in samples]

left, right = st.columns(2)

# Sky plot: azimuth around, elevation inward (90° at the centre)
with left:
    sky = go.Figure()
    sky.add_trace(go.Scatterpolar(
        r=[90 - s["elevation_deg"] for s in samples],
        theta=[s["azimuth_deg"] for s in samples],
        mode="lines", line=dict(width=3, color="#1D9E75"),
        text=[f"{c:%H:%M:%S}<br>el {s['elevation_deg']:.0f}° az {s['azimuth_deg']:.0f}°"
              for c, s in zip(clock, samples)],
        hoverinfo="text", name="Track",
    ))
    sky.add_trace(go.Scatterpolar(
        r=[90 - samples[0]["elevation_deg"], 90 - samples[-1]["elevation_deg"]],
        theta=[samples[0]["azimuth_deg"], samples[-1]["azimuth_deg"]],
        mode="markers+text", marker=dict(size=10, color=["#378ADD", "#D85A30"]),
        text=["Rise", "Set"], textposition="top center", hoverinfo="skip", name="",
    ))
    sky.update_layout(
        title="Sky track (centre = overhead)",
        polar=dict(
            angularaxis=dict(rotation=90, direction="clockwise",
                             tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                             ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                            ticktext=["90°", "60°", "30°", "0°"]),
        ),
        showlegend=False, height=430, margin=dict(l=40, r=40, t=60, b=30),
    )
    st.plotly_chart(sky, width="stretch")

# Ground track map with station and coverage circle
with right:
    track_lat, track_lon = [], []
    for s in p["samples"]:
        t = ts.from_datetime(s["time"])
        la, lo = wgs84.latlon_of(sat.at(t))
        track_lat.append(la.degrees)
        track_lon.append(lo.degrees)
    alt_km = wgs84.height_of(sat.at(p["culminate"])).km
    c_lat, c_lon = footprint_circle(lat, lon, coverage_radius_km(alt_km, horizon_deg))

    world = go.Figure()
    world.add_trace(go.Scattergeo(lat=c_lat, lon=c_lon, mode="lines",
                                  line=dict(color="#7F77DD", dash="dot"),
                                  name=f"Coverage (≥{horizon_deg}°)"))
    world.add_trace(go.Scattergeo(lat=track_lat, lon=track_lon, mode="lines",
                                  line=dict(width=3, color="#1D9E75"), name="Ground track"))
    world.add_trace(go.Scattergeo(lat=[lat], lon=[lon], mode="markers",
                                  marker=dict(size=10, color="#D85A30"), name="Station"))
    world.update_geos(projection_type="orthographic",
                      projection_rotation=dict(lon=lon, lat=lat),
                      showcountries=True, showland=True, landcolor="#E1F5EE",
                      showocean=True, oceancolor="#E6F1FB")
    world.update_layout(title=f"Ground track (altitude ≈ {alt_km:.0f} km)",
                        height=430, margin=dict(l=10, r=10, t=60, b=10),
                        legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(world, width="stretch")

# Time series: elevation, link margin, Doppler
def line_chart(y, title, ytitle, color, zero_line=False):
    fig = go.Figure(go.Scatter(x=clock, y=y, mode="lines", line=dict(color=color, width=2)))
    if zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="#E24B4A")
    fig.update_layout(title=title, yaxis_title=ytitle, height=280,
                      margin=dict(l=50, r=20, t=50, b=40))
    return fig

c1, c2, c3 = st.columns(3)
c1.plotly_chart(line_chart([s["elevation_deg"] for s in samples], "Elevation", "degrees", "#378ADD"),
                width="stretch")
c2.plotly_chart(line_chart([s["margin_db"] for s in samples], "Predicted link margin", "dB",
                           "#BA7517", zero_line=True), width="stretch")
c3.plotly_chart(line_chart([s["doppler_khz"] for s in samples], "Doppler shift", "kHz", "#7F77DD"),
                width="stretch")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Rise → set", f"{p['rise_local']:%H:%M} → {p['set_local']:%H:%M}")
d2.metric("Closest approach", f"{p['min_range']:.0f} km")
d3.metric("Peak predicted margin", f"{p['peak_margin']:.1f} dB")
dop = [s["doppler_khz"] for s in samples]
d4.metric("Doppler range", f"{min(dop):+.1f} to {max(dop):+.1f} kHz")

if show_weather:
    wx = p["wx"]
    st.markdown("**Forecast at peak**")
    w1, w2, w3, w4 = st.columns(4)
    if wx:
        w1.metric("Sky", f"{wx['emoji']} {wx['desc']}")
        w2.metric("Cloud cover", "—" if wx["cloud"] is None else f"{wx['cloud']:.0f}%")
        rain_txt = "—" if wx["rain_prob"] is None else f"{wx['rain_prob']:.0f}%"
        rain_amt = None if not wx["rain_mm"] else f"{wx['rain_mm']:.1f} mm in that hour"
        w3.metric("Chance of rain", rain_txt, rain_amt, delta_color="off")
        temp_txt = "—" if wx["temp"] is None else f"{wx['temp']:.0f} °C"
        wind_txt = None if wx["wind"] is None else f"wind {wx['wind']:.0f} km/h"
        w4.metric("Temperature", temp_txt, wind_txt, delta_color="off")
    elif forecast is not None:
        w1.caption("No forecast for this time (more than 16 days out).")
    else:
        w1.caption("Forecast unavailable.")

st.caption(f"Times in {LOCAL_TZ.key}. Rise/set measured at {horizon_deg}° elevation.")
