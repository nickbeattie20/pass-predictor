# Satellite Pass Predictor

Predicts upcoming passes of weather satellites (Meteor-M2 3/4) and the ISS over a ground station, with a predicted link budget, Doppler shift, and the weather forecast for each pass. I use it to plan reception with my RTL-SDR ground station.

**Live app:** _add your streamlit.app link here_

## What it does
- Downloads fresh orbital elements (TLEs) from CelesTrak and propagates them with SGP4 (Skyfield)
- Finds every pass in the next 6–96 hours and filters by peak elevation, time of day, cloud and rain
- For each pass: sky track, ground track with coverage circle, elevation, predicted link margin (Friis equation), and Doppler shift vs time

## Run it locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit interface
- `Subfiles/pass_predictor.py`: pass finding and sampling (azimuth, elevation, range, range rate)
- `Subfiles/link_budget.py`: free-space path loss and received power
- `Subfiles/doppler.py`: Doppler shift from range rate
- `analyze_satellite_pass.py`: command-line version that plots the best pass
- `*.tle`: offline fallback TLEs, used only if CelesTrak can't be reached

Weather data by [Open-Meteo](https://open-meteo.com/).
