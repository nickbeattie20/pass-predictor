"""Download fresh TLEs from Celestrak into the repo root (run daily by GitHub Actions).

Only overwrites a file when the download is a valid TLE, so a bad response
from Celestrak never replaces a good saved file.
"""
import sys
from pathlib import Path

import requests

NORAD_IDS = [59051, 57166, 25544]  # METEOR M2-4, METEOR M2-3, ISS
ROOT = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "pass-predictor (github.com/nickbeattie20/pass-predictor)"}

failed = 0
for norad_id in NORAD_IDS:
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        lines = [l.rstrip() for l in r.text.strip().splitlines()]
        if len(lines) < 3 or not lines[-2].startswith("1 ") or not lines[-1].startswith("2 "):
            raise ValueError(f"not a TLE: {r.text.strip()[:80]!r}")
        (ROOT / f"{norad_id}.tle").write_text("\n".join(lines[-3:]) + "\n")
        print(f"{norad_id}: updated ({lines[-3].strip()})")
    except Exception as e:
        failed += 1
        print(f"{norad_id}: FAILED ({e}), keeping existing file")

sys.exit(1 if failed == len(NORAD_IDS) else 0)
