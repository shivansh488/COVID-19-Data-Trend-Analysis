"""
Download COVID-19 time-series data from the Johns Hopkins CSSE GitHub repository.

The repository was archived on 2023-03-10, so these URLs point to the final
version of the dataset (Jan 22, 2020 – Mar 9, 2023).
"""

import os
import urllib.request
import sys

# ── URLs for the three main time-series files ────────────────────────────────
BASE_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)

FILES = {
    "time_series_covid19_confirmed_global.csv": BASE_URL + "time_series_covid19_confirmed_global.csv",
    "time_series_covid19_deaths_global.csv": BASE_URL + "time_series_covid19_deaths_global.csv",
    "time_series_covid19_recovered_global.csv": BASE_URL + "time_series_covid19_recovered_global.csv",
}

# ── Output directory ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")


def download_all(force: bool = False) -> None:
    """Download all three CSV files into data/raw/.

    Parameters
    ----------
    force : bool
        If True, re-download even if the file already exists.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    for filename, url in FILES.items():
        dest = os.path.join(RAW_DIR, filename)

        if os.path.exists(dest) and not force:
            print(f"  ✔ {filename} already exists, skipping (use --force to re-download)")
            continue

        print(f"  ⬇ Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"    Saved → {dest}  ({size_kb:.1f} KB)")
        except Exception as exc:
            print(f"    ✗ Failed to download {filename}: {exc}", file=sys.stderr)

    print("\nDone. All files are in:", RAW_DIR)


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("═" * 60)
    print("  COVID-19 Dataset Downloader (Johns Hopkins CSSE)")
    print("═" * 60)
    download_all(force=force)
