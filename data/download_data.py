"""Download the raw input files used by the pipeline into ``data/raw/``.

The files are fetched from a public GitHub repository
(f-linus/natural_gas_consumption_modelling) that archives the original
publications of the German gas market area managers and a Copernicus/C3S ERA5
country-average temperature extract. The download is pinned to one commit so
that the analysis is reproducible. Raw files are not committed to this repo.

Usage:  python data/download_data.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

COMMIT = "be4c9ed1db4e5934c48bbe03470c853721a60091"
BASE = (
    "https://raw.githubusercontent.com/f-linus/natural_gas_consumption_modelling/"
    f"{COMMIT}/data/raw/"
)

# remote file name -> local file name
FILES: dict[str, str] = {
    "NetConnect Germany natural gas consumption.csv": "ncg_consumption.csv",
    "GASPOOL natural gas consumption.csv": "gaspool_consumption.csv",
    "Trading Hub Europe  Publications  Transparency  Aggregated consumption data.csv":
        "the_consumption.csv",
    "Trading Hub Europe imbalance prices.csv": "the_imbalance_prices.csv",
    "H_ERA5_ECMW_T639_TA-_0002m_Euro_NUT0_S197901010000_E202212312300_INS_TIM_01d_"
    "NA-_noc_org_NA_NA---_NA---_NA---.csv": "era5_t2m_nuts0_daily.csv",
}

RAW_DIR = Path(__file__).resolve().parent / "raw"


def download(remote: str, local: Path) -> None:
    """Download one file unless it already exists locally."""
    if local.exists() and local.stat().st_size > 0:
        print(f"[skip] {local.name} already present")
        return
    url = BASE + urllib.parse.quote(remote)
    print(f"[get ] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed https URL)
        content = resp.read()
    local.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()[:12]
    print(f"       -> {local.name} ({len(content) / 1e6:.2f} MB, sha256 {sha}...)")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        download(remote, RAW_DIR / local)
    return 0


if __name__ == "__main__":
    sys.exit(main())
