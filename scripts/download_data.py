"""Download the UCI Appliances Energy Prediction dataset to data/raw/ and
verify it is byte-identical to the copy every recorded result was produced
from (decisions.md, ADR-018).

    python -m scripts.download_data

Only the Python standard library is used. The file is written only after
the checksum matches; an existing file with the right checksum is kept.
"""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from config.paths import RAW_DATA_PATH

URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
MEMBER = "energydata_complete.csv"
EXPECTED_MD5 = "69ef922b5fcafcd49097cfc09e07167e"  # also recorded in data/raw/energy_data_set.csv.dvc


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(target: Path = RAW_DATA_PATH) -> None:
    if target.exists() and md5(target) == EXPECTED_MD5:
        print(f"{target} already present and verified")
        return

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.zip"
        print(f"downloading {URL} (about 12 MB; the UCI server can be slow)")
        with urllib.request.urlopen(URL, timeout=600) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)

        with zipfile.ZipFile(archive) as zf:
            extracted = Path(zf.extract(MEMBER, tmp))

        actual = md5(extracted)
        if actual != EXPECTED_MD5:
            sys.exit(f"checksum mismatch: expected {EXPECTED_MD5}, got {actual}; nothing written")

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(extracted, target)
    print(f"wrote {target} (md5 {EXPECTED_MD5})")


if __name__ == "__main__":
    main()
