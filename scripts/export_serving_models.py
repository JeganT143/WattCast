"""Export the registered @champion bundle of every model family from the
local MLflow registry into models/, plus the pre-test seed history the
serving buffer starts from (decisions.md, ADR-016).

models/ is what the Streamlit app and the FastAPI service load at runtime,
so neither needs MLflow or the raw dataset. Run this after (re)registering
a champion:

    python -m scripts.export_serving_models

Before overwriting anything, the exported bundles are reloaded and must
produce bit-identical predictions to the registry bundles on the seeded
buffer; the script aborts otherwise.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from config.features import SPLIT_VAL_END
from config.mlflow_config import TRACKING_URI
from config.paths import MODELS_DIR, RAW_DATA_PATH, SEED_HISTORY_PATH
from src.serving.buffer import SEQUENCE_MODEL_RAW_HISTORY
from src.serving.bundle import LOADERS, ModelBundle, load_bundles, save_bundle
from src.serving.registry import CHAMPION_ALIAS, load_champion, registered_model_name
from src.serving.service import create_service


def _registry_bundles(tracking_uri: str) -> dict[str, ModelBundle]:
    client = MlflowClient(tracking_uri=tracking_uri)
    bundles = {}
    for family in LOADERS:
        name = registered_model_name(family)
        version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        bundle = load_champion(tracking_uri, family=family)
        schema = {
            **bundle.schema,
            "registered_model": name,
            "model_version": int(version.version),
            "run_id": version.run_id,
        }
        bundles[family] = ModelBundle(bundle.forecaster, bundle.scaler, schema)
    return bundles


def _seed_history(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    pre_test = raw[raw["date"] < pd.Timestamp(SPLIT_VAL_END)].sort_values("date")
    return pre_test.tail(SEQUENCE_MODEL_RAW_HISTORY)[["date", "Appliances"]].reset_index(drop=True)


def _predictions(bundles: dict[str, ModelBundle], seed: pd.DataFrame) -> dict[str, float]:
    service = create_service(bundles, seed)
    return {r["model_family"]: r["prediction_wh"] for r in service.predict(list(bundles))}


def main(out_dir: Path = MODELS_DIR, tracking_uri: str = TRACKING_URI, raw_path: Path = RAW_DATA_PATH) -> None:
    bundles = _registry_bundles(tracking_uri)
    seed = _seed_history(raw_path)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "models"
        for family, bundle in bundles.items():
            save_bundle(bundle, staging / family)
        seed.to_csv(staging / SEED_HISTORY_PATH.name, index=False)

        reloaded_seed = pd.read_csv(staging / SEED_HISTORY_PATH.name, parse_dates=["date"])
        expected = _predictions(bundles, seed)
        actual = _predictions(load_bundles(staging), reloaded_seed)
        for family, value in expected.items():
            if not np.array_equal(value, actual[family]):
                raise RuntimeError(f"{family}: exported prediction {actual[family]} != registry {value}")

        out_dir = Path(out_dir)
        for family in bundles:
            shutil.rmtree(out_dir / family, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, out_dir, dirs_exist_ok=True)

    for family, value in expected.items():
        print(f"exported {family:<17} v{bundles[family].schema['model_version']}  prediction check {value:.3f} Wh")
    print(f"wrote {len(seed)} seed rows ending {seed['date'].iloc[-1]} to {out_dir / SEED_HISTORY_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    main(out_dir=args.out)
