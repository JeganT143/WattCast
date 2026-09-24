import json

import pytest

from tests.golden import FIXTURE_PATH, compute

# linear_regression's predict() is a BLAS matrix-vector product. The harness now builds its input with
# np.concatenate (C-contiguous) where the old code passed DataFrame.to_numpy() (F-contiguous): identical values,
# different summation order, measured ~1-ulp metric differences. Every other model must stay bit-identical.
BLAS_BACKED = {"linear_regression"}
REL_TOL = 1e-12


def test_baselines_reproduce_golden_fixture():
    golden = json.loads(FIXTURE_PATH.read_text())["results"]
    fresh = compute()["results"]

    assert fresh.keys() == golden.keys()
    for key, expected in golden.items():
        got = fresh[key]
        assert got["n_rows"] == expected["n_rows"], key
        assert got["n_evaluated"] == expected["n_evaluated"], key
        assert got["metrics"].keys() == expected["metrics"].keys(), key
        model = key.split("|")[0]
        for name, value in expected["metrics"].items():
            if model in BLAS_BACKED:
                assert got["metrics"][name] == pytest.approx(value, rel=REL_TOL, abs=0), (key, name)
            else:
                assert got["metrics"][name] == value, (key, name)
