import json

from tests.golden import FIXTURE_PATH, compute


def test_baselines_are_bit_identical_to_golden_fixture():
    golden = json.loads(FIXTURE_PATH.read_text())
    fresh = compute()
    assert fresh["results"] == golden["results"], (
        f"golden meta: {golden['meta']}\nfresh meta:  {fresh['meta']}"
    )
