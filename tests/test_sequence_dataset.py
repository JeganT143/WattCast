import numpy as np
import pytest
import torch

from src.models.sequence_dataset import SequenceDataset


def _data(m=10, f=3):
    X = np.arange(m * f, dtype="float64").reshape(m, f)
    y = np.arange(m, dtype="float64") * 10.0
    return X, y


def test_is_a_torch_dataset():
    X, y = _data()
    assert isinstance(SequenceDataset(X, y, 4), torch.utils.data.Dataset)


@pytest.mark.parametrize("m, L, expected", [(10, 4, 7), (4, 4, 1), (10, 1, 10)])
def test_len_is_m_minus_L_plus_1(m, L, expected):
    X, y = _data(m, 3)
    assert len(SequenceDataset(X, y, L)) == expected


def test_len_is_zero_when_L_exceeds_rows():
    X, y = _data(3, 3)
    assert len(SequenceDataset(X, y, 5)) == 0


def test_item_shapes_and_dtypes():
    X, y = _data(10, 3)
    features, target = SequenceDataset(X, y, 4)[0]
    assert features.shape == (4, 3)
    assert features.dtype == torch.float32
    assert target.shape == torch.Size([])
    assert target.dtype == torch.float32


def test_every_item_pairs_the_window_with_the_target_of_its_last_row():
    X, y = _data(10, 3)
    L = 4
    ds = SequenceDataset(X, y, L)
    for i in range(len(ds)):
        features, target = ds[i]
        assert np.array_equal(features.numpy(), X[i : i + L].astype("float32")), i
        assert float(target) == y[i + L - 1], i
        # X[j, 0] = 3 * j and y[j] = 10 * j: the window's last row is exactly the row whose label this is
        assert float(features[-1, 0]) == 3.0 * (i + L - 1), i


def test_first_and_last_targets_are_pinned():
    X, y = _data(10, 3)
    ds = SequenceDataset(X, y, 4)
    assert float(ds[0][1]) == y[3]
    assert float(ds[len(ds) - 1][1]) == y[9]


def test_negative_index_matches_positive_index():
    X, y = _data(10, 3)
    ds = SequenceDataset(X, y, 4)
    last_features, last_target = ds[len(ds) - 1]
    neg_features, neg_target = ds[-1]
    assert torch.equal(last_features, neg_features)
    assert torch.equal(last_target, neg_target)


def test_out_of_range_index_raises_index_error():
    X, y = _data(10, 3)
    ds = SequenceDataset(X, y, 4)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_dataset_is_independent_of_the_input_arrays():
    X, y = _data(10, 3)
    ds = SequenceDataset(X, y, 4)
    features_before, target_before = ds[0][0].clone(), ds[0][1].clone()
    X[:] = -999.0
    y[:] = -999.0
    features_after, target_after = ds[0]
    assert torch.equal(features_before, features_after)
    assert torch.equal(target_before, target_after)


def test_length_mismatch_raises():
    X, y = _data(10, 3)
    with pytest.raises(ValueError, match="length"):
        SequenceDataset(X, y[:-1], 4)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_targets_raise(bad):
    X, y = _data(10, 3)
    y[5] = bad
    with pytest.raises(ValueError, match="non-finite"):
        SequenceDataset(X, y, 4)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_features_raise(bad):
    X, y = _data(10, 3)
    X[3, 1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        SequenceDataset(X, y, 4)


def test_non_1d_targets_raise():
    X, y = _data(10, 3)
    with pytest.raises(ValueError, match="1-D"):
        SequenceDataset(X, y.reshape(-1, 1), 4)


def test_non_2d_features_raise():
    _, y = _data(10, 3)
    with pytest.raises(ValueError, match="2-D"):
        SequenceDataset(np.zeros(10), y, 4)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_L_raises(bad):
    X, y = _data(10, 3)
    with pytest.raises(ValueError, match="positive"):
        SequenceDataset(X, y, bad)


def test_dataloader_batches_have_the_expected_shapes():
    X, y = _data(10, 3)
    loader = torch.utils.data.DataLoader(SequenceDataset(X, y, 4), batch_size=4, shuffle=False)
    features, targets = next(iter(loader))
    assert features.shape == (4, 4, 3)
    assert targets.shape == (4,)
    assert features.dtype == torch.float32
    assert targets.dtype == torch.float32


def test_dataset_is_independent_of_float32_input_arrays():
    X, y = _data(10, 3)
    X, y = X.astype("float32"), y.astype("float32")
    ds = SequenceDataset(X, y, 4)
    before = [(f.clone(), t.clone()) for f, t in ds]
    X[:] = -999.0
    y[:] = -999.0
    for i, (f_before, t_before) in enumerate(before):
        f_after, t_after = ds[i]
        assert torch.equal(f_before, f_after), i
        assert torch.equal(t_before, t_after), i
