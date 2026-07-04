from __future__ import annotations

import numpy as np

from bvs.pq import ProductQuantizer


def test_code_shape_and_dtype(dataset):
    pq = ProductQuantizer(dataset["dim"], m=8, ksub=256).train(dataset["vecs"])
    codes = pq.encode(dataset["vecs"][:1000])
    assert codes.shape == (1000, 8)
    assert codes.dtype == np.uint8
    assert pq.code_bytes_per_vector() == 8


def test_reconstruction_error_bounded(dataset):
    """PQ reconstruction MSE must be small relative to the data's own variance."""
    x = dataset["vecs"]
    pq = ProductQuantizer(dataset["dim"], m=16, ksub=256).train(x)
    mse = pq.reconstruction_error(x[:2000])
    # Total variance of the data (sum over dims of per-dim variance).
    data_energy = float(np.mean(np.sum((x - x.mean(0)) ** 2, axis=1)))
    assert mse > 0.0
    assert mse < 0.35 * data_energy, (mse, data_energy)


def test_more_subspaces_reduce_error(dataset):
    """Finer subdivision (larger M) must not increase reconstruction error."""
    x = dataset["vecs"][:4000]
    err = {}
    for m in (4, 8, 16):
        pq = ProductQuantizer(dataset["dim"], m=m, ksub=256).train(x)
        err[m] = pq.reconstruction_error(x)
    assert err[16] <= err[8] <= err[4] + 1e-6, err


def test_adc_matches_bruteforce_on_decoded(dataset):
    """ADC squared distance ~= exact distance to the *decoded* (reconstructed) db."""
    x = dataset["vecs"]
    pq = ProductQuantizer(dataset["dim"], m=16, ksub=256).train(x)
    db = x[:500]
    codes = pq.encode(db)
    recon = pq.decode(codes)
    q = dataset["queries"][0]
    table = pq.compute_distance_tables(q[None, :])[0]
    adc = ProductQuantizer.adc_distances(table, codes)
    exact = np.sum((recon - q) ** 2, axis=1)
    assert np.allclose(adc, exact, rtol=1e-3, atol=1e-2)


def test_encode_decode_roundtrip_shapes(dataset):
    pq = ProductQuantizer(dataset["dim"], m=8).train(dataset["vecs"])
    codes = pq.encode(dataset["vecs"][:10])
    recon = pq.decode(codes)
    assert recon.shape == (10, dataset["dim"])
    assert recon.dtype == np.float32


def test_invalid_dim_divisibility():
    import pytest

    with pytest.raises(ValueError):
        ProductQuantizer(dim=100, m=7)
