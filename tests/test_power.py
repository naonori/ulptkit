# tests/test_power_pklin.py
import numpy as np
import pathlib
import pytest

from ulptkit import ulpt_power_spectrum, pk_nowiggle


def _load_test_data():
    # Path relative to repo root (adjust if needed)
    fname = pathlib.Path(__file__).parent / "pklin_z0p5.npz"
    if not fname.exists():
        pytest.skip(f"Test data not found: {fname}")
    dat = np.load(fname, allow_pickle=False)
    return dat["k"], dat["P_lin"], float(dat["z"][0])


def test_ulpt_with_darkemu_data():
    k, P_lin, z = _load_test_data()

    k_out, P_ulpt = ulpt_power_spectrum(k, P_lin)[:2]

    # shape consistency
    assert k_out.shape == P_lin.shape
    assert P_ulpt.shape == P_lin.shape

    # finite values
    assert np.all(np.isfinite(P_ulpt))


def test_nowiggle_comparison():
    k, P_lin, z = _load_test_data()

    omega_c = 0.1198
    omega_b = 0.02225
    h = 0.67
    n_s = 0.9645

    P_nw = pk_nowiggle(k, omega_c, omega_b, h, n_s, P_lin)

    assert P_nw.shape == P_lin.shape
    assert np.all(np.isfinite(P_nw))

    # sanity: ULPT and no-wiggle are same order on quasi-linear scales
    k_out, P_ulpt = ulpt_power_spectrum(k, P_lin)[:2]
    ratio = P_ulpt / np.maximum(P_nw, 1e-60)
    assert np.all(ratio > 0)
    assert np.all(ratio < 1e3)


