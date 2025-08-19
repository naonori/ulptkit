# SPDX-License-Identifier: MIT
# Unified Lagrangian Perturbation Theory (ULPT) power spectrum utilities.
# Notes:
# - This module provides a ULPT power spectrum.
# - In addition, it includes Eisenstein–Hu-style "no-wiggle" helper functions.
#   These no-wiggle functions are not used in the ULPT power spectrum calculation
#   itself, but are provided solely for comparison purposes in analyses.
# - Numerical behavior is intentionally preserved for reproducibility.

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import mcfit


def ulpt_power_spectrum(k, P_lin, CC=2, P_J=None, return_components=False):
    """
    Compute the ULPT power spectrum on the input wavenumber grid.

    Parameters
    ----------
    k : array_like
        Wavenumber array [h/Mpc]. Must be 1D, strictly increasing, and match `P_lin`.
    P_lin : array_like
        Linear matter power spectrum [(Mpc/h)^3] sampled on `k`.
    CC : int, optional
        Highest CC order to include (supported: 0..4). Default is 2.
        CC=0 adds the leading correction; higher values include more terms.
    P_J : array_like or None, optional
        SPT-corrected baseline spectrum or equivalent input on `k`.
        If None, `P_lin` is used (copy) as the baseline.
        In typical analyses, `P_J` may include 1-loop pieces assembled externally
        (e.g., with FAST-PT extensions such as P_22_J, etc.).
    return_components : bool, optional
        If True, also return a list of components [P_CF, P_CC0, P_CC1, ...].

    Returns
    -------
    k : ndarray
        The input k array (returned for convenience).
    P_ulpt : ndarray
        ULPT power spectrum on `k`.
    components : list of ndarrays, optional
        Returned only when `return_components=True`:
        [P_CF, P_CC0, P_CC1, (P_CC2, ... up to requested CC)].

    Notes
    -----
    - Units follow standard large-scale structure conventions as indicated.
    - Numerical choices (e.g., small-r regularization and transform setup) are
      preserved to ensure bitwise-stable results across releases where possible.
    """
    # Use P_lin as the SPT-like baseline if not provided (do not modify user input).
    if P_J is None:
        P_J = P_lin.copy()

    # Correlation function and displacement correlators via Hankel transforms (mcfit)
    r, xi = mcfit.P2xi(k)(P_J)
    r, sigma0 = mcfit.P2xi(k, l=0)(P_lin / k**2)
    r, sigma2 = mcfit.P2xi(k, l=2)(P_lin / k**2)

    # ULPT convention: average over spatial dimensions
    sigma0 = sigma0 / 3.0
    sigma2 = sigma2 / 3.0

    # sigma0_bar = ∫ P_lin dk / (6 π^2)
    sigma0_bar = simpson(P_lin, x=k) / (6.0 * np.pi**2)

    # Small-r stabilization for sigma0(r), sigma2(r); preserves values at r_cut
    sigma0_new, sigma2_new = fix_sigma0_sigma2_small_r(
        r, sigma0, sigma2, sigma0_bar, k.max()
    )

    # Build ULPT components
    P_CF, *P_CC = compute_power_terms(
        r, xi, sigma0_new, sigma2_new, sigma0_bar, k, P_J, CC=CC
    )

    # Final spectrum = CF + sum(CC terms up to requested order)
    P_ulpt = P_CF.copy()
    for i in range(CC + 1):
        P_ulpt += P_CC[i]

    if return_components:
        return k, P_ulpt, [P_CF] + P_CC
    return k, P_ulpt


def safe_exp(x):
    """
    Exponential with clipping to prevent overflow while preserving asymptotics.

    Parameters
    ----------
    x : array_like
        Input to the exponential.

    Returns
    -------
    ndarray
        np.exp(x) with large magnitudes clipped to a safe range.
    """
    return np.exp(np.clip(x, -700.0, 700.0))


def fix_sigma0_sigma2_small_r(r, sigma0, sigma2, sigma0_bar, kmax):
    """
    Stabilize sigma0(r) and sigma2(r) at small radii via a linear ramp.

    The ramp preserves the exact values at r=r_cut and approaches:
      sigma0(0) -> sigma0_bar,  sigma2(0) -> 0

    Parameters
    ----------
    r : ndarray
        Real-space separation grid [Mpc/h] from Hankel transforms.
    sigma0, sigma2 : ndarray
        Displacement correlators (already divided by 3).
    sigma0_bar : float
        Isotropic displacement variance from P_lin.
    kmax : float
        Maximum k (not used in the current fixed-r_cut choice, kept for clarity).

    Returns
    -------
    sigma0_new, sigma2_new : ndarray
        Stabilized correlators, identical to inputs beyond r_cut.

    Notes
    -----
    - The fixed r_cut = 0.75 is retained for reproducibility of published results.
    - Vectorized implementation; numerical outputs are preserved.
    """
    # Empirically suggested alternative:
    # r_cut_alt = 10 * (2 * np.pi / kmax)
    r_cut = 0.75

    idx_cut = np.searchsorted(r, r_cut)
    if idx_cut == 0:
        return sigma0.copy(), sigma2.copy()

    # End point of the ramp (inclusive)
    if idx_cut < len(r):
        r_cut_val = r[idx_cut]
        s0_cut = sigma0[idx_cut]
        s2_cut = sigma2[idx_cut]
    else:
        r_cut_val = r[-1]
        s0_cut = sigma0[-1]
        s2_cut = sigma2[-1]

    sigma0_new = sigma0.copy()
    sigma2_new = sigma2.copy()

    # Linear ramp from r=0 to r=r_cut
    sl = slice(0, idx_cut + 1)
    frac = r[sl] / r_cut_val
    sigma0_new[sl] = sigma0_bar + (s0_cut - sigma0_bar) * frac
    sigma2_new[sl] = (s2_cut - 0.0) * frac  # enforce sigma2(0) = 0

    return sigma0_new, sigma2_new


def compute_power_terms(r, xi, sigma0, sigma2, sigma0_bar, k, P_J, CC=2):
    """
    Assemble ULPT components:
      - P_CF  : Gaussian-damped baseline
      - P_CC* : corrective CC terms up to the requested order

    Parameters
    ----------
    r : ndarray
        Real-space separation grid from Hankel transforms.
    xi : ndarray
        Correlation function corresponding to the baseline P_J.
    sigma0, sigma2 : ndarray
        Displacement correlators after small-r stabilization.
    sigma0_bar : float
        Isotropic displacement variance from P_lin.
    k : ndarray
        Target k grid for the final spectrum and interpolation.
    P_J : ndarray
        Baseline spectrum (SPT-like) on `k`.
    CC : int
        Highest CC order to include.

    Returns
    -------
    tuple
        (P_CF, P_CC0[, P_CC1[, P_CC2[, P_CC3[, P_CC4]]]])

    Performance
    -----------
    - xi2P operators are built once per multipole and reused.
    - Common factors (k^2, exponentials, sigma2 powers times xi) are precomputed.
    - Interpolations remain cubic with the same settings to preserve results.
    """
    # Precompute simple factors
    k = np.asarray(k)
    k2 = k**2
    exp_kbar = safe_exp(-k2 * sigma0_bar)

    # Precompute sigma2^n * xi terms used across CC levels
    xi_arr = xi
    s2 = sigma2
    s2_xi = s2 * xi_arr
    s2sq_xi = s2_xi * s2
    s2cu_xi = s2sq_xi * s2
    s2p4_xi = s2cu_xi * s2

    # Prepare xi2P operators (reuse)
    needed_ells = {0}
    if CC >= 1:
        needed_ells |= {2}
    if CC >= 2:
        needed_ells |= {2, 4}
    if CC >= 3:
        needed_ells |= {2, 4, 6}
    if CC >= 4:
        needed_ells |= {2, 4, 6, 8}

    xi2P_ops = {}
    kgrids = {}
    for ell in sorted(needed_ells):
        op = mcfit.xi2P(r, l=ell)
        # Initialize with zeros once to retrieve the operator's k-grid
        k_out, _ = op(np.zeros_like(r))
        xi2P_ops[ell] = op
        kgrids[ell] = k_out

    # CF term: Gaussian damping times the baseline
    P_CF = exp_kbar * P_J

    # Allocate CC arrays on demand
    P_CC_list = []
    if CC >= 0:
        P_CC0 = np.empty_like(k)
        P_CC_list.append(P_CC0)
    if CC >= 1:
        P_CC1 = np.empty_like(k)
        P_CC_list.append(P_CC1)
    if CC >= 2:
        P_CC2 = np.empty_like(k)
        P_CC_list.append(P_CC2)
    if CC >= 3:
        P_CC3 = np.empty_like(k)
        P_CC_list.append(P_CC3)
    if CC >= 4:
        P_CC4 = np.empty_like(k)
        P_CC_list.append(P_CC4)

    # Exponential difference used repeatedly
    d_sigma = (sigma0_bar - sigma0)

    # Main loop over k (order preserved for reproducibility)
    for i, ki in enumerate(k):
        e = safe_exp(-k2[i] * d_sigma)  # vector over r

        if CC >= 0:
            # CC0: (e - exp(-k^2*sigma0_bar)) * xi  -> ℓ=0 transform
            arr0 = (e - exp_kbar[i]) * xi_arr
            k0, y0 = xi2P_ops[0](arr0)
            f0 = interp1d(k0, y0, kind="cubic", bounds_error=False, fill_value="extrapolate")
            P_CC0[i] = f0(ki)

        if CC >= 1:
            # CC1: e * (sigma2 * xi) * (2 k^2)  -> ℓ=2 transform
            arr1 = e * s2_xi * (2.0 * k2[i])
            k2g, y2 = xi2P_ops[2](arr1)
            f2 = interp1d(k2g, y2, kind="cubic", bounds_error=False, fill_value="extrapolate")
            P_CC1[i] = f2(ki)

        if CC >= 2:
            # CC2: e * (sigma2^2 * xi) -> combine ℓ=0,2,4 with fixed coefficients
            arr2 = e * s2sq_xi
            k0g, y0g = xi2P_ops[0](arr2)
            k2g, y2g = xi2P_ops[2](arr2)
            k4g, y4g = xi2P_ops[4](arr2)
            f0 = interp1d(k0g, y0g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f2 = interp1d(k2g, y2g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f4 = interp1d(k4g, y4g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            coef = ((2.0 * k2[i]) ** 2) / 2.0
            P_CC2[i] = coef * (f0(ki) / 5.0 + (2.0 / 7.0) * f2(ki) + (18.0 / 35.0) * f4(ki))

        if CC >= 3:
            # CC3: e * (sigma2^3 * xi) -> ℓ=0,2,4,6
            arr3 = e * s2cu_xi
            k0g, y0g = xi2P_ops[0](arr3)
            k2g, y2g = xi2P_ops[2](arr3)
            k4g, y4g = xi2P_ops[4](arr3)
            k6g, y6g = xi2P_ops[6](arr3)
            f0 = interp1d(k0g, y0g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f2 = interp1d(k2g, y2g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f4 = interp1d(k4g, y4g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f6 = interp1d(k6g, y6g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            coef = ((2.0 * k2[i]) ** 3) / 6.0
            P_CC3[i] = coef * (
                (2.0 / 35.0) * f0(ki)
                + (291.0 / 154.0) * f2(ki)
                + (756.0 / 2695.0) * f4(ki)
                + (18.0 / 77.0) * f6(ki)
            )

        if CC >= 4:
            # CC4: e * (sigma2^4 * xi) -> ℓ=0,2,4,6,8
            arr4 = e * s2p4_xi
            k0g, y0g = xi2P_ops[0](arr4)
            k2g, y2g = xi2P_ops[2](arr4)
            k4g, y4g = xi2P_ops[4](arr4)
            k6g, y6g = xi2P_ops[6](arr4)
            k8g, y8g = xi2P_ops[8](arr4)
            f0 = interp1d(k0g, y0g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f2 = interp1d(k2g, y2g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f4 = interp1d(k4g, y4g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f6 = interp1d(k6g, y6g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            f8 = interp1d(k8g, y8g, kind="cubic", bounds_error=False, fill_value="extrapolate")
            coef = ((2.0 * k2[i]) ** 4) / 24.0
            P_CC4[i] = coef * (
                (3.0 / 35.0) * f0(ki)
                + (133.0 / 512.0) * f2(ki)
                + (687.0 / 1872.0) * f4(ki)
                + (1441.0 / 7704.0) * f6(ki)
                + (1451.0 / 14408.0) * f8(ki)
            )

    return (P_CF, *P_CC_list)


# --- No-wiggle helpers (English comments; formulas unchanged) ---

def P_nowiggle_EH(k, omega_cdm, omega_b, h, n_s):
    """
    Eisenstein–Hu-style 'no-wiggle' baseline P(k).

    Parameters
    ----------
    k : array_like
        Wavenumber grid [h/Mpc].
    omega_cdm : float
        Cold dark matter density parameter times h^2 (Ω_cdm h^2) or consistent proxy.
    omega_b : float
        Baryon density parameter times h^2 (Ω_b h^2) or consistent proxy.
    h : float
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
    n_s : float
        Scalar spectral index.

    Returns
    -------
    ndarray
        Unnormalized no-wiggle power on `k` (Eisenstein–Hu transfer with tilt).

    Notes
    -----
    The expression retains the original factors to preserve numerical results.
    """
    theta = 2.725 / 2.7
    omega_m = omega_cdm + omega_b
    obh2 = omega_b
    omh2 = omega_m
    s = 44.5 * h * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2 ** 0.75)
    alpha = 1 - 0.328 * np.log(431.0 * omh2) * (omega_b / omega_m) \
            + 0.38 * np.log(22.3 * omh2) * (omega_b / omega_m) ** 2
    gamma = (omh2 / h) * (alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))
    q = k * theta ** 2 / gamma
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    T_nw = L0 / (L0 + C0 * q ** 2)
    P_nw = k ** n_s * T_nw ** 2
    return P_nw


def pk_nowiggle(k, omega_cdm, omega_b, h, n_s, P_lin):
    """
    No-wiggle P(k) normalized to match the sigma8 of a reference P_lin.

    Parameters
    ----------
    k : array_like
        Wavenumber grid [h/Mpc].
    omega_cdm, omega_b, h, n_s : float
        Cosmological parameters as in `P_nowiggle_EH`.
    P_lin : array_like
        Reference linear power [(Mpc/h)^3] on `k`.

    Returns
    -------
    ndarray
        No-wiggle spectrum on `k`, rescaled to match sigma8(P_lin).
    """
    P_nw_tmp = P_nowiggle_EH(k, omega_cdm, omega_b, h, n_s)
    sigma8_lin = sigma8_from_pk(k, P_lin)
    sigma8_nw = sigma8_from_pk(k, P_nw_tmp)
    P_nw = P_nw_tmp * (sigma8_lin / sigma8_nw) ** 2
    return P_nw


def sigma8_from_pk(k, pk):
    """
    Compute sigma8 from P(k) using a spherical top-hat window of R = 8 Mpc/h.

    Parameters
    ----------
    k : array_like
        Wavenumber grid [h/Mpc].
    pk : array_like
        Power spectrum [(Mpc/h)^3] on `k`.

    Returns
    -------
    float
        sigma8 value derived from the input `pk`.
    """
    R = 8.0
    x = k * R
    W = np.ones_like(x)
    mask = x != 0
    W[mask] = 3.0 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask])) / (x[mask] ** 3)
    integrand = k ** 2 * pk * W ** 2
    sigma8_sq = simpson(integrand, x=k) / (2.0 * np.pi ** 2)
    return np.sqrt(sigma8_sq)

