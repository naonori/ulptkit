#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Halo Correlation-function demo (z=0.5, logM=13.5):
- Build P_J components with FAST-PT
- Combine into P_J_hh, P_J_hm using (b1,b2,b3)
- Call ulpt_power_spectrum with CC=3
- Compare ULPT (xi_hh, xi_hm) to Dark Emulator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from dark_emulator import darkemu
from fastpt.matter_power_spt import (
    P_13_reg, P_22_J, P_13_b1b3, P_22_b1b2, P_22_b2b2
)
from ulptkit import ulpt_power_spectrum, pk_nowiggle
import mcfit

# -----------------------------
# Figure style (linear axes)
# -----------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.usetex": True,      # Set False if LaTeX is not available
    "font.size": 17,
    "axes.labelsize": 19,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
})

# -----------------------------
# Configuration
# -----------------------------
z = 0.5
logM = 13.5
M = 10.0**logM
k = np.logspace(-4, 2, 600)   # [h/Mpc]; correlation will be evaluated on r from P->xi

# -----------------------------
# Cosmology (emulator)
# -----------------------------
omega_b = 0.02225
omega_cdm = 0.1198
Omega_Lambda = 0.6844
ln10e10As = 3.094
n_s = 0.9645
w = -1.0
cosmo_dark = np.array([omega_b, omega_cdm, Omega_Lambda, ln10e10As, n_s, w])

def get_consistent_h(omega_b, omega_c, Omega_Lambda, omega_nu=0.00064):
    """Return h consistent with physical densities."""
    Omega_m = 1.0 - Omega_Lambda
    return float(np.sqrt((omega_b + omega_c + omega_nu) / Omega_m))

# -----------------------------
# Emulator data and linear xi
# -----------------------------
emu = darkemu.base_class()
emu.set_cosmology(cosmo_dark)
P_lin = emu.get_pklin_from_z(k, z)
r, xi_lin = mcfit.P2xi(k)(P_lin)

xi_hh_emu = emu.get_xiauto_mass(r, M, M, z)
xi_hm_emu = emu.get_xicross_mass(r, M, z)

# -----------------------------
# Bias parameters
# -----------------------------
b1 = 2.05
b2 = -2.68
b3 = 0.59

# -----------------------------
# FAST-PT → P_J components
# -----------------------------
P_window = [0.2, 0.2]
C_window = 0.75
n_pad    = len(k)

P_J_b1b1 = P_lin + P_22_J(k, P_lin, P_window, C_window, n_pad) + P_13_reg(k, P_lin)
P_J_b1b2 = P_22_b1b2(k, P_lin, P_window, C_window, n_pad)
P_J_b2b2 = P_22_b2b2(k, P_lin, P_window, C_window, n_pad)
P_J_b1b3 = P_13_b1b3(k, P_lin)

# -----------------------------
# Combine P_J for tracers (hh, hm)
# -----------------------------
P_J_hh = (b1**2)*P_J_b1b1 + (b1*b2)*P_J_b1b2 + (b2**2)*P_J_b2b2 + (b1*b3)*P_J_b1b3
P_J_hm = (b1)*P_J_b1b1 + 0.5*b2*P_J_b1b2 + 0.5*b3*P_J_b1b3

# -----------------------------
# ULPT with combined P_J (CC=3); then xi(r) via Hankel transform
# -----------------------------
_, P_hh_ulpt = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_hh)
_, P_hm_ulpt = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_hm)

_, xi_hh_ulpt = mcfit.P2xi(k)(P_hh_ulpt)
_, xi_hm_ulpt = mcfit.P2xi(k)(P_hm_ulpt)

# -----------------------------
# Figure: top (r^2 xi), bottom (residual [%]) — linear axes
# -----------------------------
fig = plt.figure(figsize=(9.2, 6.8))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.06)

# Top panel: r^2 xi
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(r, r**2 * xi_hh_emu, "o", ms=5.5, mfc="white", mec="magenta",  mew=1.2, label=r"Emu $\xi_{\mathrm{hh}}$")
ax1.plot(r, r**2 * xi_hm_emu, "^", ms=5.5, mfc="white", mec="royalblue", mew=1.2, label=r"Emu $\xi_{\mathrm{hm}}$")
ax1.plot(r, r**2 * xi_hh_ulpt, "-",  lw=2.2, color="magenta",    label=r"ULPT $\xi_{\mathrm{hh}}$")
ax1.plot(r, r**2 * xi_hm_ulpt, "--", lw=2.2, color="royalblue",  label=r"ULPT $\xi_{\mathrm{hm}}$")
ax1.axhline(1.0, color="k", lw=1.0)
ax1.set_xlim(15, 115)
ax1.set_ylim(5, 95)
ax1.xaxis.set_major_locator(MaxNLocator(7))
ax1.yaxis.set_major_locator(MaxNLocator(7))
ax1.grid(True, alpha=0.25)
ax1.set_ylabel(r"$r^2\,\xi(r)$")
ax1.set_title(fr"Halo correlation functions at $z={z}$, $\log_{{10}}(M/M_\odot)={logM}$")
ax1.legend(loc="upper right", frameon=True)
ax1.tick_params(labelbottom=False)

# Bottom panel: percent residuals (ULPT vs Emulator)
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(r, 100.0 * (xi_hh_ulpt/np.maximum(xi_hh_emu,1e-60) - 1.0),
         color="magenta",   lw=2.2,           label=r"$\Delta \xi_{\mathrm{hh}}$")
ax2.plot(r, 100.0 * (xi_hm_ulpt/np.maximum(xi_hm_emu,1e-60) - 1.0),
         color="royalblue", lw=2.2, ls="--", label=r"$\Delta \xi_{\mathrm{hm}}$")
ax2.axhline(0, color="k", lw=1.0)
ax2.fill_between(r, -2, 2, color="forestgreen", alpha=0.18)  # ±2% band
ax2.set_xlim(15, 115)
ax2.set_ylim(-3, 3)
ax2.xaxis.set_major_locator(MaxNLocator(7))
ax2.yaxis.set_major_locator(MaxNLocator(7))
ax2.grid(True, alpha=0.25)
ax2.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$")
ax2.set_ylabel(r"$100\times(\xi_{\rm ULPT}/\xi_{\rm Emu}-1)$ [\%]", fontsize=13)

plt.tight_layout()
# plt.savefig("halo_corr_z0p5_logM13p5_linear.png", dpi=220, bbox_inches="tight")  # optional
plt.show()

