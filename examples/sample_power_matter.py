#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matter Power-spectrum demo (z=0.5):
- Build P_J components with FAST-PT
- Call ulpt_power_spectrum with CC=3
- Compare ULPT to Dark Emulator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from dark_emulator import darkemu
from fastpt.matter_power_spt import (
    P_13_reg, P_22_J
)
from ulptkit import ulpt_power_spectrum, pk_nowiggle

# -----------------------------
# Figure style (linear axes)
# -----------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.usetex": True,        # Set False if LaTeX is not available
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
k = np.logspace(-4, 2, 600)   # [h/Mpc]; plotting range is 0.01–0.30/0.41 below

# -----------------------------
# Cosmology (emulator & no-wiggle)
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

h = get_consistent_h(omega_b, omega_cdm, Omega_Lambda)

# -----------------------------
# Emulator data
# -----------------------------
emu = darkemu.base_class()
emu.set_cosmology(cosmo_dark)
P_lin = emu.get_pklin_from_z(k, z)
P_nl_emu = emu.get_pknl(k, z)
P_nw  = pk_nowiggle(k, omega_cdm, omega_b, h, n_s, P_lin)

# -----------------------------
# FAST-PT → P_J (one-loop combination for matter)
# -----------------------------
P_window = [0.2, 0.2]
C_window = 0.75
n_pad    = len(k)

P_J_1loop = P_lin + P_22_J(k, P_lin, P_window, C_window, n_pad) + P_13_reg(k, P_lin)

# -----------------------------
# ULPT with P_J (CC=3)
# -----------------------------
# ulpt_power_spectrum returns (k_out, P_ULPT, ...); use P_ULPT only.
_, P_nl_ulpt = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_1loop)

# -----------------------------
# Figure: top (P/P_nw), bottom (residual [%]) — linear axes
# -----------------------------
fig = plt.figure(figsize=(9.2, 6.8))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.06)

# Top panel: P/P_nw
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k, P_nl_emu/np.maximum(P_nw,1e-60),
         "o", ms=5.5, mfc="white", mec="royalblue", mew=1.2, label=r"Emu")
ax1.plot(k, P_nl_ulpt/np.maximum(P_nw,1e-60),
         "-", lw=2.2, color="magenta", label=r"ULPT")
ax1.axhline(1.0, color="k", lw=1.0)
ax1.set_xlim(0.01, 0.41)
ax1.set_ylim(0.83, 1.75)
ax1.yaxis.set_major_locator(MaxNLocator(7))
ax1.grid(True, alpha=0.25)
ax1.set_ylabel(r"$P/P_{\mathrm{nw}}$")
ax1.set_title(fr"Matter power spectra at $z={z}$")
ax1.legend(loc="upper left", frameon=True)
ax1.tick_params(labelbottom=False)

# Bottom panel: percent residuals (ULPT vs Emulator)
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(k, 100.0*(P_nl_ulpt/np.maximum(P_nl_emu,1e-60) - 1.0),
         color="magenta", lw=2.2, label=r"$\Delta P$")
ax2.axhline(0, color="k", lw=1.0)
ax2.fill_between(k, -3, 3, color="forestgreen", alpha=0.18)  # ±3% band
ax2.set_xlim(0.01, 0.41)
ax2.set_ylim(-5.0, 5.0)
ax2.yaxis.set_major_locator(MaxNLocator(7))
ax2.grid(True, alpha=0.25)
ax2.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
ax2.set_ylabel(r"$100\times(P_{\rm ULPT}/P_{\rm Emu}-1)$ [\%]", fontsize=13)

#plt.tight_layout()
plt.savefig("figures/power_z0p5_matter.png", dpi=220, bbox_inches="tight")
plt.show()







