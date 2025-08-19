import os
import numpy as np
import matplotlib.pyplot as plt

from dark_emulator import darkemu

from fastpt.matter_power_spt import (
    P_13_reg, P_22, P_22_J, P_13_b1b3, P_22_b1b2, P_22_b2b2
)

from ulptkit.ulpt import ulpt_power_spectrum, pk_nowiggle
from scipy.interpolate import interp1d
import mcfit

import time

start = time.time()

def get_consistent_h(omega_b, omega_c, Omega_Lambda, omega_nu=0.00064):
    Omega_m = 1.0 - Omega_Lambda
    h_squared = (omega_b + omega_c + omega_nu) / Omega_m
    return np.sqrt(h_squared)

output_dir = "Results/calc_P"
os.makedirs(output_dir, exist_ok=True)

# --- Analysis ranges ---
z_list = [0.0, 0.5, 1.0]
logM_list = [12.5, 13.0, 13.5]
k_val = np.linspace(0.01, 0.4, 40)

# --- Grids ---
kmax = 100
kmin = 1.0e-4
k = np.logspace(np.log10(kmin), np.log10(kmax), 600)

emu = darkemu.base_class()

# --- Cosmological parameters ---
omega_b = 0.02225
omega_cdm = 0.1198
Omega_Lambda = 0.6844
ln10e10As = 3.094
n_s = 0.9645
w = -1.0

h = get_consistent_h(omega_b, omega_cdm, Omega_Lambda)
#cosmo_dark = np.array([omega_b, omega_cdm, Omega_Lambda, ln10e10As, n_s, w])
#emu.set_cosmology(cosmo_dark)

z = z_list[1]

P_lin = emu.get_pklin_from_z(k, z)
P_nl = emu.get_pknl(k, z)
P_nw = pk_nowiggle(k, omega_cdm, omega_b, h, n_s, P_lin)

logM = logM_list[1]
M = 10**logM
P_hm = emu.get_phm_mass(k, M, z)
P_hh = emu.get_phh_mass(k, M, M, z)

# FAST-PT kernels
P_window = [0.2, 0.2]
C_window = 0.75
n_pad = len(k)
P22_J = P_22_J(k, P_lin, P_window, C_window, n_pad)
P13_reg = P_13_reg(k, P_lin)
P_J_b1b1 = P_lin + P22_J + P13_reg
P_J_b1b2 = P_22_b1b2(k, P_lin, P_window, C_window, n_pad)
P_J_b2b2 = P_22_b2b2(k, P_lin, P_window, C_window, n_pad)
P_J_b1b3 = P_13_b1b3(k, P_lin)

b1 = 1.44
#b2 = 1.44
#b3 = -0.77
#Ne = 69.59
b2 = 0.0
b3 = 0.0
Ne = 0.0

P_J_hh = b1**2 * P_J_b1b1 + b1 * b3 * P_J_b1b3 + b1 * b2 * P_J_b1b2 + b2**2 * P_J_b2b2
P_J_hm = b1 * P_J_b1b1 + b3 * P_J_b1b3 / 2.0 + b2 * P_J_b1b2 / 2.0

# ULPT spectrum components
_, P_hh_ulpt = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_hh)
_, P_hm_ulpt = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_hm)
P_hh_ulpt = P_hh_ulpt + Ne

end = time.time()
print(f"Elapsed time: {end - start:.3f} sec")

X = np.array([k, P_hh_ulpt, P_hm_ulpt]).T
np.savetxt("test.txt", X)

#plt.plot(k, P_hh_ulpt/P_nw/b1**2)
#plt.plot(k, P_hm_ulpt/P_nw/b1)
#plt.plot(k, P_hh/P_nw/b1**2, "o")
#plt.plot(k, P_hm/P_nw/b1, "^")
##plt.plot(k, 100*(P_ulpt-P_nl)/P_nl)
#plt.xlim(0.01,0.3)
#plt.ylim(0.8,1.5)
##plt.ylim(-8,8)
#plt.show()
#
