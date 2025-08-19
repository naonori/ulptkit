import os
import numpy as np
import matplotlib.pyplot as plt

from dark_emulator import darkemu
from fastpt.matter_power_spt import (
    P_13_reg, P_22, P_22_J, P_13_b1b3, P_13_P2, P_13_reg_ver2,
    P_22_b1b2, P_22_b2b2
)
from ulptkit.ulpt import ulpt_power_spectrum
from ulptkit.nowiggle import pk_nowiggle
from scipy.interpolate import interp1d
import mcfit

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

# --- Loop over redshift and mass ---
for z in z_list:

    # Obtain linear/nonlinear P(k) and no-wiggle
    P_lin = emu.get_pklin_from_z(k, z)
    P_nl = emu.get_pknl(k, z)
    P_nw = pk_nowiggle(k, omega_cdm, omega_b, h, n_s, P_lin)

    # FAST-PT kernels
    P_window = [0.2, 0.2]
    C_window = 0.75
    n_pad = len(k)
    P22_J = P_22_J(k, P_lin, P_window, C_window, n_pad)
    P22_b1b2 = P_22_b1b2(k, P_lin, P_window, C_window, n_pad)
    P22_b2b2 = P_22_b2b2(k, P_lin, P_window, C_window, n_pad)
    _, P22 = P_22(k, P_lin, P_window, C_window, n_pad)
    P13_reg = P_13_reg(k, P_lin)
    P13_b1b3 = P_13_b1b3(k, P_lin)
    
    # Power spectrum combinations for ULPT
    P_J_b1b1 = P_lin + P22_J + P13_reg
    P_J_b1b3 = P13_b1b3
    P_J_b1b2 = P22_b1b2
    P_J_b2b2 = P22_b2b2

    # ULPT spectrum components
    _, P_ulpt_b1b1 = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_b1b1)
    _, P_ulpt_b1b3 = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_b1b3)
    _, P_ulpt_b1b2 = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_b1b2)
    _, P_ulpt_b2b2 = ulpt_power_spectrum(k, P_lin, CC=3, P_J=P_J_b2b2)

    r, xi_ulpt_b1b1 = mcfit.P2xi(k)(P_ulpt_b1b1)
    r, xi_ulpt_b1b3 = mcfit.P2xi(k)(P_ulpt_b1b3)
    r, xi_ulpt_b1b2 = mcfit.P2xi(k)(P_ulpt_b1b2)
    r, xi_ulpt_b2b2 = mcfit.P2xi(k)(P_ulpt_b2b2)
    r, xi_nw = mcfit.P2xi(k)(P_nw)

    xi_nl = emu.get_xinl(r, z)
    
    # Interpolators for ULPT and emulated spectra
    f_ulpt_b1b1 = interp1d(k, P_ulpt_b1b1, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_ulpt_b1b3 = interp1d(k, P_ulpt_b1b3, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_ulpt_b1b2 = interp1d(k, P_ulpt_b1b2, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_ulpt_b2b2 = interp1d(k, P_ulpt_b2b2, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_nw = interp1d(k, P_nw, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_nl = interp1d(k, P_nl, kind="linear", bounds_error=False, fill_value="extrapolate")
    
    for logM in logM_list:
        M = 10**logM
        P_hm = emu.get_phm_mass(k, M, z)
        P_hh = emu.get_phh_mass(k, M, M, z)
        f_hh = interp1d(k, P_hh, kind="linear", bounds_error=False, fill_value="extrapolate")
        f_hm = interp1d(k, P_hm, kind="linear", bounds_error=False, fill_value="extrapolate")

        xi_hh = emu.get_xiauto_mass(r, M, M, z)
        xi_hm = emu.get_xicross_mass(r, M, z)
        
        # Estimate b1
        b1 = emu.get_bias_mass(M, z)[0,0]
        print("%.2f" % b1)
        
        # --- Covariance (example, update as needed) ---
        Vhh = (2000)**3
        Vhm = (1000)**3
        dk = 0.01
        n_bar = 1.0e-4
        N_mode_hh = 4.0 * np.pi * k_val**2 * dk * Vhh / (2.0 * np.pi)**3
        N_mode_hm = 4.0 * np.pi * k_val**2 * dk * Vhm / (2.0 * np.pi)**3
        cov = (2.0 / N_mode_hh) * (f_hh(k_val) + 1.0 / n_bar) ** 2
        cov_hm = (1.0 / N_mode_hm) * ((f_hh(k_val) + 1.0 / n_bar) * f_nl(k_val) + f_hm(k_val) ** 2)
        
        # --- Data dict ---
        data = {
            "k": k_val,
            "P_ulpt_b1b1": f_ulpt_b1b1(k_val),
            "P_ulpt_b1b3": f_ulpt_b1b3(k_val),
            "P_ulpt_b1b2": f_ulpt_b1b2(k_val),
            "P_ulpt_b2b2": f_ulpt_b2b2(k_val),
            "P_nw": f_nw(k_val),
            "P_nl": f_nl(k_val),
            "P_hh": f_hh(k_val),
            "P_hm": f_hm(k_val),
            "b1": b1,
            "cov": cov,
            "cov_hm": cov_hm,
        }
 
        fig = {
            "k": k,
            "P_ulpt_b1b1": P_ulpt_b1b1,
            "P_ulpt_b1b3": P_ulpt_b1b3,
            "P_ulpt_b1b2": P_ulpt_b1b2,
            "P_ulpt_b2b2": P_ulpt_b2b2,
            "P_nw": P_nw,
            "P_nl": P_nl,
            "P_hh": P_hh,
            "P_hm": P_hm,
        }  

        XX = {
            "r": r,
            "xi_ulpt_b1b1": xi_ulpt_b1b1,
            "xi_ulpt_b1b3": xi_ulpt_b1b3,
            "xi_ulpt_b1b2": xi_ulpt_b1b2,
            "xi_ulpt_b2b2": xi_ulpt_b2b2,
            "xi_nw": xi_nw,
            "xi_nl": xi_nl,
            "xi_hh": xi_hh,
            "xi_hm": xi_hm,
        }  



        # --- Save: filename reflects z and logM ---
        fname = rf"{output_dir}/data_z{z:.1f}_logM{logM:.1f}.npz"
        fname_fig = rf"{output_dir}/fig_z{z:.1f}_logM{logM:.1f}.npz"
        fname_XX = rf"{output_dir}/XX_z{z:.1f}_logM{logM:.1f}.npz"
        np.savez(fname, **data)
        np.savez(fname_fig, **fig)
        np.savez(fname_XX, **XX)
        print(f"Saved: {fname}")


