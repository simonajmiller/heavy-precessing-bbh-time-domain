import numpy as np
import argparse
import lal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from lalsimulation.nrfits.NRSur7dq4Remnant import NRSur7dq4Remnant
from gwtools import rotations   # pip install gwtools
import h5ify

def m1m2_from_mtotq(mtot, q):
    m1 = mtot / (1 + q)
    m2 = mtot - m1
    return m1, m2

def get_mag(v): 
    v_squared = [x*x for x in v]
    mag_v = np.sqrt(sum(v_squared))
    return mag_v

def unit_vector(v):
    unit_v =  v / get_mag(v)
    return unit_v

import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], 'lalsuite-extra/data/lalsimulation')

# Load parameter dict for all the different angular configurations
data_dir = '../../data/different_extrinsic_angles/'
params = h5ify.load(data_dir+'parameters_different_angles.h5')

# Define surrogate
sur = NRSur7dq4Remnant()
sur_dyn = sur._get_surrogate_dynamics

angles_vs_time_dict = {}

for k, samp in params.items(): 
    
    M = samp['mtotal'] # Detector frame total mass
    q = samp['q']
    m1, m2 = m1m2_from_mtotq(M, q) # Detector frame component masses
    eta=m1*m2/(m1+m2)/(m1+m2) # Symmetric mass ratio
    m1_SI = m1*lal.MSUN_SI    # Convert to kg
    m2_SI = m2*lal.MSUN_SI
    chi1 = samp['chi1']
    chi2 = samp['chi2']
    tilt1 = samp['tilt1']
    tilt2 = samp['tilt2']
    phi12 = samp['phi_12']
    theta_jn = samp['theta_jn']
    phi_jl = samp['phi_jl']
    dist_mpc = samp['dist']
    phi_ref = samp['phase']
    f_ref = samp['f_ref']

    # Transform spin convention 
    incl, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2, m1_SI, m2_SI, f_ref, phi_ref
    )

    # Orbital angular frequency converted to dimensionless units.
    # (Freq. scales as 1/M; this takes out the total mass dependence to get the dim-less omega_ref.
    # lal.MTSUN_SI is a solar mass in seconds.)
    omega_ref = -1 if f_ref==-1 else f_ref * np.pi* M * lal.MTSUN_SI

    # Translate between LIGO and NR conventions 
    q_sur = 1.0/q           # q = m1/m2 in NR <-> q = m2/m1 in LIGO
    chiA0 = [s1x, s1y, s1z] # subscript "A" in NR <-> subscript "1" in LIGO
    chiB0 = [s2x, s2y, s2z] # subscript "B" in NR <-> subscript "2" in LIGO

    # Get surrogate dynamics
    dyn_times, quat_sur, orbphase_sur, chiA_copr_sur, chiB_copr_sur \
        = sur_dyn(q_sur, chiA0, chiB0, [1,0,0,0], 0, omega_ref, False)

    # Direction of the orbital angular momentum, defined with respect to the
    # source frame at f_ref
    Lhat = rotations.lHat_from_quat(quat_sur).T

    # This is the phi that goes into the Ylms, and phi_ref is defined
    # like this LIGO.
    phi = np.pi/2 - phi_ref

    # N_hat = the direction of the line of sight from source to obsever,
    # also defined with respect to the source frame at f_ref
    Nhat = np.array([np.sin(incl)*np.cos(phi), \
                      np.sin(incl)*np.sin(phi), \
                      np.cos(incl)]).T

    # Take inner product of the two unit vector and arccos it to get
    # inclination as a function of time.
    incl_vs_t = np.arccos(np.sum(Lhat * Nhat, axis=1))

    # Get spin angular momentum vector at ref req with proper units
    S_A0 = np.asarray(chiA0) * m1 * m1   
    S_B0 = np.asarray(chiB0) * m2 * m2 
    S_0 = S_A0 + S_B0

    # Get orbital angular momentum vector with proper units (in PN limit)
    v0 = np.cbrt(  M * lal.MTSUN_SI * np.pi * f_ref )
    Lmag = (M*M*eta / v0)*(1.0 + v0*v0*(1.5 + eta/6.)) # 2PN expansion
    L = Lmag * Lhat

    # Add it all up to get total angular momentum at ref freq 
    # (in PN limit, it is constant in time so we don't need to time evolve)
    J = L[0] + S_0      
    Jhat = unit_vector(J) # unit vector

    # Just xy components
    Lhat_xy = np.asarray([unit_vector(Lhat[i,:-1]) for i in range(Lhat.shape[0])])
    Jhat_xy = unit_vector(Jhat[:-1])

    # Get angles between J and L as function of time
    thetajl_vs_t = np.arccos(np.sum(Lhat * Jhat, axis=1))
    phijl_vs_t = np.arccos(np.sum(Lhat_xy * Jhat_xy, axis=1))

    # Scale the dynamical times 
    # Mf = ???
    # dyn_times = dyn_times * (Mf / M)  

    # Add to ongoing dict 
    angles_vs_time_dict[k] = {
        'time_Mf':dyn_times,
        'incl':incl_vs_t, 
        'theta_JL':thetajl_vs_t, 
        'phi_JL':phijl_vs_t,
    } 

# Save
h5ify.save('angles_over_time.h5', angles_vs_time_dict)