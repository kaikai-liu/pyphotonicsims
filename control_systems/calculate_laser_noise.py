import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def S_noise_interp(freq,S,freqx,ifplot = 0):
    """

    Args:
        freq: measured frequency offset
        S: measured noise data
        freqx: modeling frequency offset
        ifplot: plot the interpolation or not

    Returns:
        Sx: interpolated noise spectrum

    """
    # convert to log10
    freq = np.log10(freq)
    S = np.log10(S)
    freqx = np.log10(freqx)
    ft = interp1d(freq,S)

    # convert back to linear
    Sx = 10**(ft(freqx))

    # plot interpolation
    if ifplot:
        plt.plot(freq,S,'.',freqx,ft(freqx),'-')

    return Sx

def get_S_TRN(freq):
    """
    TRN noise calculation for SBS resonator
    Args:
        freq: frequency offset valid from 1e-2Hz to 1e8Hz

    Returns:
        S: TRN

    """
    x = np.log10(freq)
    y = -5.489e-06*x**8 + 8.455e-05*x**7 - 1.348e-4*x**6 - 2.432e-3*x**5 + 4.441e-3*x**4 + 1.271e-2*x**3 - 0.03264*x**2 - 0.1446*x + 2.004
    return 10**y

def get_S_shot(Pc, dv):
    """
    calculate shot noise equivalent FN
    Args:
        Pc: carrier power
        dv: resonator linewidth [Hz]

    Returns:
        S: shot noise [Hz^2/Hz]
    """
    h = 6.63e-34
    v = 194e12
    return dv**2*h*v/(16*Pc)

def get_S_PD(NEP, Pin, dv, sideband = 0.1):
    """
    calculate PD noise equivalent FN
    Args:
        NEP: PD noise [W/rtHz]
        Pin: optical input power on PD [W]
        dv: resonator linewidth [Hz]
        sideband: sideband power ratio

    Returns:
        S: PD noise
    """
    D = 8*Pin*np.sqrt(sideband)/dv
    S = (NEP/D)**2
    return S