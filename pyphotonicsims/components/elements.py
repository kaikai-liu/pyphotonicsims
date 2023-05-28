from scipy.constants import c
import numpy as np

# ring resonators
def FAddThru(dw, r_in, r_ex):
    """
    Add-thru resonator cavity response function
    F = (1j*dw + (r_in - r_ex)/2)/(1j*dw + (r_in - r_ex)/2)
    Args:
        dw: detuning in [MHz]
        r_in: intrinsic loss in [MHz]
        r_ex: external loss in [MHz]

    Returns:
        F: cavity response

    """
    r = r_in + r_ex
    F = (1j * dw + (r_in - r_ex) / 2) / (1j * dw + r / 2)
    D = np.sqrt(r_in * r_ex) / (1j * dw + r / 2)

    return F, D


def FAddThruDrop(dw, r_in, r_thru, r_drop):
    """
    Add-thru-drop resonator cavity response function
    F = (1j * dw + (r_in + r_drop + r_thru) / 2) / (1j * dw + (r_in + r_drop + r_thru) / 2)
    Args:
        dw: detuning in [MHz]
        r_in: intrinsic loss in [MHz]
        r_thru: thru bus external loss in [MHz]
        r_drop: drop bus external loss in [MHz]

    Returns:
        F1: cavity thru response
        F2: cavity drop response

    """
    r = r_in + r_thru + r_drop
    F1 = (1j * dw + (r_in + r_drop - r_thru) / 2) / (1j * dw + r / 2)
    F2 = np.sqrt(r_drop * r_thru) / (1j * dw + r / 2)
    D = np.sqrt(r_in * r_thru) / (1j * dw + r / 2)
    return F1, F2, D

def FAddThruSplit(dw, r_in, r_ex, g):
    """
    Add-thru resonator cavity with CW-CCW coupling and splitting

    Args:
        dw: detuning in [MHz]
        r_in: intrinsic loss in [MHz]
        r_ex: external loss in [MHz]
        g: CW-CCW coupling rate in [MHz]
        w0: resonance offset in [MHz]
        level:

    Returns:
        F: cavity response

    """
    r = r_in + r_ex
    F1 = 1 - r_ex * (1j * dw + r / 2) / ((1j * dw + r / 2)**2 + g**2)
    F2 = g * r_ex / ((1j * dw + r / 2)**2 + g**2)
    D = np.sqrt(r_in * r_ex * (dw**2 + (r/2)**2 + g**2)) / ((1j * dw + r / 2)**2 + g**2)

    return F1, F2, D


def FAddThruFano(dw, r_in, r_ex, phi):
    """
    Add-thru Fano resonance
    Args:
        dw: detuning in [MHz]
        r_in: intrinsic loss in [MHz]
        r_ex: external loss in [MHz]
        phi: Fano phase in [rad]


    Returns:
        F: cavity response

    """
    F = np.exp(1j*phi) - r_ex / (1j * dw + (r_in + r_ex) / 2)

    return F

# MZI response
def S_MZI(dw, fsr, ka2, attn):
    """
    response of an MZI
    Args:
        dw: detuning in [MHz]
        fsr: FSR in [MHz]
        ka2: power coupling at the splitters
        attn: longer-arm attenuation [1], e.g., A = 0.5 -> -3 dB

    Returns:
        S11:
        S12:
        S21:
        S22:

    """
    A = np.sqrt(attn)
    t2 = 1 - ka2
    phi = 2 * np.pi * dw / fsr
    S11 = t2 - ka2 * A * np.exp(1j * phi)
    S12 = 1j * np.sqrt(t2 * ka2) * (1 + A * np.exp(1j * phi))
    S21 = S12
    S22 = t2 * A * np.exp(1j * phi) - ka2

    return S11, S12, S21, S22
def S_di_coupler(ka2):
    """
    response of an MZI
    Args:
        ka2:

    Returns:
        S: 2 by 2 matrix

    """
    ka = np.sqrt(ka2)
    tau = np.sqrt(1-ka2)
    S = np.array([[tau, 1j * ka], [1j * ka, tau]])
    return S

def PDB_MZI_resonator(dw, r_in, r_ex, ka2_in, ka2_out, phi):
    """
    Balanced detection of two outputs of an MZI resonator
    Args:
        dw: detuning in [MHz]
        r_in: intrinsic loss in [MHz]
        r_ex: external loss in [MHz]
        ka2_in: MZI coupler
        ka2_out: MZI coupler
        phi: MZI phase section

    Returns:
        y: balanced detection output

    """
    y1 = np.zeros(len(dw))
    y2 = np.zeros(len(dw))
    y = np.zeros(len(dw))
    for ii, dw_ii in enumerate(dw):
        F_cav, _ = FAddThru(dw_ii, r_in, r_ex)
        S1 = S_di_coupler(ka2_in)
        S2 = np.array([[F_cav, 0], [0, np.exp(1j*phi)]])
        S3 = S_di_coupler(ka2_out)
        F = np.matmul(np.matmul(S1, S2), S3)
        F = np.matmul(F, np.array([[1], [0]]))
        y1[ii] = np.abs(F[1]) ** 2
        y2[ii] = np.abs(F[0]) ** 2
    y = y1 - y2
    return y, y1, y2


def directional_coupler(dn, Lc, wl = 1550e-9):
    """
    power coupling of a directional coupler
    Args:
        dn: neff difference between symmetric and assymetric modes
        Lc: coupling length
        wl: wavelength

    Returns:

    """
    ka = np.sin(np.pi * dn * Lc / wl)
    ka2 = ka**2
    return ka2

def cavity_coupling_rate(ka2, ng, L):
    """
    calculate cavity coupling loss rate in [rad/s]
    Args:
        ka2: power coupling
        ng: group index
        L: cavity length

    Returns:

    """
    gamma_ex = np.log(1/(1 - ka2)) * c / (ng * L)
    return gamma_ex