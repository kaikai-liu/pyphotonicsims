"""
Stimulated Brillouin Scattering (SBS) laser model based on Ryan O. Behunin's paper.
Behunin, Ryan O., et al. "Fundamental noise dynamics in cascaded-order Brillouin lasers." Physical Review A 98.2 (2018): 023832.

"""

from .semiconductor_laser import LaserConst
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

class SBSLaser(LaserConst):
    """
    SBS laser model

    """
    def __init__(self, ord, r = [1.0, 1.0], L = 0.07, vST_min = 1, Aeff = 30e-12):
        super().__init__()
        self.ord = ord              # cascading order
        self.n0 = 26/0.045          # phonon occupation number 1/(exp(hv/kT)-1) ~ kT(26meV)/hv(0.045meV)
        self.r = r                  # loss rates [r_in, r_ex] in [MHz]
        self.L = L                  # cavity length
        self.Aeff = Aeff            # mode area
        self.ng = 1.5               # group index
        self.gB0 = 4.5e-13  # bulk silica Brillouin gain 0.045 m / GW

        self.threshold_calc(vST_min)

    def threshold_calc(self, vST_min = 1, ifprint = True):

        self.vST_min = vST_min
        self.vg = self.c/self.ng
        self.gamma_in = 2*np.pi*self.r[0]*1e6
        self.gamma_ex = 2*np.pi*self.r[1]*1e6
        self.gamma = self.gamma_in + self.gamma_ex
        self.Q = 2*np.pi*self.f0/self.gamma
        self.mu = vST_min*2*np.pi/self.n0
        self.GB = 2 * self.mu * self.L / (self.h * self.f0 * self.vg**2)
        self.gB = self.GB * self.Aeff
        self.rho = self.gB/self.gB0
        self.P_th = self.h * self.f0 * self.gamma**3 / (8 * self.mu * self.gamma_ex)

        if ifprint:
            print('-----------------REPORT------------------')
            print('Cavity Q:      %.2f M' % (self.Q/1e6))
            print('P_th:          %.3f mW' % (self.P_th*1e3))
            print('min FLW:       %.3f Hz' % (self.vST_min))
            print('rho:           %.3f' % (self.rho))
            print('GB:            %.3f' % (self.GB))

    def pump_sweep(self,Px):
        Fx = np.sqrt(Px/(self.h*self.f0))
        tspan = [0, 300/self.gamma]
        rtol = 1e-4
        ax = np.zeros((self.ord + 1, len(Px)))
        for ii in range(len(Fx)):
            if ii == 0:
                a_init = 1e4*np.ones(self.ord + 1)
            else:
                a_init = ax[:, ii-1]

            # ode45 integration
            sol = solve_ivp(req_sbs_laser, tspan, a_init.tolist(), args=(Fx[ii], self), rtol=rtol)
            t_sol = sol['t']
            y_sol = sol['y']
            ax[:,ii] = y_sol[:,-1]

            if ii == 0:
                t = t_sol
                at = y_sol
            else:
                t = np.hstack((t, t_sol + t[-1]))
                at = np.hstack((at, y_sol))

        Pout = self.h * self.f0 * abs(ax)**2 * self.gamma_ex
        # vST

        return Pout, ax, t, at

    def pump_sweep_visulization(self, Px):
        Pout, ax, t, at = self.pump_sweep(Px)

        legends = []
        for ii in range(self.ord):
            legends.append('S' + str(ii + 1))
        legends = tuple(legends)

        plt.figure(figsize=(4, 3))
        for ii in range(self.ord):
            plt.plot(Px*1e3, Pout[ii + 1, :]*1e3)


        plt.xlabel('Pump (mW)')
        plt.ylabel('Stokes power (mW)')
        plt.legend(legends)
        plt.title('P_th = ' + '%.3f' % (self.P_th * 1e3) + ' mW')

    def freqresp_current_mod(self,P_drive,freq1 = 1e3,freq2 = 1e10,freq_points = 1000):
        pass

def req_sbs_laser(t, a, F_pump, sbs):
    """
    Rate equation for SBS laser
    Args:
        t: time
        a: photon mode
        F_pump: pump mode
        sbs: SBSLaser object

    Returns:
        dadt

    """
    ord = sbs.ord
    gamma = sbs.gamma
    gamma_ex = sbs.gamma_ex
    mu = sbs.mu
    dadt = []
    for ii in range(ord + 1):
        if ii == 0:
            dadt.append((-gamma / 2 - mu * abs(a[ii + 1])**2 +            0          ) * a[ii] + np.sqrt(gamma_ex) * F_pump)
        elif ii == ord:
            dadt.append((-gamma / 2 -           0            + mu * abs(a[ii - 1])**2) * a[ii])
        else:
            dadt.append((-gamma / 2 - mu * abs(a[ii + 1])**2 + mu * abs(a[ii - 1])**2) * a[ii] + np.random.rand(1)[0])
    return dadt
