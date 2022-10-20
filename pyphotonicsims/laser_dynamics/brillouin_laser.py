"""
Stimulated Brillouin Scattering (SBS) laser model based on Ryan O. Behunin's paper.
Behunin, Ryan O., et al. "Fundamental noise dynamics in cascaded-order Brillouin lasers." Physical Review A 98.2 (2018): 023832.

"""

from .semiconductor_laser import LaserConst
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, ode

class SBSLaser(LaserConst):
    """
    SBS laser model

    """
    def __init__(self, ord = 4, r = [1.0, 1.0], L = 0.07, vST_min = 0.5, Aeff = 30e-12, ifprint = True):
        super().__init__()
        self.ord = ord              # cascading order
        self.n0 = 26/0.045          # phonon occupation number 1/(exp(hv/kT)-1) ~ kT(26meV)/hv(0.045meV)
        self.r = r                  # loss rates [r_in, r_ex] in [MHz]
        self.L = L                  # cavity length
        self.Aeff = Aeff            # mode area
        self.ng = 1.5               # group index
        self.gB0 = 4.5e-13  # bulk silica Brillouin gain 0.045 m / GW
        self.vST_min = vST_min

        self.vg = self.c / self.ng
        self.gamma_in = 2 * np.pi * self.r[0] * 1e6
        self.gamma_ex = 2 * np.pi * self.r[1] * 1e6
        self.gamma = self.gamma_in + self.gamma_ex
        self.Q = 2 * np.pi * self.f0 / self.gamma
        self.mu, self.GB, self.gB, self.rho, self.P_th = self.threshold_calc_from_vST(self.vST_min)

        if ifprint:
            print('-----------------REPORT------------------')
            print('Cavity Q:      %.2f M' % (self.Q/1e6))
            print('P_th:          %.3f mW' % (self.P_th*1e3))
            print('min FLW:       %.3f Hz' % (self.vST_min))
            print('S1 efficiency: %.3f' % ((self.gamma_ex/self.gamma)**2))
            print('rho:           %.3f' % (self.rho))
            print('GB:            %.3f' % (self.GB))

    def threshold_calc_from_vST(self, vST_min = 0.3):
        """
        Calculate SBS laser metrics such as from minimum ST linewidth
        Args:
            vST_min: minimum ST/fundamental/intrinsic linewidth at S1 clamping point
        Returns:

        """

        self.vST_min = vST_min
        mu = vST_min*2*np.pi/self.n0
        GB = 2 * mu * self.L / (self.h * self.f0 * self.vg**2)
        gB = GB * self.Aeff
        rho = gB/self.gB0
        P_th = self.h * self.f0 * self.gamma**3 / (8 * mu * self.gamma_ex)

        return mu, GB, gB, rho, P_th

    def threshold_calc_from_GB(self, GB):
        """
        Calculate SBS laser metrics such as from Brillouin gain
        Args:
            GB:

        Returns:

        """

        mu = GB * (self.h * self.f0 * self.vg ** 2) / (2 * self.L)
        vST_min = mu * self.n0 / (2 * np.pi)
        gB = GB * self.Aeff
        rho = gB / self.gB0
        P_th = self.h * self.f0 * self.gamma ** 3 / (8 * mu * self.gamma_ex)

        return vST_min, mu, gB, rho, P_th



    def pump_detuning_sweep(self, Px, dfx, abs_heating):
        """
        Input a list of pump power values
        Solve the SBS laser rate equation using ode45 for each pump until steady state

        Args:
            Px: an array of pump power values in [W], and dfx cannot be a list or array
            dfx: an array of pump power detuning in [MHz] and Px cannot be a list or array
            abs_heating: [f_abs, eta_abs] = abs_heating,
                    f_abs is resonator thermal redshift coefficient in [MHz/mW]
                    eta_abs is the absorption fraction in total intrinsic loss

        Returns:
            Pout: power output at steady states, Pout = array([[Pump],[S1],...[Disp],[Total]])
            ax: photon mode output at steady states
            t: time sequence points from ode solver
            at: photon mode from ode solver

        """
        tspan = [0, 300 / self.gamma]
        rtol = 1e-4

        if len(Px) > 1:
            Psweep = True
            l = len(Px)
        else:
            Psweep = False
            l = len(dfx)

        Fx = np.sqrt(Px / (self.h * self.f0))               # pump influx
        ax = np.zeros((self.ord + 1, l)) + 0*1j       # make it complex by "+ 0*1j"
        t = 0. + 0*1j                                       # make it complex by "+ 0*1j"
        at = 0. + 0*1j                                      # make it complex by "+ 0*1j"

        for ii in range(l):
            if ii == 0:
                a_init = 1e4*np.ones(self.ord + 1) + 0*1j   # make it complex by "+ 0*1j"
            else:
                a_init = ax[:, ii-1] + 0*1j                 # make it complex by "+ 0*1j"

            if Psweep:
                Fii = Fx[ii]
                dfii = dfx[0]
            else:
                Fii = Fx[0]
                dfii = dfx[ii]

            # ode45 integration
            sol = solve_ivp(req_sbs_laser, tspan, a_init.tolist(), args=(Fii, dfii, abs_heating, self), rtol=rtol)
            t_sol = sol['t']
            y_sol = sol['y']
            ax[:, ii] = y_sol[:, -1]

            if ii == 0:
                t = t_sol
                at = y_sol
            else:
                t = np.hstack((t, t_sol + t[-1]))
                at = np.hstack((at, y_sol))

        Pout = self.h * self.f0 * abs(ax)**2 * self.gamma_ex
        Pout[0, :] = self.h * self.f0 * abs(1j * np.sqrt(self.gamma_ex) * ax[0, :] + Fx) ** 2
        Pdisp = self.h * self.f0 * sum(abs(ax)**2) * self.gamma_in
        Pout = np.vstack((Pout, Pdisp, sum(Pout) + Pdisp))
        # vST need to be added


        return Pout, ax, t, at

    def pump_sweep_visulization(self, Px):
        """
        Visualization of the pump sweep with an array of pump powers

        """

        Pout, ax, t, at = self.pump_detuning_sweep(Px, np.array([0.0]), [0., 0.])

        # legneds = ['Pump', 'S1', ..., 'S5', 'Disp', 'Total']
        legends = ['Pump']
        for ii in range(self.ord):
            legends.append('S' + str(ii + 1))
        legends.append('Disp')
        legends.append('Total')

        plt.figure(figsize=(8, 7))
        plt.subplot(221)
        plt.plot(Px * 1e3, (Pout / Px).T)
        plt.xlabel('Pump (mW)')
        plt.ylabel('Efficiency')
        plt.legend(tuple(legends))
        plt.title('P' + r'$_{th}$' + ' %.3f' % (self.P_th * 1e3) + ' mW')
        plt.subplot(222)
        plt.plot(Px*1e3, abs(ax.T)**2)
        plt.xlabel('Pump (mW)')
        plt.ylabel('Photon number')
        plt.legend(tuple(legends[:-2]))
        plt.title('P' + r'$_{th}$' + ' %.3f' % (self.P_th * 1e3) + ' mW')
        plt.subplot(223)
        plt.plot(Px * 1e3, Pout[1:-2, :].T*1e3)
        plt.xlabel('Pump (mW)')
        plt.ylabel('Stokes power (mW)')
        plt.legend(tuple(legends[1:-2]))
        plt.subplot(224)
        # plt.plot(Px * 1e3, vST.T)
        plt.xlabel('Pump (mW)')
        plt.ylabel(r'\nu_{ST}' + ' (Hz)')
        #plt.legend(tuple(legends[1:-2]))

    def detuning_sweep_visulization(self, dfx, P, abs_heating = [6.0, 0.05]):
        """
        Visualization of the detuning sweep with an array of detuning in [MHz]

        Args:
            dfx: an array in [MHz]
            P: pump power in [W]
            abs_heating: [f_abs, eta_abs] = abs_heating,
                    f_abs is resonator thermal redshift coefficient in [MHz/mW]
                    eta_abs is the absorption fraction in total intrinsic loss

        Returns:

        """

        Pout, ax, t, at = self.pump_detuning_sweep(np.array([P]), dfx, abs_heating)

        # legneds = ['Pump', 'S1', ..., 'S5', 'Disp']
        legends = ['Pump']
        for ii in range(self.ord):
            legends.append('S' + str(ii + 1))
        legends.append('Disp')

        plt.plot(dfx, (Pout[:-1, :]/P).T)
        plt.xlabel('Detuning (MHz)')
        plt.ylabel('Efficiency')
        plt.legend(tuple(legends))
        plt.title('P' + r'$_{th}$' + ' %.3f' % (self.P_th * 1e3) + ' mW')

    def freqresp_current_mod(self, P_drive, freq1 = 1e3, freq2 = 1e10, freq_points = 1000):
        pass

    def sbs_freq_matching(self, f_bt_S1_pump, Vac):
        """

        Args:
            f_bt_S1_pump:
            Vac:

        Returns:

        """

def req_sbs_laser(t, a, F_pump, df, abs_heating, sbs):
    """
    Rate equation for SBS laser
    Args:
        t: time
        a: photon mode
        F_pump: pump mode
        df: detuning in [MHz]
        abs_heating: [f_abs, eta_abs] = abs_heating,
                    f_abs is resonator thermal redshift coefficient in [MHz/mW]
                    eta_abs is the absorption fraction in total intrinsic loss
        sbs: SBSLaser object

    Returns:
        dadt

    """

    ord = sbs.ord
    gamma = sbs.gamma
    gamma_ex = sbs.gamma_ex
    mu = sbs.mu

    # resonator thermal redshift coefficient in [MHz/mW]
    [f_abs, eta_abs] = abs_heating
    f_thermal = f_abs * eta_abs
    # resonance frequency shift by cavity optical power absorption and heating
    f_res = 2*np.pi*f_thermal*(sbs.h*sbs.f0*sbs.gamma_in*sum(abs(a)**2))*1e3 # multiply by 1e3 converts [W] into [mW]

    dadt = []
    for ii in range(ord + 1):
        if ii == 0:
            dadt.append((1j*2*np.pi*(df - f_res)*1e6 -gamma / 2 - mu * abs(a[ii + 1])**2 +            0          ) * a[ii] + 1j * np.sqrt(gamma_ex) * F_pump)
        elif ii == ord:
            dadt.append((1j*2*np.pi*(df - f_res)*1e6 -gamma / 2 -           0            + mu * abs(a[ii - 1])**2) * a[ii])
        else:
            dadt.append((1j*2*np.pi*(df - f_res)*1e6 -gamma / 2 - mu * abs(a[ii + 1])**2 + mu * abs(a[ii - 1])**2) * a[ii] + np.random.rand(1)[0])
    return dadt
