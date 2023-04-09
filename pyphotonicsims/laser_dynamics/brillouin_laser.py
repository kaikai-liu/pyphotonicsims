"""
Stimulated Brillouin Scattering (SBS) laser model based on Ryan O. Behunin's paper.
Behunin, Ryan O., et al. "Fundamental noise dynamics in cascaded-order Brillouin lasers." Physical Review A 98.2 (2018): 023832.

"""

from .semiconductor_laser import LaserConst
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

class SBSLaser(LaserConst):
    """
    SBS laser model

    """
    def __init__(self, lmbd = 1.55e-6, ord = 4, r = [1.0, 1.0], L = 0.07, vST_min = 0.5, Aeff = 30e-12, ifprint = True):
        super().__init__(lmbd = lmbd)
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
        self.mu, self.vST_min, self.P_th, self.GB, self.gB, self.rho = self.calc_from_vST(self.vST_min, self.L, self.gamma_in, self.gamma_ex, self.Aeff)

        if ifprint:
            print('-----------------REPORT------------------')
            print('Cavity Q:      %.2f M' % (self.Q/1e6))
            print('P_th:          %.3f mW' % (self.P_th*1e3))
            print('min FLW:       %.3f Hz' % (self.vST_min))
            print('S1 efficiency: %.3f' % ((self.gamma_ex/self.gamma)**2))
            print('rho:           %.3f' % (self.rho))
            print('GB:            %.3f' % (self.GB))

    def calc_from_vST(self, vST_min, L, gamma_in, gamma_ex, Aeff):
        """
        Calculate SBS laser metrics such as from minimum ST linewidth
        Args:
            vST_min: minimum ST/fundamental/intrinsic linewidth at S1 clamping point
            L:
            gamma_in:
            gamma_ex:
            Aeff:

        Returns:

        """

        mu = vST_min*2*np.pi/self.n0
        GB = 2 * mu * L / (self.h * self.f0 * self.vg**2)
        gB = GB * Aeff
        rho = gB/self.gB0
        P_th = self.h * self.f0 * (gamma_in + gamma_ex)**3 / (8 * mu * gamma_ex)

        return mu, vST_min, P_th, GB, gB, rho

    def calc_from_GB(self, GB, L, gamma_in, gamma_ex, Aeff):
        """
        Calculate SBS laser metrics such as from Brillouin gain
        Args:
            GB:
            L:
            gamma_in:
            gamma_ex:
            Aeff:

        Returns:

        """

        mu = GB * (self.h * self.f0 * self.vg ** 2) / (2 * L)
        vST_min = mu * self.n0 / (2 * np.pi)
        P_th = self.h * self.f0 * (gamma_in + gamma_ex) ** 3 / (8 * mu * gamma_ex)
        gB = GB * Aeff
        rho = gB / self.gB0

        return mu, vST_min, P_th, GB, gB, rho

    def calc_from_P_th(self, P_th, L, gamma_in, gamma_ex, Aeff):
        """
        Calculate SBS laser metrics such as from P_th
        Args:
            P_th: SBS S1 threshold
            L:
            gamma_in:
            gamma_ex:
            Aeff:

        Returns:

        """

        mu = self.h * self.f0 * (gamma_in + gamma_ex) ** 3 / (8 * P_th * gamma_ex)
        GB = 2 * mu * L / (self.h * self.f0 * self.vg ** 2)
        vST_min = mu * self.n0 / (2 * np.pi)
        gB = GB * Aeff
        rho = gB / self.gB0


        return mu, vST_min, P_th, GB, gB, rho

    def pump_detuning_sweep(self, Px, dfx = np.array([0.0]), abs_heating = [0.0, 0.0]):
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

        Pout, ax, t, at = self.pump_detuning_sweep(Px)

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
        plt.ylabel(r'$\nu_{ST}$' + ' (Hz)')
        #plt.legend(tuple(legends[1:-2]))

    def detuning_sweep_visulization(self, dfx, P, abs_heating = [1.0, 0.05]):
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

    def get_ng_from_sbs_freq_matching(self, f_bt_pump_S1 = 10.900e9, wl_match = 1550e-9):
        """
        calculate accurate ng and Vac from measured pump-S1 beatnote frequency
        Args:
            f_bt_pump_S1: measured pump-S1 beatnote frequency
            wl_match: minimum SBS laser threshold wavelength

        Returns:

        """
        FSR = self.c/(self.ng * self.L)
        m = round(f_bt_pump_S1 / FSR)
        ng = m * self.c / self.L / f_bt_pump_S1
        Vac = f_bt_pump_S1 * wl_match / (2 * ng)
        print(f"ng = {ng:.4f}")
        print(f"V_ac = {Vac:.1f} m/s")
        return ng, Vac

    def sbs_freq_matching_plot(self, ng = 1.5, Vac = 1808., wl_match = 1550e-9, wl1 = 1530e-9, wl2 = 1580e-9, m_plus = 0, threshold_plot = True, ifinterp = False, df1 = np.array([-10, 10]), GB1 = np.array([1., 1.])):
        """
        Show plots of SBS phase/frequency matching
        Args:
            ng: calculated from get_ng_from_sbs_freq_matching()
            Vac: calculated from get_ng_from_sbs_freq_matching()
            wl_match: wavelength where SBS frequency matching is satisfied
            wl1: plot wavelength range start
            wl2: plot wavelength range stop
            m_plus: how many resonance modes involved in SBS laser
            ifinterp: wheather or not to interpolate from similated gain spectrum
            df1: simulated GB spectrum data, in [MHz]
            GB1: simulated GB spectrum data, in [1]

        Returns:

        """
        # calculate OmgB, GB, P_th for other wavelengths [wl1, wl2]
        FSR = self.c / (self.ng * self.L)
        m = round(2 * np.pi * Vac * ng / wl_match / FSR)
        wlx = np.linspace(wl1, wl2, 200)
        OmgBx = 2 * np.pi * Vac * ng / wlx
        mlist = np.array([[m]]) if m_plus == 0 else np.linspace(m - m_plus, m + m_plus, 2 * m_plus + 1).reshape((2 * m_plus + 1, 1))
        mFSRx = mlist * self.c / (ng * self.L) * np.ones((1, len(wlx))) if m_plus else m * self.c / (ng * self.L) * np.ones((1, len(wlx)))

        GBx = self.GB*GB_spectrum_silica((mFSRx - OmgBx) * 1e-6, ifinterp = ifinterp, df1 = df1, GB1 = GB1)
        mu, vST_min, P_th, _, _, _ = self.calc_from_GB(GBx, self.L, self.gamma_in, self.gamma_ex, self.Aeff)

        plt.figure()
        plt.plot(wlx * 1e9, OmgBx * 1e-9, label = 'SBS shift')
        plt.plot(wlx * 1e9, mFSRx.T * 1e-9, label = 'mFSRx')
        plt.fill_between(wlx * 1e9, OmgBx*1e-9 - 0.05, OmgBx*1e-9 + 2 * 0.05, alpha = 0.3)
        plt.xlabel(r'$\lambda$' + ' (nm)')
        plt.ylabel('SBS shift (GHz)')
        plt.legend()
        plt.title(r'$n_g$' + ' %.5f, ' % ng + r'$\Omega_B$' + '@1550nm %.4f GHz' % (OmgBx[0]*1e-9))
        if threshold_plot:
            plt.figure(figsize = (8, 3))
            plt.subplot(121)
            plt.plot(wlx * 1e9, GBx.T)
            plt.xlabel(r'$\lambda$' + ' (nm)')
            plt.ylabel('G' + r'$_B$' + ' (1/m W)')
            plt.subplot(122)
            plt.plot(wlx * 1e9, P_th.T*1e3)
            plt.xlabel(r'$\lambda$' + ' (nm)')
            plt.ylabel('P' + r'$_{th}$' + ' (mW)')
            plt.ylim((0, 1e4*np.min(P_th)))



def GB_spectrum_silica(df_B, BW = 150., ifinterp = False, df1 = np.array([-10, 10]), GB1 = np.array([1., 1.])):
    """
    Calculate Brillouin gain at a frequency offset from Brillouin frequency shift (~10.9 GHz)
    Args:
        df_B: frequency offset from Brillouin frequency shift (~10.9 GHz), mFSR - Omg_B, in [MHz]
        BW: Brillouin gain bandwidth ~150 MHz in [MHz]
        ifinterp: wheather or not to interpolate from similated gain spectrum
        df1: simulated GB spectrum data, in [MHz]
        GB1: simulated GB spectrum data, in [1]

    Returns:
        GB

    """
    BW = BW/3
    if ifinterp:
        GB1 = GB1 / max(GB1)
        ind = np.logical_and(df_B > min(df1), df_B < max(df1))  # index for those within the interpolation range
        df_B = np.where(ind, df_B, min(df1))
        ft = interp1d(df1, GB1)
        GB = np.where(ind, ft(df_B), 0)                         # set to 0 outside the interpolation range
    else:
        # Asymmetric gain profile
        GB = np.where(df_B < 0, 1/(1 + (df_B/BW)**2), 1/(1 + (df_B/(BW*2))**2))

    return GB

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
