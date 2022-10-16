"""
Examples of different types of lasers,
such as in-plane AlGaAs laser, VSCEL laser, extended-cavity laser.

Data source:
 - in-plane AlGaAs laser: diode laser and integrated circuits, TABLE 5.1
 - VSCEL laser AlGaAs laser: diode laser and integrated circuits, TABLE 5.1
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


class LaserConst:
    """
    Define basic constants (such as speed of light c) and parameters (such as wavelength lambda)
    """

    def __init__(self):
        self.c = 2.998e8  # speed of light
        self.q = 1.6e-19  # electron charge
        self.h = 6.63e-34  # plank const
        self.lmbd = 1.55e-6  # wavelength
        self.f0 = self.c / self.lmbd  # frequency
        self.nsp = 1  # spon em factor
        self.ccm = self.c * 100


class LaserModel(LaserConst):
    """
    A semiconductor laser model object
    """

    def __init__(self, lasertype='inplane'):
        super().__init__()
        if lasertype == 'inplane':
            self.define_inplane()
        elif lasertype == 'vcsel':
            self.define_vscel()
        elif lasertype == 'ecl':
            self.define_ecl()

        self.threshold_calc()

    def threshold_calc(self):
        """
        calculate values such as threshold density, current threshold and so on

        """
        self.r_in_a = self.ccm * (self.a_in_a * self.La / self.ng_a) / (self.La + self.Lp)
        self.r_in_p = self.ccm * (self.a_in_p * self.Lp / self.ng_p) / (self.La + self.Lp)
        self.r_in = self.r_in_a + self.r_in_p  # cavity intrinsic loss rate, in Hz
        self.r_m = self.ccm * self.a_m * (self.La / self.ng_a + self.Lp / self.ng_p) \
                   / (self.La + self.Lp)  # cavity mirror loss rate, in Hz
        self.r_ex = self.r_m  # cavity coupling loss rate, in Hz
        # self.r_ex = 3*self.r_in            # cavity coupling loss rate, in Hz
        self.eta_d = self.r_ex/(self.r_in + self.r_ex) # diff efficiency
        self.eta_total = self.eta_i*self.eta_d         # total efficiency
        self.t_p = 1 / (self.r_in + self.r_ex)  # photon lifetime, in s
        self.Q = 2 * np.pi * self.f0 / (self.r_in + self.r_ex)
        self.g_th = (self.r_in + self.r_ex) / (self.Gamma * self.ccm / self.ng_a)  # threshold gain
        sol = fsolve(self.gain_model, [3e18], args=(0, self.g_th, True))
        self.N_th = sol[0]
        self.I_th = self.q * self.V / self.eta_i * \
                    (self.A * self.N_th + self.B * self.N_th ** 2 + self.C * self.N_th ** 3)  # threshold current

        # report and print
        print('-----------------REPORT------------------')
        print('Cavity Q:      %.1e' % self.Q)
        print('Active loss:   %.1f MHz' % (self.r_in_a / (2 * np.pi * 1e6)))
        print('Passive loss:  %.1f MHz' % (self.r_in_p / (2 * np.pi * 1e6)))
        print('Cavity loss:   %.1f MHz' % (self.r_in / (2 * np.pi * 1e6)))
        print('Mirror loss:   %.1f MHz' % (self.r_m / (2 * np.pi * 1e6)))
        print('g_th:          %.1f cm^(-1)' % self.g_th)
        print('N_th:          %.2fe18 cm^(-3)' % (self.N_th * 1e-18))
        print('I_th:          %.1f mA' % (self.I_th * 1e3))
        print('eta_d:         %.1f %%' % (self.eta_d * 100))
        print('eta:           %.1f %%' % (self.eta_total * 100))

    def gain_model(self, Ne, Np, g_th=1600, find_N_th=False):
        """
        gain model using the log model
        Args:
            N: a list of two variables [Ne, Np] = N

        Returns:
            g: active region gain

        """
        if find_N_th:
            g = self.g0 / (1 + self.eps * Np) * (np.log(Ne + self.N_s) - np.log(self.N_tr + self.N_s)) - g_th
        else:
            g = self.g0 / (1 + self.eps * Np) * (np.log(Ne + self.N_s) - np.log(self.N_tr + self.N_s))
        return g

    def PI_current_sweep(self, Ix, tspan):
        """

        Args:
            Ix: current sweep points, list, in A
            tspan: time span for ode, [0,tmax]

        Returns:
            Pout: laser output power, (n,) 1-D array, in W
            vST: ST linewidth, (n,) 1-D array, in Hz
            Nx: [carrier density, photon density], (n,2) 2-D array in cm^-3
            t: time seq array solved in ode45, 1-D array
            Nt: [carrier density, photon density], (n,2) 2-D array in cm^-3

        """

        rtol = 1e-4
        Nx = np.zeros((2,len(Ix)))
        for ii in range(len(Ix)):
            if ii == 0:
                N_init = [1e18, 1e13]
            else:
                N_init = Nx[:,ii-1].tolist()

            # ode45 integration
            sol = solve_ivp(req_semi_laser, tspan, N_init, args=(Ix[ii], self), rtol=rtol)
            t_sol = sol['t']
            y_sol = sol['y']
            Nx[:,ii] = y_sol[:,-1]

            if ii == 0:
                t = t_sol
                Nt = y_sol
            else:
                t = np.hstack((t, t_sol + t[-1]))
                Nt = np.hstack((Nt, y_sol))

        Pout = self.h * self.f0 * Nx[1,:] * self.Veff * self.r_ex
        vST = self.nsp * (1 + 5 ** 2) * (self.r_in + self.r_ex) / (4 * np.pi * Nx[1,:] * self.Veff)

        return Pout, vST, Nx, t, Nt

    def PI_visulization(self,Ix,plotindensity = True):
        tspan = [0, self.tspanmax]
        Pout, vST, Nx, t, Nt = self.PI_current_sweep(Ix,tspan)

        plt.figure(figsize= (8,7))
        plt.subplot(221)
        plt.plot(Ix*1e3,Nx[0,:],'.') if plotindensity else plt.plot(Ix*1e3,Nx[0,:]*self.V,'.')
        plt.xlabel('Current (mA)')
        plt.ylabel('Carrier density (cm^-3)') if plotindensity else plt.ylabel('Carrier number')
        plt.title('I_th = ' + '%.2f' % (self.I_th*1e3) + ' mA')
        plt.subplot(222)
        plt.plot(Ix*1e3,Pout*1e3,'.')
        plt.xlabel('Current (mA)')
        plt.ylabel('Output power (mW)')
        plt.title('I_th = ' + '%.2f' % (self.I_th*1e3) + ' mA')
        plt.subplot(223)
        plt.plot(Ix*1e3,Nx[1,:],'.') if plotindensity else plt.plot(Ix*1e3,Nx[1,:]*self.Veff,'.')
        plt.xlabel('Current (mA)')
        plt.ylabel('Photon density (cm^-3)') if plotindensity else plt.ylabel('Photon number')
        plt.subplot(224)
        plt.semilogy(Ix*1e3,vST,'.')
        plt.xlabel('Current (mA)')
        plt.ylabel('ST linewidth (Hz)')

    def transient_visulization(self,Ix,plotindensity = True):
        tspan = [0, self.tspanmax]
        Pout, vST, Nx, t, Nt = self.PI_current_sweep(Ix, tspan)

        plt.figure(figsize=(8, 7))
        plt.subplot(221)
        plt.plot(t * 1e9, Nt[0, :]) if plotindensity else plt.plot(t * 1e9, Nt[0, :]*self.V)
        plt.xlabel('Time (ns)')
        plt.ylabel('Carrier density (cm^-3)') if plotindensity else plt.ylabel('Carrier number')
        plt.title('I_th = ' + '%.2f' % (self.I_th * 1e3) + ' mA')
        plt.subplot(222)
        plt.plot(t * 1e9, Nt[1, :]) if plotindensity else plt.plot(t * 1e9, Nt[1, :]*self.Veff)
        plt.xlabel('Time (ns)')
        plt.ylabel('Photon density (cm^-3)') if plotindensity else plt.ylabel('Photon number')
        plt.title('I_th = ' + '%.2f' % (self.I_th * 1e3) + ' mA')
        plt.subplot(223)
        plt.plot(Ix * 1e3,'.')
        plt.xlabel('Current point')
        plt.ylabel('Current (mA)')
    def freqresp_current_mod(self,I_drive,freq1 = 1e3,freq2 = 1e10,freq_points = 1000):
        tspan = [0, self.tspanmax]
        Pout, vST, Nx, t, Nt = self.PI_current_sweep([I_drive], tspan)

        freqx = np.logspace(np.log10(freq1),np.log10(freq2),freq_points)
        omgx = 2*np.pi*freqx
        Nx = Nx[:,-1].tolist()
        dfdNS = self.differentiate_req(Nx,I_drive,dx=1e-4)
        dfdNS_freq = np.vstack((1j*omgx - dfdNS[0,0],
                                -np.ones(len(omgx))*dfdNS[0,1],
                                -np.ones(len(omgx))*dfdNS[1,0],
                                1j*omgx - dfdNS[1,1]))
        det = dfdNS_freq[0,:]*dfdNS_freq[3,:] - dfdNS_freq[1,:]*dfdNS_freq[2,:]
        n1 = self.eta_i/(self.q*self.V)*dfdNS_freq[3,:]/det
        s1 = -self.eta_i/(self.q*self.V)*dfdNS_freq[1,:]/det
        p1 = self.h * self.f0 * s1 * self.Veff * self.r_ex
        H = (dfdNS[0,0]*dfdNS_freq[1,1] - dfdNS[0,1]*dfdNS[1,0])/det

        return n1,s1,p1,H,freqx


    def differentiate_req(self, N, current, dx = 1e-4):
        """
        rate equations
        Args:
            N: Ne, Np = N
            current: driving current

        Returns:
            dfdNS: (2,2) 2-D array, [[yNN,ySN],[ySN,ySS]]

        """
        Ne, Ns = N
        dfdN = (np.array(req_semi_laser(0, [Ne*(1+dx),Ns], current, self)) - np.array(req_semi_laser(0, [Ne,Ns], current, self)))/(Ne*dx)
        dfdS = (np.array(req_semi_laser(0, [Ne,Ns*(1+dx)], current, self)) - np.array(req_semi_laser(0, [Ne,Ns], current, self)))/(Ns*dx)
        dfdNS = np.vstack((dfdN,dfdS))
        return dfdNS


    def define_inplane(self):
        """
        in-plane AlGaAs laser: diode laser and integrated circuits, TABLE 5.1

        """
        # basic parameters
        self.tspanmax = 1e-8  # ode45 similaion time span
        self.beta_sp = 8.7e-5  # spontaneous emission factor
        self.d = 80e-8  # in cm
        self.w = 2e-4  # in cm
        self.La = 250e-4  # in cm
        self.V = self.d * self.w * self.La
        self.Veff_a = 1.25e-10
        self.Lp = 0
        self.Veff_p = self.Lp * 2e-8
        self.Veff = self.Veff_a + self.Veff_p
        self.Gamma = self.V / self.Veff
        self.ng_a = 4.2  # active region group index
        self.ng_p = 1.6  # passive region group index
        self.v_g_a = self.ccm / self.ng_a

        # loss parameters
        self.a_in_a = 5  # active internal loss, in cm^-1
        self.a_in_p = 0.01 / 4.34  # extended SiN waveguide loss, in cm^-1
        self.a_in = (self.a_in_a * self.La + self.a_in_p * self.Lp) / (self.La + self.Lp)
        self.a_m = 1 / (self.La + self.Lp) * np.log(1 / 0.32)  # mirror loss, in cm^-1

        # current parameters
        self.eta_i = 0.8  # current efficiency
        self.A = 0.0  # in 1 / s
        self.B = 0.8e-10  # in cm^3 / s
        self.C = 3.5e-30  # in cm^6 / s

        # gain model parameters
        self.g0 = 1800  # in cm ^ -1
        self.N_tr = 1.8e18  # in cm ^ -3
        self.N_s = -0.4e18  # in cm ^ -3
        self.eps = 1.5e-17  # in cm ^ 3

    def define_vscel(self):
        """
        VCSEL AlGaAs laser: diode laser and integrated circuits, TABLE 5.1

        """
        # basic parameters
        self.tspanmax = 1e-8    # ode45 similaion time span
        self.beta_sp = 16.9e-5  # spontaneous emission factor
        self.d = 10e-4  # in cm
        self.w = 10e-4  # in cm
        self.La = 3 * 80e-8  # in cm
        self.V = self.d * self.w * self.La
        self.Veff_a = 0
        self.Lp = 1.15e-4
        self.Veff_p = 0
        self.Veff = 6.3e-11
        self.Gamma = self.V / self.Veff
        self.ng_a = 4.2  # active region group index
        self.ng_p = 4.2  # passive region group index
        self.v_g_a = self.ccm / self.ng_a

        # loss parameters
        self.a_in_a = 20  # active internal loss, in cm^-1
        self.a_in_p = 20  # passive loss, in cm^-1
        self.a_in = (self.a_in_a * self.La + self.a_in_p * self.Lp) / (self.La + self.Lp)
        self.a_m = 1 / (self.La + self.Lp) * np.log(1 / 0.995)  # mirror loss, in cm^-1

        # current parameters
        self.eta_i = 0.8  # current efficiency
        self.A = 0.0  # in 1 / s
        self.B = 0.8e-10  # in cm^3 / s
        self.C = 3.5e-30  # in cm^6 / s

        # gain model parameters
        self.g0 = 1800  # in cm ^ -1
        self.N_tr = 1.8e18  # in cm ^ -3
        self.N_s = -0.4e18  # in cm ^ -3
        self.eps = 1.5e-17  # in cm ^ 3

    def define_ecl(self):
        """
        extended cavity length with SiN waveguide based on in-plane laser model

        """
        # basic parameters
        self.tspanmax = 1e-7  # ode45 similaion time span
        self.beta_sp = 8.7e-5  # spontaneous emission factor
        self.d = 80e-8  # in cm
        self.w = 2e-4  # in cm
        self.La = 250e-4  # in cm
        self.V = self.d * self.w * self.La
        self.Veff_a = 1.25e-10
        self.Lp = 10.0          # cavity extension in cm
        self.Veff_p = self.Lp * 2e-8
        self.Veff = self.Veff_a + self.Veff_p
        self.Gamma = self.V / self.Veff
        self.ng_a = 4.2  # active region group index
        self.ng_p = 1.6  # passive region group index
        self.v_g_a = self.ccm / self.ng_a

        # loss parameters
        self.a_in_a = 5  # active internal loss, in cm^-1
        self.a_in_p = 0.01 / 4.34  # extended SiN waveguide loss, in cm^-1
        self.a_in = (self.a_in_a * self.La + self.a_in_p * self.Lp) / (self.La + self.Lp)
        self.a_m = 1 / (self.La + self.Lp) * np.log(1 / 0.32)  # mirror loss, in cm^-1
        self.a_m = self.a_in*2

        # current parameters
        self.eta_i = 0.8  # current efficiency
        self.A = 0.0  # in 1 / s
        self.B = 0.8e-10  # in cm^3 / s
        self.C = 3.5e-30  # in cm^6 / s

        # gain model parameters
        self.g0 = 1800  # in cm ^ -1
        self.N_tr = 1.8e18  # in cm ^ -3
        self.N_s = -0.4e18  # in cm ^ -3
        self.eps = 1.5e-17  # in cm ^ 3


def req_semi_laser(t, N, current, laser):
    """
    rate equations
    Args:
        N: Ne, Np = N
        current: driving current

    Returns:
        dNdt

    """
    Ne, Np = N
    g = laser.gain_model(Ne, Np)
    dNdt = [laser.eta_i * current / laser.q / laser.V - (
                laser.A * Ne + laser.B * Ne ** 2 + laser.C * Ne ** 3) - laser.v_g_a * g * Np,
            laser.Gamma * laser.v_g_a * (g - laser.g_th) * Np + laser.Gamma * laser.beta_sp * laser.B * Ne ** 2]

    return dNdt

