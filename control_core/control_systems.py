from control import *
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')

class ControlModule:
    """
    Model for single control module such as D, C, P with its transfer function and noise spectrum
    reference signal -> D -> C -> P -> output.
    """

    def __init__(self, freq1 = 1, freq2 = 1e7, freq_points = 500, sys_ref = None, unit_in = 'V', unit_out = 'V'):
        # basic parameters for a general control module
        freq1 = np.log10(freq1)
        freq2 = np.log10(freq2)
        self.freq_points = freq_points
        self.freqx = np.logspace(freq1, freq2, freq_points)
        self.omgx = 2*np.pi*self.freqx
        self.unit_in = unit_in
        self.unit_out = unit_out
        self.sys_ref = sys_ref
        self.sys_noise = []
        self.output_noise = np.zeros(freq_points)  # V^2/Hz
        self.sub_noise = [] # V^2/Hz
        self.ref_input_noise = np.zeros(freq_points)  # V^2/Hz

    def freqresp_update(self):
        mag, phase, omg = freqresp(self.sys_ref, self.omgx)
        self.magx_ref = mag.reshape(self.freq_points)
        self.phasex_ref = phase.reshape(self.freq_points)

        self.magx_noise = []
        self.phasex_noise = []
        for ii in range(len(self.sys_noise)):
            sys_ii = self.sys_noise[ii]
            mag, phase, omg = freqresp(sys_ii, self.omgx)
            mag = mag.reshape(self.freq_points)
            phase = phase.reshape(self.freq_points)
            self.magx_noise.append(mag)
            self.phasex_noise.append(phase)

    def plot_ref_tracking(self):
        mag = self.magx_ref

        plt.figure()
        plt.loglog(self.freqx, mag)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude ' + self.unit_in + '/' + self.unit_out)
        plt.title('Reference tracking r->y')

    def plot_noise_transfer(self):
        mag = self.magx_noise

        legends = []
        for ii in range(len(self.sys_noise)):
            legends.append('S'+str(ii+1)+'->S')
        legends = tuple(legends)

        plt.figure()
        for ii in range(len(self.sys_noise)):
            plt.loglog(self.freqx, mag[ii])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Noise transfer')
        plt.legend(legends)


    def plot_noise(self):
        ref_noise = self.magx_ref**2*self.ref_input_noise
        total_noise = ref_noise + self.output_noise
        plt.figure()
        plt.loglog(self.freqx, ref_noise, self.freqx, self.output_noise, self.freqx, total_noise)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Noise spectrum (' + self.unit_out + '$^2$/Hz)')
        plt.legend(('Reference noise', 'Output Noise', 'Total'))
        plt.title('System noise')

        legends = []
        for ii in range(len(self.sys_noise)):
            legends.append('S'+str(ii+1)+'->S')
        legends.append('Total output noise')
        legends = tuple(legends)

        plt.figure()
        for ii in range(len(self.sys_noise)):
            plt.loglog(self.freqx, self.sub_noise[ii])
        plt.loglog(self.freqx, self.output_noise)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Noise spectrum (' + self.unit_out + '$^2$/Hz)')
        plt.legend(legends)
        plt.title('Output noise decomposition')

def feedback_combine(C_list, output = 0):
    """
    6 Cobj with output after C3:
    ---> C1 ---> C2 ---> C3 ---> output
         |               |
         C6 <--- C5 <---C4
    Args:
        C_list: list of the control modules
        output: the port of the output

    Returns:
        Cobj_fb: combined feedback control module with its transfer function and noise spectrum

    """
    Cobj_fb = ControlModule()
    Cobj_fb.unit_in = C_list[0].unit_in
    Cobj_fb.unit_out = C_list[output].unit_out
    Cobj_fb.freqx = C_list[0].freqx
    Cobj_fb.omgx = C_list[0].omgx

    sys_list = []
    for Cii in C_list:
        sys_list.append(Cii.sys_ref)

    # noise transfer function parts
    sys_parts = []
    for ii in range(output+1):
        sys_list_tmp = sys_list[0:ii+1]
        sys_tmp = 1
        for sys_list_tmp_ii in sys_list_tmp:
            sys_tmp = sys_tmp/sys_list_tmp_ii
        sys_parts.append(sys_tmp)                   # [1/C1, 1/C1C2, 1/C1C2C3]

    for ii in range(output+1, len(C_list)):
        sys_list_tmp = sys_list[ii:]
        sys_tmp = 1
        for sys_list_tmp_ii in sys_list_tmp:
            sys_tmp = sys_tmp*sys_list_tmp_ii
        sys_parts.append(-sys_tmp/sys_list[ii]) # [-C5C6, -C6, -1]

    sys_tmp = 1
    for sys_list_tmp_ii in sys_list:
        sys_tmp = sys_tmp * sys_list_tmp_ii
    sys_parts.append(sys_tmp)                   # C1C2...C5C6
    sys_parts.append(1/sys_parts[output]) # C1C2C3

    # feedback system transfer function sys_ref = C1C2C3/(1+C1C2C3C4C5C6)
    Cobj_fb.sys_ref = sys_parts[-1]/(1 + sys_parts[-2])
    for ii in range(len(C_list)):
        Cobj_fb.sys_noise.append(Cobj_fb.sys_ref*sys_parts[ii])

    # feedback system noise spectrum calculation
    for ii in range(len(C_list)):
        mag, phase, omg = freqresp(Cobj_fb.sys_noise[ii], Cobj_fb.omgx)
        mag = mag.reshape(Cobj_fb.freq_points)
        Cobj_fb.sub_noise.append(mag**2*C_list[ii].output_noise)
        Cobj_fb.output_noise = Cobj_fb.output_noise + mag**2*C_list[ii].output_noise

    return Cobj_fb

def S_noise_interp(freq,S):
    pass

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