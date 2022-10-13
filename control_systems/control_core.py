from control import *
import numpy as np
import matplotlib.pyplot as plt

class ControlBlock(object):
    """
    Model for single control module such as D, C, P with its transfer function and noise spectrum
    reference signal -> D -> C -> P -> output.
    """

    def __init__(self, freq1 = 1, freq2 = 1e7, freq_points = 1000, sys_ref = None, unit_in = 'V', unit_out = 'V', label = 'PD'):
        # basic parameters for a general control module
        freq1 = np.log10(freq1)                                 # start freq
        freq2 = np.log10(freq2)                                 # stop freq
        self.freq_points = freq_points                          # number of points for frequencies
        self.freqx = np.logspace(freq1, freq2, freq_points)     # freq array
        self.omgx = 2*np.pi*self.freqx                          # angular freq array
        self.unit_in = unit_in                                  # input unit
        self.unit_out = unit_out                                # output unit
        self.label = label                                      # module label
        self.sys_ref = sys_ref                                  # transfer function of the block
        self.output_noise = np.zeros(freq_points)               # V^2/Hz
        self.input_noise = np.zeros(freq_points)                # V^2/Hz

    def freqresp_block_update(self):
        """
        Evaluate the transfer function frequency response

        """
        mag, phase, omg = freqresp(self.sys_ref, self.omgx)
        self.magx_sys_ref = mag.reshape(self.freq_points)
        self.phasex_sys_ref = phase.reshape(self.freq_points)

    def plot_ref_tracking(self):
        """
        plot control block's reference tracking transfer function

        """

        plt.figure()
        plt.loglog(self.freqx, self.magx_sys_ref)
        plt.xlabel('f (Hz)')
        plt.ylabel('Magnitude (' + self.unit_in + '/' + self.unit_out + ')')
        plt.title('Reference tracking r->y')

    def plot_block_noise(self):
        """
        plot control block's noise spectra

        """
        ref_noise = self.magx_sys_ref**2*self.input_noise
        total_noise = ref_noise + self.output_noise
        plt.figure()
        plt.loglog(self.freqx, ref_noise, self.freqx, self.output_noise, self.freqx, total_noise)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Noise spectrum (' + self.unit_out + '$^2$/Hz)')
        plt.legend(('Reference noise', 'Output Noise', 'Total'))
        plt.title('System spectrum')

    def get_sys_ref_from_data(self,freq,mag,phase):
        """
        Update sys_ref with FrequencyReponseData
        Warnings: this is not a TransferFunction object like "sys"
        Args:
            freq: data
            mag: data
            phase: data in [rad]

        Returns:

        """
        H = mag*np.exp(1j*phase)
        self.sys_ref = FrequencyResponseData(H,2*np.pi*freq,smooth=True)
    def pidtune(self):
        """
        Tuning the PID parameters of a servo based on the loop response
        Returns:
            sys_pid: optimized transfer function of a PID

        """
        pass

class ControlModule(ControlBlock):
    """
    Control module with different blocks such as D, C, P with its transfer function and noise spectrum
    reference signal -> D -> C -> P -> output.
                        |         |
                        ----------
    """

    def __init__(self,freq1 = 1, freq2 = 1e7, freq_points = 1000, sys_ref = None, unit_in = 'V', unit_out = 'V', label = 'PD'):
        # inherit all properties from ControlBlock
        super().__init__(freq1,freq2,freq_points,sys_ref,unit_in,unit_out,label)
        # add module properties such as noise spectra arrays
        self.sub_sys = []
        self.sub_noise = []
        self.sub_labels = []

    def freqresp_module_update(self):
        mag, phase, omg = freqresp(self.sys_ref, self.omgx)
        self.magx_sys_ref = mag.reshape(self.freq_points)
        self.phasex_sys_ref = phase.reshape(self.freq_points)

        self.magx_sub_sys = []
        self.phasex_sub_sys = []
        for ii in range(len(self.sub_sys)):
            sys_ii = self.sub_sys[ii]
            mag, phase, omg = freqresp(sys_ii, self.omgx)
            mag = mag.reshape(self.freq_points)
            phase = phase.reshape(self.freq_points)
            self.magx_sub_sys.append(mag)
            self.phasex_sub_sys.append(phase)

    def plot_subnoise(self):
        """
        plot noise spectrum from each sub-module

        """

        legends = []
        for ii in range(len(self.sub_sys)):
            legends.append(self.sub_labels[ii]+' S'+str(ii+1)+'->S')
        legends = tuple(legends)

        mag = self.sub_sys
        plt.figure()
        for ii in range(len(self.sub_sys)):
            plt.loglog(self.freqx, mag[ii])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Sub-module noise transfer function')
        plt.legend(legends)

    def plot_module_noise(self):
        # plot reference tracking noise and output noise
        self.plot_block_noise()

        # plot noise contribution from individual blocks
        legends = []
        for ii in range(len(self.sub_sys)):
            legends.append(self.sub_labels[ii]+' S'+str(ii+1)+'->S')
        legends.append('Total output noise')
        legends = tuple(legends)

        plt.figure()
        for ii in range(len(self.sub_sys)):
            plt.loglog(self.freqx, self.sub_noise[ii])
        plt.loglog(self.freqx, self.output_noise)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Noise spectrum (' + self.unit_out + '$^2$/Hz)')
        plt.legend(legends)
        plt.title('Output noise decomposition')

def feedback_combine(C_list, output=0):
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
    Cobj_fb = C_list[0]
    Cobj_fb.unit_in = C_list[0].unit_in
    Cobj_fb.unit_out = C_list[output].unit_out
    Cobj_fb.freqx = C_list[0].freqx
    Cobj_fb.omgx = C_list[0].omgx

    sys_list = []
    for Cii in C_list:
        sys_list.append(Cii.sys_ref)

    # noise transfer function parts
    sys_parts = []
    for ii in range(output + 1):
        sys_list_tmp = sys_list[0:ii + 1]
        sys_tmp = 1
        for sys_list_tmp_ii in sys_list_tmp:
            sys_tmp = sys_tmp / sys_list_tmp_ii
        sys_parts.append(sys_tmp)  # [1/C1, 1/C1C2, 1/C1C2C3]

    for ii in range(output + 1, len(C_list)):
        sys_list_tmp = sys_list[ii:]
        sys_tmp = 1
        for sys_list_tmp_ii in sys_list_tmp:
            sys_tmp = sys_tmp * sys_list_tmp_ii
        sys_parts.append(-sys_tmp / sys_list[ii])  # [-C5C6, -C6, -1]

    sys_tmp = 1
    for sys_list_tmp_ii in sys_list:
        sys_tmp = sys_tmp * sys_list_tmp_ii
    sys_parts.append(sys_tmp)  # C1C2...C5C6
    sys_parts.append(1 / sys_parts[output])  # C1C2C3

    # feedback system transfer function sys_ref = C1C2C3/(1+C1C2C3C4C5C6)
    Cobj_fb.sys_ref = sys_parts[-1] / (1 + sys_parts[-2])
    for ii in range(len(C_list)):
        Cobj_fb.sub_sys.append(Cobj_fb.sys_ref * sys_parts[ii])

    # feedback system noise spectrum calculation
    for ii in range(len(C_list)):
        mag, phase, omg = freqresp(Cobj_fb.sub_sys[ii], Cobj_fb.omgx)
        mag = mag.reshape(Cobj_fb.freq_points)
        Cobj_fb.sub_noise.append(mag ** 2 * C_list[ii].output_noise)
        Cobj_fb.output_noise = Cobj_fb.output_noise + mag ** 2 * C_list[ii].output_noise
        Cobj_fb.sub_labels.append(C_list[ii].label)

    return Cobj_fb

