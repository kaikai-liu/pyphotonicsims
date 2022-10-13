import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
from feedbackloop.control_systems import *
plt.style.use('notebook')

s = ctrl.tf('s')
sys = 10e6/(1 + s/(2*np.pi*1e5))
plant = ControlModule(sys_ref = sys, unit_in = 'V', unit_out = 'Hz')
plant.output_noise = np.ones(500)*1e4

sys = 0.5e-6*(1 + 2*np.pi*1e5/s)
servo = ControlModule(sys_ref = sys, unit_in = 'Hz', unit_out = 'V')

C_list = [servo,plant]
laser = feedback_combine(C_list,1)
laser.ref_input_noise = np.ones(500)*1e2
laser.freqresp_update()

laser.plot_ref_tracking()