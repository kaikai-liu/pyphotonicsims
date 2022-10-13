import control as ct
import numpy as np
import matplotlib.pyplot as plt

from control_core.control_systems import *
plt.style.use('')

s = ct.tf('s')
sys = 10e6/(1 + s/(2*np.pi*1e5))
plant = ControlModule(sys_ref = sys, unit_in = 'V', unit_out = 'Hz', label = 'plant')
plant.output_noise = np.ones(1000)*1e4

sys = 0.5e-6*(1 + 2*np.pi*1e5/s)
servo = ControlModule(sys_ref = sys, unit_in = 'Hz', unit_out = 'V', label = 'servo')

C_list = [servo,plant]
laser = feedback_combine(C_list,1)
laser.input_noise = np.ones(1000) * 1e2
laser.freqresp_module_update()

laser.plot_ref_tracking()
laser.plot_module_noise()
plt.show()