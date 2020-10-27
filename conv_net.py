import numpy as np
import spynnaker8 as sim

shape = np.array([5, 5], dtype='int')  # h, w
n_input = np.prod(shape, dtype='int')
stride = np.array([1, 1], dtype='int')  # h, w
k_shape = np.array([3, 3], dtype='int')

n_out = np.prod(np.floor((shape - k_shape) / stride) - 1, dtype='int')
sim.setup(timestep=1.)

vline = [[10.] if (idx % shape[1]) == (shape[1] // 2) else []
         for idx in range(n_input)]

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': vline}, label='input spikes')

dst = sim.Population(n_out, sim.IF_curr_exp_conv, {})

conn = sim.KernelConnector()

prj = sim.Projection(src, dst, sim.KernelConnector)