import numpy as np
import spynnaker8 as sim

shape = np.array([5, 5], dtype='int32')  # h, w
n_input = np.prod(shape, dtype='int32')
stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([3, 3], dtype='int32')
shape_out = (np.floor((shape - k_shape) / stride) - 1).astype('int32')
n_out = np.prod(shape_out, dtype='int32')
vline = [[10.] if (idx % shape[1]) == (shape[1] // 2) else []
         for idx in range(n_input)]

ws = np.zeros(k_shape)
ws[:, k_shape[1]//2] = np.arange(k_shape[0]) + 1.

run_time = 50.

sim.setup(timestep=1.)

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': vline}, label='input spikes')

dst = sim.Population(n_out, sim.IF_curr_exp_conv, {})

conn = sim.ConvolutionConnector(shape, shape_out, k_shape)

prj = sim.Projection(src, dst, conn)

sim.run(run_time)
