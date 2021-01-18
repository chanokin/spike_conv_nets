import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D

sim.SpikeSourceArray.set_model_max_atoms_per_core(n_atoms=11)

shape = np.array([5, 5], dtype='int32')  # h, w
n_input = np.prod(shape, dtype='int32')
stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([3, 3], dtype='int32')
vline = [[5. + idx%3] if (idx % shape[1]) == (shape[1] // 2) or idx == 13 else []
         for idx in range(n_input)]

for x, i in enumerate(vline):
    if x:
        print(i, x)

ws = np.zeros(k_shape)
ws[:, k_shape[1]//2] = np.arange(k_shape[0]) + 2.
ws[:, k_shape[1]//2] = -1.8
ws[:, k_shape[1]//2+1:] = 5.5

run_time = 15.

sim.setup(timestep=1.)

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': vline}, label='input spikes')

conn = sim.ConvolutionConnector(shape, ws, strides=stride)
shape_out = conn.get_post_shape()
n_out = np.prod(shape_out, dtype='int32')
dst = sim.Population(n_out, sim.IF_curr_exp_conv, {})
dst.record('v')
# syn = sim.StaticSynapse(weight=ws.flatten)

prj = sim.Projection(src, dst, conn)

sim.run(run_time)

neo = dst.get_data('v')
v = neo.segments[0].filter(name='v')[0]
print(v)

sim.end()

import matplotlib.pyplot as plt
plt.figure()
for i, vv in enumerate(v.T):
    plt.plot(vv, label=i)
plt.legend()
plt.show()

