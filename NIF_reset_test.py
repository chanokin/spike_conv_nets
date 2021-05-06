import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys

w = 0.3
sim.setup(timestep=1.)
pre_shape = (4, 4)
n_neurons = int(np.prod(pre_shape))
src = sim.Population(n_neurons, sim.SpikeSourceArray,
                     {'spike_times': [[1, 2, 3] for _ in range(n_neurons)]},
                     label='input spikes')

params = {
    'v_thresh': 1.,
    'v_reset': 0.,
    'v': 0.,
}
conn = sim.ConvolutionConnector(pre_shape, np.array([[w]]), strides=1)
post_shape = conn.get_post_shape()
# post_shape = (1, 1)
output = sim.Population(int(np.prod(post_shape)), sim.NIF_curr_exp_conv,
                        params, label="out")



proj = sim.Projection(src, output, conn)

output.record(['v', 'spikes'])
run_time = 10.
sim.run(run_time)

# neo = output.get_data()

# sim.reset()
output.set(v=0.)
src.set(spike_times=[[12, 13, 14, 15] for _ in range(n_neurons)])

sim.run(run_time)
neo2 = output.get_data()

print('end')
sim.end()

# v0 = neo.segments[0].filter(name='v')[0]
v1 = neo2.segments[0].filter(name='v')[0]

# plt.figure()
# plt.plot(v0, marker='.')

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
for i, v in enumerate(v1.T):
    v = np.array([float(vv) for vv in v])
    ax.plot(v + i, marker='+')

ax.set_xlim(0, 2*run_time)
plt.show()
