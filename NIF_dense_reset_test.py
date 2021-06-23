import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys

sim.NIF_curr_exp_pool_dense.set_model_max_atoms_per_core(10)
sim.setup(timestep=1.)
pre_shape = (5, 5)
n_neurons = int(np.prod(pre_shape))
src = sim.Population(n_neurons, sim.SpikeSourceArray,
                     {'spike_times': [[1, 2, 3] for _ in range(n_neurons)]},
                     label='input spikes')

params = {
    'v_thresh': 1.,
    'v_reset': 0.,
    'v': 0.,
}
post_size = 13
w = 1./(3 * (n_neurons + 1))
dense_w = np.ones((n_neurons, post_size)) * w
# dense_w = np.random.uniform(-0.5, 0.5, size=(n_neurons, post_size))

conn = sim.PoolDenseConnector(0, pre_shape, dense_w, post_size=post_size)
pool_shape = conn.get_post_pool_shape()
# post_shape = (1, 1)
output = sim.Population(post_size, sim.NIF_curr_exp_pool_dense,
                        params, label="out")



proj = sim.Projection(src, output, conn)

output.record(['v', 'spikes'])
run_time = 10.
sim.run(run_time)

# neo = output.get_data()

# sim.reset()
output.set(v=0.)
src.set(spike_times=[[12, 13, 14, 15]
                     # if np.random.uniform(0., 1.) <= 0.75 else []
                     for _ in range(n_neurons)])

print("-----------------------------------------------------")
print(" second half of test")
print("-----------------------------------------------------")
sim.run(run_time)
neo2 = output.get_data()

print('end')
sim.end()

# v0 = neo.segments[0].filter(name='v')[0]
v1 = neo2.segments[0].filter(name='v')[0]
spikes = neo2.segments[0].spiketrains

# plt.figure()
# plt.plot(v0, marker='.')

fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
# ax0 = plt.subplot(2, 1, 1)
for i, v in enumerate(v1.T):
    v = np.array([float(vv) for vv in v])
    ax0.plot(v + i, marker='+')
    ax0.axhline(i, linestyle='--', linewidth=1)
ax0.set_xlim(0, 2*run_time)

# fig = plt.figure()
# ax1 = plt.subplot(2, 1, 2)
for i, spks in enumerate(spikes):
    ax1.plot(spks, i * np.ones_like(spks), '.')

ax1.set_xlim(0, 2*run_time)
plt.show()
