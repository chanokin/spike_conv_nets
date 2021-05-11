import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys

VISUALIZE = bool(0)
in_shape = (11, 11)
n_input = int(np.prod(in_shape))
stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([5, 5], dtype='int32')
kernel = (np.arange(np.prod(k_shape)) + 1).reshape(k_shape) * 0.01
print(kernel)

# sys.exit()

# sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=1024)
# sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=2048)
sim.NIF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=2048)
# sim.SpikeSourcePoisson.set_model_max_atoms_per_core(n_atoms=1024)
# sim.SpikeSourcePoisson.set_model_max_atoms_per_core(n_atoms=100)

run_time = 5.

sim.setup(timestep=1.)

spike_idx = ( (in_shape[0] // 2) * in_shape[1] + (in_shape[1] // 2) )
spike_times = [[1.0] if i == spike_idx else []
               for i in range(int(np.prod(in_shape)))]

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': spike_times}, label='input spikes')

conn = sim.ConvolutionConnector(in_shape, kernel, strides=stride)

out_shape = conn.get_post_shape()
out_size = int(np.prod(out_shape))

params = {
    'v_thresh': 1.,
    'v_reset': 0.,
    'v': 0.,
    # 'v_rest': 0.,
    # 'tau_m': 1.0,
}
output = sim.Population(out_size, sim.NIF_curr_exp_conv,
                         params, label="out")

src.record('spikes')
output.record(['v', 'spikes'])
# syn = sim.StaticSynapse(weight=ws.flatten)


proj = sim.Projection(src, output, conn)

sim.run(run_time)

neo = output.get_data()

sim.end()

v = neo.segments[0].filter(name='v')[0]

vmin = 0
vmax = kernel.max()
print(v.shape)
for t, vt in enumerate(v):
    img = np.zeros(out_shape)
    img[:] = vt.reshape(out_shape)
    plt.figure()
    im = plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
plt.show()
# import plot_conv_filter_demo

ctr = img[1:-1, 1:-1]
diff = kernel - ctr
plt.figure()
im = plt.imshow(diff, vmin=-vmax, vmax=vmax)
plt.colorbar(im)

plt.show()

print()
np.testing.assert_array_almost_equal(kernel, ctr, decimal=2)
