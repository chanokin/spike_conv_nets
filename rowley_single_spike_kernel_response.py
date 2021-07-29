import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys
import field_encoding as fe

ROWS_AS_MSB = bool(1)
VISUALIZE = bool(0)
in_shape = (11, 11)
n_input = int(np.prod(in_shape))
# n_input = fe.max_coord_size(in_shape[1], in_shape[0], ROWS_AS_MSB)

stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([5, 5], dtype='int32')
kernel = (np.arange(np.prod(k_shape)) + 1).reshape(k_shape) * 0.01
kernel -= kernel.mean()
print(kernel)

run_time = 4.

sim.setup(timestep=1.)
machine = sim.get_machine()
# machine.ignore_chips()
chip11 = machine.get_chip_at(1,1)
spike_idx = fe.encode_coords((in_shape[0] // 2), (in_shape[1] // 2),
                             in_shape[1], in_shape[0], ROWS_AS_MSB)
spike_times = [[1.0] if i == spike_idx else []
               for i in range(n_input)]

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': spike_times},
                     structure=Grid2D(in_shape[1]/in_shape[0]),
                     label='input spikes'
                    )

conn = sim.ConvolutionConnector(kernel)

out_shape = conn.get_post_shape(in_shape)
out_size = int(np.prod(out_shape))
# out_size = fe.max_coord_size(out_shape[1], out_shape[0], ROWS_AS_MSB)

out_type = sim.NIF_curr_delta if bool(1) else sim.IF_curr_exp
params = {
    'v': 0,
    'v_thresh': 1,
    'v_reset': 0,
}

if out_type is sim.IF_curr_exp:
    params['v_rest'] = 0
    params['tau_m'] = 20
    params['cm'] = 0.1
    params['tau_syn_E'] = 1
    params['tau_syn_I'] = 1


output = sim.Population(out_size, out_type, params,
                        structure=Grid2D(out_shape[1]/out_shape[0]),
                        label="out"
                       )
output.set(v=0)

src.record('spikes')
output.record(['v', 'spikes'])
# syn = sim.StaticSynapse(weight=ws.flatten)


proj = sim.Projection(src, output, conn, sim.Convolution())

sim.run(run_time)

neo = output.get_data()

sim.end()

v = neo.segments[0].filter(name='v')[0]

vmin = 0
vmax = kernel.max()
print(v.shape)
for t, vt in enumerate(v):
    img = np.zeros(out_shape)
    # img[:] = vt.reshape(out_shape)
    for i, vv in enumerate(vt):
        # r, c = fe.decode_ids(i, most_significant_rows=ROWS_AS_MSB, shape=out_shape)
        # print(i, r, c)
        # if r >= out_shape[0] or c >= out_shape[1]:
        #     continue
        r, c = i // out_shape[0], i % out_shape[0]
        img[r, c] = vv

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Voltage at t = {}".format(t))
    im = plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
plt.show()
# import plot_conv_filter_demo
dko = np.abs(np.asarray(out_shape) - np.asarray(k_shape))
off = dko[0] // 2
ctr = img[off:-off, off:-off]
diff = kernel - ctr
plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_title("Difference between output centre and kernel")
im = plt.imshow(diff, vmin=-vmax, vmax=vmax)
plt.colorbar(im)

plt.show()

print()
np.testing.assert_array_almost_equal(kernel, ctr, decimal=2)
print("np.testing.assert_array_almost_equal(kernel, ctr, decimal=2) passed")

