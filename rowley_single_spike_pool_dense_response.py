import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys
import field_encoding as fe
from copy import deepcopy
ROWS_AS_MSB = bool(1)
VISUALIZE = bool(0)


def to_pynn_shape(shape):
    s = [x for x in shape]
    if len(s):
        s[0] = shape[1]
        s[1] = shape[0]

    return tuple(s)


def pynn_aspect_ratio(shape):
    return float(shape[1]) / float(shape[0])


in_shape = (3, 3)
n_input = int(np.prod(in_shape))
# n_input = fe.max_coord_size(in_shape[1], in_shape[0], ROWS_AS_MSB)

stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([9, 9], dtype='int32')
kernel = (np.arange(np.prod(k_shape)) + 1).reshape(k_shape) * 0.01
kernel -= kernel.mean()
print(kernel)

run_time = 4.


sim.setup(timestep=1.)

sim.set_number_of_neurons_per_core(sim.NIF_curr_delta_dense, (4, 4))

# spike_idx = fe.encode_coords((in_shape[0] // 2), (in_shape[1] // 2),
#                              in_shape[1], in_shape[0], ROWS_AS_MSB)
spike_idx = (in_shape[0] // 2) * in_shape[1] + (in_shape[1] // 2)
spike_times = [[1.0] if i == spike_idx else []
               for i in range(n_input)]

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': spike_times},
                     structure=Grid2D(pynn_aspect_ratio(in_shape)),
                     label='input spikes'
                    )

conn = sim.PoolDenseConnector(kernel)

out_shape = in_shape
out_size = int(np.prod(out_shape))
# out_size = fe.max_coord_size(out_shape[1], out_shape[0], ROWS_AS_MSB)

out_type = sim.NIF_curr_delta_dense
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
                        structure=Grid2D(pynn_aspect_ratio(out_shape)),
                        label="out"
                       )
output.set(v=0)

src.record('spikes')
output.record(['v', 'spikes'])
# syn = sim.StaticSynapse(weight=ws.flatten)


proj = sim.Projection(src, output, conn, sim.PoolDense())

sim.run(run_time)

neo = output.get_data()

sim.end()

v = neo.segments[0].filter(name='v')[0]


vmax = np.max(np.abs(kernel))
vmin = -vmax
print(v.shape)
for t, vt in enumerate(v):
    img = np.zeros(out_shape)
    # img[:] = vt.reshape(out_shape)
    for i, vv in enumerate(vt):
        # r, c = fe.decode_ids(i, most_significant_rows=ROWS_AS_MSB, shape=out_shape)
        # print(i, r, c)
        # if r >= out_shape[0] or c >= out_shape[1]:
        #     continue
        r, c = i // out_shape[1], i % out_shape[1]
        img[r, c] = vv

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Voltage at t = {}".format(t))
    im = plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
plt.show()
# import plot_conv_filter_demo
dko = np.abs(np.asarray(out_shape) - np.asarray(k_shape))
off0 = dko[0] // 2
off1 = dko[1] // 2
ctr = img[off0:-off0, off1:-off1]
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

