import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys

VISUALIZE = bool(0)
ROWS_AS_MSB = bool(1)

def generate_kernels(shape, w=1.0):
    def normalize(k, w):
        # k -= k.mean()
        # k /= k.var()
        k[k < 0] /= -np.sum(k[k < 0])
        k[k > 0] /= np.sum(k[k > 0])
        # k /= np.sum(k**2)
        k *= w

    def rotate(k, a):
        center = np.array(k.shape[1::-1]) / 2
        rot_mat = cv2.getRotationMatrix2D(center, a, 1.0)
        return cv2.warpAffine(k, rot_mat, k.shape[1::-1],
                              flags=cv2.INTER_LINEAR)

    v = -np.ones(shape)
    v[:, shape[1]//2] = 1.0
    normalize(v, w)

    h = -np.ones(shape)
    h[shape[0]//2, :] = 1.0
    normalize(h, w)

    a45 = rotate(h, 45)
    normalize(a45, w)
    a135 = rotate(h, 135)
    normalize(a135, w)

    return {
        'vert': v,
        # 'a45': a45,
        # 'horiz': h,
        'a135': a135
    }

##################################################
###             N E W    S H A P E             ###
##################################################
# shape = (240, 320)  # rows, columns
# shape = (120, 160)  # rows, columns
shape = (60, 80)  # rows, columns
n_input = int(np.prod(shape))
stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([5, 5], dtype='int32')
kernels = generate_kernels(k_shape, 1.5)

for k in kernels:
    print(k)
    print(kernels[k])
# plt.figure(figsize=(8, 8))
# vmax = np.max([np.max(kernels[k]) for k in kernels])
# vmin = -vmax
# for i, k in enumerate(kernels):
#     ax = plt.subplot(2, 2, i+1)
#     im = plt.imshow(kernels[k], cmap='PiYG', label=k, vmin=vmin, vmax=vmax)
#     plt.colorbar(im)
# plt.savefig("kernels.png", dpi=300)
# if VISUALIZE:
#     plt.show()

# sys.exit()


sim.setup(timestep=1.)


print("loaded spikes")
total_sim_time = 10000
n_runs = 1
run_time = total_sim_time / n_runs

spike_times = [[] for _ in range(n_input)]
start_t = 0
end_t = run_time
with open("spikes.txt", "r") as spk_f:
    while True:
        l = spk_f.readline()
        if l == "\n":
            continue
        if len(l) == 0:
            break
        srow, scol, schan, stime, sdv = l.split(", ")
        row = int(srow)
        col = int(scol)
        spike_t = int(stime)
        if spike_t < start_t:
            continue

        if spike_t >= end_t:
            break

        neuron_id = row * shape[1] + col
        spike_times[neuron_id].append(spike_t)

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': spike_times}, label='input spikes',
                     structure=Grid2D(shape[1]/shape[0]))
src.set_max_atoms_per_core((32, 16))

conns = {k: sim.ConvolutionConnector(kernels[k])
         for k in kernels}

out_shapes = {k: conns[k].get_post_shape(shape) for k in conns}
out_sizes = {k: int(np.prod(out_shapes[k])) for k in out_shapes}

params = {
    'v_thresh': 1.,
    'v_rest': 0.,
    'v_reset': 0.,
    'v': 0.,
    'tau_m': 1.0,
}

outputs = {
    k: sim.Population(out_sizes[k], sim.IF_curr_exp,
                      params, label="out_{}".format(k),
                      structure=Grid2D(out_shapes[k][1]/out_shapes[k][0]))
    for k in out_shapes
}


src.record('spikes')

for k in outputs:
    outputs[k].set_max_atoms_per_core((32, 16))
    outputs[k].record(['spikes',])
# syn = sim.StaticSynapse(weight=ws.flatten)


projs = {
    k: sim.Projection(src, outputs[k], conns[k], sim.Convolution())
    for k in outputs
}


for run_idx in range(n_runs):
    # start_t = run_idx * run_time
    # end_t = start_t + run_time
    # spike_times = [[] for _ in range(n_input)]
    # with open("spikes.txt", "r") as spk_f:
    #     while True:
    #         l = spk_f.readline()
    #         if l == "\n":
    #             continue
    #         if len(l) == 0:
    #             break
    #         srow, scol, schan, stime, sdv = l.split(", ")
    #         row = int(srow)
    #         col = int(scol)
    #         spike_t = int(stime)
    #         if spike_t < start_t:
    #             continue
    #
    #         if spike_t >= end_t:
    #             break
    #
    #         neuron_id = row * shape[1] + col
    #         spike_times[neuron_id].append(spike_t)

    # src.set(spike_times=spike_times)

    sim.run(run_time)


neos = {
    k: outputs[k].get_data()
    for k in outputs
}
neos['input'] = src.get_data()
sim.end()

np.savez_compressed("output_for_conv_filter_demo.npz",
    neos=neos, shape=shape,
    n_input=n_input,
    stride=stride, k_shape=k_shape, kernels=kernels,
    run_time=run_time, out_shapes=out_shapes,
)

# import plot_conv_filter_demo

