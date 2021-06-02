import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys
import field_encoding as fe

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
        rot_mat = cv2.getRotationMatrix2D(
            tuple(np.array(k.shape[1::-1]) // 2), a, 1.0)
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
        'a45': a45,
        'horiz': h,
        'a135': a135
    }

img_name = 'test_img'
# img_name = 'test_pulse'
img = cv2.imread('./{}.png'.format(img_name),
                 cv2.IMREAD_GRAYSCALE).astype('float')

##################################################
###             N E W    S H A P E             ###
##################################################
new_shape = (np.asarray(img.shape) * 1.0).astype('int')
# new_shape = np.array([7, 7], dtype='int')

img = cv2.resize(img, tuple(new_shape))

# new_shape = np.array([7, 7], dtype='int')
# centre =  new_shape // 2
# img = np.zeros(new_shape)
# img[centre[0], centre[1]] = 255.0


pix2rate = 500./255.
img *= pix2rate

if VISUALIZE:
    vmax = np.max(np.abs(img))
    vmin = -vmax
    plt.figure()
    im = plt.imshow(img, cmap='PiYG', vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    # plt.show()

shape = img.shape
flat = img.flatten()
n_input = int(np.prod(shape))
bits_w = int(np.ceil(np.log2(shape[1])))
bits_h = int(np.ceil(np.log2(shape[0])))
n_input = 2**(bits_h + bits_w)  # int(np.prod(shape, dtype='int32'))

rates = np.zeros((n_input, 1))
n_in_rows = shape[0]
n_in_cols = shape[1]
rows = np.repeat(np.arange(n_in_rows), n_in_cols)
cols = np.tile(np.arange(n_in_cols), n_in_rows)
ids = fe.encode_coords(rows, cols, n_in_rows, n_in_cols, ROWS_AS_MSB)
rates[ids, :] = img.flatten()[:, np.newaxis]
# centre_id = fe.encode_coords(centre[0], centre[1], new_shape[1], new_shape[0],
#                              ROWS_AS_MSB)
# rates[centre_id][0] = img[centre[0], centre[1]]


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

# sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=1024)
# sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=2048)
# sim.SpikeSourceArray.set_model_max_atoms_per_core(n_atoms=36)
# sim.SpikeSourcePoisson.set_model_max_atoms_per_core(n_atoms=18)
sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=512)
# sim.SpikeSourcePoisson.set_model_max_atoms_per_core(n_atoms=1024)
# sim.SpikeSourcePoisson.set_model_max_atoms_per_core(n_atoms=100)

run_time = 10.

sim.setup(timestep=1.)

if bool(0):
    spike_idx = [0, (n_input + shape[1])//2, shape[1] - 1, n_input - shape[1], n_input - 1]
    spike_times = [[] for idx in range(n_input)]
    for i, sidx in enumerate(spike_idx):
        spike_times[sidx] = [2*i + 1]

    src = sim.Population(n_input, sim.SpikeSourceArray,
                         {'spike_times': spike_times}, label='input spikes')
else:
    src = sim.Population(n_input, sim.SpikeSourcePoisson,
                         {'rate': rates}, label='input spikes')

conns = {k: sim.ConvolutionConnector(shape, kernels[k], strides=stride)
         for k in kernels}

as_post = {k: {r: {c: conns[k].pre_as_post(r, c)
                      for c in range(shape[1])}
               for r in range(shape[0])}
           for k in conns}

out_shapes = {k: conns[k].get_post_shape() for k in conns}
out_sizes = {k: fe.power_of_2_size(out_shapes[k][1], out_shapes[k][0], ROWS_AS_MSB)
             for k in out_shapes}

params = {
    'v_thresh': 1.,
    'v_rest': 0.,
    'v_reset': 0.,
    'v': 0.,
    'tau_m': 1.0,
}

outputs = {
    k: sim.Population(out_sizes[k], sim.IF_curr_exp_conv,
                      params, label="out_{}".format(k))
    for k in out_shapes
}

src.record('spikes')

for k in outputs:
    outputs[k].record(['v', 'spikes'])
# syn = sim.StaticSynapse(weight=ws.flatten)


projs = {
    k: sim.Projection(src, outputs[k], conns[k])
    for k in outputs
}

sim.run(run_time)

neos = {
    k: outputs[k].get_data()
    for k in outputs
}
neos['input'] = src.get_data()
sim.end()

np.savez_compressed("output_for_conv_filter_demo.npz",
    input_name=img_name,
    neos=neos, pix2rate=pix2rate, shape=shape,
    flat=flat, n_input=n_input, rates=rates,
    stride=stride, k_shape=k_shape, kernels=kernels,
    run_time=run_time, out_shapes=out_shapes,
    as_post=as_post
)

# import plot_conv_filter_demo

