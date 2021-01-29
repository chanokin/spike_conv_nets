import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt

VISUALIZE = bool(1)


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

    return {'vert': v, 'a45': a45, 'horiz': h, 'a135': a135}


img = cv2.imread('./test_img.png', cv2.IMREAD_GRAYSCALE).astype('float')
pix2rate = 100./255.

if VISUALIZE:
    img *= pix2rate
    vmax = np.max(np.abs(img))
    vmin = -vmax
    plt.figure()
    im = plt.imshow(img, cmap='PiYG', vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    # plt.show()



shape = img.shape
flat = img.flatten()
n_input = np.prod(shape, dtype='int32')
rates = [[pix] for pix in flat]

stride = np.array([2, 2], dtype='int32')  # h, w
k_shape = np.array([5, 5], dtype='int32')
kernels = generate_kernels(k_shape, 2.0)

if VISUALIZE:
    plt.figure(figsize=(8, 8))
    vmax = np.max([np.max(kernels[k]) for k in kernels])
    vmin = -vmax
    for i, k in enumerate(kernels):
        ax = plt.subplot(2, 2, i+1)
        im = plt.imshow(kernels[k], cmap='PiYG', label=k, vmin=vmin, vmax=vmax)
        plt.colorbar(im)
    plt.savefig("kernels.png", dpi=300)
    plt.show()

run_time = 50.

sim.setup(timestep=1.)

src = sim.Population(n_input, sim.SpikeSourcePoisson,
                     {'rate': rates}, label='input spikes')
src.record('spikes')

conns = {k: sim.ConvolutionConnector(shape, kernels[k], strides=stride)
         for k in kernels}


out_shapes = {k: conns[k].get_post_shape() for k in conns}
out_sizes = {k: np.prod(out_shapes[k], dtype='int32') for k in out_shapes}

params = {
    'v_thresh': 1.,
    'v_rest': 0.,
    'v_reset': 0.,
    'v': 0.,
    'tau_m': 5.0,
}
outputs = {
    k: sim.Population(out_sizes[k], sim.IF_curr_exp_conv,
                      params, label="out_{}".format(k))
    for k in out_shapes
}
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
    neos=neos, pix2rate=pix2rate, shape=shape,
    flat=flat, n_input=n_input, rates=rates,
    stride=stride, k_shape=k_shape, kernels=kernels,
    run_time=run_time, out_shapes=out_shapes)

