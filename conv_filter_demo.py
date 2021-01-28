import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt

VISUALIZE = bool(0)


def generate_kernels(shape, w):
    def normalize(k, w):
        # k -= k.mean()
        # k /= k.var()
        k[k < 0] /= np.sum(k[k < 0])
        k[k > 0] /= np.sum(k[k > 0])
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
    a135 = rotate(h, 135)

    return {'vert': v, 'a45': a45, 'horiz': h, 'a135': a135}


img = cv2.imread('./test_img.png', cv2.IMREAD_GRAYSCALE).astype('float')

if VISUALIZE:
    vmax = np.max(np.abs(img))
    vmin = -vmax
    plt.figure()
    im = plt.imshow(img, cmap='PiYG', vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.show()

pix2rate = 100./255.

shape = img.shape
flat = img.flatten()
n_input = np.prod(shape, dtype='int32')
rates = [[pix * pix2rate] for pix in flat]

stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([5, 5], dtype='int32')
kernels = generate_kernels(k_shape, 1.2)

if VISUALIZE:
    plt.figure(figsize=(8, 8))
    for i, k in enumerate(kernels):
        vmax = np.max(np.abs(kernels[k]))
        vmin = -vmax
        ax = plt.subplot(2, 2, i+1)
        im = plt.imshow(kernels[k], cmap='PiYG', label=k, vmin=vmin, vmax=vmax)
        plt.colorbar(im)

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

# print(neos)

dt = 3.
locs = {'horiz': 1, 'vert': 3, 'input': 5, 'a135': 7,  'a45': 9}
for ts in np.arange(0, run_time, dt):
    te = ts + dt
    print(ts, te)
    fig = plt.figure(figsize=(10, 10))
    for i, k in enumerate(neos):
        s = neos[k].segments[0].spiketrains
        shp = shape if k == 'input' else out_shapes[k]
        out_img = np.zeros(shp)
        w = shp[1]
        for nid, trains in enumerate(s):
            whr = np.where(
                    np.logical_and(ts <= trains, trains < te))
            if len(whr[0]):
                out_img[nid // w, nid % w] = 1.

        plt.suptitle('[{} to {})'.format(ts, te))
        ax = plt.subplot(3, 3, locs[k])
        im = plt.imshow(out_img)
        plt.colorbar(im)
        ax.set_title("filter {}".format(k))

    plt.savefig("sim_output_{:010d}.png".format(int(ts)), dpi=300)
    plt.close(fig)
    # plt.show()

# plt.show()

# for i, vv in enumerate(v.T):
#     plt.plot(vv, label=i)
# plt.legend()
# plt.show()
#
print("end")
