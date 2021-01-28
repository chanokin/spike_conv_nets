import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt

VISUALIZE = bool(1)

def generate_kernels(shape, w):
    def normalize(k, w):
        k -= k.mean()
        k /= k.var()
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
kernels = generate_kernels(k_shape, 3.)

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

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': vline}, label='input spikes')

conn = sim.ConvolutionConnector(shape, ws, strides=stride)

hh, hw = k_shape // 2
for i, x in enumerate(vline):
    if len(x):
        print(i, x)
        r, c = i // shape[1], i % shape[1]
        print("pre {}, r {}, c {}".format(i, r, c))
        postr, postc = conn.pre_as_post(r, c)
        print("postr {}, postc {}".format(postr, postc))
        for kr in range(-hh, hh+1):
            for kc in range(-hw, hw+1):
                print("row {}, col {}, w {}".format(
                    postr + kr, postc + kc,
                    ws[kr + k_shape[0] // 2, kc + k_shape[1] // 2]
                ))

shape_out = conn.get_post_shape()
n_out = np.prod(shape_out, dtype='int32')
dst = sim.Population(n_out, sim.IF_curr_exp_conv,
                     {'v_thresh': -60.0,
                      'v_reset': -80.0})
dst.record(['v', 'spikes'])
# syn = sim.StaticSynapse(weight=ws.flatten)

prj = sim.Projection(src, dst, conn)

sim.run(run_time)

neo = dst.get_data('v')
v = neo.segments[0].filter(name='v')[0]
spikes = neo.segments[0].spiketrains
print(v)
print(spikes)

sim.end()

import matplotlib.pyplot as plt
plt.figure()
for i, vv in enumerate(v.T):
    plt.plot(vv, label=i)
plt.legend()
plt.show()

