from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import cv2

data = np.load("output_for_conv_filter_demo.npz",
               allow_pickle=True)

run_time = data['run_time']
shape = data['shape']

neos = data['neos'].item()
out_shapes = data['out_shapes'].item()
# out_aug_shapes = {k: fe.get_augmented_shape(out_shapes[k])
#                   for k in out_shapes}
k_shape = data['k_shape']
stride = data['stride']

# as_post = data['as_post'].item()

# print(neos)

dt = 33.
locs = {'horiz': (1, 2), 'vert': (1, 3), 'input': 5,
        'a135': (2, 2),  'a45': (2, 3)}
char = {'horiz': '_', 'vert': '|', 'a135': '$\\backslash$',  'a45': '$/$'}
chan = {'horiz': [0], 'vert': [1], 'a135': [1, 2],  'a45': [0, 1]}
wchan = {'horiz': [1.], 'vert': [1.], 'a135': [1., 1.],  'a45': [1., 0.5]}
colors = {'input': 'gray', 'horiz': 'red', 'vert': 'blue',
          'a135': 'cyan',  'a45': 'orange'}

### RASTER PLOT

# plt.figure()
# k = 'input'
# s = neos[k].segments[0].spiketrains
# for j, times in enumerate(s):
#     plt.plot(times, (j) * np.ones_like(times), '.',
#              markersize=1., color=colors[k])
#
# for i, k in enumerate(neos):
#     if k == 'input':
#         continue
#     s = neos[k].segments[0].spiketrains
#     nspikes = np.sum([len(times) for times in s])
#     print(k, nspikes)
#     for j, times in enumerate(s):
#         plt.plot(times, (j + 0.2 * i) * np.ones_like(times), '|',
#                  markersize=5., color=colors[k])
#
# plt.savefig("{}_raster_conv_filter_demo.png".format(k), dpi=300)
# plt.show()
# sys.exit(0)

in_img = np.zeros(shape)
out_imgs = {k: np.zeros(out_shapes[k], dtype='uint8') for k in out_shapes}
fade = 0.3
cmap = 'hot'
# cmap = 'seismic_r'
name = 'dvs_emu_input'
for tidx, ts in tqdm(enumerate(np.arange(0, run_time, dt))):

    te = ts + dt
    for i, k in enumerate(neos):
        if k == 'input':
            continue

        s = neos[k].segments[0].spiketrains
        voltages = neos[k].segments[0].filter(name='v')

        shp = shape if k == 'input' else out_shapes[k]
        w = shp[1]
        out_imgs[k][:] = 0
        for nid, times in enumerate(s):
            tss = ts + dt
            tee = te + dt
            whr = np.where(
                    np.logical_and(tss <= times, times < tee))

            n_spikes = len(whr[0])
            if n_spikes > 0:
                row, col = nid // w, nid % w
                out_imgs[k][row, col] = 255#min(n_spikes * 21, 255)

        image_fname = f"./spinn_out/{name}_{k}_sim_output_{tidx:010d}.png"
        cv2.imwrite(image_fname, out_imgs[k])
