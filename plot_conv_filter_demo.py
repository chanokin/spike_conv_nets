import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data = np.load("output_for_conv_filter_demo.npz",
               allow_pickle=True)

run_time = data['run_time']
shape = data['shape']
neos = data['neos'].item()
out_shapes = data['out_shapes'].item()
k_shape = data['k_shape']
stride = data['stride']
name = data['input_name']
as_post = data['as_post'].item()

# print(neos)

dt = 1.
locs = {'horiz': (1, 2), 'vert': (1, 3), 'input': 5,
        'a135': (2, 2),  'a45': (2, 3)}
char = {'horiz': '_', 'vert': '|', 'a135': '$\\backslash$',  'a45': '$/$'}
chan = {'horiz': [0], 'vert': [1], 'a135': [1, 2],  'a45': [0, 1]}
wchan = {'horiz': [1.], 'vert': [1.], 'a135': [1., 1.],  'a45': [1., 0.5]}
colors = {'input': 'gray', 'horiz': 'red', 'vert': 'blue',
          'a135': 'cyan',  'a45': 'orange'}

### RASTER PLOT

plt.figure()
k = 'input'
s = neos[k].segments[0].spiketrains
for j, times in enumerate(s):
    plt.plot(times, (j) * np.ones_like(times), '.',
             markersize=1., color=colors[k])

for i, k in enumerate(neos):
    if k == 'input':
        continue
    s = neos[k].segments[0].spiketrains
    nspikes = np.sum([len(times) for times in s])
    print(k, nspikes)
    for j, times in enumerate(s):
        plt.plot(times, (j + 0.2 * i) * np.ones_like(times), '|',
                 markersize=5., color=colors[k])

plt.savefig("{}_raster_conv_filter_demo.png".format(name), dpi=300)
# plt.show()
# import sys
# sys.exit(0)

# neo.segments[0].filter(name='v')[0]
k = 'a45'
v = neos[k].segments[0].filter(name='v')
### IMAGE VERSION
vmax = np.max([np.max(neos[k].segments[0].filter(name='v')[0])
               for k in neos if k != 'input'])
vmin = np.max([np.min(neos[k].segments[0].filter(name='v')[0])
               for k in neos if k != 'input'])
# vmin = 0
vmax = np.max(np.abs([vmin, vmax]))
# vmax = 1.0
vmin = -vmax
# vmin = None
# vmax = None
in_img = np.zeros(shape)
out_imgs = {k: np.zeros((shape[0], shape[1])) for k in out_shapes}
fade = 0.3
cmap = 'hot'
# cmap = 'seismic_r'
for tidx, ts in enumerate(np.arange(0, run_time, dt)):
    in_img *= fade
    for k in out_imgs:
        out_imgs[k] *= 0.

    te = ts + dt
    print(ts, te)
    fig = plt.figure(figsize=(20, 10))#, constrained_layout=True)
    # fig = plt.figure(figsize=(20, 10))
    fig.suptitle('[{} to {})'.format(ts, te))
    widths = [2, 1, 1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                            height_ratios=heights)
    locations = {
        'input': spec[:, 0], 'horiz': spec[0, 1],
        'vert': spec[0, 2],  'a45': spec[1, 2],
        'a135': spec[1, 1],
    }
    for i, k in enumerate(neos):
        s = neos[k].segments[0].spiketrains
        if k != 'input':
            voltages = neos[k].segments[0].filter(name='v')[0]

        shp = shape if k == 'input' else out_shapes[k]

        if k == 'input':
            w = shp[1]
            for nid, times in enumerate(s):
                whr = np.where(
                    np.logical_and(ts <= times, times < te))
                if len(whr[0]):
                    r, c = nid // w, nid % w
                    in_img[r, c] = min(1., in_img[r, c] + 1.)

            ax = fig.add_subplot(locations[k])
            # pos = ax.get_position()
            # dx = pos.x0 - pos.x0 * 0.5
            # pos.x0 -= dx
            # ax.set_position([pos.x0, pos.y0,
            #                  pos.width, pos.height])

            im = ax.imshow(in_img, cmap="Greys_r")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(k)



        else:
            w = shp[1]
            vidx = tidx+1 if tidx+1 < len(voltages) else tidx
            vs = voltages[vidx]
            for nid, v in enumerate(vs):
                row, col = nid // w, nid % w

                row += k_shape[0] // 2
                col += k_shape[1] // 2
                # row, col = as_post[k][row][col] + 1

                out_imgs[k][row, col] = float(v)

            ax = fig.add_subplot(locations[k])
            pos = ax.get_position()
            # dx = pos.x0 - pos.x0 * 0.95
            # pos.x0 -= dx
            # ax.set_position([pos.x0, pos.y0,
            #                  pos.width, pos.height])

            im = ax.imshow(out_imgs[k], cmap=cmap, vmin=vmin, vmax=vmax,
                           alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(k)
            if k in ['vert', 'a45']:
                axins = inset_axes(ax,
                                   width="5%",  # width = 5% of parent_bbox width
                                   height="100%",  # height : 50%
                                   loc='lower left',
                                   bbox_to_anchor=(1.0, 0., 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )
                fig.colorbar(im, cax=axins)

            for nid, times in enumerate(s):
                tss = ts + dt
                tee = te + dt
                whr = np.where(
                        np.logical_and(tss <= times, times < tee))
                if len(whr[0]):
                    row, col = nid // w, nid % w
                    row += k_shape[0] // 2
                    col += k_shape[1] // 2

                    # [padr, padc] = k_shape // 2
                    # row = row * stride[0] + padr
                    # col = col * stride[1] + padc
                    print(k, row, col, ts, te, tidx)
                    ax.plot(col, row, marker="^",#char[k],
                            markersize=20.,
                            markeredgewidth=5.,
                            markerfacecolor='green',
                            markeredgecolor='red',
                            # color='black'
                            )




    # fig.show()

    # fig.tight_layout()
    fig.savefig("{}_sim_output_{:010d}.png".format(name, int(ts)), dpi=300)
    plt.close(fig)


# plt.show()

# for i, vv in enumerate(v.T):
#     plt.plot(vv, label=i)
# plt.legend()
# plt.show()
#
print("end")
