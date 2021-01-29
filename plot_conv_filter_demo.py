import numpy as np
from matplotlib import pyplot as plt

data = np.load("output_for_conv_filter_demo.npz",
               allow_pickle=True)

run_time = data['run_time']
shape = data['shape']
neos = data['neos'].item()
out_shapes = data['out_shapes'].item()
k_shape = data['k_shape']
stride = data['stride']

# print(neos)

dt = 1.
locs = {'horiz': 1, 'vert': 3, 'input': 5, 'a135': 7,  'a45': 9}
chan = {'horiz': [0], 'vert': [1], 'a135': [1, 2],  'a45': [0, 1]}
colors = {'horiz': 'red', 'vert': 'green', 'a135': 'cyan',  'a45': 'yellow'}
in_img = np.zeros(shape)
out_img = np.zeros((shape[0], shape[1], 3))
for ts in np.arange(0, run_time*0.25, dt):
    in_img *= 0.25
    out_img *= 0.25
    te = ts + dt
    print(ts, te)
    fig = plt.figure(figsize=(20, 10))
    for i, k in enumerate(neos):
        s = neos[k].segments[0].spiketrains
        shp = shape if k == 'input' else out_shapes[k]

        if k == 'input':
            w = shp[1]
            for nid, trains in enumerate(s):
                whr = np.where(
                    np.logical_and(ts <= trains, trains < te))
                if len(whr[0]):
                    r, c = nid // w, nid % w
                    in_img[r, c] = min(1., in_img[r, c] + 1.)


        else:
            w = shp[1]
            for nid, trains in enumerate(s):
                tss = ts + dt
                tee = te + dt
                whr = np.where(
                        np.logical_and(tss <= trains, trains < tee))
                if len(whr[0]):
                    row, col = nid // w, nid % w
                    [padr, padc] = k_shape // 2
                    row = row * stride[0] + padr
                    col = col * stride[1] + padc
                    for c in chan[k]:
                        # out_img[row, col, c] += 0.25
                        out_img[row, col, c] = min(1., out_img[row, col, c] + 1.0)


    plt.suptitle('[{} to {})'.format(ts, te))
    ax = plt.subplot(1, 2, 1)
    im = plt.imshow(in_img, cmap="Greys_r")
    ax.set_title("input")

    ax = plt.subplot(1, 2, 2)
    im = plt.imshow(out_img)
    ax.set_title("filters")
    rect = plt.Rectangle(
            (shape[1], shape[0] * 0.25), 5, -5, fc='gray', ec='none')
    ax.add_patch(rect)
    for y, k in enumerate(neos):
        if k == 'input':
            continue
        s = k
        if k == 'a45':
            s = "{} /".format(s)
        elif k == 'a135':
            s = "{} \\".format(s)
        elif k == 'horiz':
            s = "{} -".format(s)
        elif k == 'vert':
            s = "{} |".format(s)
        ax.text(shape[1], shape[0] * 0.25 + y, s, color=colors[k])

    # plt.show()
    plt.savefig("sim_output_{:010d}.png".format(int(ts)), dpi=300)
    plt.close(fig)


# plt.show()

# for i, vv in enumerate(v.T):
#     plt.plot(vv, label=i)
# plt.legend()
# plt.show()
#
print("end")
