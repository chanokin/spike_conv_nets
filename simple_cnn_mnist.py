import spynnaker8 as sim
import numpy as np
import mnist
import matplotlib.pyplot as plt
import plotting
import sys

filename = "simple_cnn_network_elements.npz"

data = np.load(filename, allow_pickle=True)
thresholds = dict(
    conv2d=1,#3.1836495399475098,
    conv2d_1=1,#2.9346282482147217,
    dense=1,#1.1361589431762695,
    dense_1=1,#2.435835599899292,
    dense_2=1,#2.36885929107666,
)

order0 = data['order']
order = order0[:]
ml_conns = data['conns'].item()
ml_param = data['params'].item()

print(list(data.keys()))

test_X = mnist.test_images('./')
test_y = mnist.test_labels('./')

# shape of dataset
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# plotting
# for i in range(9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(test_X[i], cmap=plt.get_cmap('gray'))
#
# plt.show()

sim.extra_models.SpikeSourcePoissonVariable.set_model_max_atoms_per_core(270)
sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=1024)
# sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=256)
# sim.IF_curr_exp_pool_dense.set_model_max_atoms_per_core(n_atoms=64)

sim.setup(timestep=1.)

np.random.seed(13)

shape_in = np.asarray([28, 28])
n_in = int(np.prod(shape_in))
n_digits = 10#0
digit_duration = 1000.0  # ms
digit_rate = 100.0  # hz
in_rates = np.zeros((n_in, n_digits))
for i in range(n_digits):
    in_rates[:, i] = test_X[i].flatten()

in_rates *= (digit_rate / in_rates.max())
in_durations = np.ones((n_in, n_digits)) * np.round(digit_duration * 0.5)
in_starts = np.repeat([np.arange(n_digits) * digit_duration],
                      n_in, axis=0)
in_params = {
    'rates': in_rates,
    'starts': in_starts,
    'durations': in_durations
}
pops = {
    'input': [sim.Population(  # put into list for ease of connection
        n_in,  # number of sources
        sim.extra_models.SpikeSourcePoissonVariable,  # source type
        in_params,
        label='mnist',
        additional_parameters={'seed': 24534}
    )]
}
sizes = {'input': [p.size for p in pops['input']]}

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
def_params = {
    'v_thresh': 1.,
    'v_rest': 0.,
    'v_reset': 0.,
    'v': 0.,
    'tau_m': 10.,
    'cm': 0.3,
}

for i, o in enumerate(order):
    if i == 0:
        continue

    par = ml_param[o]
    if 'conv2d' in o.lower():
        shape = par['shape'][0:2]
        chans = par['shape'][2]
        ps = def_params.copy()
        v = ps.pop('v')
        ps['v_thresh'] = thresholds[o] if bool(0) else par['threshold']
        n = int(np.prod(shape))
        print(o, n, shape, chans)
        pop = [sim.Population(n, sim.IF_curr_exp_conv, ps,
                              label="{}_chan_{}".format(o, ch))
               for ch in range(chans)]

        for p in pop:
            p.set(v=v)
    elif 'dense' in o.lower():
        shape = par['shape'][0:2]
        ps = def_params.copy()
        v = ps.pop('v')
        ps['v_thresh'] = thresholds[o]

        n = int(np.prod(shape))
        chans = 1
        if 'conv2d' in order[i-1]:
            chans = 4
            n = n // chans

        pop = [sim.Population(n, sim.IF_curr_exp_pool_dense, ps,
                              label="{}_chan_{}".format(o, ch))
               for ch in range(chans)]
        for p in pop:
            p.set(v=v)

    sizes[o] = [p.size for p in pop]
    pops[o] = pop

rec = [
    'input',
    'conv2d',
    'conv2d_1',
    'dense',
    'dense_1',
    'dense_2',
]

shapes = {
    'input': [28, 28],
    'conv2d': [24, 24],
    'conv2d_1': [8, 8],
    'dense': [12, 12],
    'dense_1': [8, 8],
    'dense_2': [4, 4],
}

offsets = {
    'input': 0,
    'conv2d': 0,
    'conv2d_1': 0,
    'dense': 32,
    'dense_1': 0,
    'dense_2': 0,
}

for k in rec:
    for p in pops[k][:]:
        p.record('spikes')

projs = {}
kernels = {}
dense_weights = {}

def norm_w(w, is_conv=False):
    # pos = w[w > 0]
    # pos /= np.sum(pos)
    # neg = w[w < 0]
    # neg /= (-np.sum(neg))
    # new_w = w.copy()
    # new_w[w > 0] = pos
    # new_w[w < 0] = neg
    # return new_w
    return w

for i, o in enumerate(order):
    if i == 0:
        continue
    c = ml_conns[o]
    weights = c['weights']
    pooling = 'pool' in c
    pool_area = np.asarray(c['pool']['shape']) if pooling else None
    pool_stride = np.asarray(c['pool']['strides']) if pooling else None
    wshape = c.get('shape', None)
    strides = c.get('strides', None)

    o0 = order[i-1]
    # dense_weights[o] =
    for prei, pre in enumerate(pops[o0]):
        pre_shape = np.asarray(ml_param[o0]['shape'][:2])
        if len(pre_shape) == 1:
            pre_shape = (pre.size, 1)
            n_chan = 1
        else:
            n_chan = ml_param[o0]['shape'][-1]

        wl = []
        for posti, post in enumerate(pops[o]):
            lbl = "{}_{} to {}_{}".format(o0, prei, o, posti)
            print(pre_shape, n_chan, o0, prei, o, posti)
            if 'conv2d' in o.lower():
                # print(prei, posti, wshape, c['weights'].shape)
                w = norm_w(weights[:, :, prei, posti].reshape(wshape))
                wl.append(w)
                cn = sim.ConvolutionConnector(pre_shape, w, strides=strides,
                        pooling=pool_area, pool_stride=pool_stride)
                prj = sim.Projection(pre, post, cn, label=lbl)
                projs[lbl] = prj

            elif 'dense' in o.lower():
                n_out = post.size
                sh_pre = sim.PoolDenseConnector.calc_post_pool_shape(
                            pre_shape, pooling, pool_area, pool_stride)
                size_pre = int(np.prod(sh_pre))
                if 'conv2d' in o0.lower():
                    cnv = pops[o0]
                    col0 = posti * n_out
                    col1 = col0 + n_out

                    mtx_rows = []
                    chan = prei
                    for r in np.arange(sh_pre[0]):
                        for c in np.arange(sh_pre[1]):
                            mtx_rows.append(r * sh_pre[1] * len(pops[o0]) +
                                            c * len(pops[o0]) + chan)
                    mtx_rows = np.asarray(mtx_rows)
                    pre_rows = np.repeat(np.arange(sh_pre[0]), sh_pre[1])
                    # print("pre_rows = {}".format(pre_rows))
                    pre_cols = np.tile(np.arange(sh_pre[1]), sh_pre[0])
                    # print("pre_cols = {}".format(pre_cols))
                    mtx_rows0 = (pre_rows * sh_pre[1] * n_chan +
                                 pre_cols * n_chan + prei)
                    print("mtx_rows = {}".format(mtx_rows))
                    print(np.all(mtx_rows == mtx_rows0))
                    n_rows = len(mtx_rows)
                    mtx_rows = np.repeat(mtx_rows, n_out)
                    # print("mtx_rows = {}".format(mtx_rows))
                    mtx_cols = np.arange(col0, col1)
                    print("mtx_cols = {}".format(mtx_cols))
                    mtx_cols = np.tile(mtx_cols, n_rows)
                    # print("mtx_cols = {}".format(mtx_cols))
                    ws = norm_w(weights[mtx_rows, mtx_cols].reshape((n_rows, n_out)))
                    print(ws.shape)
                    # print(ws)
                    # row0 = prei * size_pre
                    # row1 = row0 + size_pre
                    # ws = weights[row0:row1, col0:col1]

                else:
                    row0 = prei * size_pre
                    row1 = row0 + size_pre
                    ws = norm_w(weights[row0:row1, :])
                wl.append(ws)
                cn = sim.PoolDenseConnector(pre_shape, ws, n_out, pool_area,
                                            pool_stride)

                prj = sim.Projection(pre, post, cn, label=lbl)
                projs[lbl] = prj
        kernels[o] = wl

sim_time = digit_duration * (n_digits + 0.1)

neos = {}
spikes = {}
sim.run(sim_time)

for k in rec:
    neos[k] = [p.get_data() for p in pops[k]]
    spikes[k] = [x.segments[0].spiketrains for x in neos[k]]


sim.end()


rates = {}
conf_matrix = np.zeros((10, 10))
correct = 0
no_spikes = 0
ncols = 5
for si, k in enumerate(order):
    if k not in spikes:
        continue

    if k == 'dense':
        imgs, bins = plotting.spikes_to_images_list(
                                  spikes[k], shapes[k], sim_time,
                                  digit_duration, offsets[k],
                                  merge_images=True)
        # rates[k] = bins
        rates[k] = [[np.mean([len(ts) for ts in b])
                    for b in pbins] for pbins in bins]

        nimgs = len(imgs)
        nrows = nimgs // ncols + int(nimgs % ncols > 0)

        fig = plt.figure(figsize=(ncols, nrows))
        plt.suptitle("{}_{}".format(k, 0))
        for i in range(nimgs):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.imshow(imgs[i])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig("{:03d}_{}_{:03d}.png".format(si, k, 0), dpi=150)
        plt.close(fig)
        continue

    rl = []
    for pi, p in enumerate(spikes[k]):
        imgs, bins = plotting.spikes_to_images(p, shapes[k], sim_time,
                                               digit_duration)
        rl.append([np.mean([len(ts) for ts in b]) for b in bins])

        nimgs = len(imgs)
        # if 'conv2d' in k:
        #     nimgs += 1
        nimgs += 1

        nrows = nimgs // ncols + int(nimgs % ncols > 0)

        fig = plt.figure(figsize=(ncols, nrows))
        plt.suptitle("{}_{}".format(k, pi))
        for i in range(nimgs):
            if i == len(imgs):
                break

            ax = plt.subplot(nrows, ncols, i + 1)
            if k == 'dense_2':
                if np.sum(imgs[i]) == 0:
                    no_spikes += 1
                else:
                    pred = np.argmax(imgs[i])
                    corr = test_y[i]
                    if pred == corr:
                        correct += 1
                    conf_matrix[corr, pred] += 1
                    ax.set_title("{} - {}".format(test_y[i], np.argmax(imgs[i])))

            ax.imshow(imgs[i])
            ax.set_xticks([])
            ax.set_yticks([])

        if 'conv2d' in k or 'dense' in k:
            w = kernels[k][pi]
            vmax = np.max(np.abs(w))
            ax = plt.subplot(nrows, ncols, nimgs)
            im = ax.imshow(w, vmin=-vmax, vmax=vmax, cmap='PiYG')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)

        plt.savefig("{:03d}_{}_{:03d}.png".format(si, k, pi), dpi=150)
        plt.close(fig)

    rates[k] = rl

plt.figure()
plt.suptitle("confusion matrix ({})\n no spikes {} acc {:5.2f}%".format(
                n_digits, no_spikes, (100. * correct)/n_digits))
plt.imshow(conf_matrix)
plt.savefig("confusion_matrix.pdf")
# plt.show()


colors = ['red', 'green', 'blue', 'cyan', 'orange',
          'magenta', 'black', 'yellow', ]
for i, k in enumerate(rates):
    if k == 'input':
        continue
    # max_r = np.max(rates[k])
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    for j, r in enumerate(rates[k]):

        lbl = "{} {}".format(k, j)
        plt.plot(r[:-1], label=lbl, linewidth=1)

    plt.plot(rates['input'][0][:-1], linestyle=':', color=colors[0],
             label='Input', linewidth=4)


# plt.legend()
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig('average_rates_simple_cnn_mnist_layer_{}.pdf'.format(k))

ord = order
mean_rates = [np.mean(rates[rk]) for rk in ord if rk in rates]
std_rates = [np.std(rates[rk]) for rk in ord if rk in rates]
labels = [k for k in ord if k in rates]
xticks = np.arange(len(mean_rates))
fig, ax = plt.subplots(1, 1)
ax.set_title("average rates per layer")
im = plt.errorbar(xticks, mean_rates, yerr=std_rates,
                  linestyle='dotted', marker='o')
ax.set_ylabel("rate (hz)")
ax.set_xticks(xticks)
ax.set_xticklabels(labels)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# ax.set_xticklabels(labels)
plt.savefig("average_spikes_per_layer.png", dpi=300)
# plt.show()


for si, k in enumerate(order):
    if k not in spikes:
        continue

    for pi, p in enumerate(spikes[k]):
        fig = plt.figure()
        plt.suptitle("{}_{}".format(k, pi))
        ax = plt.subplot(1, 1, 1)
        if k == 'dense_2':
            for i in range(10):
                plt.axhline(i, linestyle='--', color='gray', linewidth=0.1)

        for i in np.arange(0, sim_time, digit_duration):
            plt.axvline(i, linestyle='--', color='gray', linewidth=0.5)

        for ni, ts in enumerate(p):
            ax.plot(ts, ni * np.ones_like(ts), '.b', markersize=1)
        plt.savefig("spikes_{:03d}_{}_{:03d}.png".format(si, k, pi), dpi=300)
        plt.close(fig)

# plt.show()


