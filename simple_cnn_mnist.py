import spynnaker8 as sim
import numpy as np
import mnist
import matplotlib.pyplot as plt
import plotting
import sys

filename = "simple_cnn_network_elements.npz"

data = np.load(filename, allow_pickle=True)

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
n_digits = 50
digit_duration = 500.0  # ms
digit_rate = 50.0  # hz
in_rates = np.zeros((n_in, n_digits))
for i in range(n_digits):
    in_rates[:, i] = test_X[i].flatten()

in_rates *= (digit_rate / in_rates.max())
in_durations = np.ones((n_in, n_digits)) * digit_duration
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


# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
def_params = {
    'v_thresh': 1.,
    'v_rest': 0.,
    'v_reset': 0.,
    'v': 0.,
    'tau_m': 200.#0.#np.round(digit_duration // 2.),
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
        ps['v_thresh'] = par['threshold']
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
        ps['v_thresh'] = par['threshold']

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
                w = weights[:, :, prei, posti].reshape(wshape)
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
                    col0 = posti * n_out
                    col1 = col0 + n_out

                    pre_rows = np.repeat(np.arange(sh_pre[0]), sh_pre[1])
                    # print("pre_rows = {}".format(pre_rows))
                    pre_cols = np.tile(np.arange(sh_pre[1]), sh_pre[0])
                    # print("pre_cols = {}".format(pre_cols))
                    mtx_rows = (pre_rows * sh_pre[1] * n_chan +
                                pre_cols * n_chan + prei)
                    print("mtx_rows = {}".format(mtx_rows))
                    n_rows = len(mtx_rows)
                    mtx_rows = np.repeat(mtx_rows, n_out)
                    # print("mtx_rows = {}".format(mtx_rows))
                    mtx_cols = np.arange(col0, col1)
                    print("mtx_cols = {}".format(mtx_cols))
                    mtx_cols = np.tile(mtx_cols, n_rows)
                    # print("mtx_cols = {}".format(mtx_cols))
                    ws = weights[mtx_rows, mtx_cols].reshape((n_rows, n_out))
                    print(ws.shape)
                    # print(ws)
                    # row0 = prei * size_pre
                    # row1 = row0 + size_pre
                    # ws = weights[row0:row1, col0:col1]

                else:
                    row0 = prei * size_pre
                    row1 = row0 + size_pre
                    ws = weights[row0:row1, :]
                wl.append(ws)
                cn = sim.PoolDenseConnector(pre_shape, ws, n_out, pool_area,
                                            pool_stride)

                prj = sim.Projection(pre, post, cn, label=lbl)
                projs[lbl] = prj
        kernels[o] = wl

sim_time = digit_duration * n_digits * 1.1

neos = {}
spikes = {}
sim.run(sim_time)

for k in rec:
    neos[k] = [p.get_data() for p in pops[k]]
    spikes[k] = [x.segments[0].spiketrains for x in neos[k]]


sim.end()

# sys.exit()

for si, k in enumerate(order):
    if k not in spikes:
        continue

    if k == 'dense':
        imgs, bins = plotting.spikes_to_images_list(
                                  spikes[k], shapes[k], sim_time,
                                  digit_duration, offsets[k],
                                  merge_images=True)
        nrows = 3
        nimgs = len(imgs)
        ncols = nimgs // nrows + int(nimgs % nrows > 0)

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

    for pi, p in enumerate(spikes[k]):
        imgs, bins = plotting.spikes_to_images(p, shapes[k], sim_time,
                                               digit_duration)
        nimgs = len(imgs)
        nrows = 3
        # if 'conv2d' in k:
        #     nimgs += 1
        nimgs += 1

        ncols = nimgs // nrows + int(nimgs % nrows > 0)

        fig = plt.figure(figsize=(ncols, nrows))
        plt.suptitle("{}_{}".format(k, pi))
        for i in range(nimgs):
            if i == len(imgs):
                break

            ax = plt.subplot(nrows, ncols, i + 1)
            if k == 'dense_2':
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
# plt.show()




