import spynnaker8 as sim
import numpy as np
import mnist
import matplotlib.pyplot as plt
import plotting

filename = "simple_cnn_network_elements.npz"

data = np.load(filename, allow_pickle=True)

order0 = data['order']
order = order0[:4]
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

# sim.extra_models.SpikeSourcePoissonVariable.set_model_max_atoms_per_core(300)
sim.IF_curr_exp_conv.set_model_max_atoms_per_core(n_atoms=1024)
# sim.IF_curr_exp_pool_dense.set_model_max_atoms_per_core(n_atoms=1024)

sim.setup(timestep=1.)

np.random.seed(13)

shape_in = np.asarray([28, 28])
n_in = int(np.prod(shape_in))
n_digits = 9
digit_duration = 500.0  # ms
digit_rate = 10.0  # hz
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

pops['input'][0].record('spikes')

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
def_params = {
    'v_thresh': 1.,
    'v_rest': 0.,
    'v_reset': 0.,
    'v': 0.,
    'tau_m': np.round(digit_duration // 3.),
}

for i, o in enumerate(order):
    if i == 0:
        continue

    par = ml_param[o]
    if 'conv2d' in o.lower():
        shape = par['shape'][0:2]
        chans = par['shape'][2]
        ps = def_params.copy()
        ps['v_thresh'] = par['threshold']
        n = int(np.prod(shape))
        print(o, n, shape, chans)
        pop = [sim.Population(n, sim.IF_curr_exp_conv, ps,
                              label="{}_chan_{}".format(o, chan))
               for chan in range(chans)]
    elif 'dense' in o.lower():
        shape = par['shape'][0:2]

        pop = [sim.Population(n, sim.IF_curr_exp_pool_dense, ps,
                              label="{}_chan".format(o))]


    pops[o] = pop

projs = {}
for i, o in enumerate(order):
    if i == 0:
        continue
    o0 = order[i-1]
    for prei, pre in enumerate(pops[o0]):
        pre_shape = np.asarray(ml_param[o0]['shape'][:2])
        c = ml_conns[o]
        for posti, post in enumerate(pops[o]):
            lbl = "{}_{} to {}_{}".format(o0, prei, o, posti)
            print(pre_shape, o0, prei, o, posti)
            if 'conv2d' in o.lower():
                wshape = c['shape']
                print(prei, posti, wshape, c['weights'].shape)
                strides = c['strides']
                w = c['weights'][:, :, prei, posti].reshape(wshape)
                pool_area = c['pool']['shape'] if 'pool' in c else None
                pool_stride = c['pool']['strides'] if 'pool' in c else None
                cn = sim.ConvolutionConnector(pre_shape, w, strides=strides,
                                          pooling=pool_area, pool_stride=pool_stride)
                prj = sim.Projection(pre, post, cn, label=lbl)
                projs[lbl] = prj
            elif 'dense' in o.lower():
                if len(pre_shape) == 1:
                    pre_shape = (pre_shape[0], 1)
                n_out = post.size
                pooling = 'pool' in c
                pool_area = np.asarray(c['pool']['shape']) if pooling else None
                pool_stride = np.asarray(c['pool']['strides']) if pooling else None
                sh_pre = sim.PoolDenseConnector.calc_post_pool_shape(
                            pre_shape, pooling, pool_area, pool_stride)
                size_pre = int(np.prod(sh_pre))
                sw = size_pre * prei
                ew = sw + size_pre
                ws = c['weights'][sw:ew, :]
                cn = sim.PoolDenseConnector(pre_shape, ws, n_out, pool_area,
                                            pool_stride)

                prj = sim.Projection(pre, post, cn, label=lbl)
                projs[lbl] = prj


sim_time = digit_duration * n_digits * 1.1

sim.run(sim_time)

neo = pops['input'][0].get_data()
spikes = neo.segments[0].spiketrains

sim.end()

imgs, bins = plotting.spikes_to_images(spikes, shape_in, sim_time,
                                       digit_duration//2)
nrows = 3
nimgs = len(imgs)
ncols = nimgs // nrows + int(nimgs % nrows > 0)

fig = plt.figure(figsize=(ncols, nrows))
for i in range(nimgs):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.imshow(imgs[i])
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


