#import tensorflow as tf
from tensorflow.keras import models#, layers, datasets
from ml_genn import Model
# from ml_genn.layers import InputType
# from ml_genn.norm import DataNorm, SpikeNorm
# from ml_genn.utils import parse_arguments, raster_plot
import numpy as np

from bifrost.extract.mlgenn.extractor import extract_all
from bifrost.extract.mlgenn.to_ir import (to_neuron_layer, to_connection)
from bifrost.export.ml_genn import MLGeNNContext
from bifrost.export.population import export_layer_neuron
from bifrost.export.connection import export_connection

def to_dict(np_file):
    d = {}
    for k in np_file.keys():
        try:
            d[k] = np_file[k].item()
        except:
            d[k] = np_file[k]
    return d

param_trans = {
    'neurons': {
        'threshold': ('extra_global_params', 'Vthr'),
        'shape': 'shape',
    },
    'input': {
        'shape': 'shape',
    },
    'conv2d': {
        'weights': 'weights',
        'shape': 'conv_size',
        'padding': 'conv_padding',
        'strides': 'conv_strides',
        'n_filters': 'filters',
    },
    'avepool2d': {
        'shape': 'pool_size',
        'padding': 'pool_padding',
        'strides': 'pool_strides',
    },
    'dense': {
        'weights': 'weights',
        'size': 'units',
    }
}


def get_neuron_params(lyr, translations=param_trans):
    print(lyr.__class__.__name__)
    k = 'neurons' if lyr.__class__.__name__ != 'InputLayer' else 'input'
    ft = translations[k]
    d = {}
    for k in ft:
        o = lyr if k == 'shape' else lyr.neurons.nrn
        if isinstance(ft[k], str):
            d[k] = getattr(o, ft[k])
        else:
            tmp = getattr(o, ft[k][0])
            d[k] = tmp[ft[k][1]].view[0]
    return d


def get_connectivity(lyr, translations=param_trans):
    if lyr.__class__.__name__ == 'InputLayer':
        return {}

    tp = str(type(lyr.downstream_synapses[0])).lower()
    if 'conv2d' in tp:
        k = 'conv2d'
    else:  # 'dense' in tp:
        k = 'dense'

    ft = translations[k]
    d = {}
    ws = lyr.upstream_synapses[0]
    for k in ft:
        if isinstance(ft[k], str):
            d[k] = getattr(ws, ft[k])
            d[k] = d[k].value if 'padding' in k else d[k]
        else:
            tmp = getattr(ws, ft[k][0])
            d[k] = tmp[ft[k][1]]

    if 'pool' in tp:
        ft = translations['avepool2d']
        pd = {}
        for k in ft:
            if isinstance(ft[k], str):
                pd[k] = getattr(ws, ft[k])
                pd[k] = pd[k].value if 'padding' in k else pd[k]
            else:
                tmp = getattr(ws, ft[k][0])
                pd[k] = tmp[ft[k][1]]
        d['pool'] = pd
    return d



tf_model = models.load_model('simple_cnn_tf_model.tf')
mlg_model = Model.convert_tf_model(tf_model, input_type='poisson', 
                connectivity_type='procedural')
mlg_model.compile(dt=1.0, batch_size=1, rng_seed=0)

net_params = extract_all(mlg_model)
np.savez_compressed('simple_cnn_params.npz', **net_params)

del net_params

cnn_params = np.load('simple_cnn_params.npz', allow_pickle=True)
dcnn_params = to_dict(cnn_params)


lyr_map = {}
layers = []
for i, k in enumerate(dcnn_params.keys()):
    lyr = dcnn_params[k]
    # print(lyr.keys())
    # print(lyr['params'].keys())
    if i > 0:
        lyr = to_neuron_layer(i, dcnn_params)
        layers.append(lyr)
        lyr_map[str(lyr)] = k

ctx = MLGeNNContext(lyr_map)
layer_statements = [export_layer_neuron(l, ctx, pop_join_str=",\n    ") for l in layers]
imp_set = set()
for s in layer_statements:
    imp_set |= set(s.imports)

prm_set = set()
for s in layer_statements:
    prm_set |= set(s.preambles)

for s in imp_set:
    print(s)

for s in prm_set:
    print(s)

for s in layer_statements:
    print(s.value)

conns = []
for prei, pre in enumerate(layers[:-1]):
    post = layers[prei+1]
    conns.append(to_connection(pre, post, dcnn_params))

conn_statements = [export_connection(c, ctx) for c in conns]

for cs in conn_statements:
    print(cs.value)

print(mlg_model)
params = {}
order = []
for layer in mlg_model.layers:
    name = layer.name
    params[name] = get_neuron_params(layer)
    # conns[name] = get_connectivity(layer)
    # order.append(name)

for n in order:
    print(n)
    for p in conns[n]:
        if p == 'weights':
            continue
        print(p, conns[n][p])
    # print(conns[n])
    print(params[n])

np.savez_compressed("simple_cnn_network_elements.npz",
    params=params, conns=conns, order=order)
