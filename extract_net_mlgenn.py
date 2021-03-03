
param_trans = {
    'neurons': {
        'threshold': ('global_params', 'Vthr'),
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
    ft = translations['neurons']
    d = {}
    ns = lyr.neurons
    for k in ft:
        if isinstance(ft[k], str):
            d[k] = getattr(ns, ft[k])
        else:
            tmp = getattr(ns, ft[k][0])
            d[k] = tmp[ft[k][1]]
    return d


def get_connectivity(lyr, translations=param_trans):
    tp = type(lyr).lower()
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
        else:
            tmp = getattr(ws, ft[k][0])
            d[k] = tmp[ft[k][1]]

    if 'avepool2d' in tp:
        ft = translations['avepool2d']
        for k in ft:
            if isinstance(ft[k], str):
                d[k] = getattr(ws, ft[k])
            else:
                tmp = getattr(ws, ft[k][0])
                d[k] = tmp[ft[k][1]]

    return d

