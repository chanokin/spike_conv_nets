from copy import deepcopy

import torch
import norse.torch
from torch import nn
from norse.torch.module.lif import LIFCell
from norse.torch.module.leaky_integrator import LICell
from collections import OrderedDict
from pprint import pprint
import numpy as np


class Parser:
    CONV2D_KEYS = ('in_channels', 'kernel_size', 'out_channels',
                   'padding', 'stride', 'weight')
    AVGPOOL2D_KEYS = ('kernel_size', 'stride')
    LICELL_KEYS = ('alpha', 'tau_mem_inv', 'tau_syn_inv',
                   'v_leak', 'v_reset', 'v_th')
    CELL_TYPES = {
        'Conv2dLICell': 'IF_curr_exp_conv'
    }

    CELL_TRANSLATIONS = {
        'Conv2dLICell': {
            'tau_mem_inv': ('tau_m', lambda x: 1./x),
            # 'tau_syn_inv': ('tau_syn', lambda x: 1./x),
            'v_leak': ('v_rest', ),
            'v_reset': ('v_reset'),
            'v_th': ('v_thresh')
        }
    }
    #shape_pre, weights_kernel, strides, padding,
                 # pooling=None, pool_stride=None,
    CONNECT_TRANSLATIONS = {
        'AvgPool2d': {
            'kernel_size': ('pooling'),
            'stride': ('pool_stride')
        },
        'Conv2d': {
            'in_shape': ('shape_pre'),
            # this currently has the order out_channels, in_channels, rows, cols
            # we need to test if it needs flipping as in tensorflow
            'weight': ('weights_kernel', lambda x: x.detach().numpy()),
            'stride': ('strides'),
            'padding': ('padding'),
        }
    }

    def __init__(self, model, dummy_data, pynn):
        self.pynn = pynn
        self.input_data = dummy_data
        self.x = None
        self.s = None
        self.model = model
        self.input_shape = (model.height, model.width)
        self.layer_dicts = {}
        self.layers_count = 0
        self.extract_layers()

    def extract_layers(self):
        if not hasattr(self.model, "layers"):
            raise Exception("Please define layers as an ordered list of "
                            "blocks/layers of your PyTorch model")

        self.x = self.input_data[:, 0]
        self.layer_dicts = OrderedDict(
                            {i: self.get_pop_type_and_params(l, self.x, self.s)
                                for i, l in enumerate(self.model.layers)
                                if not isinstance(l, nn.BatchNorm2d)})

        # pprint(self.layer_dicts)

    def get_pop_type_and_params(self, layer, x, s):
        d = {
            'name': layer.__class__.__name__,
            'in_shape': x.shape
        }

        try:
            x = layer(x, s)
        except:
            x = layer(x)

        if len(x) == 1:
            self.x = x
        else:
            self.x = x[0]
            self.s = x[1]

        if isinstance(layer, nn.Conv2d):
            d.update({k: getattr(layer, k) for k in self.CONV2D_KEYS})
        elif isinstance(layer, nn.AvgPool2d):
            d.update({k: getattr(layer, k) for k in self.AVGPOOL2D_KEYS})
        elif isinstance(layer, LICell):
            d.update({k: getattr(layer.p, k) for k in self.LICELL_KEYS})

        d['out_shape'] = self.x.shape
        self.layers_count += 1
        return d

    def generate_pynn_populations_dicts(self):
        sim = self.pynn
        pops = {}
        count = 0
        prev_i = 0
        for i, d in self.layer_dicts.items():
            if i == 0:
                continue

            n0 = self.layer_dicts[prev_i]['name']
            n = d['name']
            prev_i = i
            comp_name = "{}{}".format(n0, n)
            if comp_name not in self.CELL_TYPES:
                continue
            ct = getattr(sim, self.CELL_TYPES[comp_name])
            osh = d['out_shape']
            if len(osh) > 2:
                n_channels = osh[1]
                size = int(np.prod(osh[-2:]))
            else:
                n_channels = 1
                size = int(np.prod(osh))

            params = self.parse_cell_parameters(comp_name, d)
            label = "{}_{}__size_{}".format(n, count, size)
            # size, cellclass, cellparams =

            pops[i] = {
                'size': size,
                'cellclass': ct,
                'cellparams': params,
                'label': label,
                'n_channels': n_channels
            }
            count += 1

        return OrderedDict(pops)

    def generate_pynn_projections_dicts(self, pops):
        post = None
        pre = 0
        sim = self.pynn
        prjs = {}
        count = 0
        last_i = 0
        conn_layers_dicts = []
        for i, d in self.layer_dicts.items():
            if i == 0:
                conn_layers_dicts.append(d)
                continue

            n = d['name']
            n0 = self.layer_dicts[last_i]['name']
            comp_cell = "{}{}".format(n0, n)
            last_i = i

            if comp_cell not in self.CELL_TYPES:
                conn_layers_dicts.append(d)

            if comp_cell in self.CELL_TYPES:
                post = i
                if pre == 0:
                    '''this means the first pre, a.k.a. the input'''
                else:
                    '''everything else :)'''
                pj = {
                    'pre': pre, 'post': post
                }

                for j, c in enumerate(conn_layers_dicts):
                    nn = c['name']
                    if nn in self.CONNECT_TRANSLATIONS:
                        pj.update(self.parse_conn_params(nn, c))

                prjs[i] = pj
                conn_layers_dicts[:] = []
                pre = i

                count += 1

        return OrderedDict(prjs)

    def generate_pynn_dictionaries(self):
        pops = self.generate_pynn_populations_dicts()
        prjs = self.generate_pynn_projections_dicts(pops)
        return pops, prjs

    def generate_pynn_objects(self, pops, prjs):
        sim = self.pynn
        sim.setup()

        return pops, prjs

    def pynn_dict_to_object(self, obj_type, dictionary):
        if obj_type == 'population':
            return self.pynn.Population(**dictionary)
        else:
            conn_d = dictionary['connector']
            proj_d = dictionary['projection']
            conn = self.pynn.Connector(**conn_d)
            proj_d['connector'] = conn
            return self.pynn.Projection(**proj_d)

    def parse_translations(self, in_dict, translations):
        trans = {v[0]: v[1](in_dict[k]) if len(v) == 2 else in_dict[k]
                 for k, v in translations.items()}
        return trans

    def parse_cell_parameters(self, name, layer_dict):
        tr = self.CELL_TRANSLATIONS[name]
        return self.parse_translations(layer_dict, tr)

    def parse_conn_params(self, name, layer_dict):
        tr = self.CONNECT_TRANSLATIONS[name]
        return self.parse_translations(layer_dict, tr)

    def generate_pynn_populations(self, pops_dicts):
        pops = {}
        for i, di in pops_dicts.items():
            d = deepcopy(di)
            n_channels = d.pop('n_channels')
            lbl = di['label']
            ps = []
            for j in range(n_channels):
                d['label'] = '{}_ch{}'.format(lbl, j)
                ps.append(self.pynn_dict_to_object('population', d))

            pops[i] = ps

        return OrderedDict(pops)

    def generate_pynn_projections(self, pynn_pops, projs_dicts):
        pass
