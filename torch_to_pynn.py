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
        'Conv2dLICell': 'IF_curr_delta_conv'
    }

    CELL_TRANSLATIONS = {
        'Conv2dLICell': {
            'tau_mem_inv': ('tau_m', lambda x: np.asscalar(1./x.detach().numpy())),
            # 'tau_syn_inv': ('tau_syn', lambda x: 1./x),
            'v_leak': ('v_rest', lambda x: np.asscalar(x.detach().numpy())),
            'v_reset': ('v_reset', lambda x: np.asscalar(x.detach().numpy())),
            'v_th': ('v_thresh', lambda x: np.asscalar(x.detach().numpy()))
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

    def __init__(self, model, dummy_data, pynn, timestep=1, min_delay=1, max_delay=144):
        self.timestep = timestep
        self.min_delay = min_delay
        self.max_delay = max_delay
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

                    if 'conv2d' in nn.lower() or 'dense' in nn.lower():
                        pj['name'] = nn

                prjs[i] = pj
                conn_layers_dicts[:] = []
                pre = i

                count += 1

        return OrderedDict(prjs)

    def generate_pynn_dictionaries(self):
        pops = self.generate_pynn_populations_dicts()
        prjs = self.generate_pynn_projections_dicts(pops)
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
        trans = {}
        for k, v in translations.items():
            if len(v) == 2:
                trans[v[0]] = v[1](in_dict[k])
            else:
                trans[v] = in_dict[k]

        return trans

    def parse_cell_parameters(self, name, layer_dict):
        tr = self.CELL_TRANSLATIONS[name]
        return self.parse_translations(layer_dict, tr)

    def parse_conn_params(self, name, layer_dict):
        tr = self.CONNECT_TRANSLATIONS[name]
        return self.parse_translations(layer_dict, tr)

    def parse_input_dicts(self, input_dicts):
        return input_dicts

    def generate_pynn_objects(self, input_dicts, pops_dicts, projs_dicts):
        sim = self.pynn
        sim.setup(timestep=self.timestep, min_delay=self.min_delay,
                  max_delay=self.max_delay)
        inputs = self.parse_input_dicts(input_dicts)
        pops = self.generate_pynn_populations(pops_dicts)
        prjs = self.generate_pynn_projections(inputs, pops, projs_dicts)

        return pops, prjs

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

    def generate_pynn_projections(self, inputs, pynn_pops, projs_dicts):
        sim = self.pynn
        prjs = {}
        for i, pr in projs_dicts.items():
            p = deepcopy(pr)
            conn_type = p.pop('name').lower()
            w_key = 'weights_kernel' if 'conv2d' in conn_type else 'weights'
            pre = p.pop('pre')
            post = p.pop('post')
            ws = p.pop(w_key)
            p['shape_pre'] = p['shape_pre'][2:]
            n_pre = ws.shape[1]
            n_post = ws.shape[0]
            pre_pops = inputs if pre == 0 else pynn_pops[pre]

            if pre == 0:
                print('TODO: pre {} indicates the input, not implemented yet'.format(pre))
                # continue

            post_pops = pynn_pops[post]
            pre_d = {}
            for pre_i, pre_pop in enumerate(pre_pops):
                post_d = {}
                for post_i, post_pop in enumerate(post_pops):
                    conn_d = deepcopy(p)
                    conn_d[w_key] = ws[post_i, pre_i].copy()
                    if 'conv2d' in conn_type:
                        conn = sim.ConvolutionOrigConnector(**conn_d)
                    else:
                        print('TODO: dense connector parsing not fully implemented')
                        conn = sim.DensePoolConnector(**conn_d)
                    label = '{}_from_{}[{}]_to_{}[{}]'.format(
                                            conn_type, pre, pre_i, post, post_i)
                    post_d[post_i] = sim.Projection(
                        pre_pop, post_pop, conn, label=label)
                pre_d[pre_i] = post_d

            prjs[i] = pre_d

        return prjs
