from copy import deepcopy

import torch
import norse.torch
from torch import nn
from torchinfo import summary
from norse.torch.module.sequential import SequentialState
from norse.torch.module.lif import LIFCell
from norse.torch.module.leaky_integrator import LICell
from norse.torch.functional.leaky_integrator import LIParameters
from norse.torch.functional.lif import LIFParameters
from collections import OrderedDict
from pprint import pprint
import numpy as np
def detach(x):
    return np.asscalar(x.detach().numpy())
class Parser:
    CONV2D_KEYS = ('in_channels', 'kernel_size', 'out_channels',
                   'padding', 'stride', 'weight')
    AVGPOOL2D_KEYS = ('kernel_size', 'stride')
    LICELL_KEYS = ('tau_mem_inv', 'tau_syn_inv',
                   'v_leak')
    LIFCELL_KEYS = ('tau_mem_inv', 'tau_syn_inv',
                    'v_leak', 'v_reset', 'v_th')
    CELL_TYPES = {
        'Conv2dLICell': 'IF_curr_exp',
        'Conv2dLIFCell': 'IF_curr_exp'
    }

    CELL_TRANSLATIONS = {
        'Conv2dLICell': {
            'tau_mem_inv': ('tau_m', lambda x: detach(1./x)),
            'tau_syn_inv': ('tau_syn', lambda x: detach(1./x)),
            'v_leak': ('v_rest', lambda x: detach(x)),
            'v_reset': ('v_reset', lambda x: detach(x)),
            'v_th': ('v_thresh', lambda x: detach(x))
        },
        'Conv2dLIFCell': {
            'tau_mem_inv': ('tau_m', lambda x: detach(1./x)),
            'tau_syn_inv': ('tau_syn', lambda x: detach(1./x)),
            'v_leak': ('v_rest', lambda x: detach(x)),
            'v_reset': ('v_reset', lambda x: detach(x)),
            'v_th': ('v_thresh', lambda x: detach(x))
        }
        # todo: currently only conv layers, deconv (ConvTranspose) may need
        #  other setup as they are followed by a conv one!
    }
    #shape_pre, weights_kernel, strides, padding,
                 # pooling=None, pool_stride=None,
    CONNECT_TRANSLATIONS = {
        'AvgPool2d': {
            'kernel_size': ('pool_shape'),
            'stride': ('pool_stride')
        },
        'Conv2d': {
            'in_shape': ('shape_pre'),
            # this currently has the order out_channels, in_channels, rows, cols
            # we need to test if it needs flipping as in tensorflow
            'weight': ('kernel_weights', lambda x: x.detach().numpy()),
            'stride': ('strides'),
            'padding': ('padding'),
        }
    }

    def __init__(self, model, dummy_data, timestep=1, min_delay=1, max_delay=144):
        self.timestep = timestep
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.input_data = dummy_data
        self.x = None
        self.s = None
        self.model = model
        self.input_shape = (model.height, model.width)
        self.layer_dicts = {}
        self.layers_count = 0
        self.extract_layers()

    def extract_layers(self):
        m = self.model
        s = summary(m, self.input_data.shape, verbose=0)

        ld = OrderedDict({i: self.get_pop_type_and_params(l)
                             for i, l in enumerate(s.summary_list)
                             if not (isinstance(l.module, nn.BatchNorm2d) or
                                     isinstance(l.module, SequentialState) or
                                     i == 0)})

        self.layer_dicts = ld
        # pprint(self.layer_dicts)

    def get_pop_type_and_params(self, layer):
        d = {
            'name': layer.class_name,
            'in_shape': layer.input_size,
            'out_shape': layer.output_size,
        }

        module = layer.module

        if isinstance(module, nn.Conv2d):
            d.update({k: getattr(module, k) for k in self.CONV2D_KEYS})
        elif isinstance(module, nn.ConvTranspose2d):
            d.update({k: getattr(module, k) for k in self.CONV2D_KEYS})
        elif isinstance(module, nn.AvgPool2d):
            d.update({k: getattr(module, k) for k in self.AVGPOOL2D_KEYS})
        # for 'neuron' module types we base our selection on parameter types
        # apparently that's a thing!
        elif isinstance(module.p, LIParameters):
            d.update({k: getattr(module.p, k) for k in self.LICELL_KEYS})
        elif isinstance(module.p, LIFParameters):
            d.update({k: getattr(module.p, k) for k in self.LIFCELL_KEYS})

        self.layers_count += 1
        return d

    def generate_pynn_populations_dicts(self):
        pops = {}
        count = 0
        prev_i = 0
        keys = list(self.layer_dicts.keys())
        for i, [k, d] in enumerate(self.layer_dicts.items()):
            if i == 0:
                continue

            k0 = keys[prev_i]
            n0 = self.layer_dicts[k0]['name']
            n = d['name']
            prev_i = i
            comp_name = "{}{}".format(n0, n)
            if comp_name not in self.CELL_TYPES:
                continue

            osh = d['out_shape']
            if len(osh) > 2:
                n_channels = osh[1]
                size = int(np.prod(osh[-2:]))
                shape = osh[-2:]
            else:
                n_channels = 1
                size = int(np.prod(osh))
                shape = [size, 1]

            params = self.parse_cell_parameters(comp_name, d)
            label = "{}_{}__size_{}".format(n, count, size)
            # size, cellclass, cellparams =
            cell_class = self.CELL_TYPES[comp_name]
            pops[i] = {
                'size': size,
                'cellclass': cell_class,
                'cellparams': params,
                'label': label,
                'n_channels': n_channels,
                'shape': shape
            }
            count += 1

        return OrderedDict(pops)

    def generate_pynn_projections_dicts(self, pops):
        post = None
        pre = 0
        prjs = {}
        count = 0
        last_i = 0
        conn_layers_dicts = []
        keys = list(self.layer_dicts.keys())
        for i, [k, d] in enumerate(self.layer_dicts.items()):
            if i == 0:
                conn_layers_dicts.append(d)
                continue

            n = d['name']
            k0 = keys[last_i]
            n0 = self.layer_dicts[k0]['name']
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

    def __pynn_grid2d_ratio(self, shape):
        # width / height
        return shape[1] / shape[0]

    def pynn_dict_to_object(self, sim, obj_type, dictionary):
        if obj_type == 'population':

            d = deepcopy(dictionary)
            ct = d['cellclass']
            cell = getattr(sim, ct)
            d['cellclass'] = cell
            shape = d.pop('shape')
            d['structure'] = sim.Grid2D(self.__pynn_grid2d_ratio(shape))
            return sim.Population(**d)
        else:
            conn_d = dictionary['connector']
            proj_d = dictionary['projection']
            # conn = sim.Connector(**conn_d)
            conn = dictionary['c']
            proj_d['connector'] = conn
            return sim.Projection(**proj_d)

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

    def generate_pynn_objects(self, sim, input_dicts, pops_dicts, projs_dicts):
        sim.setup(timestep=self.timestep, min_delay=self.min_delay,
                  max_delay=self.max_delay)
        inputs = self.parse_input_dicts(input_dicts)
        pops = self.generate_pynn_populations(sim, pops_dicts)
        prjs = self.generate_pynn_projections(sim, inputs, pops, projs_dicts)

        return pops, prjs

    def generate_pynn_populations(self, sim, pops_dicts):
        pops = {}
        for i, di in pops_dicts.items():
            d = deepcopy(di)
            n_channels = d.pop('n_channels')
            lbl = di['label']
            ps = []
            for j in range(n_channels):
                d['label'] = '{}_ch{}'.format(lbl, j)
                ps.append(self.pynn_dict_to_object(sim, 'population', d))

            pops[i] = ps

        return OrderedDict(pops)

    def generate_pynn_projections(self, sim, inputs, pynn_pops, projs_dicts):
        prjs = {}
        for i, pr in projs_dicts.items():
            p = deepcopy(pr)
            conn_type = p.pop('name').lower()
            w_key = 'kernel_weights' if 'conv2d' in conn_type else 'weights'
            pre = p.pop('pre')
            post = p.pop('post')
            ws = p.pop(w_key)
            shape_pre = p.pop('shape_pre')
            n_pre = ws.shape[1]
            n_post = ws.shape[0]
            pre_pops = inputs if pre == 0 else pynn_pops[pre]

            if pre == 0:
                print("TODO: pre {} indicates the input, parsing projection "
                      "not yet implemented".format(pre))
                # continue

            post_pops = pynn_pops[post]
            pre_d = {}
            for pre_i, pre_pop in enumerate(pre_pops):
                post_d = {}
                for post_i, post_pop in enumerate(post_pops):
                    conn_d = deepcopy(p)
                    conn_d[w_key] = ws[post_i, pre_i].copy()
                    if 'conv2d' in conn_type:
                        conn = sim.ConvolutionConnector(**conn_d)
                        synapse = sim.Convolution()
                    else:
                        print('TODO: dense connector parsing not fully implemented')
                        conn = sim.PoolDenseConnector(**conn_d)
                        synapse = sim.PoolDense()
                    label = '{}_from_{}[{}]_to_{}[{}]'.format(
                                            conn_type, pre, pre_i, post, post_i)
                    post_d[post_i] = sim.Projection(
                        pre_pop, post_pop, conn, synapse, label=label)
                pre_d[pre_i] = post_d

            prjs[i] = pre_d

        return prjs
