import torch
import norse.torch
from torch import nn
from norse.torch.module.lif import LIFCell
from norse.torch.module.leaky_integrator import LICell


class Parser:
    CONV2D_KEYS = ['in_channels', 'kernel_size', 'out_channels', 'padding',
                   'stride', 'weight']
    AVGPOOL2D_KEYS = ['kernel_size', 'stride']
    LICELL_KEYS = ['alpha', 'tau_mem_inv', 'tau_syn_inv', 'v_leak', 'v_reset', 'v_th']

    def __init__(self, model):
        self.model = model
        self.input_shape = (model.height, model.width)
        self.pops = {}
        self.conns = {}

        self.extract_layers()

    def extract_layers(self):
        if not hasattr(self.model, "layers"):
            raise Exception("Please define layers as an ordered list of "
                            "blocks/layers of your PyTorch model")
        self.pops = {i: self.get_pop_type_and_params(l)
                     for i, l in enumerate(self.model.layers)
                     if not isinstance(l, nn.BatchNorm2d)}

        self.conns = {i: self.get_incoming_connectivity(l)
                      for i, l in enumerate(self.model.layers)
                      if not isinstance(l, nn.BatchNorm2d)}

    def get_pop_type_and_params(self, layer):
        d = {'name': layer.__class__.__name__.lower()}
        if isinstance(layer, nn.Conv2d):
            d.update({k: getattr(layer, k) for k in self.CONV2D_KEYS})
        elif isinstance(layer, nn.AvgPool2d):
            d.update({k: getattr(layer, k) for k in self.AVGPOOL2D_KEYS})
        elif isinstance(layer, LICell):
            d.update({k: getattr(layer.p, k) for k in self.LICELL_KEYS})

        return d

    def get_incoming_connectivity(self, layer):
        print(layer)

    def generate_pynn_populations(self, pops: dict):
        pass

    def generate_pynn_projections(self, conns: dict, pynn_pops: dict):
        pass