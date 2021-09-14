import numpy as np
import torch
import norse.torch
# from models import snn
from bifrost.ir import OutputLayer, EthernetOutput, SpiNNakerSPIFInput

from torch_to_spynn import Parser
import torch.nn as nn
from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module.leaky_integrator import LICell
import pytorch_lightning as pl
import copy
# from pynn_object_serialisation.functions import (
#     intercept_simulator, restore_simulator_from_file
# )

def set_parameter_buffers(model):
    for k0 in model._modules:
        if isinstance(model._modules[k0], SequentialState):
            for k1 in model._modules[k0]._modules:
                set_parameter_buffers_per_layer(model._modules[k0]._modules[k1])
        elif not isinstance(model._modules[k0], torch.nn.CrossEntropyLoss):
            set_parameter_buffers_per_layer(model._modules[k0])


def set_parameter_buffers_per_layer(module):
    _li = ['tau_mem_inv', 'tau_syn_inv', 'v_leak']
    _lif = _li + ['v_reset', 'v_th', 'alpha']
    params = {
        'Conv2d': [
            'kernel_size', 'in_channels', 'out_channels',
            'output_padding', 'padding', 'stride'
        ],
        'LICell': {
            'LIFParameters': _lif,
            'LIParameters': _li,
        },
        'LIFCell': {
            'LIFParameters': _lif,
            'LIParameters': _li
        },
        'AvgPool2d': [
            'kernel_size', 'padding', 'stride'
        ],
    }
    mod_name = module.__class__.__name__
    if mod_name not in params:
        return

    if mod_name in ('LICell', 'LIFCell'):
        param_name = module.p.__class__.__name__
        param_list = params[mod_name][param_name]
    else:
        param_list = params[mod_name]

    for p in param_list:
        frm = module.p if mod_name == 'LICell' else module
        val = getattr(frm, p)

        module._buffers[p] = val


class DVSModelSimple2(pl.LightningModule):
    def __init__(
        self,
        n_class,
        n_channels,
        height,
        width,
        class_weights=None,
        method="super",
        alpha=100,
        dt=0.001,
    ):
        super().__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # block 1
        self.block1 = SequentialState(
            nn.Conv2d(n_channels, 8, 3, padding=1, bias=False),
            LICell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.BatchNorm2d(8),
        )

        # block 2
        self.block2 = SequentialState(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            LICell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.BatchNorm2d(16),
        )

        # block 3
        self.block3 = SequentialState(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            # LICell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            # nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
            # nn.BatchNorm2d(32),
        )

        # dense
        # self.dense = SequentialState(
        #     nn.Conv2d(32, 32, 7, padding=3, bias=False),
        #     LICell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
        #     # nn.BatchNorm2d(32),
        # )

        # self.score_block2 = nn.Conv2d(16, n_class, 1, bias=False)
        # self.deconv_block2 = nn.ConvTranspose2d(
        #     n_class, n_class, 8, stride=4, padding=2, bias=False
        # )
        #
        # self.score_dense = nn.Conv2d(32, n_class, 1, bias=False)
        # self.deconv_dense = nn.ConvTranspose2d(
        #     n_class, n_class, 16, stride=8, padding=4, bias=False
        # )
        #
        self.final = LICell(dt=dt)


    def forward(self, x):
        state_block1 = state_block2 = state_block3 = state_dense = state_final = None

        # output              batch,      class,        height,     width
        output = torch.empty(
            (x.shape[0], self.n_class, x.shape[1], x.shape[2], x.shape[3]),
            device=self.device,
        )

        # for each frame
        for i in range(1):
            frame = x[:, :, :, :]

            out_block1, state_block1 = self.block1(frame, state_block1)  # 1/2
            out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
            out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
            # out_dense, state_dense = self.dense(out_block3, state_dense)

            ####### WITH FEATURE FUSION
            # out_score_block2 = self.score_block2(out_block2)
            # out_deconv_block2 = self.deconv_block2(out_score_block2)

            # out_score_dense = checkpoint(self.score_dense, out_dense)
            # out_score_dense = self.score_dense(out_dense)

            # out_deconv_dense = self.deconv_dense(out_score_dense)
            #
            # out_deconv = out_deconv_block2 + out_deconv_dense
            # #######
            #
            # out_final, state_final = self.final(out_deconv, state_final)
            out_final, state_final = self.final(out_block3, state_final)
            #
            # output[:, :, i, :, :] = out_final

            # self.log("input_mean", frame.mean())
            # self.log("out_block1_mean", out_block1.mean())
            # self.log("out_block2_mean", out_block2.mean())
            # self.log("out_block3_mean", out_block3.mean())
            # self.log("out_dense_mean", out_dense.mean())

        # return output
        return out_final

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        z = self.forward(x)

        loss = self.loss_fn(z, y)
        # Log the loss
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-5)


width = height = 32
n_class = 9
n_in_channels = 1
n_samples = 1
n_frames_per_sample = 1
# checkpoint = torch.load('epoch=131-step=123815.ckpt')
m = DVSModelSimple2(n_class, n_in_channels, height, width)
m.eval()
dummy = torch.randn(n_samples, n_in_channels, height, width)
set_parameter_buffers(m)
x = m.forward(dummy)

childern = list(m.children())
parameters = list(m.parameters())
named_params = list(m.named_parameters())
state_dict = m.state_dict(keep_vars=True)
torch.save({'state_dict': state_dict}, 'torch_net_checkpoint.ptck')

# print(m.children())
# print(m.parameters())
modules = dict(m.named_modules())

shapes = {}
x = dummy
for i, k in enumerate(modules):
    if (isinstance(modules[k], (torch.nn.CrossEntropyLoss, SequentialState)) or
        k == ''):
        continue
    # print(k)
    x = modules[k](x)
    if isinstance(x, tuple):
        x = x[0]
    shapes[k] = x.data.shape

from bifrost.export.torch import TorchContext
from bifrost.parse.parse_torch import torch_to_network, torch_to_context
from bifrost.ir.input import InputLayer, DummyTestInputSource
from bifrost.exporter import export_network


# inp = InputLayer("in", height * width, 1, DummyTestInputSource([height, width]))
inp = InputLayer("in", height * width, 1, SpiNNakerSPIFInput([height, width]))
out = OutputLayer("out", 1, 1, sink=EthernetOutput())

net = torch_to_network(m, inp, out, {'runtime': 0.0})
ctx, net_dict = torch_to_context(net, m)
with open("torch_as_spynn_net.py", 'w') as f:
    f.write(export_network(net, ctx,))

np.savez_compressed('torch_net_dict.npz', **net_dict)
print(net)
# n_samples = 1
# n_frames_per_sample = 1
# dummy = torch.randn(n_samples, n_frames_per_sample, n_in_channels, height, width)
# parser = Parser(m, dummy)
# pynn_pops_d, pynn_projs_d = parser.generate_pynn_dictionaries()
#
# from pprint import pprint
# pprint(pynn_pops_d)
# pprint(pynn_projs_d)
# # do pynn setup
# import spynnaker8 as sim
# # input_dict = {}
# # pynn_pops, pynn_projs = parser.generate_pynn_objects(
# #                             sim, input_dict, pynn_pops_d, pynn_projs_d)
#
# # intercept_simulator(sim, "test_torch_network")
#
# # sim.run(0)
#
# # -------------------------------------------------------------------- #
# # -------------------------------------------------------------------- #
# # -------------------------------------------------------------------- #
# # onnx_path = 'dvs.onnx'
# # torch.onnx.export(m, dummy, onnx_path,
# #                   verbose=True, opset_version=11)
#
# # import onnx
# # from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
# # import sys
# # import os
# # import matplotlib.pyplot as plt
# #
# #
# # onnx_model = onnx.load(onnx_path)
# # # print(onnx_model.SerializeToString())
# #
# # graph = onnx_model.graph
# # node = graph.node
# #
# # # for n in node:
# # #     print(n.name)
# #
# # onnx.checker.check_model(onnx_model)
# #
# # pydot_graph = GetPydotGraph(onnx_model.graph, name=onnx_model.graph.name, rankdir="TB",
# #                             node_producer=GetOpNodeProducer("docstring"))
# #
# # dot_graph_fname = "graph.dot"
# # pydot_graph.write_dot(dot_graph_fname)
# # os.system('dot -O -Tpng {}'.format(dot_graph_fname))
# #
# # image = plt.imread("{}.png".format(dot_graph_fname))
# # plt.imshow(image)
# # plt.axis('off')
# # plt.show()
#
# # -------------------------------------------------------------------- #
# # -------------------------------------------------------------------- #
# # -------------------------------------------------------------------- #
#
# # from snntoolbox.utils.utils import import_configparser
# # from snntoolbox.simulation.target_simulators.spiNNaker_target_sim import SNN
# # configparser = import_configparser()
# # config = configparser.ConfigParser()
# # config['paths'] = {
# #     'path_wd': '.',             # Path to model.
# # }
# #
# # config['tools'] = {
# #     'evaluate_ann': False,           # Test ANN on dataset before conversion.
# #     # Normalize weights for full dynamic range.
# #     'normalize': False,
# #     'scale_weights_exp': False
# # }
# #
# # config['input'] = {
# #     'poisson_input': True,           # Images are encodes as spike trains.
# #     'input_rate': 100,
# #     'dataset_format': 'poisson',
# #     'num_poisson_events_per_sample': 100,
# # }
# #
# # config['cell'] = {
# #     'v': 0.0,
# #     'v_thresh': 1.0
# # }
# #
# # config['simulation'] = {
# #     # Chooses execution backend of SNN toolbox.
# #     'simulator': 'spiNNaker',
# #     'duration': 50,                 # Number of time steps to run each sample.
# #     'num_to_test': 5,               # How many test samples to run.
# #     'batch_size': 1,                # Batch size for simulation.
# #     # SpiNNaker seems to require 0.1 for comparable results.
# #     'dt': 0.1,
# #     'early_stopping': True
# # }
# #
# # config['restrictions'] = {
# #     'simulators_pynn': {
# #         'simulator': 'spiNNaker'
# #     },
# #     'cellparams_pynn': {
# #         'v': 0.0,
# #         'v_thresh': 1.0
# #     }
# # }
# #
# # config['output'] = {
# #     'plot_vars': {                  # Various plots (slows down simulation).
# #     },
# #     'log_vars': {
# #     }
# # }
# #
# # snn_parser = SNN(config)
# # snn_parser.build(m)
# # snn_parser.save('.', 'snn_model')
#
#
#
