import torch
import norse.torch
# from models import snn
from torch_to_pynn import Parser
import torch.nn as nn
from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module.leaky_integrator import LICell
import spynnaker8 as sim
import pytorch_lightning as pl

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
            LICell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.BatchNorm2d(32),
        )

        # dense
        self.dense = SequentialState(
            nn.Conv2d(32, 32, 7, padding=3, bias=False),
            LICell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(32),
        )

        self.score_block2 = nn.Conv2d(16, n_class, 1, bias=False)
        self.deconv_block2 = nn.ConvTranspose2d(
            n_class, n_class, 8, stride=4, padding=2, bias=False
        )

        self.score_dense = nn.Conv2d(32, n_class, 1, bias=False)
        self.deconv_dense = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, padding=4, bias=False
        )

        self.final = LICell(dt=dt)

    @property
    def layers(self):
        bs = [
            self.block1, self.block2, self.block3, self.dense, #self.score_dense
        ]
        lyrs = []
        for b in bs:
            if isinstance(b, SequentialState):
                bls = [b._modules[l] for l in sorted(b._modules)]
            else:
                bls = [b]
            lyrs.extend(bls)

        return lyrs

    def forward(self, x):
        state_block1 = state_block2 = state_block3 = state_dense = state_final = None

        # output              batch,      class,        frame,      height,     width
        output = torch.empty(
            (x.shape[0], self.n_class, x.shape[1], x.shape[3], x.shape[4]),
            device=self.device,
        )

        # for each frame
        for i in range(x.shape[1]):
            frame = x[:, i, :, :, :]

            out_block1, state_block1 = self.block1(frame, state_block1)  # 1/2
            out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
            out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
            out_dense, state_dense = self.dense(out_block3, state_dense)

            ####### WITH FEATURE FUSION
            out_score_block2 = self.score_block2(out_block2)
            out_deconv_block2 = self.deconv_block2(out_score_block2)

            # out_score_dense = checkpoint(self.score_dense, out_dense)
            out_score_dense = self.score_dense(out_dense)
            out_deconv_dense = self.deconv_dense(out_score_dense)

            out_deconv = out_deconv_block2 + out_deconv_dense
            #######

            out_final, state_final = self.final(out_deconv, state_final)

            output[:, :, i, :, :] = out_final

            self.log("input_mean", frame.mean())
            self.log("out_block1_mean", out_block1.mean())
            self.log("out_block2_mean", out_block2.mean())
            self.log("out_block3_mean", out_block3.mean())
            self.log("out_dense_mean", out_dense.mean())

        return output

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


width = height = 128
n_class = 9
n_in_channels = 2

m = DVSModelSimple2(n_class, n_in_channels, height, width)

n_samples = 1
n_frames_per_sample = 1
dummy = torch.randn(n_samples, n_frames_per_sample, n_in_channels, height, width)
parser = Parser(m, dummy, sim)
pynn_pops_d, pynn_projs_d = parser.generate_pynn_dictionaries()

# do pynn setup
pynn_pops, pynn_projs = parser.generate_pynn_objects(pynn_pops_d, pynn_projs_d)


# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
onnx_path = 'dvs.onnx'
torch.onnx.export(m, dummy, onnx_path,
                  verbose=True, opset_version=11)

# import onnx
# from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
# import sys
# import os
# import matplotlib.pyplot as plt
#
#
# onnx_model = onnx.load(onnx_path)
# # print(onnx_model.SerializeToString())
#
# graph = onnx_model.graph
# node = graph.node
#
# # for n in node:
# #     print(n.name)
#
# onnx.checker.check_model(onnx_model)
#
# pydot_graph = GetPydotGraph(onnx_model.graph, name=onnx_model.graph.name, rankdir="TB",
#                             node_producer=GetOpNodeProducer("docstring"))
#
# dot_graph_fname = "graph.dot"
# pydot_graph.write_dot(dot_graph_fname)
# os.system('dot -O -Tpng {}'.format(dot_graph_fname))
#
# image = plt.imread("{}.png".format(dot_graph_fname))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

# from snntoolbox.utils.utils import import_configparser
# from snntoolbox.simulation.target_simulators.spiNNaker_target_sim import SNN
# configparser = import_configparser()
# config = configparser.ConfigParser()
# config['paths'] = {
#     'path_wd': '.',             # Path to model.
# }
#
# config['tools'] = {
#     'evaluate_ann': False,           # Test ANN on dataset before conversion.
#     # Normalize weights for full dynamic range.
#     'normalize': False,
#     'scale_weights_exp': False
# }
#
# config['input'] = {
#     'poisson_input': True,           # Images are encodes as spike trains.
#     'input_rate': 100,
#     'dataset_format': 'poisson',
#     'num_poisson_events_per_sample': 100,
# }
#
# config['cell'] = {
#     'v': 0.0,
#     'v_thresh': 1.0
# }
#
# config['simulation'] = {
#     # Chooses execution backend of SNN toolbox.
#     'simulator': 'spiNNaker',
#     'duration': 50,                 # Number of time steps to run each sample.
#     'num_to_test': 5,               # How many test samples to run.
#     'batch_size': 1,                # Batch size for simulation.
#     # SpiNNaker seems to require 0.1 for comparable results.
#     'dt': 0.1,
#     'early_stopping': True
# }
#
# config['restrictions'] = {
#     'simulators_pynn': {
#         'simulator': 'spiNNaker'
#     },
#     'cellparams_pynn': {
#         'v': 0.0,
#         'v_thresh': 1.0
#     }
# }
#
# config['output'] = {
#     'plot_vars': {                  # Various plots (slows down simulation).
#     },
#     'log_vars': {
#     }
# }
#
# snn_parser = SNN(config)
# snn_parser.build(m)
# snn_parser.save('.', 'snn_model')



