import numpy as np

import torch
import torch.utils.data

from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch import SequentialState

class LIFConvNet(torch.nn.Module):
    def __init__(self, input_features, seq_length, input_scale=1,
                 model="super", only_first_spike=False,):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.only_first_spike = only_first_spike
        self.input_features = input_features

        self.conv1 = torch.nn.Conv2d(1, 16, 5, 1)
        self.lif1 = LIFCell(p=LIFParameters(method=model, alpha=100.0),)

        self.pool1 = torch.nn.AvgPool2d(2)
        self.conv2 = torch.nn.Conv2d(16, 8, 5, 1)
        self.lif2 = LIFCell(p=LIFParameters(method=model, alpha=100.0),)

        self.pool2 = torch.nn.AvgPool2d(2)

        self.dense1 = torch.nn.Linear(128, 64)
        self.lif3 = LIFCell(p=LIFParameters(method=model, alpha=100.0),)

        self.dense2 = torch.nn.Linear(64, 10)
        self.lif4 = LIFCell(p=LIFParameters(method=model, alpha=100.0),)

        self.voltages = None
        self.seq_length = seq_length
        self.input_scale = input_scale

    def forward(self, x):
        x = self.constant_current_encoder(
            x.view(-1, self.input_features) * self.input_scale
        )
        batch_size = x.shape[1]
        seq_length = x.shape[0]

        if self.only_first_spike:
            # delete all spikes except for first
            zeros = torch.zeros_like(x.cpu()).detach().numpy()
            idxs = x.cpu().nonzero().detach().numpy()
            spike_counter = np.zeros((1, 28 * 28))
            for t, batch, nrn in idxs:
                if spike_counter[batch, nrn] == 0:
                    zeros[t, batch, nrn] = 1
                    spike_counter[batch, nrn] += 1
            x = torch.from_numpy(zeros).to(x.device)

        if self.voltages is None:
            self.voltages = torch.zeros(
                seq_length, batch_size, 10, device=x.device
            )
        voltages = self.voltages

        x = x.reshape(seq_length, batch_size, 1, 28, 28)
        for in_step in range(seq_length):
            x_this_step = x[in_step, :]
            z = self.conv1(x_this_step)
            z, s = self.lif1(z)

            z = self.pool1(z)
            z = self.conv2(z)
            z, s = self.lif2(z)

            z = self.pool2(z)  # batch, 8 channels, 4x4 neurons

            # flatten
            z = z.view(batch_size, -1)

            z = self.dense1(z)
            z, s = self.lif3(z)

            z = self.dense2(z)
            z, s = self.lif4(z)

            voltages[in_step, :] = s.v


        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)

        return log_p_y

