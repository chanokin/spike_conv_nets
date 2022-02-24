import numpy as np

import torch
import torch.utils.data

from norse.torch.module.encode import PoissonEncoder
from norse.torch.functional.lif import LIFParameters
from norse.torch.functional.leaky_integrator import LIParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module.leaky_integrator import LICell
from norse.torch import SequentialState
import pytorch_lightning as pl


class LIFConvNet(pl.LightningModule):
    def __init__(self, input_features, seq_length, input_scale=1,
                 model="super", only_first_spike=False, optimizer='adam',
                 learning_rate=2e-3, threshold=0.7):
        super(LIFConvNet, self).__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.input_encoder = PoissonEncoder(seq_length=seq_length, f_max=500)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        bias = True
        self.conv1 = torch.nn.Conv2d(1, 16, 5, 1, bias=bias)
        self.lif1 = LIFCell(
            p=LIFParameters(method=model, alpha=1.0,
                            v_th=torch.as_tensor(threshold)
            ),
        )

        self.pool1 = torch.nn.AvgPool2d(2)

        self.conv2 = torch.nn.Conv2d(16, 8, 5, 1, bias=bias)
        self.lif2 = LIFCell(
            p=LIFParameters(method=model, alpha=1.0,
                v_th=torch.as_tensor(threshold)
            ),
        )


        self.pool2 = torch.nn.AvgPool2d(2)

        self.dense1 = torch.nn.Linear(128, 256, bias=bias)
        self.lif3 = LIFCell(
            p=LIFParameters(method=model, alpha=1.0,
                            v_th=torch.as_tensor(threshold)
            ),
        )

        self.dense2 = torch.nn.Linear(256, 64, bias=bias)
        self.lif4 = LIFCell(
            p=LIFParameters(method=model, alpha=1.0,
                            v_th=torch.as_tensor(threshold)
            ),
        )

        self.dense3 = torch.nn.Linear(64, 10, bias=bias)
        self.lif5 = LICell(p=LIParameters(),)
#        self.lif5 = LIFCell(
#            p=LIFParameters(method=model, alpha=1.0,
#                            v_th=torch.as_tensor(threshold)),
#        )

        self.seq_length = seq_length
        self.input_scale = input_scale
        self.voltages = None
        self.all_spike_counts = None
        self.all_weight_means = None

    def forward(self, x):

        batch_size = x.shape[0]
        seq_length = self.seq_length
        x = self.input_encoder( x.view((-1, self.input_features)) )
        x = x.reshape(seq_length, batch_size, 1, 28, 28)

        s1 = None
        s2 = None
        s3 = None
        s4 = None
        s5 = None

        output = []
        all_spikes = []
        all_weights = []
        for in_step in range(seq_length):
            x_this_step = x[in_step, :]
            # print(f"input = {x_this_step.mean()}")
            z = self.conv1(x_this_step)
            all_weights.append(self.conv1.weight.detach())
            # print(f"conv1 = {z.mean()}")
            z, s1 = self.lif1(z, s1)
            all_spikes.append(z.clone())
            # print(f"lif11 = {z.mean()}")

            z = self.pool1(z)
            # print(f"pool1 = {z.mean()}")
            z = self.conv2(z)
            all_weights.append(self.conv2.weight.detach())
            # print(f"conv2 = {z.mean()}")
            z, s2 = self.lif2(z, s2)
            all_spikes.append(z.clone())
            # print(f"lif2 = {z.mean()}")

            z = self.pool2(z)  # batch, 8 channels, 4x4 neurons
            # print(f"pool2 = {z.mean()}")

            # flatten
            # z = z.view(batch_size, -1)
            # print(z.shape)
            z = torch.flatten(z, start_dim=1)
            # print(z.shape)

            z = self.dense1(z)
            all_weights.append(self.dense1.weight.detach())
            # print(f"dense1 = {z.mean()}")
            z, s3 = self.lif3(z, s3)
            all_spikes.append(z.clone())
            # print(f"lif3 = {z.mean()}")

            z = self.dense2(z)
            all_weights.append(self.dense2.weight.detach())
            # print(f"dense2 = {z.mean()}")
            z, s4 = self.lif4(z, s4)
            all_spikes.append(z.clone())
            # print(f"lif4 = {z.mean()}")

            z = self.dense3(z)
            all_weights.append(self.dense3.weight.detach())
            # print(f"dense3 = {z.mean()}")
            # z = torch.nn.functional.relu(z)
            z, s5 = self.lif5(z, s5)
            all_spikes.append(z.clone())
            # print(f"lif5 = {z.mean()}")

            output.append(z)#.detach())

        # return voltages
        self.voltages = torch.stack(output)
        #self.all_spike_counts = [torch.sum(s, dim=0) for s in all_spikes]
        #self.all_weight_means = [torch.mean(torch.abs(w)) for w in all_weights]
        #out_spikes = torch.stack(output)
        #spike_sum = torch.sum(out_spikes, dim=0)
        #return spike_sum
        # print(self.voltages.shape)
        m, _ = torch.max(self.voltages, dim=0)
        # print(m.shape)
        return m
        # log_p_y = torch.nn.functional.log_softmax(spike_sum, dim=1)
        # return log_p_y

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        correct = out.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        acc = float(correct) / total

        batch_dict = {
            "correct": correct,
            "total": total,
            "accuracy": acc
        }

        return batch_dict

    def test_epoch_end(self, outputs):
        all_accuracies = [d["accuracy"] for d in outputs]
        print(all_accuracies)
        avg_acc = torch.mean(torch.tensor(all_accuracies))
        print(f"accuracy {avg_acc}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        log_p_y = torch.nn.functional.log_softmax(out, dim=1)
        loss = torch.nn.functional.nll_loss(log_p_y, y)

        # print()

        # punish too much activity in the output
        #net_spikes = self.all_spike_counts
        # print(f"spike_shapes = {[s.shape for s in net_spikes]}\n")
        #net_spike_averages = torch.stack([torch.mean(s) for s in net_spikes])
        #average_spiking = torch.mean(net_spike_averages)
        # print(f"average_spiking = {average_spiking}\n")

        # print(f"all_weight_means = {self.all_weight_means}\n")
        #average_weights = torch.mean(torch.stack(self.all_weight_means))
        # print(f"average_weights = {average_weights}\n")

        # print(net_spike_averages)
        #activity_loss = average_spiking * 0.001
        # print(f"activity_loss = {activity_loss}")

        #loss += activity_loss

        correct = out.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        logs={"train_loss": loss.detach()}
        batch_dictionary={
            "loss": loss,
            "log": logs,
            "correct": correct,
            "total": total
        }

        return batch_dictionary

    def training_epoch_end(self, outputs):
        print(outputs[-1])

    def configure_optimizers(self):
        if self.optimizer == "adam":
            opt = torch.optim.Adam
        else:
            opt = torch.optim.SGD

        return opt(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)