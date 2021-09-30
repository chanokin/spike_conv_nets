import numpy as np
import torch
import torchvision
from mnist_norse import LIFConvNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from bifrost.extract.torch.parameter_buffers import set_parameter_buffers


def retry_jittered_backoff(f, num_retries=5):
    # Based on:
    # https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    import time
    import random
    cap = 1.0                  # max sleep time is 1s
    base = 0.01                # initial sleep time is 10ms
    sleep = base               # initial sleep time is 10ms

    for i in range(num_retries):
        try:
            return f()
        except RuntimeError as e:
            if i == num_retries - 1:
                raise e
            else:
                continue
        time.sleep(sleep)
        sleep = min(cap, random.uniform(base, sleep * 3))


def pick_gpu():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        try:
            torch.ones(1).cuda()
        except RuntimeError:
            continue
        return i
    raise RuntimeError("No GPUs available.")


epochs = 2
batch_size = 32
seq_length = 200  # time steps
learning_rate = 2e-3
act_model = 'super'
optimizer = 'adam'
random_seed = 1374
device = 'cpu' if bool(0) else 'cuda'

torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

device = torch.device(device)

# First we create and transform the dataset
data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root=".",
        train=True,
        # download=True,
        transform=data_transform,
    ),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root=".",
        train=False,
        transform=data_transform,
    ),
    batch_size=batch_size,
)

# Second, we create the PyTorch Lightning module
model = LIFConvNet(
    28 * 28,  # Standard MNIST size
    seq_length=seq_length,
    model=act_model,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# pl.Trainer.from_argparse_args()
trainer = pl.Trainer(gpus=[0],
                     max_epochs=epochs)
trainer.fit(model, train_loader)

set_parameter_buffers(model)
model_path = "mnist-final.pt"
torch.save(
    dict(model=model.state_dict(),
         optimizer=optimizer, ),
    model_path,
)