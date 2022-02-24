import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from mnist_norse import LIFConvNet


def retry_jittered_backoff(f, num_retries=5):
    # Based on:
    # https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
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


epochs = 50
batch_size = 32
n_dataset_workers = 8
seq_length = 100  # time steps
learning_rate = 2e-3
act_model = 'super'
optimizer = 'adam'
random_seed = 1337
device = 'cpu' if bool(0) else 'cuda'

torch.manual_seed(random_seed)
np.random.seed(random_seed)
gpus = None
if device == 'cuda':
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    gpus = [0]
    device = torch.device(device)

# First we create and transform the dataset
data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Normalize((0.,), (1.,)),
    ]
)

total_train_size = 60000
train_size = 60000
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.random_split(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=data_transform,
        ),
        [train_size, total_train_size - train_size]
    )[0],
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_dataset_workers,
)

total_test_size = 10000
test_size = 10000
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.random_split(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            transform=data_transform,
        ),
        [test_size, total_test_size - test_size]
    )[0],
    batch_size=batch_size,
)

# Second, we create the PyTorch Lightning module
model = LIFConvNet(
    28 * 28,  # Standard MNIST size
    seq_length=seq_length,
    model=act_model,
).to(device)

filename = "/home/chanokin/sussex/spike_conv_nets/spike_conv_nets/mnist_norse/mnist-final-50-poisson-volt_out.pt"
checkpoint = torch.load(filename,
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'], strict=False)

trainer = pl.Trainer(gpus=gpus,
                     max_epochs=epochs,
                    #  callbacks=[checkpoint_callback],
                    #  fast_dev_run=True
                    )

trainer.test(model, test_loader)

# set_parameter_buffers(model)
# model_path = "mnist-final.pt"
# torch.save(
#     dict(state_dict=model.state_dict(keep_vars=True),
#          optimizer=optimizer, ),
#     model_path,
# )
