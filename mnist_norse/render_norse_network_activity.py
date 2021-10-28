import numpy as np
import torch
import torchvision
from mnist_norse import LIFConvNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from bifrost.extract.torch.parameter_buffers import set_parameter_buffers
import collections
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

random_seed = 1337
seq_length = 100  # time steps
learning_rate = 2e-3
act_model = 'super'
device = 'cpu' if bool(1) else 'cuda'
batch_size = 8


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
        torchvision.transforms.Normalize((0.,), (1.,)),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


total_test_size = 10000
test_size = 10000
dataset = torchvision.datasets.MNIST(
    root=".",
    train=False,
    transform=data_transform,
)
targets = dataset.targets[:batch_size]
batch = dataset.data[:batch_size].to(device).numpy() #/ 255.
dataset = torch.hstack([dataset.transform(s) for s in batch]).view((batch_size, 28*28))
# dataset = torch.hstack([s for s in batch]).view((batch_size, 28*28))
print(dataset.min(), dataset.max())


# Second, we create the PyTorch Lightning module
model = LIFConvNet(
    28 * 28,  # Standard MNIST size
    seq_length=seq_length,
    model=act_model,
).to(device)

filename = 'mnist-final-50.pt'
checkpoint = torch.load(filename,
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'], strict=False)

# a dictionary that keeps saving the activations as they come
activations = collections.defaultdict(list)
def save_activation(name, mod, inp, out):
    if isinstance(out, tuple):
        activations[name].append(out[0].cpu())
    else:
        activations[name].append(out.cpu())


# Registering hooks for all the Conv2d layers
# Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
#  called repeatedly at different stages of the forward pass (like RELUs), this will save different
#  activations. Editing the forward pass code to save activations is the way to go for these cases.
for name, m in model.named_modules():
    if name == '':
        continue

    # partial to assign the layer name to each hook
    m.register_forward_hook(partial(save_activation, name))


# forward pass through the full dataset
out = model(dataset)

# concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

# just print out the sizes of the saved activations as a sanity check
for k,v in activations.items():
    print (k, v.size(), v.mean(), v.std())

keys = list(activations.keys())
input = activations[keys[0]]
n_cols = 1
n_rows = 1
n_imgs = n_rows * n_cols
n_steps_per_image = seq_length // n_imgs
for i in range(batch_size):
    inn = input[:, i].reshape((-1, 28, 28))
    fig, axes = plt.subplots(n_rows, n_cols)
    if isinstance(axes, np.ndarray):
        axes = axes.reshape((n_rows, n_cols))
    for ax_row in range(n_rows):
        for ax_col in range(n_cols):
            img_idx = (ax_row * n_cols + ax_col)
            start_idx = img_idx * n_steps_per_image
            end_idx = min(start_idx + n_steps_per_image, seq_length)
            if isinstance(axes, np.ndarray):
                ax = axes[ax_row, ax_col]
            else:
                ax = axes
            img = inn[start_idx:end_idx].sum(axis=0)
            mn = img[img > 0].mean()
            st = img[img > 0].std()
            ax.set_title(f"mean {mn:2.6f}  std {st:2.6f}")
            im = ax.imshow(img)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
