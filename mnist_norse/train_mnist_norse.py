import numpy as np
import torch
import torchvision
from mnist_norse import LIFConvNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

epochs = 10
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
trainer = pl.Trainer(gpus=1, max_epochs=epochs)
trainer.fit(model, train_loader)

#
# writer = SummaryWriter()
#
# training_losses = []
# mean_losses = []
# test_losses = []
# accuracies = []
#
# for epoch in range(epochs):
#     training_loss, mean_loss = train(
#         model,
#         device,
#         train_loader,
#         optimizer,
#         epoch,
#         clip_grad=False,
#         grad_clip_value=1.0,
#         epochs=1,
#         log_interval=1,
#         do_plot=True,
#         plot_interval=1,
#         seq_length=seq_length,
#         writer=writer,
#     )
#     test_loss, accuracy = test(
#         model, device, test_loader, epoch, method=act_model, writer=writer
#     )
#
#     training_losses += training_loss
#     mean_losses.append(mean_loss)
#     test_losses.append(test_loss)
#     accuracies.append(accuracy)
#
#     max_accuracy = np.max(np.array(accuracies))
#
#
#     model_path = f"mnist-{epoch}.pt"
#     save(
#         model_path,
#         model=model,
#         optimizer=optimizer,
#         epoch=epoch,
#         is_best=accuracy > max_accuracy,
#     )
#
# model_path = "mnist-final.pt"
# save(
#     model_path,
#     epoch=epoch,
#     model=model,
#     optimizer=optimizer,
#     is_best=accuracy > max_accuracy,
# )