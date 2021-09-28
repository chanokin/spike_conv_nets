import numpy as np
import torch
import torchvision
from mnist_norse import LIFConvNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def test(model, device, test_loader, epoch, method, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set {method}: Average loss: {test_loss:.4f}, \
            Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
    )
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)

    return test_loss, accuracy


def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "is_best": is_best,
        },
        path,
    )


def train(model, device, train_loader, optimizer, epoch, clip_grad,
          grad_clip_value, epochs, log_interval, do_plot, plot_interval,
          seq_length, writer,):
    torch.autograd.set_detect_anomaly(True)

    model.train()
    losses = []

    batch_len = len(train_loader)
    step = batch_len * epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward(retain_graph=True)

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        step += 1

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    epochs,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        if step % log_interval == 0:
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", accuracy.item(), step)

            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram(tag, value.data.cpu().numpy(), step)
                if value.grad is not None:
                    writer.add_histogram(
                        tag + "/grad", value.grad.data.cpu().numpy(), step
                    )

        if do_plot and batch_idx % plot_interval == 0:
            ts = np.arange(0, seq_length)
            fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
            axs = axs.reshape(-1)  # flatten
            for nrn in range(10):
                one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
                ax = axs[nrn]
                ax.plot(ts, one_trace)
            plt.xlabel("Time [s]")
            plt.ylabel("Membrane Potential")

            writer.add_figure("Voltages/output", fig, step)

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

epochs = 10
batch_size = 32
seq_length = 200  # time steps
learning_rate = 2e-3
act_model = 'super'
optimizer = 'adam'
random_seed = 1374
device = 'cpu'

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
        download=True,
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

writer = SummaryWriter()

training_losses = []
mean_losses = []
test_losses = []
accuracies = []

for epoch in range(epochs):
    training_loss, mean_loss = train(
        model,
        device,
        train_loader,
        optimizer,
        epoch,
        clip_grad=False,
        grad_clip_value=1.0,
        epochs=1,
        log_interval=1,
        do_plot=True,
        plot_interval=1,
        seq_length=seq_length,
        writer=writer,
    )
    test_loss, accuracy = test(
        model, device, test_loader, epoch, method=act_model, writer=writer
    )

    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

    max_accuracy = np.max(np.array(accuracies))


    model_path = f"mnist-{epoch}.pt"
    save(
        model_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        is_best=accuracy > max_accuracy,
    )

model_path = "mnist-final.pt"
save(
    model_path,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
    is_best=accuracy > max_accuracy,
)