"""
@file VAE.py
@author Ryan Missel
@source https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb

Handles the full pipeline of training a VAE on the given pulsar data before running it through an SOM
to get all of its cluster graphs
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchsummary as sm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import VAE


""" Arg parsing and Data setup"""

# Arg parsing
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--device', '-g', type=int, default=0, help='which GPU to run on')
parser.add_argument('--checkpt', type=str, default='None', help='checkpoint to resume training from')

parser.add_argument('--epochs', '-e', type=int, default=50, help='how many epochs to run')
parser.add_argument('--batch', '-b', type=int, default=128, help='size of batch')
parser.add_argument('--resume', '-r', type=bool, default=False, help='whether to resume training')

parser.add_argument('--month', '-m', type=str, default='jan', help='month of obs')
parser.add_argument('--day', '-d', type=str, default='28', help='day of obs')
parser.add_argument('--antenna', '-a', type=str, default='A1', help='antenna of obs')

args = parser.parse_args()

# Get vars
month, day, antenna = args.month, args.day, args.antenna

# Set seed if given
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.device)


def get_window(signals):
    center = np.argmax(np.mean(signals, axis=0))
    print("Window: ({}, {})".format(center - 100, center + 100))
    return signals[:, center - 100:center + 100]


# Load in the given dataset
dataset = np.load("data/{}{}/{}/{}{}{}.npy".format(month, day, antenna, month, day, antenna), allow_pickle=True)
print("Loaded in cache dataset: ", "data/{}{}/{}/{}{}{}.npy".format(month, day, antenna, month, day, antenna))

# Get the centered window of the signal
dataset = get_window(dataset)

# Transform datasets into loaders
kwargs = {'num_workers': 0, 'pin_memory': True}
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, **kwargs)
testloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, **kwargs)

# Get the mean signal of the raw dataset as a sanity check on the window
plt.plot(np.mean(dataset, axis=0))
plt.title("Raw Dataset Mean Signal")
plt.show()
plt.close()


""" Model setup """
mse_lambda = 0.5
kld_lambda = 0.5


def loss_fn(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = mse_lambda * F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -kld_lambda * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# Build the model and load in a checkpoint if given
model = VAE(device=device)

if args.resume is True:
    model.load_state_dict(torch.load('models/{}{}{}_vae.torch'.format(month, day, antenna)))

model = model.to(device)
sm.summary(model, (200,), args.batch)

# Define optimizer to use
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model for the given epochs
print("=> Starting training...")
for epoch in range(args.epochs):
    epochloss = 0
    bceloss = 0
    kldloss = 0

    for idx, signals in enumerate(dataloader):
        signals = signals.to(device).float()

        recon_signals, h, mu, logvar = model(signals)
        loss, bce, kld = loss_fn(recon_signals, signals, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochloss += loss.item()
        bceloss += bce.item()
        kldloss += kld.item()

    torch.save(model.state_dict(), 'models/{}{}{}_vae.torch'.format(month, day, antenna))
    to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1, args.epochs, epochloss / args.batch,
                                                                bceloss / args.batch, kldloss / args.batch)
    print(to_print)

print("=> Complete!")

torch.save(model.state_dict(), 'models/{}{}{}_vae.torch'.format(month, day, antenna))
print("=> Model saved at ", 'models/{}{}{}_vae.torch'.format(month, day, antenna))


""" Evaluation """


# Reconstruct the entire dataset
recons = None
for idx, signals in enumerate(testloader):
    signals = signals.to(device).float()

    recon_signals, _, _, _ = model(signals)
    recon_signals = recon_signals.detach().cpu().numpy()

    if recons is None:
        recons = recon_signals

    else:
        recons = np.vstack((recons, recon_signals))

np.savetxt('reconstructed/{}{}{}recon_vae.csv'.format(month, day, antenna), recons, delimiter=',')

# Handles plotting a few one-on-one signals to raw to check reconstruction
srange = (0, 2)
for signal, reconsig in zip(dataset[srange[0]:srange[1]], recons[srange[0]:srange[1]]):
    plt.plot(signal)
    plt.plot(reconsig)
    # plt.close()
    plt.show()


# Handles plotting the mean signals of the raw data and the reconstruction
plt.plot(np.mean(recons, axis=0))
plt.plot(np.mean(dataset, axis=0))
plt.title("Mean Signals")
plt.legend(('recon', 'data'))
# plt.close()
plt.show()