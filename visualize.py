import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

from model import VAE, DAE


def get_window(signals):
    center = np.argmax(np.mean(signals, axis=0))
    print("Window: ({}, {})".format(center - 100, center + 100))
    return signals[:, center - 100:center + 100]


# Defines models to run
models = [
    # ['jan', '21', 'A1', 'noClean'],
    # ['jan', '21', 'A2', 'noClean'],
    # ['jan', '21', 'A1', 'rfi'],
    # ['jan', '21', 'A2', 'rfi'],
    # ['jan', '21', 'A1', 'rfiFreq'],
    # ['jan', '21', 'A2', 'rfiFreq'],
    #
    # ['jan', '24', 'A1', 'noClean'],
    # ['jan', '24', 'A2', 'noClean'],
    # ['jan', '24', 'A1', 'rfi'],
    # ['jan', '24', 'A2', 'rfi'],
    # ['jan', '24', 'A1', 'rfiFreq'],
    # ['jan', '24', 'A2', 'rfiFreq'],

    # ['jan', '28', 'A1', 'noClean'],
    # ['jan', '28', 'A2', 'noClean'],
    # ['jan', '28', 'A1', 'rfi'],
    # ['jan', '28', 'A2', 'rfi'],
    # ['jan', '28', 'A1', 'rfiFreq'],
    # ['jan', '28', 'A2', 'rfiFreq'],

    ['jan', '28', 'A1', 'rfiFreq'],
    # ['jan', '28', 'A2', 'rfiFind'],
]

# Run over the defined models
for model in models:
    month, day, antenna, set_type = model[0], model[1], model[2], model[3]
    print(" ---- {}{}{}{} ----".format(month, day, antenna, set_type))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)

    # Load in the given dataset
    dataset = np.load("data/{}{}/{}/{}{}.npy".format(month, day, set_type, set_type, antenna), allow_pickle=True)
    print("Loaded in cache dataset: ", "data/{}{}_{}.npy".format(month, antenna, set_type))

    # Get the centered window of the signal
    dataset = get_window(dataset)

    # Transform datasets into loaders
    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, **kwargs)

    # Build the model and load in a checkpoint if given
    model = DAE(device=device)
    model.load_state_dict(torch.load('models/{}{}{}{}.torch'.format(month, day, antenna, set_type), map_location=device))
    model = model.to(device)

    # Reconstruct the entire dataset
    recons = None
    for idx, signals in enumerate(testloader):
        if idx * 128 > 1000:
            break

        signals = signals.to(0).float()

        recon_signals = model(signals)
        recon_signals = recon_signals.detach().cpu().numpy()

        if recons is None:
            recons = recon_signals
        else:
            recons = np.vstack((recons, recon_signals))

    print("Done reconstructing, now plotting.")

    # Make folder is it does not exist
    if not os.path.exists("figs/{}{}{}{}/".format(month, day, antenna, set_type)):
        os.makedirs("figs/{}{}{}{}/".format(month, day, antenna, set_type))

    # Handles plotting a few one-on-one signals to raw to check reconstruction
    srange = np.random.randint(0, 1000, 50)
    idx = 0
    for idx, signal, reconsig in zip(srange, dataset[srange], recons[srange]):
        plt.figure(1)
        plt.plot(signal)
        plt.plot(reconsig)
        plt.legend(('Raw', 'Reconstructed'))
        plt.title('VAE Reconstruction of Signal {} on January 28 {}'.format(idx, antenna))
        plt.savefig('figs/{}{}{}{}/signal{}.png'.format(month, day, antenna, set_type, idx))
        plt.close()

        plt.figure(1)
        plt.plot(signal)
        # plt.legend(('Raw'))
        plt.title('Signal {} on January 28 {}'.format(idx, antenna))
        plt.savefig('figs/{}{}{}{}/rawsignal{}.png'.format(month, day, antenna, set_type, idx))
        plt.close()

    print(" ---- ")