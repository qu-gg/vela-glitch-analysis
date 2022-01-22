"""
@file VAE.py
@author Ryan Missel
@source https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb

Handles the full pipeline of training a VAE on the given pulsar data before running it through an SOM
to get all of its cluster graphs
"""
import os
import torch
import argparse
import pandas as pd
import torch.nn.functional as F

from model import VAE
from minisom import MiniSom
from util.plot_functions import *


def get_window(signals):
    center = np.argmax(np.mean(signals, axis=0))
    print("Window: ({}, {})".format(center - 100, center + 100))
    return signals[:, center - 100:center + 100]


def loss_fn(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = 0.5 * F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def run_model(epochs, resume, batch, month, day, antenna):
    # Load in the given dataset
    dataset = np.load("data/{}{}/{}/{}{}{}.npy".format(month, day, antenna, month, day, antenna), allow_pickle=True)
    print("Loaded in cache dataset: ", "data/{}{}/{}/{}{}{}.npy".format(month, day, antenna, month, day, antenna))

    # Get the centered window of the signal
    dataset = get_window(dataset)

    # Transform datasets into loaders
    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, **kwargs)

    # Get the mean signal of the raw dataset as a sanity check on the window
    plt.plot(np.mean(dataset, axis=0))
    plt.title("Raw Dataset Mean Signal")
    plt.close()

    # Build the model and load in a checkpoint if given
    model = VAE(device=device)

    if resume is True:
        model.load_state_dict(torch.load('models/{}{}{}_vae.torch'.format(month, day, antenna)))

    model = model.to(device)

    # Define optimizer to use
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model for the given epochs
    print("=> Starting training...")
    for epoch in range(epochs):
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
        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1, epochs, epochloss / batch,
                                                                    bceloss / batch, kldloss / batch)
        print(to_print)

    print("=> Complete!")

    torch.save(model.state_dict(), 'models/{}{}{}_vae.torch'.format(month, day, antenna))
    print("=> Model saved at ", 'models/{}{}{}_vae.torch'.format(month, day, antenna))

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

    np.savetxt('reconstructed/{}{}{}_recon.csv'.format(month, day, antenna), recons, delimiter=',')

    # Create folder if it doesn't exist
    trainset = '- {}{}{}'.format(month, day, antenna)
    if not os.path.exists("graphs/{}/".format(trainset)):
        os.makedirs('graphs/{}/'.format(trainset))

    # Handles plotting the mean signals of the raw data and the reconstruction
    plt.figure(1)
    plt.plot(np.mean(recons, axis=0))
    plt.plot(np.mean(dataset, axis=0))
    plt.title("Mean Signals for {}{}{}".format(month, day, antenna))
    plt.legend(('recon', 'data'))
    plt.savefig('graphs/- {}{}{}/meanSets.png'.format(month, day, antenna))
    plt.close()


def run_som(som_shape, month, day, antenna):
    trainset = '- {}{}{}'.format(month, day, antenna)

    # Load in dataset and get the centered window of the signal
    raw = np.load("data/{}{}/{}/{}{}{}.npy".format(month, day, antenna, month, day, antenna), allow_pickle=True)
    raw = get_window(raw)

    # Load in reconstructions
    data = pd.read_csv('reconstructed/{}{}{}_recon.csv'.format(month, day, antenna), header=None).to_numpy()
    print("Raw {} Data {}".format(raw.shape, data.shape))

    # Train SOM
    som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=data.shape[1], sigma=0.3, learning_rate=0.5)
    som.train(data, 100000, verbose=False)

    # Extract coordinates for each cluster and assignment
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

    # Setting up folders for graph saving
    if not os.path.exists("graphs/"):
        os.makedirs('graphs/')

    if not os.path.exists("graphs/{}/".format(trainset)):
        os.makedirs('graphs/{}/'.format(trainset))

    folder = "graphs/{}/{}_{}/".format(trainset, som_shape[0], som_shape[1])
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Gets statistics to be used in multiple plots
    shapes, indices, xs, ys, (maxes, maxes_std), (argmaxes, argmaxes_std), (widths, widths_std), (skews, skews_std), \
    (mses, mses_std) = cluster_statistics(folder, som_shape, data, raw, cluster_index, save=True)

    # Plotting the centroid signals of each node
    plot_centroids(folder, som, som_shape, data, cluster_index)

    # Mean plots of each cluster plotted on each other
    plot_means(folder, som_shape, data, raw, cluster_index, "{}{}{}".format(month, day, antenna))

    # Plotting the raw and recon mean of each cluster over each other
    plot_mean_comparison(folder, trainset, som_shape, data, raw, cluster_index)

    # Grid graph of the physical features using colormap
    # plot_gridgraph(folder, som_shape, xs, ys, maxes, argmaxes, widths, skews, mses)


if __name__ == '__main__':
    # Arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--device', '-g', type=int, default=0, help='which GPU to run on')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='how many epochs to run')
    parser.add_argument('--batch', '-b', type=int, default=128, help='size of batch')
    parser.add_argument('--resume', '-r', type=bool, default=False, help='whether to resume training')
    args = parser.parse_args()

    # Set seed if given
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    # Defines SOM shapes to run over
    som_shapes = [
        (2, 2), (3, 2)
    ]

    # Defines models to run
    models = [
        # ['jan', '21', 'A1', 'noClean'],
        # ['jan', '21', 'A2', 'noClean'],
        # ['jan', '21', 'A1', 'rfi'],
        # ['jan', '21', 'A2', 'rfi'],
        # ['jan', '21', 'A1', 'rfiFreq'],
        # ['jan', '21', 'A2', 'rfiFreq'],

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

        ['july', '19', 'A1'],
        # ['july', '20', 'A1'],
        # ['july', '21', 'A1'],
        # ['july', '23', 'A1'],
        ['july', '24', 'A1'],
        ['july', '26', 'A1'],
    ]

    # Loop over each model and SOM shape for that model
    for i, model in enumerate(models):
        run_model(args.epochs, args.resume, args.batch, model[0], model[1], model[2])

        for som_shape in som_shapes:
            run_som(som_shape, model[0], model[1], model[2])

