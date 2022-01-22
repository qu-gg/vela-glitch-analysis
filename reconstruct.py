"""
@file reconstruct.py
@author Ryan Missel

Handles first loading in the VAE model and reconstructing the given dataset
Next it fits an SOM to the reconstructions and builds up some plots based on the clustering formed
"""
import os
import torch
import pandas
import argparse
import scipy.io as sio

from model import VAE, DAE
from minisom import MiniSom
from util.plot_functions import *


""" Arg parsing and Data setup"""
mon1 = 'jan'
ant1 = 1
day1 = 28
mon2 = 'jan'
ant2 = 1
day2 = 28
modeltype = 'vae'

if '{}{}{}'.format(mon1, day1, ant1) != '{}{}{}'.format(mon2, day2, ant2):
    cross = True
    print("Crossed")
else:
    cross = False

# Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--device', '-g', type=int, default=0, help='which GPU to run on')
parser.add_argument('--cross', '-c', type=bool, default=cross, help='whether the sets are on a different day')
parser.add_argument('--checkpt', type=str, default='{}{}A{}rfiFind'.format(mon1, day1, ant1),
                    help='checkpoint to resume training from')
parser.add_argument('--dataset', type=str, default='{}{}/rfiFind/rfiFindA{}'.format(mon2, day2, ant2), help='which dataset to test')
parser.add_argument('--batch', '-b', type=int, default=128, help='size of batch')
args = parser.parse_args()

som_shape = (2, 3)
checkpath = args.checkpt.split('/')[-1].split('.')[0]
datapath = args.dataset.split('/')[0]
print(checkpath, datapath)

# Set seed if given
np.random.seed(123234)
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
dataset = np.load("data/{}.npy".format(args.dataset), allow_pickle=True)

# Get the centered window of the signal
dataset = get_window(dataset)

# Transform datasets into loaders
kwargs = {'num_workers': 0, 'pin_memory': True}
testloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, **kwargs)

# Build the model and load in a checkpoint if given
if modeltype == 'vae':
    model = VAE(device=device)
elif modeltype == 'dae':
    model = DAE(device=device)
elif modeltype == 'deae':
    model = DAE(device=device, denoise=True)
else:
    raise NotImplementedError

model.load_state_dict(torch.load('models/{}_{}.torch'.format(args.checkpt, modeltype), map_location=device))
model = model.to(device)


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


# Full data path
if args.cross is True:
    fullpath = 'graphs/- {}_{}/{}_{}_{}/'.format(checkpath, datapath, som_shape[0], som_shape[1], modeltype)
else:
    fullpath = 'graphs/- {}/{}_{}_{}/'.format(checkpath, som_shape[0], som_shape[1], modeltype)

# Setting up folders for graph saving
if not os.path.exists(fullpath):
    os.makedirs(fullpath)

# Handles plotting the mean signals of the raw data and the reconstruction
plot_set_mean(fullpath, recons, dataset)

# Train SOM
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=dataset.shape[1], sigma=0.3,
              learning_rate=0.5, random_seed=args.seed)

som.train(recons, 10000, verbose=False)

# Extract coordinates for each cluster and assignment
winner_coordinates = np.array([som.winner(x) for x in recons]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

print(np.unique(cluster_index, return_counts=True))
unq, counts = np.unique(cluster_index, return_counts=True)
mapper = np.array(unq)[np.argpartition(counts, 3)]


def remap(x):
    if x == mapper[0]:
        return 0
    elif x == mapper[1]:
        return 1
    elif x == mapper[2]:
        return 2
    elif x == mapper[3]:
        return 3


def remap2(x):
    """
    Used in a select few plots that had clusters messed up
    """
    if x == mapper[0]:
        return 1
    elif x == mapper[1]:
        return 0
    elif x == mapper[2]:
        return 2
    elif x == mapper[3]:
        return 3


if mon2 + str(day2) + 'A' + str(ant2) in ['jan21A1', 'jan24A2', 'march29A2']:
    cluster_index = np.vectorize(remap2)(cluster_index)
else:
    cluster_index = np.vectorize(remap)(cluster_index)

print(np.unique(cluster_index, return_counts=True))

# Save cluster IDs to file
np.savetxt('{}/clusterIDs.csv'.format(fullpath), cluster_index + 1, delimiter=',')

""" Plotting """
# Plot one reconstruction example per cluster
for _ in range(3):
    for clus in range(som_shape[0] * som_shape[1]):
        idxs = np.random.choice(np.where(cluster_index == clus)[0])

        signal = dataset[idxs]
        reconsig = recons[idxs]

        plt.figure(1)
        plt.plot(signal)
        plt.plot(reconsig)
        plt.legend(('Raw', 'Reconstructed'))
        plt.title('Single VAE Reconstruction from Cluster #{} on Jan. 28 A{}'.format(clus + 1, ant1))
        plt.savefig('{}/{}signalCluster{}.png'.format(fullpath, idxs, clus + 1))
        plt.close()

        plt.figure(1)
        plt.plot(signal)
        # plt.legend(('Raw'))
        plt.title('Signal {} on January 28 A{}'.format(idxs, ant1))
        plt.savefig('{}/{}rawsignalCluster{}.png'.format(fullpath, idxs, clus + 1))
        plt.close()

        plt.figure(1)
        plt.plot(reconsig)
        # plt.legend(('Raw'))
        plt.title('Recon {} on January 28 A{}'.format(idxs, ant1))
        plt.savefig('{}/{}reconsigCluster{}.png'.format(fullpath, idxs, clus + 1))
        plt.close()

# Gets statistics to be used in multiple plots
shapes, indices, xs, ys, (maxes, maxes_std), (argmaxes, argmaxes_std), (widths, widths_std), (skews, skews_std), (mses, mses_std) \
    = cluster_statistics(fullpath, som_shape, recons, dataset, cluster_index, save=True)

combined = np.array([maxes, maxes_std, argmaxes, argmaxes_std, widths, widths_std, skews, skews_std, mses, mses_std]).T
print(combined.shape)
df = pandas.DataFrame(np.array([['total'] + [i for i in range(som_shape[0] * som_shape[1])],
                                maxes, maxes_std, argmaxes, argmaxes_std, widths, widths_std,
                                skews, skews_std, mses, mses_std]).T,
                      columns=['cluster', 'maxes', 'maxes_std', 'argmaxes', 'argmaxes_std',
                               'widths', 'widths_std', 'skews', 'skews_std', 'mses', 'mses_std'])
print(df)

df.to_csv('{}/som{}{}_cluster_values.csv'.format(fullpath, som_shape[0], som_shape[1]))
exit(0)


print(indices)

# Extract stats to latex table
metrics = [
    ['peak loc', argmaxes, argmaxes_std],
    ['peak height', maxes, maxes_std],
    ['peak width', widths, widths_std],
    ['peak skew', skews, skews_std],
    ['MSE', mses, mses_std]
]

f = open("{}/statistics_as_latex.txt".format(fullpath), 'w')
f.write(checkpath + "\n")
f.write("\\begin{table*}\n")
f.write("\t \\centering\n")
f.write("\t \\caption{SOM Clustering for [] with Antenna [].}\n")
f.write("\t \\label{tab:[]}\n")
f.write("\t \\begin{tabular}{llllll}\n")
f.write("\t \t \\hline\n")
f.write("\t \t Cluster \\#& 0 & 1 & 2 & 3 & 4 \\\\ \n")
f.write("\t \t \\hline \n")
f.write("\t \t \\# pulses & {} & {} & {} & {} & {} \\\\ \n".format(shapes[0][0], shapes[1][0],
                                                              shapes[2][0], shapes[3][0], shapes[4][0]))

for metric in metrics:
    name, metvals, metstds = metric

    if name is 'MSE':
        f.write("\t \t {} & ${:0.5f} \\pm {:0.5f}$ & ${:0.5f} \\pm {:0.5f}$ & ${:0.5f} "
                "\\pm {:0.5f}$ & ${:0.5f} \\pm {:0.5f}$ & ${:0.5f} \\pm {:0.5f}$\\\\ \n".format(
                name, metvals[0], metstds[0], metvals[1], metstds[1], metvals[2], metstds[2],
                metvals[3], metstds[3], metvals[4], metstds[4]
            ))
    else:
        f.write("\t \t {} & ${:0.2f} \\pm {:0.2f}$ & ${:0.2f} \\pm {:0.2f}$ & ${:0.2f} "
                "\\pm {:0.2f}$ & ${:0.2f} \\pm {:0.2f}$ & ${:0.2f} \\pm {:0.2f}$ \\\\ \n".format(
            name, metvals[0], metstds[0], metvals[1], metstds[1], metvals[2], metstds[2],
            metvals[3], metstds[3], metvals[4], metstds[4]
        ))

f.write("\t \t \\hline \n")
f.write("\t \\end{tabular} \n")
f.write("\end{table*} \n")


# Save each cluster index to separate file
sets = dict()
sets['raw'] = dataset
sets['reconstructions'] = recons
for i, idxs in enumerate(indices):
    sets['cluster{}indices'.format(i)] = idxs

sio.savemat('{}/{}.mat'.format(fullpath, checkpath), sets)

# Plotting the centroid signals of each node
plot = True
if plot is True:
    plot_centroids(fullpath, som, som_shape, recons, cluster_index)

# Mean plots of each cluster plotted on each other
plot = True
if plot is True:
    string = '{} {} A{}'.format(mon2.capitalize(), day2, ant2) if mon2 is 'march' else \
        'January {} A{}'.format(day2, ant2)
    plot_means(fullpath, som_shape, recons, dataset, cluster_index, string)

# Plotting the raw and recon mean of each cluster over each other
plot = True
if plot is True:
    plot_mean_comparison(fullpath, checkpath, som_shape, recons, dataset, cluster_index)

# Grid graph of the physical features using colormap
plot = False
if plot is True:
    plot_gridgraph(fullpath, som_shape, xs, ys, maxes, argmaxes, widths, skews, mses)
