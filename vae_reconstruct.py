"""
@file vae_reconstruct.py
@author Ryan Missel

Handles first loading in the VAE model and reconstructing the given dataset
Next it fits an SOM to the reconstructions and builds up some plots based on the clustering formed
"""
import os
import torch
import pandas
import argparse
import scipy.io as sio

from minisom import MiniSom
from vae_model import VAE, DAE
from util.plot_functions import *
from util.util_functions import *


""" Arg parsing and Data setup"""
# Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=255, help='random seed')
parser.add_argument('--device', '-g', type=int, default=0, help='which GPU to run on')
parser.add_argument('--batch', '-b', type=int, default=128, help='size of batch')
parser.add_argument('--center', '-c', type=int, default=0, help='whether to load in the centered data')

parser.add_argument('--mon1', type=str, default='july', help='month trained on')
parser.add_argument('--day1', type=int, default=22, help='day trained on')
parser.add_argument('--ant1', type=str, default="A23", help='antenna trained on')

parser.add_argument('--mon2', type=str, default='july', help='month to test on')
parser.add_argument('--day2', type=int, default=22, help='day to test on')
parser.add_argument('--ant2', type=str, default="A23", help='antenna to test on')

parser.add_argument('--som1', type=int, default=2, help='size of the som')
parser.add_argument('--som2', type=int, default=3, help='size of the som')
args = parser.parse_args()

# Check whether the days are a cross between days or not
if '{}{}{}'.format(args.mon1, args.day1, args.ant1) != '{}{}{}'.format(args.mon2, args.day2, args.ant2):
    cross = True
    print(f"=> Model trained on {args.mon1}{args.day1}{args.ant1}, testing on {args.mon2}{args.day2}{args.ant2}")
else:
    cross = False

# Which model is used - DAE or VAE
modeltype = 'vae'

# Build checkpt and data paths
checkpt = '{}{}{}'.format(args.mon1, args.day1, args.ant1)
checkpath = checkpt.split('/')[-1].split('.')[0]

dataset = '{}{}/{}'.format(args.mon2, args.day2, args.ant2)
datapath = dataset.split('/')[0]
print(f"=> Checkpoint path: {checkpath}")
print(f"=> Dataset path: {datapath}")

# Build SOM Shape
som_shape = (args.som1, args.som2)

# Set seed if given
np.random.seed(123234)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.device)

# Load in the given dataset
if args.center == 0:
    dataset = np.load("data/{}.npy".format(dataset), allow_pickle=True)
else:
    dataset = np.load("data/{}center.npy".format(dataset), allow_pickle=True)

# Day-specific signal dropping that were noise
if args.day1 == 21:
    print(f"=> Dataset shape: {dataset.shape}")
    to_drop = [3780,  3798,  4834,  4867,  6482,  6483,  6485,  6489,
               6494, 6503,  6509,  6515,  6516, 18895, 28812, 30289]
    dataset = np.delete(dataset, to_drop, axis=0)
    print(f"=> Dataset shape: {dataset.shape}")

# Get the centered window of the signal
dataset = get_window(dataset)

# Transform datasets into loaders
kwargs = {'num_workers': 0, 'pin_memory': True}
testloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, **kwargs)

# Build the model and load in a checkpoint if given
model = VAE(device=device)
model.load_state_dict(torch.load('models/{}_{}.torch'.format(checkpt, modeltype), map_location=device))
model = model.to(device)


""" Evaluation """
# Reconstruct the entire dataset
recons = None
for idx, signals in enumerate(testloader):
    signals = signals.to(device).float()

    recon_signals, latent_z, _, _, _ = model(signals)
    recon_signals = recon_signals.detach().cpu().numpy()

    if recons is None:
        recons = recon_signals
    else:
        recons = np.vstack((recons, recon_signals))

# Full path for graph output
if cross is True:
    fullpath = 'graphs/- {}_{}/{}_{}_{}/'.format(checkpath, datapath, modeltype, som_shape[0], som_shape[1])
else:
    fullpath = 'graphs/- {}/{}_{}_{}/'.format(checkpath, modeltype, som_shape[0], som_shape[1])
print(f"=>Graph output path: {fullpath}")

# Setting up folders for graph saving
if not os.path.exists(fullpath):
    os.makedirs(fullpath)

# Handles plotting the mean signals of the raw data and the reconstruction
plot_set_mean(fullpath, recons, dataset)

# Save the reconstructions to the relevant folder
np.save(f"{fullpath}/recons.npy", recons)
np.save(f"{fullpath}/raw.npy", dataset)

# Train SOM
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=dataset.shape[1], sigma=0.3,
              learning_rate=0.5, random_seed=args.seed)

som.train(recons, 10000, verbose=False)

# Extract coordinates for each cluster and assignment
winner_coordinates = np.array([som.winner(x) for x in recons]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

# Get maxes across clusters
maxes = get_maxes(dataset, cluster_index, som_shape)
resort = list(np.array(maxes).argsort()[::-1])

# Remap to sort via peak height
map_dict = {float(k): float(v) for k, v in zip(resort, range(som_shape[0] * som_shape[1]))}
cluster_index = vector_map(cluster_index, map_dict)
cluster_index = cluster_index.astype(np.int64)

# Save cluster IDs to file
np.savetxt('{}/clusterIDs.csv'.format(fullpath), cluster_index + 1, delimiter=',')

""" Statistics """
# Gets statistics to be used in multiple plots
shapes, indices, xs, ys, (maxes, maxes_std), (argmaxes, argmaxes_std), (widths, widths_std), (skews, skews_std), (mses, mses_std) \
    = cluster_statistics(fullpath, som_shape, recons, dataset, cluster_index, save=True)

# Combined and save as a CSV over all
combined = np.array([maxes, maxes_std, argmaxes, argmaxes_std, widths, widths_std, skews, skews_std, mses, mses_std]).T
df = pandas.DataFrame(np.array([['total'] + [i for i in range(som_shape[0] * som_shape[1])],
                                maxes, maxes_std, argmaxes, argmaxes_std, widths, widths_std,
                                skews, skews_std, mses, mses_std]).T,
                      columns=['cluster', 'maxes', 'maxes_std', 'argmaxes', 'argmaxes_std',
                               'widths', 'widths_std', 'skews', 'skews_std', 'mses', 'mses_std'])
df.to_csv('{}/som{}{}_cluster_values.csv'.format(fullpath, som_shape[0], som_shape[1]))

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
f.write("\t \\caption{SOM Clustering for " + f"{args.mon1.title()} {args.day1} " + "with Antenna " + f"{args.ant1}" + ".}\n")
f.write("\t \\label{tab:[]}\n")
f.write("\t \\begin{tabular}{cclllll}\n")
f.write("\t \t \\hline\n")
f.write("\t \t Cluster \\# & \\# Pulses & Peak Loc & Peak Height & Peak Width & Peak Skew & MSE \\\\ \n")
f.write("\t \t \\hline \n")

for cidx, pulses in zip(range(1 + (som_shape[0] * som_shape[1])), shapes):
    num_pulses = pulses[0]
    peak_loc_mean, peak_loc_std = metrics[0][1][cidx], metrics[0][2][cidx]
    peak_height_mean, peak_height_std = metrics[1][1][cidx], metrics[1][2][cidx]
    peak_width_mean, peak_width_std = metrics[2][1][cidx], metrics[2][2][cidx]
    peak_skew_mean, peak_skew_std = metrics[3][1][cidx], metrics[3][2][cidx]
    mse_mean, mse_std = metrics[4][1][cidx], metrics[4][2][cidx]

    f.write(
        "\t \t {} & {} & ${:0.2f} \\pm {:0.2f}$ & ${:0.2f} \\pm {:0.2f}$ & ${:0.2f} \\pm {:0.2f}$ & ${:0.2f} \\pm {:0.2f}$ & ${:0.5f} \\pm {:0.5f}$  \\\\ \n"
            .format(cidx, num_pulses, peak_loc_mean, peak_loc_std, peak_height_mean,peak_height_std,
                    peak_width_mean, peak_width_std, peak_skew_mean, peak_skew_std, mse_mean, mse_std)
    )

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

""" Plotting """
if not os.path.exists('{}/examples/'.format(fullpath)):
    os.mkdir('{}/examples/'.format(fullpath))

# Plot one reconstruction example per cluster
for _ in range(3):
    for clus in range(som_shape[0] * som_shape[1]):
        if len(np.where(cluster_index == clus)[0]) == 0:
            continue

        idxs = np.random.choice(np.where(cluster_index == clus)[0])
        signal = dataset[idxs]
        reconsig = recons[idxs]

        plt.figure(1)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.plot(signal)
        plt.plot(reconsig)
        plt.legend(('Raw', 'Reconstructed'), fontsize=11)
        plt.title('Cluster #{} Single Reconstruction on {} {} {}'.format(clus + 1, args.mon1.title(), args.day1, args.ant1), fontdict={'fontsize': 14, 'fontweight': 3})
        plt.savefig('{}/examples/{}signalCluster{}.png'.format(fullpath, idxs, clus + 1))
        plt.close()

# Plotting the centroid signals of each node
plot = True
if plot is True:
    plot_centroids(fullpath, som, som_shape, recons, cluster_index)

# Mean plots of each cluster plotted on each other
plot = True
if plot is True:
    string = '{} {} {}'.format(args.mon2.title(), args.day2, args.ant1)
    plot_means(fullpath, som_shape, recons, dataset, cluster_index, string)

# Mean plots of each cluster plotted on each other *scaled to a common vertical axis*
plot = True
if plot is True:
    string = '{} {} {}'.format(args.mon2.title(), args.day2, args.ant1)
    plot_means(fullpath, som_shape, recons, dataset, cluster_index, string, vertical_fix=True)

# Plotting the raw and recon mean of each cluster over each other
plot = True
if plot is True:
    plot_mean_comparison(fullpath, checkpath, som_shape, recons, dataset, cluster_index)

# Grid graph of the physical features using colormap
plot = False
if plot is True:
    plot_gridgraph(fullpath, som_shape, xs, ys, maxes, argmaxes, widths, skews, mses)
