"""
@file raw_som_reconstruct.py
@author Ryan Missel

Handles performing the SOM task on the raw signals rather than reconstructions for comparison
"""
import os
import torch
import pandas
import argparse
import scipy.io as sio

from minisom import MiniSom
from util.plot_functions import *
from util.util_functions import *

""" Arg parsing and Data setup"""
# Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--device', '-g', type=int, default=0, help='which GPU to run on')
parser.add_argument('--batch', '-b', type=int, default=128, help='size of batch')
parser.add_argument('--center', '-c', type=int, default=0, help='size of batch')

parser.add_argument('--mon1', type=str, default='july', help='size of batch')
parser.add_argument('--day1', type=int, default=22, help='size of batch')
parser.add_argument('--ant1', type=str, default="A23", help='size of batch')

parser.add_argument('--mon2', type=str, default='july', help='size of batch')
parser.add_argument('--day2', type=int, default=22, help='size of batch')
parser.add_argument('--ant2', type=str, default="A23", help='size of batch')

parser.add_argument('--som1', type=int, default=2, help='size of batch')
parser.add_argument('--som2', type=int, default=2, help='size of batch')

args = parser.parse_args()

# Check whether the days are a cross or not
if '{}{}{}'.format(args.mon1, args.day1, args.ant1) != '{}{}{}'.format(args.mon2, args.day2, args.ant2):
    cross = True
    print("Crossed")
else:
    cross = False

# Which model is used - DAE or VAE
modeltype = 'vae'

# Build checkpt and data paths
checkpt = '{}{}{}'.format(args.mon1, args.day1, args.ant1)
checkpath = checkpt.split('/')[-1].split('.')[0]

dataset = '{}{}/{}'.format(args.mon2, args.day2, args.ant2)
datapath = dataset.split('/')[0]
print(checkpath, datapath)

# Build antenna title if there is a split
if len(args.ant2[1:]) > 1:
    antenna_string = f"A{args.ant2[1]} Split {args.ant2[2:]}"
else:
    antenna_string = args.ant1

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

# Get the centered window of the signal
dataset = get_window(dataset)

# Full path for graph output
if cross is True:
    fullpath = 'graphs/- {}_{}/raw_{}_{}/'.format(checkpath, datapath, som_shape[0], som_shape[1], modeltype)
else:
    fullpath = 'graphs/- {}/raw_{}_{}/'.format(checkpath, som_shape[0], som_shape[1], modeltype)
print(f"=>Graph output path: {fullpath}")

# Setting up folders for graph saving
if not os.path.exists(fullpath):
    os.makedirs(fullpath)

# Save the raw data to the relevant folder
np.save(f"{fullpath}/raw.npy", dataset)

# Train SOM
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=dataset.shape[1], sigma=0.3,
              learning_rate=0.5, random_seed=args.seed)

som.train(dataset, 10000, verbose=False)

# Extract coordinates for each cluster and assignment
winner_coordinates = np.array([som.winner(x) for x in dataset]).T
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

""" Plotting """
if not os.path.exists('{}/examples/'.format(fullpath)):
    os.mkdir('{}/examples/'.format(fullpath))

# Gets statistics to be used in multiple plots
shapes, indices, xs, ys, (maxes, maxes_std), (argmaxes, argmaxes_std), (widths, widths_std), (skews, skews_std), (mses, mses_std) \
    = cluster_statistics(fullpath, som_shape, dataset, dataset, cluster_index, save=True)

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
f.write("\t \\caption{SOM Clustering for " + f"{args.mon1.title()} {args.day1} " + "with Antenna " + f"{antenna_string}" + ".}\n")
f.write("\t \\label{tab:[]}\n")
f.write("\t \\begin{tabular}{llllll}\n")
f.write("\t \t \\hline\n")
f.write("\t \t Cluster \\#& 0 & 1 & 2 & 3 & 4 \\\\ \n")
f.write("\t \t \\hline \n")
f.write("\t \t \\# pulses & {} & {} & {} & {} & {} \\\\ \n".format(shapes[0][0], shapes[1][0],
                                                              shapes[2][0], shapes[3][0], shapes[4][0]))

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
sets['reconstructions'] = dataset
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
        reconsig = dataset[idxs]

        plt.figure(1)
        plt.plot(signal)
        plt.plot(reconsig)
        plt.legend(('Raw', 'Reconstructed'))
        plt.title('Single VAE Reconstruction from Cluster #{} on {} {} {}'.format(clus + 1, args.mon1, args.day1, antenna_string))
        plt.savefig('{}/examples/{}signalCluster{}.png'.format(fullpath, idxs, clus + 1))
        plt.close()

# Plotting the centroid signals of each node
plot = True
if plot is True:
    plot_centroids(fullpath, som, som_shape, dataset, cluster_index)

# Mean plots of each cluster plotted on each other
plot = True
if plot is True:
    string = '{} {} {} (Centered)'.format(args.mon2, args.day2, antenna_string)
    plot_means(fullpath, som_shape, dataset, dataset, cluster_index, string)

# Mean plots of each cluster plotted on each other *scaled to a common vertical axis*
plot = True
if plot is True:
    string = '{} {} {} (Centered)'.format(args.mon2, args.day2, antenna_string)
    plot_means(fullpath, som_shape, dataset, dataset, cluster_index, string, vertical_fix=True)

# Plotting the raw and recon mean of each cluster over each other
plot = True
if plot is True:
    plot_mean_comparison(fullpath, checkpath, som_shape, dataset, dataset, cluster_index)

# Grid graph of the physical features using colormap
plot = False
if plot is True:
    plot_gridgraph(fullpath, som_shape, xs, ys, maxes, argmaxes, widths, skews, mses)
