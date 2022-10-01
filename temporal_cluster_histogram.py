"""
@file temporaL_cluster_histogram.py
@author Ryan Missel

Handles chunking the time windows of a day of signals into bins
and then plotting their per-bin cluster histograms
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# Setting plt font sizes
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

""" Arg parsing and Data setup"""
# Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('--mon', type=str, default='july', help='month trained on ')
parser.add_argument('--day', type=int, default=21, help='day trained on')
parser.add_argument('--ant', type=str, default="A1", help='antenna trained on ')
parser.add_argument('--mode', type=str, default='vae', help='which model type to use - VAE or DAE')
parser.add_argument('--som_one', type=int, default=2, help='shape of som dimension 1')
parser.add_argument('--som_two', type=int, default=3, help='shape of som dimension 1')
parser.add_argument('--reverse', type=int, default=0, help='whether to reverse the indices of the clusters')
args = parser.parse_args()

# Load in cluster IDS
clusters = pd.read_csv(f"graphs/- {args.mon}{args.day}{args.ant}/{args.mode}_{args.som_one}_{args.som_two}/clusterIDs.csv", header=None).to_numpy()

# Reverse indices if flagged
if args.reverse == 1:
    clusters = clusters[::-1]

# Get the number of clusters and plot out number of samples per cluster
num_clusters = np.unique(clusters).shape[0]
print(f"=>CLuster shape: {clusters.shape}")
print(f"=>Cluster counts: {np.unique(clusters, return_counts=True)}")

# Define color array for plotting
colors = ['purple', 'orange', 'blue', 'green', 'red', 'yellow', 'black', 'cyan', 'm']

""" Histgram plotting """
# Split into chunks
window_size = 5000
chunks = np.array_split(clusters, indices_or_sections=clusters.shape[0] // window_size)
print(chunks[0].shape)

# For each cluster, put in the chunk count
tags = []
splits = [[] for _ in range(num_clusters)]
for idx, chunk in enumerate(chunks):
    ids, unq = np.unique(chunk, return_counts=True)
    for uidx in range(len(unq)):
        splits[int(ids[uidx]) - 1].append(unq[uidx])

    for cidx in range(1, num_clusters + 1):
        if float(cidx) not in ids:
            splits[cidx - 1].append(0)

    tags.append(f"{(idx + 1) * window_size}")

# Create a dataframe of each chunk
cdict = {}
for idx, sp in enumerate(splits):
    cdict[str(idx)] = sp

df = pd.DataFrame(cdict, index=tags)

# Plot a hisotgram of each chunk over time
df.plot.bar(rot=0, figsize=(20, 7), width=0.90, color=colors, edgecolor='black')
plt.title(f"Time-Window Cluster Number Analysis for {args.mon.title()} {args.day}",
          fontdict={'fontsize': 25, 'fontweight': 5})
plt.xlabel("Signal IDX over time", fontdict={'fontsize': 20, 'fontweight': 5})
plt.ylabel("Number of signals in cluster", fontdict={'fontsize': 20, 'fontweight': 5})
plt.legend([f"Cluster {i + 1}" for i in range(num_clusters)], loc='upper right')
plt.tight_layout()
plt.savefig(f"graphs/- {args.mon}{args.day}{args.ant}/{args.mode}_{args.som_one}_{args.som_two}/histogram_clusters_{window_size}_reverse{args.reverse}.png")

""" Frequency scatter plot """
# Build density color list for each point
window_size = 5000

# for every signal, get the number of neighbhors within the window size
density = []
for idx, point in tqdm(enumerate(clusters)):
    neighbors = clusters[max(0, idx - window_size):min(len(clusters), idx + window_size)]

    d = 0
    for n in neighbors:
        if point == n:
            d += 1
    density.append(d)

# Plot a scatter plot of frequency
plt.figure(figsize=(20, 7))
plt.scatter(range(clusters.shape[0]), clusters, c=density, cmap="hot")
plt.title(f"Frequency Plot of Clusters for {args.mon.title()} {args.day} on {args.som_one * args.som_two} clusters",
          fontdict={'fontsize': 25, 'fontweight': 3})
plt.yticks(range(1, num_clusters + 1))
plt.colorbar()
plt.xlabel("Signal IDX over time", fontdict={'fontsize': 20, 'fontweight': 3})
plt.ylabel("Cluster ID", fontdict={'fontsize': 20, 'fontweight': 3})
plt.savefig(f"graphs/- {args.mon}{args.day}{args.ant}/{args.mode}_{args.som_one}_{args.som_two}/frequency_density_plot_window{window_size}_reverse{args.reverse}.png")
