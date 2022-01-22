"""
@file plot_functions.py
@author Ryan Missel

Holds general plotting and saving functions for the SOM clustering
"""
import numpy as np
import scipy.signal as sig
from scipy.stats import skew
import matplotlib.pyplot as plt


def cluster_statistics(outpath, som_shape, recons, dataset, cluster_index, save=True):
    """
    Handles getting the statistics per cluster and saving it to a text file
    :param outpath: folder to save statistics.txt
    :param som_shape: shape of the SOM mmap
    :param recons: reconstructed dataset
    :param dataset: raw dataset
    :param cluster_index: clusters by SOM
    :param save: whether to write out the statistics file
    :returns: arrays of statistics found
    """
    shapes, indices = [], []
    maxes, maxes_std = [], []
    argmaxes, argmaxes_std = [], []
    widths, widths_std = [], []
    skews, skews_std = [], []
    mses, mses_std = [], []
    xs, ys = [], []

    # Gets global statistics on the full set
    # Get metrics for this subset
    maxs = np.max(recons, axis=1)
    amaxs = np.argmax(recons, axis=1)
    sk = skew(recons, axis=1)
    mse = (recons - dataset) ** 2 / recons.shape[0]
    ws = []

    # Get peak width at half maximum calculation
    for subsig in recons:
        peaks, _ = sig.find_peaks(subsig, height=max(recons[0]))
        results_half = sig.peak_widths(subsig, peaks, rel_height=0.5)

        if len(results_half[0]) == 0:
            continue
        elif len(results_half[0]) == 1:
            ws.append(results_half[0][0])
        else:
            ws.append(max(results_half[0]))

    # Append means of subset metrics to arrays
    widths.append(np.mean(ws))
    argmaxes.append(np.mean(amaxs))
    maxes.append(np.mean(maxs))
    skews.append(np.mean(sk))
    mses.append(np.mean(mse))

    # Append std of subset metrics to arrays
    widths_std.append(np.std(ws))
    argmaxes_std.append(np.std(amaxs))
    maxes_std.append(np.std(maxs))
    skews_std.append(np.std(sk))
    mses_std.append(np.std(mse))

    shapes.append(cluster_index.shape)
    indices.append(cluster_index)

    # Get cluster-specific metrics
    for idx in np.unique(cluster_index):
        # Extract subset for both reconstructions and the raw signals
        subset = recons[cluster_index == idx]
        rawsub = dataset[cluster_index == idx]

        # Get metrics for this subset
        maxs = np.max(subset, axis=1)
        amaxs = np.argmax(subset, axis=1)
        sk = skew(subset, axis=1)
        mse = (subset - rawsub) ** 2 / subset.shape[0]
        ws = []

        # Get peak width at half maximum calculation
        for subsig in subset:
            peaks, _ = sig.find_peaks(subsig, height=max(subset[0]))
            results_half = sig.peak_widths(subsig, peaks, rel_height=0.5)

            if len(results_half[0]) == 0:
                continue
            elif len(results_half[0]) == 1:
                ws.append(results_half[0][0])
            else:
                ws.append(max(results_half[0]))

        # Append means of subset metrics to arrays
        widths.append(np.mean(ws))
        argmaxes.append(np.mean(amaxs))
        maxes.append(np.mean(maxs))
        skews.append(np.mean(sk))
        mses.append(np.mean(mse))

        # Append std of subset metrics to arrays
        widths_std.append(np.std(ws))
        argmaxes_std.append(np.std(amaxs))
        maxes_std.append(np.std(maxs))
        skews_std.append(np.std(sk))
        mses_std.append(np.std(mse))

        # Append x/y cluster indices
        shapes.append(np.where(cluster_index == idx)[0].shape)
        indices.append(np.where(cluster_index == idx)[0])
        # xs.append(x)
        # ys.append(y)

    # for x in range(som_shape[0]):
    #     for y in range(som_shape[1]):
    #         # Extract subset for both reconstructions and the raw signals
    #         subset = recons[cluster_index == (x * som_shape[0] + y)]
    #         rawsub = dataset[cluster_index == (x * som_shape[0] + y)]
    #
    #         # Get metrics for this subset
    #         maxs = np.max(subset, axis=1)
    #         amaxs = np.argmax(subset, axis=1)
    #         sk = skew(subset, axis=1)
    #         mse = (subset - rawsub) ** 2 / subset.shape[0]
    #         ws = []
    #
    #         # Get peak width at half maximum calculation
    #         for subsig in subset:
    #             peaks, _ = sig.find_peaks(subsig, height=max(subset[0]))
    #             results_half = sig.peak_widths(subsig, peaks, rel_height=0.5)
    #
    #             if len(results_half[0]) == 0:
    #                 continue
    #             elif len(results_half[0]) == 1:
    #                 ws.append(results_half[0][0])
    #             else:
    #                 ws.append(max(results_half[0]))
    #
    #         # Append means of subset metrics to arrays
    #         widths.append(np.mean(ws))
    #         argmaxes.append(np.mean(amaxs))
    #         maxes.append(np.mean(maxs))
    #         skews.append(np.mean(sk))
    #         mses.append(np.mean(mse))
    #
    #         # Append std of subset metrics to arrays
    #         widths_std.append(np.std(ws))
    #         argmaxes_std.append(np.std(amaxs))
    #         maxes_std.append(np.std(maxs))
    #         skews_std.append(np.std(sk))
    #         mses_std.append(np.std(mse))
    #
    #         # Append x/y cluster indices
    #         shapes.append(np.where(cluster_index == (x * som_shape[0] + y))[0].shape)
    #         indices.append(np.where(cluster_index == (x * som_shape[0] + y))[0])
    #         xs.append(x)
    #         ys.append(y)

    # Open statistics file to write to
    if save is True:
        f = open("{}/statistics.txt".format(outpath), 'w')
        f.write("Cluster counts: {}\n\n".format(np.unique(cluster_index, return_counts=True)))

        for i, c in enumerate(np.unique(cluster_index)):
            subset = recons[cluster_index == c]
            f.write("Cluster {} Stats - {}\n".format(i, subset.shape[0]))
            f.write("Average peak loc: {:0.2f} +- {:0.2f}\n".format(argmaxes[c], argmaxes_std[c]))
            f.write("Average peak height: {:0.2f} +- {:0.2f}\n".format(maxes[c], maxes_std[c]))
            f.write("Average peak width: {:0.2f} +- {:0.2f}\n".format(widths[c], widths_std[c]))
            f.write("Average peak skew: {:0.2f} +- {:0.2f}\n".format(skews[c], skews_std[c]))
            f.write("Average MSE: {:0.5f} +- {:0.5f}\n".format(mses[c], mses_std[c]))
            f.write("\n")

        f.close()

    return shapes, indices, xs, ys, (maxes, maxes_std), (argmaxes, argmaxes_std), (widths, widths_std), (skews, skews_std), (mses, mses_std)


def plot_centroids(outpath, som, som_shape, recons, cluster_index):
    """
    Plots the centroid signal of each cluster in the reconstruction
    :param outpath: path to save file
    :param som: trained SOM object
    :param som_shape: shape of the SOM
    :param recons: reconstructed dataset
    :param cluster_index: cluster assignments
    """
    plt.figure(1)
    print("Som weight shape: ", np.reshape(som.get_weights(), [-1, recons.shape[1]]).shape)
    for weight in np.reshape(som.get_weights(), [-1, recons.shape[1]]):
        plt.plot(weight)

    if som_shape[0] * som_shape[1] < 10:
        plt.legend(["Cent {}".format(c) for c in zip(np.unique(cluster_index))])
    plt.title("Signals of each Centroid")
    plt.savefig("{}/centroidsignals.png".format(outpath))
    plt.close()


def plot_set_mean(outpath, recons, dataset):
    """
    Plots the global set mean between the raw data and the reconstructions
    :param outpath: path to save, which is then stepped up from
    :param recons: reconstructed dataset
    :param dataset: raw dataset
    :return: none
    """
    plt.plot(np.mean(recons, axis=0))
    plt.plot(np.mean(dataset, axis=0))
    plt.title("Mean Signals")
    plt.legend(('recon', 'data'))
    plt.savefig('{}/../meanSets.png'.format(outpath))
    plt.close()


def plot_means(outpath, som_shape, recons, dataset, cluster_index, name):
    """
    Plots an aggregate of the mean cluster signals over each other
    :param outpath: path to save file
    :param som_shape: shape of the SOM
    :param recons: reconstructed dataset
    :param dataset: raw dataset
    :param cluster_index: cluster assignments
    :return: none
    """
    plt.figure(1)
    subsetsizes = []
    plotset = []
    legend = []

    colors = ['purple', 'orange', 'blue', 'green', 'red', 'yellow', 'black']

    plt.plot(np.mean(recons, axis=0), linestyle='dashed', color=colors[-1])
    legend.append("Cluster 0 - {}".format(recons.shape[0]))

    for i, c in enumerate(np.unique(cluster_index)):
        subset = recons[cluster_index == c]
        subsetsizes.append(subset.shape[0])

        plotset.append((dataset[cluster_index == c][0],
                        recons[cluster_index == c][0]))

        plt.plot(np.mean(subset, axis=0), color=colors[i])

    if som_shape[0] * som_shape[1] < 10:
        for c, s in zip(np.unique(cluster_index), subsetsizes):
            legend.append("Cluster {} - {}".format(c + 1, s))

    plt.legend(legend)
    plt.title("Mean Cluster Reconstruction for {}".format(name))
    plt.savefig("{}/meansignals.png".format(outpath))
    plt.close()


def plot_mean_comparison(outpath, checkpath, som_shape, recons, dataset, cluster_index):
    """

    :param outpath:
    :param checkpath:
    :param som_shape:
    :param recons:
    :param dataset:
    :param cluster_index:
    :return:
    """
    plt.figure(1, figsize=(15, 15))
    fig, axs = plt.subplots(som_shape[0], som_shape[1])
    fig.tight_layout(h_pad=2)
    fig.suptitle("Mean Sig Comparison for {}".format(checkpath))
    fig.subplots_adjust(top=0.85)

    # Iterate over each cluster, plotting the mean of the raw and recon signal
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            # Get 1-D index of the 2-D shape
            c = i * som_shape[0] + j + i

            # Extract and plot mean signals of this cluster to the relevant subplot
            raw_subset = dataset[cluster_index == c]
            recon_subset = recons[cluster_index == c]

            # print(np.where(cluster_index == c))
            # print(raw_subset.shape)
            #
            # plt.figure(5)
            # plt.plot(raw_subset[0])
            # plt.show()
            # plt.close()

            axs[i, j].plot(np.mean(recon_subset, axis=0))
            axs[i, j].plot(np.mean(raw_subset, axis=0))
            axs[i, j].set_title('Cluster {}'.format(c, raw_subset.shape[0]))

    plt.savefig("{}/meanSigComp.png".format(outpath))
    plt.close()


def plot_gridgraph(outpath, som_shape, xs, ys, maxes, argmaxes, widths, skews, mses):
    plt.figure(2)
    plt.ylim([som_shape[0] - 0.5, -0.5])
    plt.xlim([-0.5, som_shape[1] - 0.5])
    plt.title('SOM Graph of Amplitude')
    plt.scatter(ys, xs, c=maxes, cmap='viridis', s=300)
    plt.colorbar()
    plt.savefig("{}/SOMamplitudes.png".format(outpath))
    plt.close()

    plt.figure(3)
    plt.ylim([som_shape[0] - 0.5, -0.5])
    plt.xlim([-0.5, som_shape[1] - 0.5])
    plt.title('SOM Graph of Width')
    plt.scatter(ys, xs, c=widths, cmap='viridis', s=300)
    plt.colorbar()
    plt.savefig("{}/SOMwidths.png".format(outpath))
    plt.close()

    plt.figure(4)
    plt.ylim([som_shape[0] - 0.5, -0.5])
    plt.xlim([-0.5, som_shape[1] - 0.5])
    plt.title('SOM Graph of Peak Location')
    plt.scatter(ys, xs, c=argmaxes, cmap='viridis', s=300)
    plt.colorbar()
    plt.savefig("{}/SOMpeaklocs.png".format(outpath))
    plt.close()

    plt.figure(5)
    plt.ylim([som_shape[0] - 0.5, -0.5])
    plt.xlim([-0.5, som_shape[1] - 0.5])
    plt.title('SOM Graph of Skew')
    plt.scatter(ys, xs, c=skews, cmap='viridis', s=300)
    plt.colorbar()
    plt.savefig("{}/SOMskews.png".format(outpath))
    plt.close()

    plt.figure(6)
    plt.ylim([som_shape[0] - 0.5, -0.5])
    plt.xlim([-0.5, som_shape[1] - 0.5])
    plt.title('SOM Graph of MSE')
    plt.scatter(ys, xs, c=mses, cmap='viridis', s=300)
    plt.colorbar()
    plt.savefig("{}/SOMmses.png".format(outpath))
    plt.close()
