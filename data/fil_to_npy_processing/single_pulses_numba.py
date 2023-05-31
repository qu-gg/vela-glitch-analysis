import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from period import findP
from tqdm import tqdm
import glob

np.set_printoptions(threshold=25)


def single_pulses_numba(n_files, n_pulses, polycos, bestprof, MJD0, n_bins):
    ''' This funtion converts the output files from waterfall.py into
    a single csv files with all the single pulses (row = one pulse). 
    Also plots the folded pulse to check.
    
    Args:
    n_files: (int) number of imput files counting from 0
    n_pulses: (int) how many single pulses you want
    period: (float) period of the pulse in seconds obtained from presto
    n_bins: (int) how many points per single pulse. default: 1220
    
    Returns: nothing
    
    '''

    # times array
    file_name_1 = 'jan21/times_{}.csv'
    times_array = glob.glob("jan21/times*csv")
    print("Number of Times Files:")
    print(len(times_array))
    df_list_1 = []
    for i in tqdm(range(len(times_array))):
        df_list_1.append(pd.read_csv(file_name_1.format(i), header=None))
    times = pd.concat(df_list_1).values.T.flatten()

    # intensity array
    file_name_2 = 'jan21/original_{}.csv'
    originals_array = glob.glob("jan21/original*csv")
    df_list_2 = []
    for i in tqdm(range(len(originals_array))):
        df_list_2.append(pd.read_csv(file_name_2.format(i), header=None))
    originals = pd.concat(df_list_2).values.T.flatten()

    # find the MJD0
    MJD0_bestprof = np.genfromtxt(bestprof, comments="none", dtype=float, skip_header=3, max_rows=1, usecols=(3))

    # create a new vector of times to match the period:
    print('In single_pulses.py')
    print('MJD0_header = ' + str(MJD0))
    print('MJD0_bestpr = ' + str(MJD0_bestprof))
    print('n_bins = ' + str(n_bins))
    print('n_pulses = ' + str(n_pulses))
    new_times = np.zeros(n_bins * n_pulses)  # vector with (corrected) time in seconds since the start of the observation
    new_times[0] = times[0]
    MJD_times = np.zeros(n_pulses)  # array with the MJD at the beggining of each individual pulse
    periods = np.zeros(n_pulses)  # array with the instantaneous period of each individual pulse

    for n in tqdm(range(0, n_pulses)):
        temp = new_times[n * n_bins]  # we find the time (in seconds since the beggining of the observation) at the beggining of each pulse
        period = findP(polycos, MJD0, temp)  # we find the instantaneous period corresponding to that single pulse
        new_dt = period / n_bins  # we divide the period of that single pulse into bins

        for i in range(0, n_bins):  # for each bin in that single pulse
            if n * n_bins + i + 1 == n_bins * n_pulses:
                continue  # to avoid going out of index range

            new_times[n * n_bins + i + 1] = new_times[n * n_bins + i] + new_dt  # we save the corrected time for that bin

        MJD_times[n] = MJD0 + temp * 1. / (24. * 60. * 60.)  # guille test
        periods[n] = period  # guille test

    # Interpolation
    new_data = np.reshape(np.interp(new_times, times, originals), [n_pulses, n_bins])

    plt.plot(np.mean(new_data, axis=0))
    plt.show()
    plt.close()

    # Write table
    obs_data = bestprof[bestprof.find('_A') + 1:bestprof.find('.pfd')]  # get the antenna and date
    output_csv = "sp_" + obs_data + ".csv"  # name of the output .csv

    np.savetxt(output_csv, new_data, delimiter=',')  # save as ascii file
    np.save(output_csv.replace(".csv", ".npy"), new_data)  # save as binary file
    np.savetxt("sp_MJD_" + obs_data + ".csv", np.column_stack((MJD_times, periods)), delimiter=' ')
