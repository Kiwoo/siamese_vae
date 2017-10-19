"""
Attempting to make this return a similar plot as in the GAIL paper, Figure 1,
and also to return a table with results. You need to supply a results file.
Example:

    python scripts/plot_results.py imitation_runs/classic/checkpoints/results.h5

Note that this will (unless we're using Humanoid) contain more than one task for
us to parse through.

(c) June 2017 by Daniel Seita
"""

import argparse
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from misc_util import header

# Some matplotilb options I like to use
plt.style.use('seaborn-darkgrid')
FIGDIR = 'figures/'
title_size = 22
tick_size = 18
legend_size = 17
ysize = 18
xsize = 18
lw = 3
ms = 12
mew = 5
error_region_alpha = 0.25

# Meh, a bit of manual stuff.

colors = {'red', 'blue', 'yellow', 'black'}


def main():
    """ Here, `resultfile` should be an `.h5` file with all results from a
    category of trials, e.g. classic imitation runs. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('result_h5file', type=str)
    args = parser.parse_args()

    with pd.HDFStore(args.result_h5file, 'r') as f:
        iter_log = f['iter_log'].values
        loss1_log = f['loss1_log'].values
        loss2_log = f['loss2_log'].values
        loss3_log = f['loss3_log'].values

        # print ret_mean_log

        iter_log = np.squeeze(iter_log)
        loss1_log = np.squeeze(loss1_log)
        loss2_log = np.squeeze(loss2_log)
        loss3_log = np.squeeze(loss3_log)
        c = 'red'
        header("Check Dimension")
        header('iter_log : {}'.format(np.shape(iter_log)))
        header('loss1_log : {}'.format(np.shape(loss1_log)))

        fig = plt.figure(figsize=(10,8))
        plt.plot(iter_log, loss1_log, '-', lw=lw, color=c,
                        markersize=ms, mew=mew, label="loss1_log")
        plt.title("Feature_Match_Loss", fontsize=title_size)
        plt.xlabel("Number of Iteration", fontsize=ysize)
        plt.ylabel("Loss1", fontsize=xsize)
        plt.legend(loc='lower right', ncol=2, prop={'size':legend_size})
        plt.tick_params(axis='x', labelsize=tick_size)
        plt.tick_params(axis='y', labelsize=tick_size)
        plt.tight_layout()
        plt.savefig("Loss1"+'.png')

if __name__ == '__main__':
    main()
