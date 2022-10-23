"""Test for propagation.
"""
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

import diskpropsim as dps


def compute_crosscorrelation(in1, in2):
    max_len = max(in1.size, in2.size)
    correlations = signal.correlate(in1, in2) / max_len
    lags = signal.correlation_lags(in1.size, in2.size)
    return lags, correlations


def set_lagrange_correlation_function(lags, correlations, maxlags):
    ids = np.where(np.abs(lags) < maxlags)
    return lags[ids], correlations[ids]


def main():

    num_step = 10000
    r_out = 101
    r_in = 30

    def initial_state_func(size):
        return np.random.poisson(10, size)

    def observation_func(state):
        return np.random.poisson(state)

    dp = dps.DiskPropagation(r_in, r_out, 1.05)
    dp.initialize(initial_state_func)
    dp.run_simulation(num_step)
    state = dp.state_

    variable_irdisk = np.sum(state[:, 0:1, :], axis=(1, 2))
    variable_adaf = np.sum(state[:, 1:, :], axis=(1, 2))
    variable_irdisk = stats.zscore(variable_irdisk)
    variable_adaf = stats.zscore(variable_adaf)

    lags, correlations = compute_crosscorrelation(
        variable_adaf, variable_irdisk)
    lags, correlations = set_lagrange_correlation_function(
        lags, correlations, 100)

    fig, ax = plt.subplots(3)
    time = np.arange(variable_irdisk.size)
    ax[0].plot(time, variable_irdisk)
    ax[0].set_ylabel("Counts\nirdisk")

    ax[1].plot(time, variable_adaf)
    ax[1].set_ylabel("Counts\nADAF")

    ax[2].scatter(lags, correlations)
    ax[2].set_ylabel("CCF\nLag (s)")
    plt.show()


if __name__ == "__main__":
    main()
