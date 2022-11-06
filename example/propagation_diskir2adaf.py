"""Test for propagation.
"""
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

import diskflow_variability as dfv


def compute_crosscorrelation(in1, in2, whitening=True):
    max_len = max(in1.size, in2.size)
    in1_w = stats.zscore(in1)
    in2_w = stats.zscore(in2)
    correlations = signal.correlate(in1_w, in2_w) / max_len
    lags = signal.correlation_lags(in1.size, in2.size)
    return lags, correlations


def set_lagrange_correlation_function(lags, correlations, maxlags):
    ids = np.where(np.abs(lags) < maxlags)
    return lags[ids], correlations[ids]


def main():

    num_steps = 3000
    r3 = 150
    r2 = 100
    r1 = 30

    v_diskir = 1
    v_adaf = 1

    num_steps_diskir = num_steps + (r2 - r1) + 1
    num_steps_adaf = num_steps

    np.random.seed(1)

    def initial_annulus_generator_diskir(size):
        while True:
            yield np.random.poisson(10, size)

    dp_diskir = dfv.DiskPropagation(r2, r3, 0.90)
    dp_diskir.initialize(initial_annulus_generator_diskir)
    dp_diskir.run_simulation(num_steps_diskir, v_diskir)

    def initial_annulus_generator_adaf(size):
        for snapshot in dp_diskir.state_:
            yield snapshot[-1][-size:]

    dp_adaf = dfv.DiskPropagation(r1, r2, 1.10)
    dp_adaf.initialize(initial_annulus_generator_adaf)
    dp_adaf.run_simulation(num_steps_adaf, v_adaf)

    def observation_func(state):
        return np.random.poisson(state)

    idx_start_diskir = r2 - r1 + 1
    obs_diskir = dp_diskir.observe(
        observation_func)[idx_start_diskir:idx_start_diskir+num_steps]
    obs_adaf = dp_adaf.observe(observation_func)[:num_steps]

    times = np.arange(num_steps)
    countrate_diskir = np.sum(obs_diskir, axis=(1, 2))
    countrate_adaf = np.sum(obs_adaf, axis=(1, 2))

    fig, ax = plt.subplots(3, figsize=(6, 6))
    ax[0].plot(times, countrate_diskir)
    ax[0].set_title("diskir; (Rin, Rout) = ({}, {})".format(r2, r3))
    ax[0].set_ylabel("Counts")

    ax[1].plot(times, countrate_adaf)
    ax[1].set_title("ADAF; (Rin, Rout) = ({}, {})".format(r1, r2))
    ax[1].set_ylabel("Counts")
    ax[1].set_xlabel("Time")

    lags, correlations = compute_crosscorrelation(
        countrate_adaf, countrate_diskir)
    lags, correlations = set_lagrange_correlation_function(
        lags, correlations, 200)
    ax[2].scatter(lags, correlations)
    ax[2].set_xlabel("Lag")
    ax[2].set_ylabel("CCF")

    fig.align_ylabels()

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
