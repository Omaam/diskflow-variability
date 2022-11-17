"""Test for propagation.
"""
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import diskflowsim as dfs


def main():

    num_step = 1000
    r_in = 30

    np.random.seed(0)

    def initial_state_func(size):
        while 1:
            yield np.random.poisson(10, size)

    def observation_func(state):
        return np.random.poisson(state)

    fig, ax = plt.subplots(3, 2, figsize=(8, 5),
                           sharex="col")
    for i, r_out in enumerate([50, 100, 200]):

        dp = dfs.DiskPropagation(r_in, r_out, 1.00)
        dp.initialize(initial_state_func)
        dp.run_simulation(num_step)

        obs = dp.observe(observation_func)

        time = np.arange(num_step + 1)
        obs_count_rate = np.sum(obs, axis=(1, 2))
        ax[i, 0].plot(time, obs_count_rate)
        ax[i, 0].set_ylabel("(Rin, Rout) =\n({}, {})".format(r_in, r_out))

        freqs, powers = signal.welch(obs_count_rate)
        ax[i, 1].step(freqs, powers, where="mid", color="k")
        ax[i, 1].set_xscale("log")
        ax[i, 1].set_yscale("log")

    ax[0, 0].set_title("Variability")
    ax[0, 1].set_title("Power Spectrum")
    ax[1, 1].set_ylabel("Power")
    ax[2, 0].set_xlabel("Time (sample)")
    ax[2, 1].set_xlabel("Frequency (Hz/sample)")
    fig.supylabel("Counts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
