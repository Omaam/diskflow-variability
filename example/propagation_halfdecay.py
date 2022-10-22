"""Test for propagation.
"""
import matplotlib.pyplot as plt
import numpy as np

import diskpropsim as dps


def main():

    num_step = 10000
    r_out = 100

    def initial_state_func(size):
        return np.random.poisson(10, size)

    fig, ax = plt.subplots(5, figsize=(10, 6), sharex="col")
    for i, r_in in enumerate([50, 60, 70, 80, 90]):

        dp = dps.DiskPropagation(r_in, r_out)
        dp.initialize(initial_state_func)
        dp.run_simulation(num_step)
        state = dp.state_

        time = np.arange(num_step + 1)
        count_rate = np.sum(state, axis=(1, 2))
        ax[i].plot(time, count_rate)
        ax[i].set_title("Rin = {}, Rout = {}".format(r_in, r_out))
        ax[i].set_ylabel("Counts")

    ax[i].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
