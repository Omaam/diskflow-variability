"""Test for propagation.
"""
import matplotlib.pyplot as plt
import numpy as np

import diskflow_variability as dfv


def main():

    num_step = 1000
    r_out = 100

    def initial_state_func(size):
        return np.ones(size)

    def observation_func(state):
        return np.random.poisson(state)

    fig, ax = plt.subplots(5, 2, figsize=(10, 12),
                           sharex="col", sharey="row")
    for i, r_in in enumerate([50, 60, 70, 80, 90]):

        dp = dfv.DiskPropagation(r_in, r_out, 0.95)
        dp.initialize(initial_state_func)
        dp.run_simulation(num_step)

        state = dp.state_
        obs = dp.observe(observation_func)

        time = np.arange(num_step + 1)
        obs_count_rate = np.sum(obs, axis=(1, 2))
        state_count_rate = np.sum(state, axis=(1, 2))
        ax[i, 0].plot(time, state_count_rate)
        ax[i, 1].plot(time, obs_count_rate)
        ax[i, 0].set_ylabel("(Rin, Rout) =\n({}, {})".format(r_in, r_out))

    ax[0, 0].set_title("State")
    ax[0, 1].set_title("Observation")
    fig.suptitle("Disk propagation for diskbb")
    fig.supxlabel("Time")
    fig.supylabel("Counts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
