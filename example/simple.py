"""Test for propagation.
"""
import matplotlib.pyplot as plt
import numpy as np

import diskpropsim as dps


def main():

    num_step = 1000
    r_in = 30
    r_out = 100

    def initial_state_func(size):
        return np.random.poisson(10, size)

    def observation_func(state):
        return np.random.poisson(state)

    dp = dps.DiskPropagation(r_in, r_out, 1.00)
    dp.initialize(initial_state_func)
    dp.run_simulation(num_step)

    state = dp.state_
    obs = dp.observe(observation_func)

    fig, ax = plt.subplots(2, figsize=(6, 6))
    time = np.arange(num_step + 1)
    obs_count_rate = np.sum(obs, axis=(1, 2))
    state_count_rate = np.sum(state, axis=(1, 2))
    ax[0].plot(time, state_count_rate)
    ax[0].set_title("(Rin, Rout) = ({}, {})".format(r_in, r_out))
    ax[0].set_ylabel("State")

    ax[1].plot(time, obs_count_rate)
    ax[1].set_ylabel("Observation")

    fig.supxlabel("Time")
    fig.supylabel("Counts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
