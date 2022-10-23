"""Test for propagation.
"""
from tqdm import trange
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

    dp = dps.DiskPropagation(r_in, r_out, 0.95)
    dp.initialize(initial_state_func)
    dp.run_simulation(num_step)

    obs = dp.observe(observation_func)

    for i in trange(10):
        dps.plot_snapshot(obs[i], r_in, r_out)
        idx_rjust = str(i).rjust(4, "0")
        plt.savefig(f"animation/animation_{idx_rjust}.png")
        plt.close()
        # plt.show()


if __name__ == "__main__":
    main()