"""Test for propagation.
"""
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

import diskflow_variability as dfv


def main():

    num_step = 1000
    r_in = 30
    r_out = 100

    def initial_state_func(size):
        return np.random.poisson(10, size)

    def observation_func(state):
        return np.random.poisson(state)

    dp = dfv.DiskPropagation(r_in, r_out, 1.05)
    dp.initialize(initial_state_func)
    dp.run_simulation(num_step)

    obs = dp.observe(observation_func)

    for i in trange(10):
        dfv.plot_snapshot(obs[i], r_in, r_out)
        idx_rjust = str(i).rjust(4, "0")
        plt.savefig(f"animation/animation_{idx_rjust}.png")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
