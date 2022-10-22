"""Test for propagation.
"""
import matplotlib.pyplot as plt
import numpy as np

import diskpropsim as dps


def main():
    dp = dps.DiskPropagation(3, 100)
    dp.initialize_poisson(10)
    dp.elapse(500)
    state = dp.state_

    fig, ax = plt.subplots()
    time = np.arange(501)
    count_rate = np.sum(state, axis=(1, 2))
    ax.plot(time, count_rate)
    ax.set_xlabel("Time")
    ax.set_ylabel("Counts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
