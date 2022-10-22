"""Test for propagation from scratch.

Based on this code, DiskPropagation class is made.
"""
from scipy import signal
import numpy as np


def main():

    r_in, r_out = 3, 7
    num_elapse = 10

    num_anulus = r_out - r_in + 1
    num_segments = r_out
    num_frame = num_segments + num_elapse
    state = np.zeros((num_frame, num_anulus, num_segments))

    state_shape = (num_anulus, num_segments)

    weights = np.ones(2)[None, :] / 2
    trim_matrix = np.triu(np.ones(state_shape))

    # frame = 0
    # Burn-in pahse.
    state[0, 0] = np.random.poisson(10, num_segments)
    for i in range(1, num_segments, 1):
        next_state = signal.convolve2d(state[i-1], weights, "same")
        next_state = np.roll(next_state, shift=1, axis=0)
        next_state[0] = np.random.poisson(10, num_segments)
        state[i] = next_state
    state = state[num_segments:]

    for i in range(1, num_elapse, 1):
        next_state = signal.convolve2d(state[i-1], weights, "same")
        next_state = np.roll(next_state, shift=1, axis=0)
        next_state[0] = np.random.poisson(10, num_segments)
        state[i] = next_state

    state *= trim_matrix

    print(state)


if __name__ == "__main__":
    main()
