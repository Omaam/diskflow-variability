"""Disk probagation simulation.
"""
from scipy import signal
import numpy as np


class DiskPropagation:
    """Disk propagation class.
    """
    def __init__(self, r_in, r_out, decay_ratio=0.5):
        self.r_in = r_in
        self.r_out = r_out
        self.num_anulus = r_out - r_in + 1
        self.num_segments = r_out
        self.decay_ratio = decay_ratio
        self.time = 0

    def elapse(self, num_frame):
        self.state = np.zeros(
            (
                num_frame + 1,
                self.num_anulus,
                self.num_segments
            )
        )
        self.time = 0

        self.state[0] = self.initial_state
        for _ in range(num_frame):
            self._update()

        self.state_ = self._extract_state()

    def initialize_poisson(self, lam):
        self._do_burnin(lam)

    def _do_burnin(self, lam):
        """Do burnin before doing simulation.

        For burnion, the number of updates equals the number of segments.
        """
        self.state = np.zeros(
            (
                self.num_segments + 1,
                self.num_anulus,
                self.num_segments)
        )
        self.state[0, 0] = np.random.poisson(lam, self.num_segments)

        for _ in range(self.num_segments):
            self._update()

        self.state_ = None
        self.initial_state = self._extract_state()[-1]

    def _extract_state(self):
        trim_matrix = np.ones((self.num_anulus, self.num_segments))
        trim_matrix = np.triu(trim_matrix)
        return self.state * trim_matrix

    def _update(self):

        weights = np.ones(2)[None, :] * self.decay_ratio

        self.time += 1

        i = self.time
        next_state = signal.convolve2d(self.state[i-1], weights, "same")
        next_state = np.roll(next_state, shift=1, axis=0)
        next_state[0] = np.random.poisson(10, self.num_segments)
        self.state[self.time] = next_state
