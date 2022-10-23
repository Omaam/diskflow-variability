"""Disk probagation simulation.
"""
from scipy import signal
import numpy as np


class DiskPropagation:
    """Disk propagation class.
    """
    def __init__(self, r_in, r_out, decay_ratio=1.0):
        self.r_in = r_in
        self.r_out = r_out
        self.num_anulus = r_out - r_in + 1
        self.num_segments = r_out
        self.decay_ratio = decay_ratio
        self.time = 0

        self.state = None
        self.state_ = None

    def initialize(self, initial_state_func: callable):
        """Initialize using initial state function.

        Args:
            * initial_state_func:
                initial_state_func must be designed to return
                values by passing the size, like
                initial_state_func(5) -> out [1, 4, 3, 4, 2]
        """
        self.initial_state_func = initial_state_func
        self._run_burnin()

    def observe(self, observation_func):
        return observation_func(self.state_)

    def reset(self):
        self.time = 0
        self.state = None
        self.state_ = None

    def run_simulation(self, num_step):
        self.state = np.zeros(
            (
                num_step + 1,
                self.num_anulus,
                self.num_segments
            )
        )
        self.time = 0

        self.state[0] = self.initial_state
        for _ in range(num_step):
            self._update()

        self.state_ = self._extract_state()

    def _extract_state(self):
        trim_matrix = np.ones((self.num_anulus, self.num_segments))
        trim_matrix = np.triu(trim_matrix)
        return self.state * trim_matrix

    def _run_burnin(self):
        """Do burnin before doing simulation.

        For burnion, the number of updates equals the number of segments.
        """
        self.state = np.zeros(
            (self.num_segments + 1, self.num_anulus, self.num_segments))
        self.state[0, 0] = self.initial_state_func(self.num_segments)

        for _ in range(self.num_segments):
            self._update()

        self.state_ = None
        self.initial_state = self._extract_state()[-1]

    def _update(self):
        weights = (np.ones(2)[None, :] / 2) * self.decay_ratio
        self.time += 1

        i = self.time
        next_state = signal.convolve2d(self.state[i-1], weights, "same")
        next_state = np.roll(next_state, shift=1, axis=0)
        next_state[0] = self.initial_state_func(self.num_segments)
        self.state[self.time] = next_state
