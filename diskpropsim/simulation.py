"""Disk probagation simulation.
"""
from scipy import signal
import numpy as np


class DiskPropagation:
    """Disk propagation class.

    Args:
        * r_in:
            Inner radius.
        * r_out:
            Outer radius.
        * total_convolve_rate:
            Total convolve rate. This corresponding the energy
            changing rate, thus positive value means energy will
            increase as the variabilities move inwards, vice versa.

    Attribution:
        * r_in:
            Inner radius.
        * r_out:
            Outer radius.
        * num_anulus:
            Number of anulus, calculated from r_in and r_out.
        * num_segments:
            Number of segments in each anulus.
        * total_convolve_rate:
            Total convolve rate. This corresponding the energy
            changing rate, thus positive value means energy will
            increase as the variabilities move inwards, vice versa.
        * time:
            Time.
        * initial_state:
            Initial state. If exists, used in self.run_simulation.
        * state:
            State.
        * state_:
            Estimated state, where values only exists on upper triangle.
    """
    def __init__(self, r_in: int, r_out: int,
                 total_convolve_rate: float = 0.5):
        self.r_in = r_in
        self.r_out = r_out
        self.num_anulus = r_out - r_in + 1
        self.num_segments = r_out
        self.total_convolve_rate = total_convolve_rate
        self.time = 0

        self.initial_state = None
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

        # To initialize, self.num_anulus times updates are required.
        self.run_simulation(self.num_anulus)

        self.initial_state = self.state[-1]
        self.time = 0

    def observe(self, observation_func):
        """Observe by passing observation function.
        """
        return observation_func(self.state_)

    def reset(self):
        """Reset time and state.
        """
        self.time = 0
        self.state = None
        self.state_ = None

    def run_simulation(self, num_step):
        """num_step times updates are done to run simulation.
        """
        self.state = np.zeros(
            (num_step + 1,
             self.num_anulus,
             self.num_segments)
        )

        # If self.initial_state estimated beforehand exists,
        # use it, or compute by self.initial_state_func.
        if self.initial_state is not None:
            self.state[0] = self.initial_state
        else:
            self.state[0] = self.initial_state_func(self.num_segments)

        for _ in range(num_step):
            next_state = self._update(self.state[self.time])
            self.time += 1
            self.state[self.time] = next_state
        self.state_ = self._extract_state()

    def _extract_state(self):
        """Extract state by multiplying upper triangle matrix.
        """
        trim_matrix = np.ones((self.num_anulus, self.num_segments))
        trim_matrix = np.triu(trim_matrix)
        return self.state * trim_matrix

    def _update(self, current_state):
        """Update state.
        """
        weights = np.ones(2)[None, :] * self.total_convolve_rate / 2
        next_state = signal.convolve2d(current_state, weights, "same")
        next_state = np.roll(next_state, shift=1, axis=0)
        next_state[0] = self.initial_state_func(self.num_segments)
        return next_state
