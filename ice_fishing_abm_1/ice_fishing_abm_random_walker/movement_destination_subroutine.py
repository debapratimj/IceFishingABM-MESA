import numpy as np
class ExplorationStrategy:
    def __init__(self, grid_size: int = 100, ucb_beta=0.2, tau=0.01):
        self.grid_size = grid_size
        self.ucb_beta = ucb_beta
        self.softmax_tau = tau
        self.mesh = np.array(np.meshgrid(range(self.grid_size), range(self.grid_size))).reshape(2, -1).T
        self.mesh_indices = np.arange(0, self.mesh.shape[0])
        self.belief_softmax = np.random.uniform(0, 1, (self.grid_size, self.grid_size))
        self.belief_softmax /= np.sum(self.belief_softmax)  # Normalize the distribution
        self.other_agent_locs = np.empty((0, 2))
        self.destination = None

    def choose_destination(self, success_locs, failure_locs, other_agent_locs):
        """
        Select destination randomly using a generic softmax approach
        """
        self._check_input(success_locs)
        self._check_input(failure_locs)
        self._check_input(other_agent_locs)
        self.other_agent_locs = other_agent_locs
        self.destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]
        return self.destination

    def _check_input(self, input_data):
        assert input_data.ndim == 2, "Input data must have shape (n data points, 2)"
        assert input_data.shape[1] == 2, "Input data must have shape (n data points, 2)"


class RandomWalkerExplorationStrategy(ExplorationStrategy):
    def __init__(self, motion_type="Brownian", sigma: float = 1.0, mu: float = 1.5, dmin: float = 1.0, L: float = 10.0,
                 C: float = 1.0, alpha: float = 0.1, random_state: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.motion_type = motion_type  # 'Brownian' or 'Levy'
        self.sigma = sigma  # Brownian step size
        self.mu = mu  # Levy exponent
        self.dmin = dmin
        self.L = L
        self.C = C
        self.alpha = alpha
        self.random_state = random_state

    def choose_destination(self, social_locs, catch_locs, loss_locs, social_cue=False):
        """
        Choose destination based on motion type (Brownian or Levy)
        """
        self._check_input(social_locs)
        self._check_input(catch_locs)
        self._check_input(loss_locs)

        if self.motion_type == "Brownian":
            self.destination = self._brownian_motion()
        elif self.motion_type == "Levy":
            self.destination = self._levy_flight()
        
        # Handle social cue if detected
        if social_cue:
            self.destination = self._adjust_for_social_cue(self.destination, social_locs)
        
        return self.destination

    def _brownian_motion(self):
        """
        Implement Brownian motion, sampling distance and angle
        """
        # Sample the non-negative distance from a normal distribution
        distance = np.abs(np.random.normal(0, self.sigma))
        # Sample the angle uniformly from [0, 2π]
        angle = np.random.uniform(0, 2 * np.pi)
        # Convert polar coordinates (distance, angle) to Cartesian displacement
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)

        # Find new destination
        destination = self._get_new_position(dx, dy)
        return destination

    def _levy_flight(self):
        """
        Implement Levy flight, sampling distance and angle
        """
        # Sample distance from a power-law distribution with exponent mu
        distance = self.C * np.random.uniform(self.dmin, self.L) ** (-self.mu)
        # Sample the angle uniformly from [0, 2π]
        angle = np.random.uniform(0, 2 * np.pi)
        # Convert polar coordinates (distance, angle) to Cartesian displacement
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)

        # Find new destination
        destination = self._get_new_position(dx, dy)
        return destination

    def _adjust_for_social_cue(self, current_destination, social_locs):
        """
        Adjust destination based on social cues, if detected
        """
        if social_locs.size > 0:
            # Adjust the destination based on proximity to social cues
            delta_d = np.linalg.norm(current_destination - social_locs, axis=1).min()
            prob_social = np.exp(-self.alpha * delta_d)

            # With probability prob_social, switch to the social cue location
            if np.random.rand() < prob_social:
                current_destination = social_locs[np.random.choice(social_locs.shape[0])]

        return current_destination

    def _get_new_position(self, dx, dy):
        """
        Compute the new position from a displacement (dx, dy)
        """
        # Current position (we assume the walker starts from the center of the grid)
        current_position = np.array([self.grid_size // 2, self.grid_size // 2])

        # Calculate new position with periodic boundary conditions (wrap around the grid)
        new_x = (current_position[0] + dx) % self.grid_size
        new_y = (current_position[1] + dy) % self.grid_size

        return np.array([new_x, new_y])
