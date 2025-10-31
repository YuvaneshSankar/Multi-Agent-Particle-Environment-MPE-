import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated exploration noise.
    This helps with exploration in continuous action spaces by generating smooth noise.
    """

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        """
        Initialize Ornstein-Uhlenbeck noise process.

        Args:
            size: Dimension of the noise (action dimension)
            mu: Mean of the process (equilibrium value)
            theta: Rate of mean reversion
            sigma: Volatility parameter
            dt: Time step
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.copy(self.mu)

    def reset(self):
        """Reset internal state to mean."""
        self.state = np.copy(self.mu)

    def sample(self):
        """
        Generate noise sample using Ornstein-Uhlenbeck process.

        The process is defined by: dx = θ(μ - x)dt + σdW
        where dW is a Wiener process (random noise).

        Returns:
            noise: Noise sample
        """
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state += dx
        return self.state


class GaussianNoise:
    """
    Simple Gaussian noise for exploration.
    Alternative to OU noise, often works just as well and is simpler.
    """

    def __init__(self, size, mu=0.0, sigma=0.2):
        """
        Initialize Gaussian noise.

        Args:
            size: Dimension of the noise (action dimension)
            mu: Mean of the Gaussian
            sigma: Standard deviation of the Gaussian
        """
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def reset(self):
        """Reset (no state for Gaussian noise)."""
        pass

    def sample(self):
        """
        Generate noise sample from Gaussian distribution.

        Returns:
            noise: Noise sample
        """
        return np.random.normal(self.mu, self.sigma, self.size)
