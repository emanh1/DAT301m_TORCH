"""
Consistency-loss ramp schedule — Eq. 10 of the SSMD paper.

λ is:
  • Ramped *up*   during the first quarter of training  (Gaussian ramp-up)
  • Held at 1.0   during the middle half
  • Ramped *down* during the final quarter              (Gaussian ramp-down)

This prevents the consistency loss from dominating before the model has
learned reasonable representations.
"""

import math


def consistency_lambda(j: int, N: int) -> float:
    """
    Compute the consistency-loss weight λ at iteration j out of N total.

    Eq. 10:
        λ = exp(-5 (1 - 4j/N)²)           if  0  ≤ j < N/4
          = 1                              if  N/4 ≤ j < 3N/4
          = exp(-12.5 (1 - 7(N-j)/N)²)   if  3N/4 ≤ j ≤ N

    Args:
        j : Current training iteration (0-indexed).
        N : Total number of training iterations.

    Returns:
        λ ∈ (0, 1]
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if j < 0 or j > N:
        raise ValueError(f"j={j} must be in [0, N={N}].")

    if j < N / 4:
        # Ramp up
        return math.exp(-5.0 * (1.0 - 4.0 * j / N) ** 2)
    elif j < 3 * N / 4:
        # Plateau
        return 1.0
    else:
        # Ramp down
        return math.exp(-12.5 * (1.0 - 7.0 * (N - j) / N) ** 2)


class ConsistencyScheduler:
    """
    Convenience wrapper to step through λ values during training.

    Args:
        total_iterations: Total training steps N.
    """

    def __init__(self, total_iterations: int):
        self.N = total_iterations
        self.step = 0

    def get_lambda(self) -> float:
        return consistency_lambda(self.step, self.N)

    def advance(self) -> float:
        """Advance one step and return the current λ."""
        lam = self.get_lambda()
        self.step = min(self.step + 1, self.N)
        return lam

    def __repr__(self) -> str:
        return (f"ConsistencyScheduler(N={self.N}, "
                f"step={self.step}, λ={self.get_lambda():.4f})")
