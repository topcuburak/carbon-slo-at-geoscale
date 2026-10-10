"""Arrival-time distribution generators for the load generator.

Each generator is a callable that returns the next inter-arrival time
in **seconds**.  Two distributions are supported:

  - **uniform**: draws from U(min_ms, max_ms) then converts to seconds.
  - **poisson**: exponential inter-arrivals with rate λ (req/s),
    giving a Poisson arrival process.
"""

import random
from typing import Callable

DISTRIBUTIONS = ("uniform", "poisson")


def make_arrival_generator(
  distribution: str,
  min_interval_ms: float,
  max_interval_ms: float,
  arrival_rate: float,
) -> Callable[[], float]:
  """Return a callable that produces inter-arrival times in seconds."""
  if distribution == "uniform":
    def _gen() -> float:
      return random.uniform(min_interval_ms, max_interval_ms) / 1000.0
    return _gen
  elif distribution == "poisson":
    # Poisson process: inter-arrival times are exponentially distributed
    # with rate λ (requests/sec).  random.expovariate(λ) returns seconds.
    def _gen() -> float:
      return random.expovariate(arrival_rate)
    return _gen
  else:
    raise ValueError(
      f"Unknown arrival distribution {distribution!r}. "
      f"Available: {DISTRIBUTIONS}"
    )
