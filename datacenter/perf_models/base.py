"""Abstract base class for latency/power backends.

All latency backends (lookup table, predictor, etc.) must implement
this interface so they can be used interchangeably by the GPU simulation
and provisioner latency estimation.
"""

from abc import ABC, abstractmethod


class LatencyBackend(ABC):
    """Interface for TPOT / TTFT / Power estimation."""

    @abstractmethod
    def tpot(self, batch_size: int) -> float:
        """Return time-per-output-token in seconds for *batch_size*."""
        ...

    @abstractmethod
    def ttft(self, batch_size: int, in_tokens: int) -> float:
        """Return time-to-first-token in seconds."""
        ...

    def ttft_total_prefill(self, total_prefill_tokens: int) -> float:
        """Return TTFT (seconds) based on total prefill tokens in the batch.

        Uses a piecewise power-law model fitted from benchmark data.
        Falls back to the per-BS ttft() method if not implemented.
        """
        raise NotImplementedError

    def summary(self) -> str:
        """Human-readable summary for logging."""
        return f"{self.__class__.__name__}"
