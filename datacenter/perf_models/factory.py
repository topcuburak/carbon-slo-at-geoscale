"""Factory for creating latency backends.

Selects between CSV-based lookup and formula-based predictor
based on the ``mode`` parameter.

Usage::

    from datacenter.perf_models.factory import create_latency_backend

    # Lookup (default): interpolated from benchmark CSV
    backend = create_latency_backend("lookup", csv_path="bench.csv")

    # Predictor: auto-fit curves from the same CSV
    backend = create_latency_backend("predictor", csv_path="bench.csv")

    # Predictor with direct coefficients (no CSV needed)
    backend = create_latency_backend("predictor",
        tpot_coeffs={"a": 0.185, "b": 0.021, "c": 0.008},
        prefill_coeffs={"a": -0.002, "b": 0.074, "c": 0.621, "d": -13.281},
    )
"""

from datacenter.perf_models.base import LatencyBackend
from datacenter.perf_models.lookup.latency import LatencyLookup
from datacenter.perf_models.lookup.predictor import LatencyPredictor


def create_latency_backend(mode: str = "lookup", **kwargs) -> LatencyBackend:
    """Create a latency backend by name.

    Args:
        mode: ``"lookup"`` for CSV interpolation, ``"predictor"`` for formulas.
        **kwargs: Forwarded to the backend constructor.
            For lookup: ``csv_path``
            For predictor with CSV: ``csv_path``
            For predictor with coefficients: ``tpot_coeffs``, ``prefill_coeffs``
    """
    if mode == "lookup":
        return LatencyLookup(
            kwargs["csv_path"],
            step_ms_lookup_json=kwargs.get("step_ms_lookup_json"),
            ttft_lookup_json=kwargs.get("ttft_lookup_json"),
            model_name=kwargs.get("model_name"),
        )
    elif mode == "predictor":
        if "csv_path" in kwargs:
            return LatencyPredictor.from_csv(kwargs["csv_path"])
        else:
            return LatencyPredictor(
                tpot_coeffs=kwargs["tpot_coeffs"],
                prefill_coeffs=kwargs["prefill_coeffs"],
            )
    else:
        raise ValueError(f"Unknown latency mode {mode!r}. Use 'lookup' or 'predictor'.")
