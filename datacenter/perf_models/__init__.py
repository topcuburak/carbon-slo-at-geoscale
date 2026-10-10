"""How long a step takes and what it costs, for a given (model, TP).

Two families, and the split is what they are fitted to rather than a style
preference:

  analytical/   Coefficients fitted on the calibration campaigns, in
                `analytical/fit_data/`. `latency.py` gives TTFT and TPOT,
                `energy.py` gives joules per step. A (model, TP) with no
                artifact RAISES -- there is deliberately no fallback table,
                because the one that used to be here was TP-blind and returned
                a plausible wrong number instead of an error. This is the pair
                the sweep runs on.

  lookup/       Driven by the benchmark measurements themselves, in
                `lookup/table_data/`. `latency.py` interpolates the tables,
                `predictor.py` fits curves to the same CSV. Reachable through
                `factory.py` and used by lookup/validation/ and the single-DC test;
                the sweep does not use either.

`base.py` is the interface both satisfy, `factory.py` picks between the lookup
family's two members, and `model_specs.py` holds the KV geometry and weight
sizes that neither family fits -- they come from the model architecture.
"""
