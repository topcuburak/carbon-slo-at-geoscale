"""Fitted coefficients: latency and energy from the calibration campaigns.

Artifacts live in `fit_data/`, one file per (model, TP):
`latency_fit_tp{TP}_{model}.json` and `energy_fit_tp{TP}_{model}.json`. Five
pairs are fitted -- llama3_8B and qwen2.5_7B at TP=1 and TP=2, mixtral_8x7b at
TP=4 -- and anything else raises at construction rather than falling back.
"""
