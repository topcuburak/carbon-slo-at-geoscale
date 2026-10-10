"""Per-model KV-cache geometry.

The simulator tracks the KV cache in TOKENS because that is what the engine
counts and what the calibration measured. Energy, cache capacity and any
future transfer cost depend on BYTES, and bytes per token is a property of the
architecture, not of the model size:

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes

The leading 2 is K and V. Grouped-query attention decouples this from the
parameter count: Llama-2-70B has roughly 8.75x the parameters of
Llama-3.1-8B but only 2.5x the KV per token, because both pin n_kv_heads=8
and head_dim=128 and differ only in depth (80 layers against 32).

This is why the pre-registered "model size drives e_kv" hypothesis failed by
+68%: size is the wrong variable. Dividing the measured e_kv by the bytes
below brings three architectures within 20% of each other (3.46, 3.23 and
2.59 x 1e-10 J/byte), against a spread of 2.4x on the raw per-token numbers.

Values are from each model's config.json (num_hidden_layers,
num_key_value_heads, and head_dim, or hidden_size / num_attention_heads when
head_dim is absent). `verify_against_hf` re-derives them from the live config
so the table cannot drift silently.
"""

from typing import Dict, Optional

# dtype_bytes is 2 everywhere because every calibration campaign ran fp16.
# A quantised KV cache would change this and nothing else in the formula.
KV_GEOMETRY: Dict[str, dict] = {
    # weight_bytes: parameters x dtype_bytes, what the engine loads before any
    # KV exists. Needed to size the cache: vLLM gives KV whatever is left of
    # (devices x HBM x gpu_memory_utilization) after the weights.
    "llama3_8B": dict(
        hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        n_layers=32, n_kv_heads=8, head_dim=128, dtype_bytes=2,
        weight_bytes=int(8.03e9 * 2)),
    "llama3.2_3B": dict(
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        n_layers=28, n_kv_heads=8, head_dim=128, dtype_bytes=2,
        weight_bytes=int(3.21e9 * 2)),
    "qwen2.5_7B": dict(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        n_layers=28, n_kv_heads=4, head_dim=128, dtype_bytes=2,
        weight_bytes=int(7.62e9 * 2)),
    "mixtral_8x7b": dict(
        hf_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        n_layers=32, n_kv_heads=8, head_dim=128, dtype_bytes=2,
        weight_bytes=int(46.7e9 * 2)),
    # Not calibrated yet; listed because it is the sharpest test of the
    # bytes-per-token relation we have queued.
    "llama2_70B": dict(
        hf_id="meta-llama/Llama-2-70b-chat-hf",
        n_layers=80, n_kv_heads=8, head_dim=128, dtype_bytes=2),
    # Named in the current paper draft but never measured in the energy
    # campaign. Kept so the geometry is on record if it is measured or
    # removed. Note it predates GQA in this family: n_kv_heads == n_heads,
    # which is why its KV per token exceeds Llama-3.1-8B's despite being
    # a smaller model.
    "deepseek_moe_16b": dict(
        hf_id="deepseek-ai/deepseek-moe-16b-chat",
        n_layers=28, n_kv_heads=16, head_dim=128, dtype_bytes=2),
}


def weight_bytes(model_name: str) -> Optional[int]:
    """Bytes the model's parameters occupy, or None for an unknown model."""
    spec = KV_GEOMETRY.get(model_name)
    return spec.get("weight_bytes") if spec else None


def kv_bytes_per_token(model_name: str) -> Optional[int]:
    """Bytes of KV cache one token occupies, or None for an unknown model.

    None rather than a default: a wrong constant here would scale every
    energy number silently, and a missing model is a configuration error
    worth surfacing.
    """
    g = KV_GEOMETRY.get(model_name)
    if g is None:
        return None
    return 2 * g["n_layers"] * g["n_kv_heads"] * g["head_dim"] * g["dtype_bytes"]


def kv_bytes(model_name: str, n_tokens: int) -> Optional[int]:
    """KV bytes held by `n_tokens` cached tokens."""
    b = kv_bytes_per_token(model_name)
    return None if b is None else b * n_tokens


def verify_against_hf(model_name: str) -> dict:
    """Re-derive the geometry from the model's live config.json.

    Returns {"ok": bool, "table": ..., "hf": ..., "note": ...}. Needs network
    or a warm HF cache; the table is authoritative when neither is available.
    """
    g = KV_GEOMETRY.get(model_name)
    if g is None:
        return {"ok": False, "note": f"unknown model {model_name!r}"}
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(g["hf_id"])
    except Exception as e:
        return {"ok": False, "note": f"could not load config: {type(e).__name__}: {e}"}
    n_heads = getattr(c, "num_attention_heads", None)
    hf = {
        "n_layers": getattr(c, "num_hidden_layers", None),
        "n_kv_heads": getattr(c, "num_key_value_heads", None) or n_heads,
        "head_dim": (getattr(c, "head_dim", None)
                     or (getattr(c, "hidden_size", 0) // n_heads if n_heads else None)),
    }
    same = all(hf[k] == g[k] for k in hf if hf[k] is not None)
    return {"ok": same, "table": {k: g[k] for k in hf}, "hf": hf,
            "bytes_per_token_table": kv_bytes_per_token(model_name),
            "bytes_per_token_hf": (2 * hf["n_layers"] * hf["n_kv_heads"]
                                   * hf["head_dim"] * g["dtype_bytes"]
                                   if all(hf.values()) else None)}


if __name__ == "__main__":
    print(f"{'model':<18}{'layers':>7}{'kv_h':>6}{'head_dim':>10}"
          f"{'KiB/token':>11}{'MiB @ 2048 tok':>16}")
    for name in KV_GEOMETRY:
        g = KV_GEOMETRY[name]
        b = kv_bytes_per_token(name)
        print(f"{name:<18}{g['n_layers']:>7}{g['n_kv_heads']:>6}{g['head_dim']:>10}"
              f"{b/1024:>11.0f}{b*2048/1024/1024:>16.1f}")
