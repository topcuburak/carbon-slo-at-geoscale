#!/usr/bin/env python3
"""Configuration: what a run actually ran with.

Every knob reaches a process through this chain -- environment variable, then
YAML, then a hardcoded default -- and a knob that silently keeps its default is
indistinguishable in the results from one that was set. That has already
happened twice: `max_num_seqs` was never passed by the scheduler launcher, so
the manual path ran the ILP with the cap at 0 while the sweep passed 32; and
`ttft_budget_ms: 0` is falsy, so the SLA variable was never exported at all.

Both of those are shape-of-the-chain problems, which is what these check.

Run:  PYTHONPATH=.:proto python3 scripts/test/test_config.py
"""

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))

from scheduler.config_loader import (_DEFAULTS, _PARAM_MAP,        # noqa: E402
                                     load_scheduler_config)

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok    {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except BaseException as e:                                 # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


@contextmanager
def env(**kw):
    """Set env vars for the block, and put the environment back afterwards."""
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def yaml_file(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(text)
    f.close()
    return f.name


def clean_env(**kw):
    """Every mapped env var cleared, then the ones asked for set."""
    base = {v[0]: None for v in _PARAM_MAP.values()}
    base.update(kw)
    return env(**base)


EMPTY = yaml_file("scheduler: {}\n")


# ─────────────────────────────── precedence ─────────────────────────────────

def test_a_default_is_used_when_nothing_sets_the_knob():
    with clean_env():
        cfg = load_scheduler_config(EMPTY)
    assert cfg["routing_policy"] == _DEFAULTS["routing_policy"], cfg["routing_policy"]
    assert cfg["grpc_port"] == _DEFAULTS["grpc_port"], cfg["grpc_port"]


def test_yaml_beats_the_default():
    path = yaml_file("scheduler:\n  routing:\n    policy: least_loaded\n")
    with clean_env():
        cfg = load_scheduler_config(path)
    assert cfg["routing_policy"] == "least_loaded", cfg["routing_policy"]


def test_the_environment_beats_yaml():
    path = yaml_file("scheduler:\n  routing:\n    policy: least_loaded\n")
    with clean_env(ROUTING_POLICY="round_robin"):
        cfg = load_scheduler_config(path)
    assert cfg["routing_policy"] == "round_robin", \
        "YAML overrode the environment; the sweep drives everything by env"


def test_a_knob_yaml_does_not_mention_keeps_its_default():
    path = yaml_file("scheduler:\n  routing:\n    policy: least_loaded\n")
    with clean_env():
        cfg = load_scheduler_config(path)
    assert cfg["grpc_port"] == _DEFAULTS["grpc_port"], cfg["grpc_port"]


def test_every_mapped_knob_can_be_set_from_the_environment():
    # The gap that bit twice: a knob with a place in the map but no path from
    # the launcher. This checks the map end of it -- that every entry really
    # does read its variable.
    samples = {bool: "true", int: "7", float: "1.5", str: "x", type(None): "x"}
    for key, (env_var, _) in _PARAM_MAP.items():
        want_raw = samples[type(_DEFAULTS[key])]
        with clean_env(**{env_var: want_raw}):
            cfg = load_scheduler_config(EMPTY)
        assert str(cfg[key]).lower() != str(_DEFAULTS[key]).lower() \
            or str(_DEFAULTS[key]).lower() == want_raw.lower(), \
            f"{env_var} did not reach cfg[{key!r}] (still {cfg[key]!r})"


def test_every_mapped_knob_can_be_set_from_yaml():
    for key, (_, path_keys) in _PARAM_MAP.items():
        # build the nested document this key is read from
        leaf = {bool: True, int: 7, float: 1.5, str: "x", type(None): "x"}[
            type(_DEFAULTS[key])]
        doc, node = {}, None
        cur = doc
        for k in path_keys[:-1]:
            cur[k] = {}
            cur = cur[k]
        cur[path_keys[-1]] = leaf
        import yaml as _y
        path = yaml_file(_y.safe_dump(doc))
        with clean_env():
            cfg = load_scheduler_config(path)
        assert cfg[key] == leaf, \
            f"yaml path {'.'.join(path_keys)} did not reach cfg[{key!r}]"


# ─────────────────────────── type coercion ──────────────────────────────────

def test_a_numeric_knob_comes_back_a_number_not_a_string():
    # Environment variables are strings. A port or a budget left as "50052"
    # compares wrong and formats wrong, and the failure is far from here.
    with clean_env(GRPC_PORT="50999", TTFT_BUDGET_MS="2000",
                   ROUTING_ALPHA="0.25"):
        cfg = load_scheduler_config(EMPTY)
    assert cfg["grpc_port"] == 50999 and isinstance(cfg["grpc_port"], int)
    assert cfg["ttft_budget_ms"] == 2000 and isinstance(cfg["ttft_budget_ms"], int)
    assert cfg["routing_alpha"] == 0.25 and isinstance(cfg["routing_alpha"], float)


def test_a_boolean_knob_reads_the_usual_spellings():
    for raw in ("true", "True", "1", "yes"):
        with clean_env(SIMULATE_RTT=raw):
            assert load_scheduler_config(EMPTY)["simulate_rtt"] is True, raw
    for raw in ("false", "False", "0", "no", ""):
        with clean_env(SIMULATE_RTT=raw):
            assert load_scheduler_config(EMPTY)["simulate_rtt"] is False, raw


def test_a_zero_budget_still_reaches_the_config():
    # It is falsy, which is exactly how it got dropped on the way OUT of the
    # launcher. Coming in, it must survive.
    with clean_env(TTFT_BUDGET_MS="0", TPOT_BUDGET_MS="0", MAX_BATCH_SIZE="0"):
        cfg = load_scheduler_config(EMPTY)
    assert cfg["ttft_budget_ms"] == 0
    assert cfg["tpot_budget_ms"] == 0
    assert cfg["max_num_seqs"] == 0


def test_a_nonsense_number_fails_loudly_rather_than_silently_defaulting():
    with clean_env(GRPC_PORT="not-a-port"):
        try:
            load_scheduler_config(EMPTY)
        except ValueError:
            return
    raise AssertionError("a malformed port was accepted and something defaulted")


# ────────────────────────── finding the file ────────────────────────────────

def test_an_explicit_path_wins_over_the_environment():
    a = yaml_file("scheduler:\n  routing:\n    policy: least_loaded\n")
    b = yaml_file("scheduler:\n  routing:\n    policy: round_robin\n")
    with clean_env(SCHEDULER_CONFIG=b):
        assert load_scheduler_config(a)["routing_policy"] == "least_loaded"


def test_the_environment_names_the_file_when_no_path_is_given():
    b = yaml_file("scheduler:\n  routing:\n    policy: round_robin_local\n")
    with clean_env(SCHEDULER_CONFIG=b):
        assert load_scheduler_config()["routing_policy"] == "round_robin_local"


def test_a_path_that_does_not_exist_falls_back_to_defaults_not_a_crash():
    with clean_env():
        cfg = load_scheduler_config("/nowhere/at/all.yaml")
    assert cfg["routing_policy"] == _DEFAULTS["routing_policy"]


def test_the_shipped_config_loads():
    # scheduler/config.yaml is read at runtime by the server; a syntax error in
    # it would only show up when a scheduler starts.
    with clean_env():
        cfg = load_scheduler_config(str(ROOT / "scheduler" / "config.yaml"))
    assert cfg["routing_policy"], cfg
    assert isinstance(cfg["grpc_port"], int), cfg["grpc_port"]


def test_an_empty_document_is_not_a_crash():
    for text in ("", "scheduler:\n", "{}\n"):
        with clean_env():
            cfg = load_scheduler_config(yaml_file(text))
        assert cfg["routing_policy"] == _DEFAULTS["routing_policy"], text


# ─────────────── the two paths that must agree with each other ──────────────

def test_the_launcher_and_the_loader_agree_on_every_variable_name():
    # scheduler/launch.py exports environment variables; config_loader reads
    # them. A name that exists on only one side is a knob that looks set and is
    # not -- which is what happened to MAX_BATCH_SIZE.
    src = (ROOT / "scheduler" / "launch.py").read_text()
    exported = {v[0] for v in _PARAM_MAP.values() if f'"{v[0]}"' in src}
    assert "MAX_BATCH_SIZE" in exported, \
        "the launcher no longer exports MAX_BATCH_SIZE; the ILP cap is 0 again"
    assert "ROUTING_POLICY" in exported and "OSL_SOURCE" in exported, exported


def test_the_fleet_yaml_keys_the_launcher_accepts_are_real_flags():
    # start.sh refuses an unknown key by name, but only against its own list.
    # This checks that list against the launcher's actual flags.
    import re
    sh = (ROOT / "scheduler" / "start.sh").read_text()
    keys = set(re.findall(r'"([a-z_]+)"', sh.split("KEYS = {")[1].split("}")[0]))
    launch = (ROOT / "scheduler" / "launch.py").read_text()
    for k in keys:
        flag = "--" + k.replace("_", "-")
        assert flag in launch, f"start.sh accepts {k!r} but there is no {flag}"


def test_the_shipped_fleet_yaml_only_uses_keys_the_launcher_knows():
    import re
    import yaml as _y
    sh = (ROOT / "scheduler" / "start.sh").read_text()
    keys = set(re.findall(r'"([a-z_]+)"', sh.split("KEYS = {")[1].split("}")[0]))
    doc = _y.safe_load((ROOT / "scheduler" / "fleet.yaml").read_text()) or {}
    unknown = set(doc.get("fleet") or {}) - keys
    assert not unknown, f"fleet.yaml sets keys start.sh would refuse: {unknown}"


@check("test_no_comment_sits_inside_a_shell_continuation")
def _():
    # A `\`-continued line followed by a comment silently TRUNCATES the
    # command: the shell joins them, everything after `#` is discarded, and the
    # next line starts a fresh command. `bash -n` passes -- it is valid syntax,
    # just not the command anyone wrote.
    #
    # It has cost two runs. Once by dropping SIM_DURATION_S from the energy
    # call, giving sim_duration_s=0.0; and once by swallowing
    # `PYTHONPATH=proto:.` from the datacentre launch, so every DC died with
    # ModuleNotFoundError: inference_pb2 and the sweep aborted on the first
    # policy.
    import glob
    bad = []
    for path in glob.glob(str(ROOT / "scripts" / "**" / "*.sh"), recursive=True):
        prev = ""
        for n, line in enumerate(open(path), 1):
            # A backslash at the end of a COMMENT continues nothing -- usage
            # examples inside header comments are full of them.
            prev_is_code = bool(prev.strip()) and not prev.lstrip().startswith("#")
            if (line.lstrip().startswith("#") and prev_is_code
                    and prev.rstrip().endswith("\\")):
                bad.append(f"{path.split('/')[-1]}:{n}")
            prev = line
    assert not bad, f"comment inside a line continuation: {', '.join(bad)}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        check(t.__name__)(t)
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    sys.exit(1 if FAIL else 0)
