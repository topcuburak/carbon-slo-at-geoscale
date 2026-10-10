# Everything that checks the simulator, in one place

Two kinds, and the difference is whether anything can fail.

## Assertions — seconds, no processes, pass or fail

    test_inference_engine.py    the engine. Tokens conserved, the admission
                                  bound respected, counters consistent while
                                  running, energy under the device ceiling,
                                  long prompts split at the cap, step time
                                  matching the model when uncontended.
                                  7 checks, 12 assertions
    test_datacenter.py            the layers above it. Topology and host-power
                                  allocation, next_node's rotation, dispatch
                                  reporting where a request really ran, counter
                                  aggregation, the energy split, carbon, a dead
                                  engine giving its counters back, and a static
                                  check that no Datacenter method reaches past a
                                  node into an engine. 24 checks, 52 assertions

    python scripts/test/test_inference_engine.py
    python scripts/test/test_datacenter.py

These are CONSISTENCY checks, not accuracy ones: they ask whether the simulator
contradicts itself -- whether a number reported by one route matches the same
number reached by another. Whether it matches hardware is
`testbed/phase2_endtoend/within_node_sim.py`, replaying real traces, which takes
minutes.

The datacentre suite was itself checked by breaking the code: nine deliberate
defects, three of them bugs that were really present that week, and it caught
eight. The ninth had no coverage -- nothing induced an engine fault -- which is
why `test_a_dead_engine_gives_its_counters_back` exists.

## Fleet checks — real processes over gRPC, they report

    smoke_test_topology.sh   smallest useful fleet at one arrival rate. Ends
                             with a verdict on scheduler p90
    smoke_test_models.sh     the same, once per model
    quick_test.sh            round_robin against carbon_aware at one alpha
    local_test.sh            stands the fleet up and leaves it running
    sweep_test.sh            a short sweep across policies and alpha
    test_verbosity.sh        2 groups x 2 DCs, high log verbosity
    validate_single_dc.sh    latency accuracy on one datacentre across arrival
                             rates. Expects the repository root as cwd

## Not here

    scripts/benchmarks/   in-process runs that print a table and assert nothing
    testbed/              the real 4xA100 box
