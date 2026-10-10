# Runs that produce a table, not a verdict

Three in-process simulations that were named `test_*` and kept beside the
assertion suites,
where they answered no question anyone ran the suite to ask. Between them they
are 1116 lines and contain zero assertions: they drive the simulator and print
what happened.

    single_dc.py     one datacentre, latency against a benchmark CSV.
                     Needs --csv; it is a CLI tool
    multi_dc.py      several datacentres, load-aware routing. Prints avg/p50/
                     p99 per policy
    carbon_aware.py  the carbon/latency trade: round-robin against carbon_aware
                     at several alpha, with energy, carbon and forwarding rate

Read the numbers; nothing here fails. The checks that do fail are `../test/`.

These run the simulator IN PROCESS -- no gRPC, no fleet, no ports. That is what
separates them from `../test/`, which stands the real distributed system up and
talks to it over the wire. Both print numbers; only one of them needs the
system running.
