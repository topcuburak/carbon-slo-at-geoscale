"""Provisioner: the loop that gives every scheduler its view of the fleet.

It makes no placement decision of its own. It polls the datacentres, prices
congestion into an RTT matrix, and pushes the result to every scheduler -- so a
defect here corrupts the input of every policy at once rather than one of them.

    peers.py       who the fleet is and how to reach them
    polling.py     asking them, and remembering the last answer
    congestion.py  turning load into RTT
    snapshot.py    delivering it, in one piece or in waves
    loop.py        the cadence, and the guarantee it never stops
    service.py     the read-only gRPC surface
    server.py      entry point
"""
