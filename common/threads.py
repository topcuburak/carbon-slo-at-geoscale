"""Thread stack sizing for the simulator's servers.

Every level of this simulator is synchronous: a request in flight parks a
thread in the load generator, in the scheduler that took it, in the scheduler
it was forwarded to, and in the datacentre serving it. Thread count therefore
tracks CONCURRENCY, not request rate, and concurrency is what a concentrating
policy multiplies -- pile the fleet's work onto two sites and the queues there
hold far more requests at once than an even spread would.

**This does not raise the thread ceiling, and it is not the fix for
`RuntimeError: can't start new thread`.** Measured, spawning threads until
failure on a 16-core / 15 GB machine:

    stack  8192 KB  ->  17892 threads,  53754 maps,  VmSize 140.3 GB
    stack   512 KB  ->  17892 threads,  53739 maps,  VmSize   9.2 GB

Identical ceiling. What binds is `vm.max_map_count`, 65530 by default: each
thread costs about three mappings, and 65530/3.66 is 17900. Raising it is a
sysctl and needs root:

    sudo sysctl -w vm.max_map_count=262144

What the smaller stack IS worth is the third column -- 140 GB of reserved
address space down to 9.2 GB. On a 15 GB box that is real memory pressure
removed even though none of it was resident, and it costs nothing: these
handlers block on a semaphore or a socket and do no deep recursion. 512 KB is
ample. Set SIM_THREAD_STACK_KB to change it, or 0 to keep the platform default.

Why any of this matters: two runs have died on the thread ceiling, at 400 req/s
spread evenly and at 200 req/s once `carbon_aware` alpha=0.0 concentrated the
load onto two sites. Both times a scheduler stopped serving and the failure was
the bad kind -- gRPC went quiet, the load generator's 2000 s per-request
timeout kept waiting, and the run HUNG rather than failed. Fourteen minutes in,
every datacentre was empty and nothing said so.

Call `set_stack_size()` at the top of a server's main(), before anything
creates a thread -- the setting applies only to threads started afterwards.
"""

import logging
import os
import threading

log = logging.getLogger(__name__)

DEFAULT_STACK_KB = int(os.environ.get("SIM_THREAD_STACK_KB", "512"))


def set_stack_size(kb: int | None = None) -> int:
    """Shrink the stack reserved per thread. Returns the value in use.

    A platform that refuses the size (it must be a multiple of the page size
    and at least 32 KB) keeps the default rather than failing the process:
    running with fat threads is worse than not running, but only just.
    """
    kb = DEFAULT_STACK_KB if kb is None else kb
    if kb <= 0:
        return 0
    try:
        threading.stack_size(kb * 1024)
        return kb
    except (ValueError, RuntimeError) as e:
        log.warning("thread stack size %d KB refused (%s); keeping the "
                    "platform default, which is 8 MB on Linux and limits this "
                    "process to roughly 15k threads", kb, e)
        return 0
