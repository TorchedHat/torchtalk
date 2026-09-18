"""Start a worker pool with the given start method, print worker pids, then idle."""

import multiprocessing
import sys
import time

from torchtalk.analysis.cpp_call_graph import _worker_init, _worker_initargs


def work(_):
    time.sleep(300)


if __name__ == "__main__":
    ctx = multiprocessing.get_context(sys.argv[1])
    pool = ctx.Pool(2, _worker_init, _worker_initargs(None))
    print(*(p.pid for p in multiprocessing.active_children()), flush=True)
    pool.map_async(work, range(2))
    time.sleep(300)
