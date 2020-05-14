import logging
from concurrent.futures import ThreadPoolExecutor

_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def run_cpu_bound(target, args=()):
    logging.info('run_cpu_bound: %s%r', target, args)
    if args:
        def target(): return target(*args)
    _EXECUTOR.submit(target)
