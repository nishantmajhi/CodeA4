"""
RQ worker entrypoint. Run as its own process, separate from the Flask web
server (see Backend Plan §1 — Flask never runs pipeline stages inline).

    python worker.py
"""
from rq import Worker

from extensions import get_queue, get_redis

if __name__ == "__main__":
    queue = get_queue()
    worker = Worker([queue], connection=get_redis())
    worker.work()
