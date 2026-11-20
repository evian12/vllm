# SPDX-License-Identifier: Apache-2.0
"""Conservative read-only eligibility for one fixed-H Ready coalescing wait."""


def can_defer_ready(*, window_open, steps, horizon, capacity, ready_ids,
                    worker_ready_ids, leased_ids, draft_wait_ids, inflight_ids,
                    waiting_count, running_rows):
    if (not window_open or horizon != 8 or capacity != 4
            or not 0 < steps < horizon or waiting_count or inflight_ids
            or not 0 < len(ready_ids) < capacity):
        return False
    running = [rid for rid, healthy in running_rows]
    if len(set(running)) != len(running) or not all(ok for _, ok in running_rows):
        return False
    running = set(running)
    ready, worker, leased = set(ready_ids), set(worker_ready_ids), set(leased_ids)
    if (ready & leased or worker & ready or not worker <= leased
            or not ready <= running or not leased <= running or not worker <= running
            or not running <= ready | leased | set(draft_wait_ids)):
        return False
    # Already-completed Worker candidates guarantee a full batch at the next
    # existing H boundary. Never wait speculatively for confidence/arrivals.
    return len(ready) + len(worker) >= capacity
