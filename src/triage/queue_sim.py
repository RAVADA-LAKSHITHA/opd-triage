"""Simple OPD queue simulation: FCFS vs priority-by-triage-level.

Non-preemptive, `doctors` servers, Poisson arrivals, fixed consult time.
Parameters are ASSUMPTIONS until the real arrival log is available.
"""
import heapq

import numpy as np


def simulate(levels, arrival_gap=4.0, doctors=2, service=7.0, priority=True, seed=0):
    """Return wait time (minutes) per patient. Lower level number = served first."""
    levels = np.asarray(levels)
    n = len(levels)
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(arrival_gap, n))
    free = [0.0] * doctors
    heapq.heapify(free)
    waiting, waits = [], np.zeros(n)
    i = done = 0
    while done < n:
        d = heapq.heappop(free)
        if not waiting and i < n and arrivals[i] > d:
            d = arrivals[i]                      # doctor idle until next arrival
        while i < n and arrivals[i] <= d:
            key = (int(levels[i]) if priority else 0, arrivals[i], i)
            heapq.heappush(waiting, key)
            i += 1
        _, _, idx = heapq.heappop(waiting)
        waits[idx] = d - arrivals[idx]
        heapq.heappush(free, d + service)
        done += 1
    return waits


def compare(true_emergency, priority_levels, **kw):
    true_emergency = np.asarray(true_emergency).astype(bool)
    fcfs = simulate(priority_levels, priority=False, **kw)
    prio = simulate(priority_levels, priority=True, **kw)
    ef, ep = float(fcfs[true_emergency].mean()), float(prio[true_emergency].mean())
    return {
        "emergency_wait_fcfs": ef,
        "emergency_wait_priority": ep,
        "emergency_wait_reduction_pct": 100 * (ef - ep) / ef if ef > 0 else 0.0,
        "all_wait_fcfs": float(fcfs.mean()),
        "all_wait_priority": float(prio.mean()),
    }