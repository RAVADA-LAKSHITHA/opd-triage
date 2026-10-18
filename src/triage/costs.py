"""Asymmetric triage cost matrix. COST[true-1, pred-1]. Levels: 1 (most urgent) .. 5."""
import numpy as np

LEVELS = [1, 2, 3, 4, 5]
OVER_COST = 1.0        # per level of over-triage (sent too urgent)
ABSTAIN_COST = 2.0     # cost of "cannot assess - send to nurse" (small: nurse looks at patient)


def under_weight(true_level: int) -> float:
    """Per-level cost of UNDER-triage. True emergencies are ~20x worse than a mild over-triage."""
    if true_level <= 2:
        return 20.0
    if true_level == 3:
        return 3.0
    return 1.0


def build_cost_matrix() -> np.ndarray:
    C = np.zeros((5, 5))
    for t in LEVELS:
        for p in LEVELS:
            if p < t:
                C[t - 1, p - 1] = (t - p) * OVER_COST
            elif p > t:
                C[t - 1, p - 1] = (p - t) * under_weight(t)
    return C


COST = build_cost_matrix()