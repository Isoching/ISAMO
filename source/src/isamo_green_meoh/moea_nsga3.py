from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class NSGA3Result:
    X: np.ndarray
    F: np.ndarray  # minimization objectives
    history: List[Dict[str, float]]
    pareto_indices: np.ndarray
    engine: str = "custom.nsga3_like"


def generate_reference_directions_3obj(n_partitions: int) -> np.ndarray:
    dirs = []
    p = int(n_partitions)
    for i in range(p + 1):
        for j in range(p + 1 - i):
            k = p - i - j
            dirs.append([i / p, j / p, k / p])
    return np.asarray(dirs, dtype=float)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    # Specialized for 3 objectives to avoid repeated NumPy allocation overhead in O(N^2) sorting.
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return (a0 <= b0 and a1 <= b1 and a2 <= b2) and (a0 < b0 or a1 < b1 or a2 < b2)


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def _normalize_objectives(F: np.ndarray) -> np.ndarray:
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    scale = np.where((f_max - f_min) < 1e-12, 1.0, f_max - f_min)
    return (F - f_min) / scale


def _associate_to_ref_dirs(F_norm: np.ndarray, ref_dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = ref_dirs / np.clip(np.linalg.norm(ref_dirs, axis=1, keepdims=True), 1e-12, None)
    point_norm = np.linalg.norm(F_norm, axis=1, keepdims=True)
    point_safe = np.where(point_norm < 1e-12, 1.0, point_norm)
    proj = (F_norm @ ref.T)
    dist_sq = np.clip(np.sum(F_norm**2, axis=1, keepdims=True) - proj**2, 0.0, None)
    perp_dist = np.sqrt(dist_sq)
    assoc = np.argmin(perp_dist, axis=1)
    dmin = perp_dist[np.arange(len(F_norm)), assoc]
    return assoc, dmin


def _binary_tournament(ranks: np.ndarray, rng: np.random.Generator, n_select: int) -> np.ndarray:
    n = len(ranks)
    selected = np.empty(n_select, dtype=int)
    for i in range(n_select):
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if ranks[a] < ranks[b]:
            selected[i] = a
        elif ranks[b] < ranks[a]:
            selected[i] = b
        else:
            selected[i] = a if rng.random() < 0.5 else b
    return selected


def _blend_crossover(
    P1: np.ndarray, P2: np.ndarray, bounds: np.ndarray, prob: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = P1.shape
    C1 = P1.copy()
    C2 = P2.copy()
    do_cross = rng.random(n) < prob
    alpha = rng.random((n, d))
    C1[do_cross] = alpha[do_cross] * P1[do_cross] + (1.0 - alpha[do_cross]) * P2[do_cross]
    C2[do_cross] = alpha[do_cross] * P2[do_cross] + (1.0 - alpha[do_cross]) * P1[do_cross]
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    np.clip(C1, lb, ub, out=C1)
    np.clip(C2, lb, ub, out=C2)
    return C1, C2


def _gaussian_mutation(
    X: np.ndarray,
    bounds: np.ndarray,
    prob: float,
    scale: float,
    gen_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    Y = X.copy()
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    span = (ub - lb)
    sigma = span * scale * (0.65 + 0.35 * (1.0 - gen_frac))
    mask = rng.random(Y.shape) < prob
    noise = rng.normal(0.0, sigma, size=Y.shape)
    Y = np.where(mask, Y + noise, Y)
    np.clip(Y, lb, ub, out=Y)
    return Y


def _environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    pop_size: int,
    ref_dirs: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    fronts = fast_non_dominated_sort(F)
    selected: List[int] = []
    last_front: List[int] = []

    for front in fronts:
        if len(selected) + len(front) < pop_size:
            selected.extend(front)
        else:
            last_front = front
            break

    if len(selected) == pop_size:
        return X[selected], F[selected]

    remaining = pop_size - len(selected)
    if remaining <= 0:
        return X[selected[:pop_size]], F[selected[:pop_size]]

    all_idx = np.array(selected + last_front, dtype=int)
    F_norm_all = _normalize_objectives(F[all_idx])
    assoc_all, dist_all = _associate_to_ref_dirs(F_norm_all, ref_dirs)

    n_selected = len(selected)
    assoc_sel = assoc_all[:n_selected] if n_selected > 0 else np.array([], dtype=int)
    niche_count = np.zeros(len(ref_dirs), dtype=int)
    if len(assoc_sel) > 0:
        vals, counts = np.unique(assoc_sel, return_counts=True)
        niche_count[vals] = counts

    cand_local_idx = np.arange(n_selected, len(all_idx), dtype=int)
    pools: Dict[int, List[int]] = {}
    for c in cand_local_idx:
        r = int(assoc_all[c])
        pools.setdefault(r, []).append(int(c))

    picked_local: List[int] = []
    while len(picked_local) < remaining:
        available_dirs = [r for r, members in pools.items() if members]
        if not available_dirs:
            break
        min_count = min(int(niche_count[r]) for r in available_dirs)
        candidate_dirs = [r for r in available_dirs if niche_count[r] == min_count]
        r = int(candidate_dirs[int(rng.integers(0, len(candidate_dirs)))])
        members = pools[r]
        if niche_count[r] == 0:
            chosen_pos = int(np.argmin([dist_all[m] for m in members]))
        else:
            chosen_pos = int(rng.integers(0, len(members)))
        chosen_local = members.pop(chosen_pos)
        picked_local.append(chosen_local)
        niche_count[r] += 1

    if len(picked_local) < remaining:
        leftovers = [m for members in pools.values() for m in members]
        rng.shuffle(leftovers)
        picked_local.extend(leftovers[: remaining - len(picked_local)])

    chosen_idx = list(selected) + [int(all_idx[i]) for i in picked_local[:remaining]]
    return X[chosen_idx], F[chosen_idx]


def nsga3_optimize(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[Tuple[float, float]],
    pop_size: int = 96,
    n_generations: int = 50,
    n_reference_partitions: int = 12,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.2,
    mutation_scale: float = 0.08,
    random_state: int = 42,
) -> NSGA3Result:
    rng = np.random.default_rng(random_state)
    bounds_arr = np.asarray(bounds, dtype=float)
    dim = bounds_arr.shape[0]
    lb = bounds_arr[:, 0]
    ub = bounds_arr[:, 1]
    ref_dirs = generate_reference_directions_3obj(n_reference_partitions)

    X = rng.uniform(lb, ub, size=(pop_size, dim))
    F = evaluate_fn(X)
    history: List[Dict[str, float]] = []

    for gen in range(n_generations):
        fronts = fast_non_dominated_sort(F)
        rank = np.full(pop_size, fill_value=len(fronts), dtype=int)
        for i, front in enumerate(fronts):
            rank[front] = i

        parent_idx = _binary_tournament(rank, rng, n_select=pop_size)
        mating_pool = X[parent_idx]
        P1 = mating_pool[0::2]
        P2 = mating_pool[1::2]
        if len(P1) != len(P2):
            P2 = np.vstack([P2, mating_pool[:1]])
        C1, C2 = _blend_crossover(P1, P2, bounds_arr, crossover_prob, rng)
        offspring = np.vstack([C1, C2])[:pop_size]
        offspring = _gaussian_mutation(
            offspring, bounds_arr, mutation_prob, mutation_scale, gen_frac=gen / max(1, n_generations - 1), rng=rng
        )
        F_off = evaluate_fn(offspring)

        X_comb = np.vstack([X, offspring])
        F_comb = np.vstack([F, F_off])
        X, F = _environmental_selection(X_comb, F_comb, pop_size=pop_size, ref_dirs=ref_dirs, rng=rng)

        fronts_now = fast_non_dominated_sort(F)
        first_front = fronts_now[0]
        history.append(
            {
                "generation": int(gen),
                "nondominated_count": int(len(first_front)),
                "f1_min": float(F[:, 0].min()),
                "f2_min": float(F[:, 1].min()),
                "f3_min": float(F[:, 2].min()),
            }
        )
        if gen in (0, n_generations - 1) or ((gen + 1) % 5 == 0):
            print(
                "    optimizer gen {}/{} | nondominated={}".format(
                    gen + 1, n_generations, len(first_front)
                )
            )

    pareto_idx = np.asarray(fast_non_dominated_sort(F)[0], dtype=int)
    return NSGA3Result(X=X, F=F, history=history, pareto_indices=pareto_idx, engine="custom.nsga3_like")
