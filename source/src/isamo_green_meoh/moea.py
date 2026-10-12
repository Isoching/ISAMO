from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .moea_nsga3 import NSGA3Result, fast_non_dominated_sort, nsga3_optimize as nsga3_optimize_fallback


def _nsga3_optimize_pymoo(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[Tuple[float, float]],
    pop_size: int,
    n_generations: int,
    n_reference_partitions: int,
    crossover_prob: float,
    mutation_prob: float,
    mutation_scale: float,
    random_state: int,
    history_callback: Optional[Callable[[int, np.ndarray, np.ndarray], Dict[str, float]]] = None,
) -> NSGA3Result:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    bounds_arr = np.asarray(bounds, dtype=float)
    xl = bounds_arr[:, 0]
    xu = bounds_arr[:, 1]

    class _SurrogateProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(bounds), n_obj=3, n_ieq_constr=0, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = evaluate_fn(np.asarray(X, dtype=float))

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_reference_partitions)
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=20),
        mutation=PM(prob=mutation_prob, eta=20),
        eliminate_duplicates=True,
    )

    history: List[Dict[str, float]] = []

    class _LightweightHistory(Callback):
        """Store only scalar diagnostics, not deep copies of the algorithm/models."""

        def notify(self, current_algorithm):
            try:
                pop_X = np.asarray(current_algorithm.pop.get("X"), dtype=float)
                pop_F = np.asarray(current_algorithm.pop.get("F"), dtype=float)
                nd = NonDominatedSorting(method="efficient_non_dominated_sort").do(
                    pop_F,
                    only_non_dominated_front=True,
                )
                generation = len(history)
                row = {
                    "generation": int(generation),
                    "nondominated_count": int(len(nd)),
                    "f1_min": float(pop_F[:, 0].min()),
                    "f2_min": float(pop_F[:, 1].min()),
                    "f3_min": float(pop_F[:, 2].min()),
                }
                if history_callback is not None:
                    extra = history_callback(generation, pop_X, pop_F)
                    if extra:
                        row.update(extra)
                history.append(row)
            except Exception:
                # Optimisation must not fail because an optional diagnostic did.
                pass

    res = minimize(
        _SurrogateProblem(),
        algorithm,
        termination=("n_gen", int(n_generations)),
        seed=int(random_state),
        verbose=False,
        save_history=False,
        callback=_LightweightHistory(),
    )

    if res.X is None or res.F is None:
        raise RuntimeError("pymoo returned no solutions.")

    X = np.asarray(res.X, dtype=float)
    F = np.asarray(res.F, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if F.ndim == 1:
        F = F.reshape(1, -1)

    return NSGA3Result(
        X=X,
        F=F,
        history=history,
        pareto_indices=np.arange(len(F), dtype=int),
        engine="pymoo.nsga3",
    )


def nsga3_optimize_auto(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[Tuple[float, float]],
    pop_size: int = 96,
    n_generations: int = 50,
    n_reference_partitions: int = 12,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.2,
    mutation_scale: float = 0.08,
    random_state: int = 42,
    prefer_pymoo: bool = True,
    history_callback: Optional[Callable[[int, np.ndarray, np.ndarray], Dict[str, float]]] = None,
) -> NSGA3Result:
    if prefer_pymoo:
        try:
            print("    optimizer backend: pymoo NSGA-III")
            return _nsga3_optimize_pymoo(
                evaluate_fn=evaluate_fn,
                bounds=bounds,
                pop_size=pop_size,
                n_generations=n_generations,
                n_reference_partitions=n_reference_partitions,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                mutation_scale=mutation_scale,
                random_state=random_state,
                history_callback=history_callback,
            )
        except Exception as e:
            print(f"    pymoo NSGA-III unavailable/failed ({type(e).__name__}); falling back to custom NSGA-III-like.")

    result = nsga3_optimize_fallback(
        evaluate_fn=evaluate_fn,
        bounds=bounds,
        pop_size=pop_size,
        n_generations=n_generations,
        n_reference_partitions=n_reference_partitions,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        mutation_scale=mutation_scale,
        random_state=random_state,
    )
    result.engine = "custom.nsga3_like"
    return result
