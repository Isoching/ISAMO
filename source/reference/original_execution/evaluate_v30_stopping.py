from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["carbon_utilization", "specific_energy_kwh_per_kg_meoh", "capex_proxy"]


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing stopping-ledger columns: {missing}")


def numeric_limits(specification: object, expected: int = 1) -> list[float]:
    """Extract pre-registered numeric limits from a protocol condition string."""
    values = [float(value) for value in re.findall(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?", str(specification))]
    if len(values) < expected:
        raise ValueError(
            f"Protocol condition does not contain {expected} numeric limit(s): {specification!r}"
        )
    return values[:expected]


def stopping_thresholds(protocol: dict[str, object]) -> dict[str, float | int]:
    stop = protocol["iterative_stop"]
    numeric = stop.get("numeric_conditions")
    if numeric is not None:
        return {
            "consecutive_rounds_required": int(stop["consecutive_rounds_required"]),
            "nrmse_abs_change_pp": float(numeric["nrmse_abs_change_pp_lt"]),
            "mean_mae_relative_change_pct": float(
                numeric["mean_absolute_paired_mae_change_pct_lt"]
            ),
            "hypervolume_relative_change_pct": float(
                numeric["hypervolume_relative_change_pct_lt"]
            ),
            "nrmse_deterioration_pp": float(numeric["nrmse_deterioration_pp_max"]),
            "front_distance": float(numeric["mean_symmetric_front_distance_lt"]),
            "top10_overlap_pct": float(numeric["top10_overlap_pct_min"]),
            "selected_objective_movement": float(
                numeric["selected_objective_movement_lt"]
            ),
            "legacy_relative_worsening_pct": float(
                numeric["legacy_relative_worsening_pct_max"]
            ),
            "legacy_absolute_worsening_pp": float(
                numeric["legacy_absolute_worsening_pp_max"]
            ),
            "latest_batch_convergence_pct": float(
                numeric["latest_batch_convergence_pct_min"]
            ),
        }
    conditions = stop["conditions"]
    legacy_relative, legacy_absolute = numeric_limits(
        conditions["legacy_holdout_guardrail"], expected=2
    )
    return {
        "consecutive_rounds_required": int(stop["consecutive_rounds_required"]),
        "nrmse_abs_change_pp": numeric_limits(
            conditions["each_target_STOP_NRMSE_change_abs_pp"]
        )[0],
        "mean_mae_relative_change_pct": numeric_limits(
            conditions["STOP_mean_paired_MAE_relative_improvement_pct"]
        )[0],
        "hypervolume_relative_change_pct": numeric_limits(
            conditions["fixed_reference_hypervolume_absolute_relative_change_pct"]
        )[0],
        "nrmse_deterioration_pp": numeric_limits(
            conditions["no_STOP_target_NRMSE_deterioration_pp"]
        )[0],
        "front_distance": numeric_limits(
            conditions["mean_symmetric_nearest_front_distance"]
        )[0],
        "top10_overlap_pct": numeric_limits(
            conditions["TOPSIS_and_VIKOR_top10_overlap_pct"]
        )[0],
        "selected_objective_movement": numeric_limits(
            conditions["each_selected_objective_vector_normalised_movement"]
        )[0],
        "legacy_relative_worsening_pct": legacy_relative,
        "legacy_absolute_worsening_pp": legacy_absolute,
        "latest_batch_convergence_pct": numeric_limits(
            conditions["latest_gproms_batch_convergence_pct"]
        )[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen V30 active-learning stopping contract")
    parser.add_argument("--round-metrics", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = pd.read_csv(args.round_metrics, float_precision="round_trip").sort_values("al_round").reset_index(drop=True)
    if metrics.empty:
        raise ValueError("Round metrics table is empty")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    minimum_rounds = int(protocol["active_learning"]["minimum_rounds"])
    maximum_rounds = int(protocol["active_learning"]["maximum_rounds"])
    thresholds = stopping_thresholds(protocol)
    consecutive_required = int(thresholds["consecutive_rounds_required"])
    if consecutive_required < 1:
        raise ValueError("consecutive_rounds_required must be a positive integer")
    required = [
        "al_round",
        "fixed_reference_hypervolume",
        "mean_symmetric_nearest_front_distance",
        "topsis_top10_overlap_pct",
        "vikor_top10_overlap_pct",
        "topsis_selected_objective_movement",
        "vikor_selected_objective_movement",
        "latest_gproms_batch_convergence_pct",
    ]
    for target in TARGETS:
        required += [
            f"stop_{target}_nrmse_pct",
            f"stop_{target}_mae",
            f"legacy_{target}_nrmse_pct",
            f"baseline_legacy_{target}_nrmse_pct",
        ]
    require_columns(metrics, required)
    numeric = metrics[required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Round metrics contain blank, non-numeric or non-finite required values")
    metrics[required] = numeric
    nonnegative = [
        "mean_symmetric_nearest_front_distance",
        "topsis_selected_objective_movement",
        "vikor_selected_objective_movement",
    ]
    for target in TARGETS:
        nonnegative.extend(
            [
                f"stop_{target}_nrmse_pct",
                f"stop_{target}_mae",
                f"legacy_{target}_nrmse_pct",
            ]
        )
        if (metrics[f"baseline_legacy_{target}_nrmse_pct"] <= 0).any():
            raise ValueError(f"baseline_legacy_{target}_nrmse_pct must be positive")
    if (metrics[nonnegative] < 0).any().any():
        raise ValueError("Error, distance and movement metrics must be non-negative")
    if (metrics["fixed_reference_hypervolume"] <= 0).any():
        raise ValueError("fixed_reference_hypervolume must be positive")
    for column in [
        "topsis_top10_overlap_pct",
        "vikor_top10_overlap_pct",
        "latest_gproms_batch_convergence_pct",
    ]:
        if ((metrics[column] < 0) | (metrics[column] > 100)).any():
            raise ValueError(f"{column} must be between 0 and 100")
    round_values = metrics["al_round"].to_numpy(dtype=float)
    if not np.equal(round_values, np.floor(round_values)).all():
        raise ValueError("al_round values must be exact integers; fractional rounds are invalid")
    rounds = round_values.astype(int).tolist()
    if rounds != list(range(rounds[0], rounds[0] + len(rounds))) or rounds[0] != 1:
        raise ValueError("Round metrics must contain consecutive rounds starting at 1")
    if rounds[-1] > maximum_rounds:
        raise ValueError(f"Metrics exceed maximum round {maximum_rounds}")

    evaluations: list[dict[str, object]] = []
    for index in range(1, len(metrics)):
        previous = metrics.iloc[index - 1]
        current = metrics.iloc[index]
        row: dict[str, object] = {"al_round": int(current["al_round"])}
        nrmse_plateau = []
        nrmse_no_deterioration = []
        mae_change = []
        global_guardrail = []
        for target in TARGETS:
            previous_nrmse = float(previous[f"stop_{target}_nrmse_pct"])
            current_nrmse = float(current[f"stop_{target}_nrmse_pct"])
            nrmse_plateau.append(
                abs(current_nrmse - previous_nrmse) < float(thresholds["nrmse_abs_change_pp"])
            )
            nrmse_no_deterioration.append(
                current_nrmse - previous_nrmse
                <= float(thresholds["nrmse_deterioration_pp"])
            )
            previous_mae = float(previous[f"stop_{target}_mae"])
            current_mae = float(current[f"stop_{target}_mae"])
            mae_change.append(100.0 * abs(current_mae - previous_mae) / max(abs(previous_mae), 1e-12))
            legacy = float(current[f"legacy_{target}_nrmse_pct"])
            baseline = float(current[f"baseline_legacy_{target}_nrmse_pct"])
            relative_worsening = 100.0 * (legacy - baseline) / max(abs(baseline), 1e-12)
            absolute_worsening = legacy - baseline
            global_guardrail.append(
                relative_worsening <= float(thresholds["legacy_relative_worsening_pct"])
                and absolute_worsening <= float(thresholds["legacy_absolute_worsening_pp"])
            )
        hv_previous = float(previous["fixed_reference_hypervolume"])
        hv_current = float(current["fixed_reference_hypervolume"])
        hv_abs_relative_change_pct = 100.0 * abs(hv_current - hv_previous) / max(abs(hv_previous), 1e-12)
        checks = {
            "each_target_nrmse_abs_change_within_protocol": all(nrmse_plateau),
            "no_target_nrmse_deterioration_beyond_protocol": all(nrmse_no_deterioration),
            "mean_abs_relative_mae_change_within_protocol": float(np.mean(mae_change))
            < float(thresholds["mean_mae_relative_change_pct"]),
            "hypervolume_abs_relative_change_within_protocol": hv_abs_relative_change_pct
            < float(thresholds["hypervolume_relative_change_pct"]),
            "mean_symmetric_front_distance_within_protocol": float(
                current["mean_symmetric_nearest_front_distance"]
            )
            < float(thresholds["front_distance"]),
            "topsis_top10_overlap_within_protocol": float(current["topsis_top10_overlap_pct"])
            >= float(thresholds["top10_overlap_pct"]),
            "vikor_top10_overlap_within_protocol": float(current["vikor_top10_overlap_pct"])
            >= float(thresholds["top10_overlap_pct"]),
            "topsis_movement_within_protocol": float(
                current["topsis_selected_objective_movement"]
            )
            < float(thresholds["selected_objective_movement"]),
            "vikor_movement_within_protocol": float(
                current["vikor_selected_objective_movement"]
            )
            < float(thresholds["selected_objective_movement"]),
            "legacy_global_guardrails_all_pass": all(global_guardrail),
            "latest_batch_convergence_within_protocol": float(
                current["latest_gproms_batch_convergence_pct"]
            )
            >= float(thresholds["latest_batch_convergence_pct"]),
        }
        row.update(checks)
        row["transition_all_conditions_pass"] = all(checks.values())
        row["hypervolume_abs_relative_change_pct"] = hv_abs_relative_change_pct
        row["mean_abs_relative_mae_change_pct"] = float(np.mean(mae_change))
        evaluations.append(row)

    evaluation = pd.DataFrame(evaluations)
    qualifying_sequence = False
    stop_round: int | None = None
    if len(evaluation) >= consecutive_required:
        transitions = evaluation["transition_all_conditions_pass"].astype(bool).to_numpy()
        for index in range(consecutive_required - 1, len(transitions)):
            window = transitions[index - consecutive_required + 1 : index + 1]
            if bool(np.all(window)):
                candidate_round = int(evaluation.iloc[index]["al_round"])
                if candidate_round >= minimum_rounds:
                    qualifying_sequence = True
                    stop_round = candidate_round
                    break
    if qualifying_sequence:
        development_decision = "STOP_AND_FREEZE_CANDIDATE_MODEL"
    elif rounds[-1] >= maximum_rounds:
        development_decision = "BUDGET_EXHAUSTED_NO_PLATEAU"
    else:
        development_decision = "CONTINUE_ACTIVE_LEARNING"
    decision = {
        "minimum_rounds": minimum_rounds,
        "maximum_rounds": maximum_rounds,
        "latest_round_supplied": rounds[-1],
        "consecutive_qualifying_transitions_required": consecutive_required,
        "required_qualifying_transition_sequence_met": qualifying_sequence,
        "stop_round": stop_round,
        "development_stop_decision": development_decision,
        "thresholds_applied": thresholds,
        "important": "This is a development stopping decision, not final success. Final acceptance still requires unopened independent tests and V30 confirmation96.",
        "transition_evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
