from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from .core import PROJECT, model_directory, new_output, predict, read, sha256, validate_splits, verify_inference, write_json


def verify_manifest() -> int:
    manifest = json.loads((PROJECT / "SOURCE_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        path = (PROJECT / entry["path"]).resolve()
        if not path.is_relative_to(PROJECT.resolve()) or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Package asset differs: {entry['path']}")
    return len(manifest["files"])


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "select-active":
        sys.argv = ["select_v30_batch.py", *sys.argv[2:]]
        runpy.run_path(str(PROJECT / "reference/original_execution/select_v30_batch.py"), run_name="__main__")
        return
    parser = argparse.ArgumentParser(description="Portable V30 ISAMO data, training, optimisation and interpretation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify", help="Verify package bytes, frozen data identities and result metrics")
    p.add_argument("--models", type=Path, help="Also verify original frozen weights and 140 sampled predictions")
    p.add_argument("--report", type=Path)
    p = sub.add_parser("evaluate", help="Recompute published primary metrics from recorded pointwise predictions")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("train", help="Retrain the original selected learner with real permitted development data")
    p.add_argument("--variant", choices=["pooled_all_al", "pooled_existing60"], default="pooled_all_al")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("predict", help="Predict with frozen or newly trained models")
    p.add_argument("--models", type=Path)
    p.add_argument("--input", type=Path, default=PROJECT / "data/confirmation_94.csv")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("optimise", help="Run strict pymoo NSGA-III with integer cell counts")
    p.add_argument("--models", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--population", type=int, default=256)
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--partitions", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("rank", help="Rank the fixed final archive with TOPSIS and VIKOR")
    p.add_argument("--input", type=Path, default=PROJECT / "data/manuscript_fixed_pool_front.csv")
    p.add_argument("--weights", nargs=3, type=float, default=[1, 1, 1])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("calibrate", help="Refit five-fold grouped cross-conformal using development rows only")
    p.add_argument("--models", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("explain", help="Permutation-SHAP with a reproducible candidate-space background")
    p.add_argument("--models", type=Path)
    p.add_argument("--input", type=Path, default=PROJECT / "data/manuscript_decision_anchors.csv")
    p.add_argument("--eval-rows", type=int, default=5)
    p.add_argument("--background", type=int, default=24)
    p.add_argument("--permutations", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plot", help="Render confirmation parity plots from the frozen recorded evidence")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select-active", help="Original path-configurable AL/confirmation selector")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "select-active":
        sys.argv = ["select_v30_batch.py", *args.arguments]
        runpy.run_path(str(PROJECT / "reference/original_execution/select_v30_batch.py"), run_name="__main__")
        return
    from .analysis import evaluate, plot_validation, rank
    if args.command == "verify":
        report = {"status": "PASS", "manifest_files": verify_manifest(), "splits": validate_splits(),
                  "recomputed_primary_metrics": evaluate().to_dict("records")}
        if args.models:
            report["frozen_inference"] = verify_inference(model_directory(args.models))
        if args.report:
            if args.report.exists():
                raise FileExistsError("Verification report already exists")
            write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return
    output = new_output(args.output)
    if args.command == "evaluate":
        evaluate().to_csv(output / "validation_metrics_recomputed.csv", index=False)
    elif args.command == "rank":
        rank(read(args.input), args.weights).to_csv(output / "pareto_ranked.csv", index=False)
    elif args.command == "plot":
        plot_validation(output)
    elif args.command == "predict":
        frame = read(args.input)
        prediction = predict(model_directory(args.models), frame)
        for column in prediction:
            name = f"{column.removesuffix('_std')}_std_prequery" if column.endswith("_std") else f"{column}_pred"
            frame[name] = prediction[column].to_numpy()
        frame.to_csv(output / "predictions.csv", index=False)
    elif args.command == "calibrate":
        from .calibration import calibrate
        calibrate(model_directory(args.models), output)
    else:
        from .workflow import explain, optimise, train
        if args.command == "train":
            train(output, args.variant)
        elif args.command == "optimise":
            if min(args.population, args.generations, args.partitions) < 1:
                raise ValueError("Positive population, generation and reference-partition counts are required")
            optimise(model_directory(args.models), output, args.seeds, args.population, args.generations, args.partitions)
        elif args.command == "explain":
            if min(args.eval_rows, args.background, args.permutations) < 1:
                raise ValueError("Positive explanation sample counts are required")
            explain(model_directory(args.models), args.input, output, args.eval_rows, args.background, args.permutations)


if __name__ == "__main__":
    main()
