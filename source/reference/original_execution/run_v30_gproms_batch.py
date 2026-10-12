"""Fail-closed V30 gPROMS batch runner contract and schema dry-run.

The dry-run writes only audit evidence.  It never launches gPROMS and never
requests a label.  Live execution remains separately gated by the aggregate
Phase-0 approval and the approved GUI/JAB operating procedure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNNER_VERSION = "V30-gproms-runner-1.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate_contract(script_path: Path) -> Any:
    gate_path = script_path.with_name("evaluate_v30_phase0_gate.py")
    spec = importlib.util.spec_from_file_location("v30_phase0_gate_for_runner", gate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the frozen Phase-0 schema contract: {gate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def audit_queue(queue: Path, gate: Any) -> dict[str, Any]:
    with queue.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    missing = [name for name in gate.REQUIRED_RUNNER_INPUT_COLUMNS if name not in header]
    if missing:
        raise ValueError(f"Queue lacks required runner columns: {missing}")
    if not rows:
        raise ValueError("Queue must contain at least one design")

    record_ids: list[str] = []
    input_hashes: list[str] = []
    mapping_columns = {
        "Current_density_eCO2R_Acm2": "J_eCO2R",
        "WE_Energy_input_rate_KW": "P_WE",
        "Number_of_cells_stacked_in_series_WE": "N_WE",
        "Number_of_stacks_eCO2R": "N_eCO2R",
        "Outlet_pressure": "P_MeOH",
        "Outlet_pressure_cap": "P_Cap",
        "T_eCO2R_gproms_input": "T_eCO2R",
    }
    for number, row in enumerate(rows, start=2):
        record_id = row.get("record_id", "").strip()
        if not record_id:
            raise ValueError(f"Queue row {number}: blank record_id")
        if not row.get("dataset_role", "").strip():
            raise ValueError(f"Queue row {number}: blank dataset_role")
        canonical = gate.canonical_feature_string(row)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if row.get("input_row_sha256") != digest:
            raise ValueError(f"Queue row {number}: input_row_sha256 mismatch")
        for mapped, source in mapping_columns.items():
            if not math.isclose(
                float(row[mapped]), float(row[source]), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"Queue row {number}: mapping mismatch for {mapped}")
        if not math.isclose(
            float(row["Utility_constant_temperature"]),
            float(row["T_utility"]) - 273.15,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Queue row {number}: mapping mismatch for Utility_constant_temperature"
            )
        record_ids.append(record_id)
        input_hashes.append(digest)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Queue contains duplicate record_id values")
    if len(set(input_hashes)) != len(input_hashes):
        raise ValueError("Queue contains duplicate canonical input hashes")
    return {
        "rows": len(rows),
        "record_ids_unique": True,
        "input_hashes_unique": True,
        "all_input_hashes_recomputed": True,
        "all_gproms_mappings_recomputed": True,
    }


def dry_run(
    script_path: Path,
    queue: Path,
    project: Path,
    output_dir: Path,
) -> Path:
    gate = load_gate_contract(script_path)
    queue = queue.resolve()
    project = project.resolve()
    output_dir = output_dir.resolve()
    if not queue.is_file():
        raise FileNotFoundError(queue)
    if not project.is_file():
        raise FileNotFoundError(project)
    if sha256(project) != gate.EXPECTED_GPROMS_PROJECT_SHA256:
        raise ValueError("Project is not the approved frozen B7E0ED3D gPROMS project")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty evidence directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    queue_audit = audit_queue(queue, gate)
    output_schema_path = output_dir / "v30_gproms_return_schema.json"
    write_json(
        output_schema_path,
        {
            "schema_version": "V30-gproms-return-schema-1.0.0",
            "columns": list(gate.REQUIRED_RUNNER_OUTPUT_COLUMNS),
        },
    )

    runner_binding = bind(script_path)
    project_binding = bind(project)
    queue_binding = bind(queue)
    schema_binding = bind(output_schema_path)
    report_path = output_dir / "runner_schema_dry_run_report.json"
    write_json(
        report_path,
        {
            "schema_version": "V30-gproms-runner-schema-dry-run-report-1.0.0",
            "runner_version": RUNNER_VERSION,
            "status": "PASS",
            "dry_run": True,
            "writes_performed": False,
            "evidence_files_written": True,
            "gproms_process_launched": False,
            "labels_requested": 0,
            "runner_sha256": runner_binding["sha256"],
            "frozen_project_sha256": project_binding["sha256"],
            "input_queue_sha256": queue_binding["sha256"],
            "output_schema_sha256": schema_binding["sha256"],
            "queue_audit": queue_audit,
            "execution_boundary": (
                "SCHEMA_ONLY_NO_PROCESS_LAUNCH; live GUI/JAB execution requires a separate "
                "aggregate Phase-0 result with authorizes_phase1_execution=true"
            ),
        },
    )
    log_path = output_dir / "runner_schema_dry_run.log"
    log_path.write_text(
        "\n".join(
            [
                f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
                f"runner_version={RUNNER_VERSION}",
                "status=PASS",
                "dry_run=true",
                "gproms_process_launched=false",
                "labels_requested=0",
                f"queue_rows={queue_audit['rows']}",
                f"frozen_project_sha256={project_binding['sha256']}",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    evidence_path = output_dir / "phase0_gproms_runner_schema_dry_run_evidence.json"
    artifacts = {
        "runner": runner_binding,
        "frozen_project": project_binding,
        "input_queue": queue_binding,
        "output_schema": schema_binding,
        "dry_run_report": bind(report_path),
        "dry_run_log": bind(log_path),
    }
    write_json(
        evidence_path,
        {
            "schema_version": "V30-gproms-runner-schema-dry-run-evidence-1.0.0",
            "record_status": "FINAL_EVIDENCE",
            "template": False,
            "dry_run": True,
            "dry_run_status": "PASS",
            "gproms_process_launched": False,
            "labels_requested": 0,
            "required_input_columns": list(gate.REQUIRED_RUNNER_INPUT_COLUMNS),
            "required_return_columns": list(gate.REQUIRED_RUNNER_OUTPUT_COLUMNS),
            "artifacts": artifacts,
        },
    )
    return evidence_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the V30 gPROMS batch mapping and materialise schema-only evidence"
    )
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = dry_run(Path(__file__).resolve(), args.queue, args.project, args.output_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "dry_run": True,
                "gproms_process_launched": False,
                "labels_requested": 0,
                "evidence": str(evidence),
                "evidence_sha256": sha256(evidence),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
