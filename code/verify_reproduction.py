from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def close(actual: float, expected: float, tolerance: float = 5e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def main(package_root: Path) -> None:
    expected = json.loads((package_root / "expected_results" / "expected_key_results.json").read_text(encoding="utf-8"))
    result_path = package_root / "reproduced_outputs" / "Reproduced_Model_Results.xlsx"
    summary_path = package_root / "reproduced_outputs" / "reproduction_summary.json"
    performance = pd.read_excel(result_path, sheet_name="Test_Performance")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    first_order = performance.loc[performance["model"] == "First-order Markov"].iloc[0]

    checks = {
        "modeling_routes": summary["modeling_routes"] == expected["modeling_routes"],
        "dated_modeling_routes": summary["dated_modeling_routes"] == expected["dated_modeling_routes"],
        "training_routes": summary["training_routes"] == expected["training_routes"],
        "validation_routes": summary["validation_routes"] == expected["validation_routes"],
        "test_routes": summary["test_routes"] == expected["test_routes"],
        "test_transitions": summary["test_transitions"] == expected["test_transitions"],
        "first_order_hit_at_10": close(first_order["Hit@10"], expected["first_order_hit_at_10"]),
        "first_order_ndcg_at_10": close(first_order["NDCG@10"], expected["first_order_ndcg_at_10"]),
        "candidate_cutoff": summary["candidate_cutoff"] == expected["final_candidate_cutoff"],
    }
    report = {"all_checks_passed": all(checks.values()), "checks": checks}
    output = package_root / "reproduced_outputs" / "verification_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify key reproduced values against the archived targets.")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    main(arguments.package_root.resolve())
