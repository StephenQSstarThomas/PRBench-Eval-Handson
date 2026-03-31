"""Grader module for evaluating paper reproduction results against ground truth."""

import csv
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

# Expected CSV files and their key properties
EXPECTED_FILES = {
    "fig1_convergence.csv": {
        "description": "Average plaquette convergence at beta=2.3",
        "required_columns": [
            "iteration", "L4_cold", "L4_hot", "L6_cold", "L6_hot",
            "L8_cold", "L8_hot", "L10_cold", "L10_hot",
        ],
        "min_rows": 31,
    },
    "fig2_evolution.csv": {
        "description": "Plaquette evolution at different beta values",
        "required_columns": [
            "iteration", "beta_1.2", "beta_1.6", "beta_2.0",
            "beta_2.4", "beta_2.8", "beta_3.2",
        ],
        "min_rows": 31,
    },
    "fig3_wilson_vs_size.csv": {
        "description": "Wilson loops vs lattice size",
        "required_columns": ["loop_size", "lattice_size", "mean", "std"],
        "min_rows": 12,  # Not all loop sizes fit on small lattices
    },
    "fig4_wilson_vs_beta.csv": {
        "description": "Wilson loops vs beta",
        "required_columns": ["beta", "W1x1", "W2x2", "W3x3", "W4x4", "W5x5"],
        "min_rows": 14,
    },
    "fig5_wilson_data.csv": {
        "description": "Wilson loop data for string tension fitting",
        "required_columns": ["beta", "loop_size", "wilson_mean", "wilson_std"],
        "min_rows": 20,
    },
    "fig6_string_tension.csv": {
        "description": "String tension vs beta",
        "required_columns": ["beta", "string_tension"],
        "min_rows": 12,
    },
}

EXPECTED_CODE_FILES = [
    "su2_lattice.py",
    "fig1_convergence.py",
    "fig2_evolution.py",
    "fig3_wilson_vs_size.py",
    "fig4_wilson_vs_beta.py",
    "fig5_wilson_fits.py",
    "fig6_string_tension.py",
]

# Relative tolerance for numerical comparison
RELATIVE_TOLERANCE = 0.30  # 30% tolerance for Monte Carlo stochastic results
ABSOLUTE_TOLERANCE = 0.05  # absolute tolerance for values near zero


def load_csv(path: str) -> list[dict[str, str]]:
    """Load a CSV file, skipping comment lines starting with #."""
    rows = []
    with open(path, "r") as f:
        # Skip comment lines
        lines = [line for line in f if not line.strip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(row)
    return rows


def compare_values(expected: float, actual: float) -> tuple[bool, float]:
    """Compare two float values with relative + absolute tolerance.

    Returns (is_close, relative_error).
    """
    if expected == 0.0:
        err = abs(actual)
        return err < ABSOLUTE_TOLERANCE, err
    rel_err = abs(actual - expected) / abs(expected)
    abs_err = abs(actual - expected)
    is_close = rel_err < RELATIVE_TOLERANCE or abs_err < ABSOLUTE_TOLERANCE
    return is_close, rel_err


def grade_csv_file(
    gt_path: str, agent_path: str, file_spec: dict
) -> dict[str, Any]:
    """Grade a single CSV file against ground truth.

    Returns a dict with:
      - score: float 0.0-1.0
      - file_exists: bool
      - columns_correct: bool
      - row_count_ok: bool
      - value_accuracy: float (fraction of values within tolerance)
      - details: str
    """
    result: dict[str, Any] = {
        "score": 0.0,
        "file_exists": False,
        "columns_correct": False,
        "row_count_ok": False,
        "value_accuracy": 0.0,
        "details": "",
    }

    # Check file exists
    if not os.path.exists(agent_path):
        result["details"] = f"File not found: {agent_path}"
        return result
    result["file_exists"] = True

    # Load both CSVs
    try:
        gt_rows = load_csv(gt_path)
        agent_rows = load_csv(agent_path)
    except Exception as e:
        result["details"] = f"Error loading CSV: {e}"
        result["score"] = 0.05  # small credit for file existing
        return result

    # Check columns
    if not agent_rows:
        result["details"] = "Agent CSV is empty"
        result["score"] = 0.05
        return result

    agent_columns = list(agent_rows[0].keys())
    required = file_spec["required_columns"]
    missing_cols = [c for c in required if c not in agent_columns]
    if missing_cols:
        result["details"] = f"Missing columns: {missing_cols}"
        result["score"] = 0.1
        return result
    result["columns_correct"] = True

    # Check row count
    min_rows = file_spec["min_rows"]
    if len(agent_rows) < min_rows:
        result["details"] = (
            f"Too few rows: {len(agent_rows)} < {min_rows}"
        )
        result["score"] = 0.15
        return result
    result["row_count_ok"] = True

    # Compare values
    total_values = 0
    close_values = 0
    total_rel_error = 0.0
    comparison_errors = []

    # Compare row by row for matching keys
    num_compare_rows = min(len(gt_rows), len(agent_rows))
    for i in range(num_compare_rows):
        for col in required:
            if col in gt_rows[i] and col in agent_rows[i]:
                gt_str = gt_rows[i][col].strip()
                agent_str = agent_rows[i][col].strip()
                # Skip empty cells (e.g., missing W5x5 for extreme beta)
                if not gt_str or not agent_str:
                    continue
                try:
                    gt_val = float(gt_str)
                    agent_val = float(agent_str)
                    total_values += 1
                    is_close, rel_err = compare_values(gt_val, agent_val)
                    if is_close:
                        close_values += 1
                    else:
                        if len(comparison_errors) < 5:
                            comparison_errors.append(
                                f"  Row {i}, col '{col}': expected={gt_val:.6f}, got={agent_val:.6f}, err={rel_err:.2%}"
                            )
                    if not math.isnan(rel_err):
                        total_rel_error += min(rel_err, 1.0)
                except (ValueError, TypeError):
                    total_values += 1

    if total_values == 0:
        result["details"] = "No comparable numeric values found"
        result["score"] = 0.2
        return result

    accuracy = close_values / total_values
    result["value_accuracy"] = accuracy

    # Score: 20% for structure, 80% for accuracy
    structure_score = 0.2  # file exists + correct columns + correct rows
    accuracy_score = 0.8 * accuracy
    result["score"] = structure_score + accuracy_score

    details_parts = [
        f"Values within tolerance: {close_values}/{total_values} ({accuracy:.1%})",
    ]
    if comparison_errors:
        details_parts.append("Sample mismatches:")
        details_parts.extend(comparison_errors)
    result["details"] = "\n".join(details_parts)

    return result


def grade_analysis_md(analysis_path: str) -> dict[str, Any]:
    """Grade the ANALYSIS.md file for completeness of methodology description.

    Checks for presence of key concepts that should be discussed.
    """
    result: dict[str, Any] = {
        "score": 0.0,
        "file_exists": False,
        "keyword_hits": {},
        "details": "",
    }

    if not os.path.exists(analysis_path):
        result["details"] = "ANALYSIS.md not found"
        return result
    result["file_exists"] = True

    with open(analysis_path, "r") as f:
        content = f.read().lower()

    # Key concepts that should be present in a thorough analysis
    key_concepts = {
        "su2_group": ["su(2)", "su2", "special unitary"],
        "quaternion": ["quaternion", "a_0", "a0", "four-vector"],
        "plaquette": ["plaquette"],
        "heat_bath": ["heat bath", "heat-bath", "heatbath"],
        "wilson_loop": ["wilson loop", "wilson_loop"],
        "string_tension": ["string tension"],
        "asymptotic_freedom": ["asymptotic freedom"],
        "coupling": ["coupling", "beta", "β"],
        "lattice_action": ["lattice action", "action", "plaquette action"],
        "monte_carlo": ["monte carlo"],
        "thermalization": ["thermalization", "thermaliz"],
        "rejection_sampling": ["rejection", "sampling", "accept"],
        "confinement": ["confinement", "confin"],
        "periodic_boundary": ["periodic", "boundary"],
    }

    hits = {}
    for concept, keywords in key_concepts.items():
        hits[concept] = any(kw in content for kw in keywords)

    result["keyword_hits"] = hits
    hit_count = sum(1 for v in hits.values() if v)
    total = len(key_concepts)
    result["score"] = hit_count / total

    missing = [k for k, v in hits.items() if not v]
    result["details"] = (
        f"Concept coverage: {hit_count}/{total}\n"
        f"Missing: {missing if missing else 'none'}"
    )
    return result


def grade_code_files(reproduction_dir: str) -> dict[str, Any]:
    """Grade the existence and basic quality of code files."""
    result: dict[str, Any] = {
        "score": 0.0,
        "files_found": [],
        "files_missing": [],
        "details": "",
    }

    found = 0
    for fname in EXPECTED_CODE_FILES:
        path = os.path.join(reproduction_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 100:
            result["files_found"].append(fname)
            found += 1
        else:
            result["files_missing"].append(fname)

    total = len(EXPECTED_CODE_FILES)
    result["score"] = found / total if total > 0 else 0.0
    result["details"] = (
        f"Code files found: {found}/{total}\n"
        f"Missing: {result['files_missing'] if result['files_missing'] else 'none'}"
    )
    return result


def grade_task(
    task_dir: str,
    workspace_dir: str,
) -> dict[str, Any]:
    """Run full grading for a paper reproduction task.

    Args:
        task_dir: Path to the task directory containing ground truth data/
        workspace_dir: Path to the agent's workspace containing reproduction/ and data/

    Returns:
        Comprehensive grading result dict.
    """
    gt_data_dir = os.path.join(task_dir, "data")
    agent_data_dir = os.path.join(workspace_dir, "data")
    agent_repro_dir = os.path.join(workspace_dir, "reproduction")
    analysis_path = os.path.join(agent_repro_dir, "ANALYSIS.md")

    overall_result: dict[str, Any] = {
        "task_id": os.path.basename(task_dir),
        "csv_grades": {},
        "analysis_grade": {},
        "code_grade": {},
        "overall_score": 0.0,
        "summary": "",
    }

    # Grade each CSV file (60% of total score)
    csv_scores = []
    for fname, spec in EXPECTED_FILES.items():
        gt_path = os.path.join(gt_data_dir, fname)
        agent_path = os.path.join(agent_data_dir, fname)
        grade = grade_csv_file(gt_path, agent_path, spec)
        overall_result["csv_grades"][fname] = grade
        csv_scores.append(grade["score"])
        logger.info(f"CSV grade for {fname}: {grade['score']:.2f} - {grade['details']}")

    avg_csv_score = sum(csv_scores) / len(csv_scores) if csv_scores else 0.0

    # Grade ANALYSIS.md (25% of total score)
    analysis_grade = grade_analysis_md(analysis_path)
    overall_result["analysis_grade"] = analysis_grade
    logger.info(f"Analysis grade: {analysis_grade['score']:.2f} - {analysis_grade['details']}")

    # Grade code files (15% of total score)
    code_grade = grade_code_files(agent_repro_dir)
    overall_result["code_grade"] = code_grade
    logger.info(f"Code grade: {code_grade['score']:.2f} - {code_grade['details']}")

    # Weighted overall score
    overall_result["overall_score"] = (
        0.60 * avg_csv_score
        + 0.25 * analysis_grade["score"]
        + 0.15 * code_grade["score"]
    )

    overall_result["summary"] = (
        f"Overall Score: {overall_result['overall_score']:.2%}\n"
        f"  CSV Data Accuracy (60%): {avg_csv_score:.2%}\n"
        f"  Analysis Completeness (25%): {analysis_grade['score']:.2%}\n"
        f"  Code Completeness (15%): {code_grade['score']:.2%}"
    )

    return overall_result
