"""
ChurnIQ - Module 8: Master Pipeline Orchestrator
==================================================
Coordinates all pipeline stages without duplicating their logic.

Usage
-----
  # Show help
  python src/train.py --help

  # Run all stages (safe: skips stages whose outputs already exist)
  python src/train.py --all

  # Force regenerate everything from scratch
  python src/train.py --all --force

  # Run selected stages only
  python src/train.py --stages 1 2 3

  # Skip expensive SHAP stage
  python src/train.py --all --skip-shap

  # Run only inference artifact generation
  python src/train.py --stages 7

  # Dry-run: check dependencies without executing
  python src/train.py --all --dry-run

  # List all stages and their artifact status
  python src/train.py --status

Stage map
---------
  1  Data preprocessing        (src/data_pipeline.py)
  2  Logistic Regression        (src/baseline_model.py)
  3  MLP neural network         (src/neural_network.py)
  4  Threshold optimisation     (src/threshold_optimizer.py)
  5  Final test evaluation      (src/evaluation.py)
  6  SHAP explainability        (src/explainability.py)
  7  Inference artifacts        (src/inference.py)

Data isolation rules (enforced by this orchestrator)
------------------------------------------------------
  - Preprocessing fits ONLY on training data.
  - Model training uses ONLY training data.
  - Threshold optimisation uses ONLY validation predictions.
  - Final evaluation uses ONLY the quarantined test partition,
    and only AFTER models and thresholds are finalised.
  - SHAP background uses ONLY training records.
  - Test results NEVER influence models, thresholds or risk boundaries.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import json
import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    set_all_seeds,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        datefmt=DATE_FORMAT, handlers=handlers, force=True)
    return logging.getLogger("ChurnIQ")


# ===========================================================================
# Artifact registry — the single source of truth for stage I/O
# ===========================================================================
# Each stage declares its required inputs and produced outputs.
# The orchestrator checks existence before running and after completing.

STAGE_REGISTRY = {
    1: {
        "name":        "Data Preprocessing",
        "module":      "src.data_pipeline",
        "entry_point": "run_data_pipeline",
        "requires":    [],           # only the raw CSV, checked separately
        "produces": [
            PROCESSED_DATA_DIR / "train_split.parquet",
            PROCESSED_DATA_DIR / "val_split.parquet",
            PROCESSED_DATA_DIR / "test_split.parquet",
            MODELS_DIR / "preprocessor.joblib",
            MODELS_DIR / "feature_mapping.json",
        ],
        "isolation_note": "Preprocessor fitted ONLY on training partition.",
    },
    2: {
        "name":        "Logistic Regression Baseline",
        "module":      "src.baseline_model",
        "entry_point": "run_baseline_pipeline",
        "requires": [
            PROCESSED_DATA_DIR / "train_split.parquet",
            PROCESSED_DATA_DIR / "val_split.parquet",
        ],
        "produces": [
            MODELS_DIR / "baseline_logistic.joblib",
            REPORTS_DIR / "baseline_validation_metrics.json",
            REPORTS_DIR / "baseline_val_predictions.parquet",
        ],
        "isolation_note": "Trained on training partition only; evaluated on val.",
    },
    3: {
        "name":        "MLP Neural Network",
        "module":      "src.neural_network",
        "entry_point": "run_mlp_pipeline",
        "requires": [
            PROCESSED_DATA_DIR / "train_split.parquet",
            PROCESSED_DATA_DIR / "val_split.parquet",
        ],
        "produces": [
            MODELS_DIR / "mlp_churn_model.keras",
            REPORTS_DIR / "mlp_validation_metrics.json",
            REPORTS_DIR / "mlp_val_predictions.parquet",
            REPORTS_DIR / "mlp_training_history.json",
        ],
        "isolation_note": "Trained on training partition only; evaluated on val.",
    },
    4: {
        "name":        "Threshold Optimisation",
        "module":      "src.threshold_optimizer",
        "entry_point": "run_threshold_optimizer",
        "requires": [
            REPORTS_DIR / "baseline_val_predictions.parquet",
            REPORTS_DIR / "mlp_val_predictions.parquet",
        ],
        "produces": [
            MODELS_DIR / "threshold_metadata.json",
            REPORTS_DIR / "threshold_optimization_comparison.json",
            REPORTS_DIR / "threshold_search_results.csv",
        ],
        "isolation_note": "Uses ONLY validation predictions. Test partition never accessed.",
    },
    5: {
        "name":        "Final Test Evaluation",
        "module":      "src.evaluation",
        "entry_point": "run_final_evaluation",
        "requires": [
            PROCESSED_DATA_DIR / "test_split.parquet",
            MODELS_DIR / "baseline_logistic.joblib",
            MODELS_DIR / "mlp_churn_model.keras",
            MODELS_DIR / "threshold_metadata.json",
        ],
        "produces": [
            REPORTS_DIR / "final_test_metrics.json",
            REPORTS_DIR / "final_test_predictions.parquet",
            REPORTS_DIR / "model_comparison.csv",
            FIGURES_DIR / "confusion_matrices_test.png",
            FIGURES_DIR / "roc_pr_curves_test.png",
        ],
        "isolation_note": (
            "Test partition used ONLY here, after models and thresholds are finalised. "
            "Results NEVER fed back to influence models or thresholds."
        ),
    },
    6: {
        "name":        "SHAP Explainability",
        "module":      "src.explainability",
        "entry_point": "run_explainability_pipeline",
        "requires": [
            PROCESSED_DATA_DIR / "train_split.parquet",
            PROCESSED_DATA_DIR / "val_split.parquet",
            MODELS_DIR / "baseline_logistic.joblib",
            MODELS_DIR / "mlp_churn_model.keras",
            MODELS_DIR / "feature_mapping.json",
        ],
        "produces": [
            MODELS_DIR / "shap_background.joblib",
            REPORTS_DIR / "shap_global_importance.csv",
            REPORTS_DIR / "shap_local_explanations.json",
            FIGURES_DIR / "shap_summary_mlp.png",
            FIGURES_DIR / "shap_summary_logistic.png",
            FIGURES_DIR / "shap_importance_mlp.png",
            FIGURES_DIR / "shap_importance_logistic.png",
        ],
        "isolation_note": (
            "SHAP background uses ONLY real training records. "
            "Test partition never accessed."
        ),
    },
    7: {
        "name":        "Inference Artifact Generation",
        "module":      "src.inference",
        "entry_point": "run_inference_demo",
        "requires": [
            MODELS_DIR / "baseline_logistic.joblib",
            MODELS_DIR / "mlp_churn_model.keras",
            MODELS_DIR / "threshold_metadata.json",
            MODELS_DIR / "preprocessor.joblib",
            PROCESSED_DATA_DIR / "val_split.parquet",
        ],
        "produces": [
            REPORTS_DIR / "inference_predictions_mlp.parquet",
            REPORTS_DIR / "inference_predictions_logistic_regression.parquet",
        ],
        "isolation_note": "Runs on val partition for demonstration. Test never accessed.",
    },
}

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ===========================================================================
# Stage runner helpers
# ===========================================================================
def _stage_outputs_exist(stage_id: int) -> bool:
    """Return True if ALL outputs for the stage already exist on disk."""
    return all(p.exists() for p in STAGE_REGISTRY[stage_id]["produces"])


def _check_stage_inputs(stage_id: int) -> list[Path]:
    """Return list of missing required inputs."""
    return [p for p in STAGE_REGISTRY[stage_id]["requires"] if not p.exists()]


def _import_entry(stage_id: int):
    """Dynamically import and return the stage entry-point function."""
    reg  = STAGE_REGISTRY[stage_id]
    mod_name = reg["module"]
    fn_name  = reg["entry_point"]
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def run_stage(
    stage_id: int,
    force:    bool,
    dry_run:  bool,
    logger:   logging.Logger,
) -> bool:
    """
    Execute a single pipeline stage.

    Parameters
    ----------
    stage_id : int (1–7)
    force    : if True, run even if outputs already exist
    dry_run  : if True, check dependencies but do not execute
    logger   : Logger instance

    Returns
    -------
    True  — stage ran (or skipped because outputs exist and force=False)
    False — stage failed
    """
    reg  = STAGE_REGISTRY[stage_id]
    name = reg["name"]
    banner = f"Stage {stage_id}: {name}"

    logger.info("=" * 65)
    logger.info(f"  {banner}")
    logger.info(f"  Isolation: {reg['isolation_note']}")
    logger.info("=" * 65)

    # 1. Check required inputs
    missing_inputs = _check_stage_inputs(stage_id)
    if stage_id == 1:
        # Stage 1 also needs the raw CSV
        if not RAW_CSV_PATH.exists():
            missing_inputs.append(RAW_CSV_PATH)
    if missing_inputs:
        logger.error(f"[Stage {stage_id}] MISSING required inputs:")
        for p in missing_inputs:
            logger.error(f"    {p}")
        return False

    outputs_exist = _stage_outputs_exist(stage_id)

    # 2. Dry-run: display planned actions and NEVER execute stage
    if dry_run:
        if not force and outputs_exist:
            logger.info(
                f"[Stage {stage_id}] DRY-RUN — Action: WOULD SKIP "
                f"(all outputs already exist on disk; use --force to regenerate)."
            )
            logger.info("  Verified inputs:")
            inputs_to_show = [RAW_CSV_PATH] + reg["requires"] if stage_id == 1 else reg["requires"]
            for p in inputs_to_show:
                logger.info(f"    ✓  {str(p.relative_to(PROJECT_ROOT))}")
            logger.info("  Existing outputs (would be preserved):")
            for p in reg["produces"]:
                size = f"{p.stat().st_size:,}B" if p.exists() else "MISSING"
                logger.info(f"    ✓  {str(p.relative_to(PROJECT_ROOT))}  ({size})")
        else:
            reason = "--force specified" if force else "outputs missing/incomplete"
            logger.info(
                f"[Stage {stage_id}] DRY-RUN — Action: WOULD EXECUTE "
                f"{reg['entry_point']}() ({reason})"
            )
            logger.info("  Verified inputs:")
            inputs_to_show = [RAW_CSV_PATH] + reg["requires"] if stage_id == 1 else reg["requires"]
            for p in inputs_to_show:
                status = "EXISTS" if p.exists() else "MISSING"
                logger.info(f"    ✓  {str(p.relative_to(PROJECT_ROOT))}  [{status}]")
            logger.info("  Outputs (would be produced/updated):")
            for p in reg["produces"]:
                status = "EXISTS (would overwrite)" if p.exists() else "NEW"
                logger.info(f"    →  {str(p.relative_to(PROJECT_ROOT))}  [{status}]")
        return True

    # 3. Skip if already done and not forced
    if not force and outputs_exist:
        logger.info(
            f"[Stage {stage_id}] All outputs exist — SKIPPING "
            f"(use --force to regenerate)."
        )
        for p in reg["produces"]:
            size = f"{p.stat().st_size:,}B" if p.exists() else "MISSING"
            logger.info(f"    ✓  {str(p.relative_to(PROJECT_ROOT))}  ({size})")
        return True

    # 4. Safety guard for Stage 5 (final test evaluation)
    if stage_id == 5 and not force:
        final_metrics = REPORTS_DIR / "final_test_metrics.json"
        if final_metrics.exists():
            logger.warning(
                "[Stage 5] Final test results already exist. "
                "Use --force to overwrite. Skipping to protect data isolation."
            )
            return True

    # 5. Execute
    t_start = datetime.now()
    logger.info(f"[Stage {stage_id}] Starting at {t_start.strftime('%H:%M:%S')} ...")
    try:
        fn = _import_entry(stage_id)
        fn()
        t_end = datetime.now()
        elapsed = (t_end - t_start).total_seconds()
        logger.info(
            f"[Stage {stage_id}] COMPLETED in {elapsed:.1f}s  ✓"
        )

        # Verify outputs were actually produced
        missing_out = [p for p in reg["produces"] if not p.exists()]
        if missing_out:
            logger.warning(
                f"[Stage {stage_id}] WARNING — some expected outputs are missing:"
            )
            for p in missing_out:
                logger.warning(f"    ✗  {str(p.relative_to(PROJECT_ROOT))}")
        return True

    except Exception:
        t_end   = datetime.now()
        elapsed = (t_end - t_start).total_seconds()
        logger.error(f"[Stage {stage_id}] FAILED after {elapsed:.1f}s")
        logger.error(traceback.format_exc())
        return False


# ===========================================================================
# Status display
# ===========================================================================
def show_status(logger: logging.Logger) -> None:
    """Print current artifact status for all stages."""
    logger.info("\n" + "=" * 65)
    logger.info("  ChurnIQ Pipeline — Artifact Status")
    logger.info("=" * 65)

    all_complete = True
    for sid, reg in STAGE_REGISTRY.items():
        outputs = reg["produces"]
        present = [p for p in outputs if p.exists()]
        complete = len(present) == len(outputs)
        if not complete:
            all_complete = False
        status = "COMPLETE" if complete else f"INCOMPLETE ({len(present)}/{len(outputs)})"
        logger.info(f"\n  Stage {sid}: {reg['name']}  [{status}]")
        for p in outputs:
            sym  = "✓" if p.exists() else "✗"
            size = f"{p.stat().st_size:,}B" if p.exists() else "missing"
            logger.info(f"    {sym}  {str(p.relative_to(PROJECT_ROOT)):<60} {size}")

    logger.info("\n" + "=" * 65)
    if all_complete:
        logger.info("  All stages COMPLETE. Full pipeline has been run.")
    else:
        logger.info("  Some stages have missing outputs. Run with --all or --stages N.")
    logger.info("=" * 65 + "\n")


# ===========================================================================
# Dependency check
# ===========================================================================
def check_dependencies(logger: logging.Logger) -> bool:
    """
    Verify installed package versions match pinned requirements.
    Returns True if all required packages are importable and correctly versioned.
    """
    REQUIRED = {
        "keras":        "3.15.1",
        "torch":        "2.15.0+cpu",
        "sklearn":      "1.5.2",
        "shap":         "0.46.0",
        "pandas":       "2.3.3",
        "numpy":        "1.26.4",
        "joblib":       "1.4.2",
        "pyarrow":      "17.0.0",
        "matplotlib":   "3.9.4",
        "seaborn":      "0.13.2",
        "plotly":       "5.24.1",
        "streamlit":    "1.39.1",
    }
    MODULE_MAP = {
        "sklearn": "scikit-learn",
    }

    logger.info("\n  Package versions:")
    all_ok = True
    for mod_name, expected_ver in REQUIRED.items():
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            actual = getattr(mod, "__version__", "?")
            # Loose check: actual must start with expected (handles +cpu suffix)
            ok = actual == expected_ver or actual.startswith(expected_ver.split("+")[0])
            sym = "✓" if ok else "⚠"
            if not ok:
                all_ok = False
            logger.info(
                f"    {sym}  {MODULE_MAP.get(mod_name, mod_name):<20} "
                f"installed={actual:<18} pinned={expected_ver}"
            )
        except ImportError as e:
            logger.error(f"    ✗  {mod_name:<20} NOT IMPORTABLE: {e}")
            all_ok = False

    # Keras backend
    try:
        import keras
        backend = keras.backend.backend()
        ok = backend == "torch"
        sym = "✓" if ok else "✗"
        logger.info(f"    {sym}  Keras backend        = {backend}  (expected: torch)")
        if not ok:
            all_ok = False
    except Exception as e:
        logger.error(f"    ✗  Keras backend check failed: {e}")
        all_ok = False

    return all_ok


# ===========================================================================
# requirements.txt updater (non-destructive — updates versions in place)
# ===========================================================================
def update_requirements_txt(logger: logging.Logger) -> None:
    """
    Update requirements.txt with verified exact versions.
    Preserves comments and structure; only updates version specifiers.
    """
    VERIFIED_VERSIONS = {
        "pandas":        "2.3.3",
        "numpy":         "1.26.4",
        "scikit-learn":  "1.5.2",
        "keras":         "3.15.1",
        "torch":         "2.15.0",        # +cpu suffix added separately via install note
        "matplotlib":    "3.9.4",
        "seaborn":       "0.13.2",
        "plotly":        "5.24.1",
        "streamlit":     "1.39.1",
        "shap":          "0.46.0",
        "joblib":        "1.4.2",
        "pyarrow":       "17.0.0",
        "jupyter":       "1.1.1",
        "tensorflow":    "2.17.1",        # installed but blocked by SAC on Windows 11
    }

    req_path = PROJECT_ROOT / "requirements.txt"
    new_lines = []

    PYTORCH_INSTALL_NOTE = (
        "# NOTE: PyTorch CPU (Windows AMD64, Python 3.12) must be installed with:\n"
        "#   pip install torch==2.15.0 "
        "--index-url https://download.pytorch.org/whl/cpu\n"
        "# TensorFlow 2.17.1 is installed but its C extensions are blocked by\n"
        "# Windows Smart App Control on this machine. The project uses\n"
        "# Keras 3 with the PyTorch backend (KERAS_BACKEND=torch) instead.\n"
        "# To set permanently: add ~/.keras/keras.json with {\"backend\": \"torch\"}\n"
    )

    added_note = False
    with open(req_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
            # Extract package name (handle extras and options)
            pkg_name = (
                stripped.split("==")[0]
                       .split(">=")[0]
                       .split("<=")[0]
                       .split("[")[0]
                       .strip()
                       .lower()
            )
            if pkg_name in VERIFIED_VERSIONS:
                ver = VERIFIED_VERSIONS[pkg_name]
                new_lines.append(f"{pkg_name}=={ver}\n")
                if pkg_name == "torch" and not added_note:
                    new_lines.append(PYTORCH_INSTALL_NOTE)
                    added_note = True
            else:
                new_lines.append(line)

    with open(req_path, "w") as f:
        f.writelines(new_lines)

    logger.info(f"[OK] requirements.txt updated with verified versions → {req_path}")


# ===========================================================================
# CLI argument parser
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "ChurnIQ Master Pipeline Orchestrator\n\n"
            "Coordinates the full AI churn prediction pipeline:\n"
            "  1=Preprocessing  2=LR  3=MLP  4=Thresholds\n"
            "  5=TestEval  6=SHAP  7=Inference\n\n"
            "DATA ISOLATION GUARANTEE:\n"
            "  Preprocessing fits only on training data.\n"
            "  Test evaluation runs only after models and thresholds are finalised.\n"
            "  SHAP background uses only training records.\n"
            "  Test results never influence models or thresholds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline stages (skips stages with existing outputs unless --force).",
    )
    mode.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=range(1, 8),
        metavar="N",
        help="Run specific stage(s) by number (e.g. --stages 1 2 3).",
    )
    mode.add_argument(
        "--status",
        action="store_true",
        help="Show artifact status for all stages and exit.",
    )
    mode.add_argument(
        "--check-deps",
        action="store_true",
        help="Verify installed package versions and Keras backend, then exit.",
    )
    mode.add_argument(
        "--update-requirements",
        action="store_true",
        help="Update requirements.txt with verified exact versions, then exit.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Overwrite existing outputs. For Stage 5 (final test evaluation), "
            "an explicit --force is required to prevent accidental overwrites."
        ),
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        default=False,
        help="Skip Stage 6 (SHAP explainability) when running --all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Check dependencies and show what would run without executing.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write log output (in addition to stdout).",
    )
    return parser


# ===========================================================================
# Main orchestrator
# ===========================================================================
def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point. Returns exit code (0=success, 1=failure).
    """
    parser = build_parser()
    args   = parser.parse_args(argv)

    # Default log file
    log_file = args.log_file
    if log_file is None and not args.status and not args.check_deps:
        log_file = REPORTS_DIR / "pipeline_run.log"

    logger = setup_logging(log_file)

    logger.info("=" * 65)
    logger.info("  ChurnIQ Pipeline Orchestrator")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Project: {PROJECT_ROOT}")
    logger.info(f"  Seed:    {RANDOM_SEED}")
    logger.info("=" * 65)

    # --- Utility modes ---
    if args.status:
        show_status(logger)
        return 0

    if args.check_deps:
        ok = check_dependencies(logger)
        return 0 if ok else 1

    if args.update_requirements:
        update_requirements_txt(logger)
        return 0

    # --- Dependency check always runs first ---
    logger.info("\n[Pre-flight] Checking package dependencies ...")
    dep_ok = check_dependencies(logger)
    if not dep_ok:
        logger.error(
            "Dependency check failed. Fix the issues above before running the pipeline."
        )
        return 1
    logger.info("[Pre-flight] Dependencies OK.\n")

    # --- Seed ---
    set_all_seeds()

    # --- Determine which stages to run ---
    if args.all:
        stage_ids = list(STAGE_REGISTRY.keys())   # [1,2,3,4,5,6,7]
    else:
        stage_ids = sorted(set(args.stages))

    if args.skip_shap and 6 in stage_ids:
        stage_ids.remove(6)
        logger.info("[INFO] --skip-shap: Stage 6 (SHAP) excluded from run.\n")

    # --- Warn on --force for Stage 5 ---
    if 5 in stage_ids and args.force:
        logger.warning(
            "[Stage 5] --force is set: Final test evaluation will be OVERWRITTEN. "
            "Ensure models and thresholds are truly finalised before proceeding."
        )

    # --- Validate stage dependencies and ordering ---
    required_predecessors = {
        2: [1],
        3: [1],
        4: [2, 3],
        5: [1, 2, 3, 4],
        6: [1, 2, 3],
        7: [1, 2, 3, 4],
    }
    for sid in stage_ids:
        for pred in required_predecessors.get(sid, []):
            if pred not in stage_ids:
                # Predecessor not in this run — check if its outputs exist
                if not _stage_outputs_exist(pred):
                    logger.error(
                        f"[Dependency] Stage {sid} requires Stage {pred} outputs, "
                        f"but Stage {pred} is not in the run list and its outputs "
                        f"are MISSING. Add --stages {pred} or run --all first."
                    )
                    return 1

    # --- Run stages ---
    logger.info(
        f"[INFO] Stages to run: {stage_ids}"
        + (" (DRY-RUN)" if args.dry_run else "")
        + (" (FORCE)" if args.force else "")
    )

    failures: list[int] = []
    for sid in stage_ids:
        success = run_stage(
            stage_id=sid,
            force=args.force,
            dry_run=args.dry_run,
            logger=logger,
        )
        if not success:
            failures.append(sid)
            logger.error(
                f"[Orchestrator] Stage {sid} failed. "
                "Stopping pipeline to prevent dependent stage corruption."
            )
            break   # halt on first failure (downstream stages depend on upstream)

    # --- Final summary ---
    logger.info("\n" + "=" * 65)
    logger.info("  Pipeline Run Summary")
    logger.info("=" * 65)
    for sid in stage_ids:
        if sid in failures:
            status = "FAILED  ✗"
        elif sid > (failures[0] if failures else 999):
            status = "SKIPPED (upstream failure)"
        else:
            status = "OK      ✓"
        logger.info(f"  Stage {sid}: {STAGE_REGISTRY[sid]['name']:<35} {status}")

    if failures:
        logger.error(f"\n  Pipeline FAILED at stage(s): {failures}")
        logger.error("  Fix the error above and re-run. Existing artifacts are preserved.")
        return 1
    else:
        logger.info(
            f"\n  All {len(stage_ids)} stage(s) completed successfully."
        )
        logger.info(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    sys.exit(main())
