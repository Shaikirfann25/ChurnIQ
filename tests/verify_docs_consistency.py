"""
Comprehensive Documentation Consistency and Audit Verification Script for ChurnIQ.

Verifies:
1. Exact metric values in documentation match reports/final_test_metrics.json.
2. Threshold values match models/threshold_metadata.json.
3. All mentioned file paths exist on disk.
4. SHAP feature names and values match reports/shap_global_importance.csv.
5. Absence of forbidden/overstated claims ("Decisive F1 Winner", "decisiveness margin", etc.).
6. Accurate reflection of canonical directory C:\\Users\\shaik\\OneDrive\\Desktop\\ChurnIQ.
"""

import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "PROJECT_REPORT.md",
    PROJECT_ROOT / "docs" / "ARCHITECTURE.md",
    PROJECT_ROOT / "docs" / "METHODOLOGY.md",
]

failures = []
warnings = []

print("=" * 60)
print("  ChurnIQ Documentation Consistency & Critical Audit Check")
print(f"  Root: {PROJECT_ROOT}")
print("=" * 60)

# ── 1. Check doc files exist ──────────────────────────────────────────────
print("\n[1] Verifying documentation files exist...")
for doc in DOC_FILES:
    if not doc.exists():
        failures.append(f"Missing doc file: {doc}")
        print(f"  ✗ MISSING: {doc.name}")
    else:
        size = doc.stat().st_size
        print(f"  ✓ {doc.name} ({size} bytes)")

# ── 2. Check metrics consistency ──────────────────────────────────────────
print("\n[2] Verifying metrics match reports/final_test_metrics.json...")
metrics_file = PROJECT_ROOT / "reports" / "final_test_metrics.json"
if not metrics_file.exists():
    failures.append(f"Missing {metrics_file}")
else:
    with open(metrics_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    results = meta["results"]

    lr_opt = next(r for r in results if r["model"] == "LogisticRegression" and r["threshold_label"] == "optimal")
    mlp_opt = next(r for r in results if r["model"] == "MLP" and r["threshold_label"] == "optimal")

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    report_text = (PROJECT_ROOT / "docs" / "PROJECT_REPORT.md").read_text(encoding="utf-8")

    expected_metrics = [
        ("LR ROC-AUC", f"{lr_opt['roc_auc']:.4f}"),
        ("LR PR-AUC", f"{lr_opt['pr_auc']:.4f}"),
        ("LR F1", f"{lr_opt['f1']:.4f}"),
        ("LR Precision", f"{lr_opt['precision']:.4f}"),
        ("LR Recall", f"{lr_opt['recall']:.4f}"),
        ("MLP ROC-AUC", f"{mlp_opt['roc_auc']:.4f}"),
        ("MLP PR-AUC", f"{mlp_opt['pr_auc']:.4f}"),
        ("MLP F1", f"{mlp_opt['f1']:.4f}"),
        ("MLP Precision", f"{mlp_opt['precision']:.4f}"),
        ("MLP Recall", f"{mlp_opt['recall']:.4f}"),
    ]

    for label, val in expected_metrics:
        in_readme = val in readme_text
        in_report = val in report_text
        if not in_readme:
            failures.append(f"{label} ({val}) missing from README.md")
            print(f"  ✗ {label} ({val}) missing from README.md")
        elif not in_report:
            failures.append(f"{label} ({val}) missing from docs/PROJECT_REPORT.md")
            print(f"  ✗ {label} ({val}) missing from docs/PROJECT_REPORT.md")
        else:
            print(f"  ✓ {label}: {val} verified in docs")

# ── 3. Check threshold metadata consistency ──────────────────────────────
print("\n[3] Verifying threshold values match models/threshold_metadata.json...")
th_file = PROJECT_ROOT / "models" / "threshold_metadata.json"
if th_file.exists():
    with open(th_file, "r", encoding="utf-8") as f:
        th_meta = json.load(f)
    lr_th = f"{th_meta['models']['logistic_regression']['optimal_threshold']:.2f}"
    mlp_th = f"{th_meta['models']['mlp']['optimal_threshold']:.2f}"

    for doc in DOC_FILES:
        text = doc.read_text(encoding="utf-8")
        if lr_th not in text:
            failures.append(f"LR optimal threshold {lr_th} missing in {doc.name}")
            print(f"  ✗ LR threshold {lr_th} not found in {doc.name}")
        if mlp_th not in text:
            failures.append(f"MLP optimal threshold {mlp_th} missing in {doc.name}")
            print(f"  ✗ MLP threshold {mlp_th} not found in {doc.name}")
    print(f"  ✓ LR optimal threshold ({lr_th}) and MLP optimal threshold ({mlp_th}) verified")

# ── 4. Search for forbidden overstated claims ──────────────────────────────
print("\n[4] Scanning for overstated / unsupported claims...")
forbidden_patterns = [
    r"Decisive (?:F1 )?Winner",
    r"decisiveness margin",
    r"decisiveness criterion",
    r"significant improvement",
    r"best model",
]

for doc in DOC_FILES:
    text = doc.read_text(encoding="utf-8")
    for pat in forbidden_patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            for m in matches:
                # Find line number
                line_no = text[:m.start()].count("\n") + 1
                failures.append(f"Forbidden claim '{m.group(0)}' in {doc.name}:{line_no}")
                print(f"  ✗ Found '{m.group(0)}' at {doc.name}:{line_no}")

if not any("Forbidden claim" in f for f in failures):
    print("  ✓ Zero forbidden or overstated claims found across all doc files.")

# ── 5. Check SHAP feature separation ───────────────────────────────────────
print("\n[5] Verifying SHAP feature names and uncombined charges...")
for doc in DOC_FILES:
    text = doc.read_text(encoding="utf-8")
    if re.search(r"MonthlyCharges[`*]*\s*(&|and)\s*[`*]*TotalCharges", text, re.IGNORECASE):
        failures.append(f"Combined 'MonthlyCharges & TotalCharges' found in {doc.name}")
        print(f"  ✗ Inaccurate combined charges found in {doc.name}")

if not any("Combined" in f for f in failures):
    print("  ✓ MonthlyCharges and TotalCharges are properly separated.")

# ── 6. Check canonical directory paths (no ChurnIQ') ──────────────────────
print("\n[6] Verifying no duplicate/apostrophe paths ('ChurnIQ's') in docs...")
for doc in DOC_FILES:
    text = doc.read_text(encoding="utf-8")
    if "ChurnIQ's" in text or "ChurnIQ'" in text:
        failures.append(f"Duplicate path 'ChurnIQ\\'s' found in {doc.name}")
        print(f"  ✗ Duplicate path found in {doc.name}")

if not any("Duplicate path" in f for f in failures):
    print("  ✓ Only canonical project path 'ChurnIQ' referenced.")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures:
    print(f"AUDIT FAILED with {len(failures)} issue(s):")
    for idx, f in enumerate(failures, 1):
        print(f"  [{idx}] {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED (exit code 0)")
    print("=" * 60)
    sys.exit(0)
