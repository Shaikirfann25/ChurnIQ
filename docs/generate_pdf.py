"""
Generate Publication-Quality Academic PDF from docs/PROJECT_REPORT.md for ChurnIQ.
Uses Pandoc + MathJax + Headless Chrome with professional academic styling.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_MD = DOCS_DIR / "PROJECT_REPORT.md"
OUTPUT_HTML = DOCS_DIR / "PROJECT_REPORT_RENDERED.html"
OUTPUT_PDF = DOCS_DIR / "ChurnIQ_Academic_Project_Report.pdf"

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def find_browser():
    for p in CHROME_PATHS:
        if Path(p).exists():
            return p
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found.")


def generate_html():
    print(f"[1/3] Converting {REPORT_MD.name} to HTML via Pandoc...")
    # Convert with Pandoc with mathjax and gfm extensions
    cmd = [
        "pandoc",
        str(REPORT_MD),
        "-s",
        "--mathjax",
        "-f", "gfm",
        "-t", "html5",
        "-o", str(OUTPUT_HTML),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Pandoc error:", res.stderr)
        sys.exit(1)

    html = OUTPUT_HTML.read_text(encoding="utf-8")

    # Replace relative image paths ../reports/figures/ with absolute file:/// URIs
    fig_dir = (PROJECT_ROOT / "reports" / "figures").resolve()
    
    def replace_img_src(match):
        img_name = Path(match.group(1)).name
        abs_img_path = (fig_dir / img_name).as_uri()
        return f'src="{abs_img_path}"'

    html = re.sub(r'src=["\'](?:\.\./)?reports/figures/([^"\']+)["\']', replace_img_src, html)

    # Academic CSS Stylesheet
    academic_css = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
  size: A4 portrait;
  margin: 14mm 13mm 14mm 13mm;
  @bottom-right {
    content: counter(page);
  }
}

*, *:before, *:after {
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: #1e293b;
  background: #ffffff;
  font-size: 9.5pt;
  line-height: 1.48;
  margin: 0 auto;
  max-width: 100%;
}

/* Page Break Management */
.page-break {
  page-break-before: always;
}

h1 {
  color: #0f172a;
  font-size: 15pt;
  font-weight: 700;
  border-bottom: 2px solid #2563eb;
  padding-bottom: 4px;
  margin-top: 24pt;
  margin-bottom: 10pt;
  page-break-before: always;
}

/* Prevent first H1 from breaking page */
h1:first-of-type, body > h1:nth-of-type(1) {
  page-break-before: avoid !important;
  margin-top: 0;
}

h2 {
  color: #1e3a8a;
  font-size: 12.5pt;
  font-weight: 600;
  margin-top: 16pt;
  margin-bottom: 4.5pt;
  page-break-after: avoid;
}

h3 {
  color: #1e40af;
  font-size: 11pt;
  font-weight: 600;
  margin-top: 12pt;
  margin-bottom: 6pt;
  page-break-after: avoid;
}

h4 {
  color: #334155;
  font-size: 10pt;
  font-weight: 600;
  margin-top: 10pt;
  margin-bottom: 4pt;
  page-break-after: avoid;
}

p {
  margin-top: 0;
  margin-bottom: 4.5pt;
  text-align: justify;
}

/* Title Block Styling */
.title-block {
  text-align: center;
  border: 1.5px solid #2563eb;
  background: #f8fafc;
  padding: 16pt 18pt;
  border-radius: 6px;
  margin-bottom: 18pt;
  page-break-inside: avoid;
}

.title-block h1 {
  border-bottom: none;
  color: #1e3a8a;
  font-size: 18pt;
  margin: 0 0 6pt 0;
  padding: 0;
}

.title-block h2 {
  color: #475569;
  font-size: 12pt;
  font-weight: 500;
  margin: 0 0 12pt 0;
}

.meta-table {
  width: 100%;
  margin: 10pt auto 0 auto;
  border-collapse: collapse;
  font-size: 9pt;
}

.meta-table td {
  padding: 3pt 6pt;
  border: none;
}

.meta-table td.label {
  font-weight: 600;
  color: #1e3a8a;
  width: 25%;
  text-align: right;
}

.meta-table td.value {
  color: #0f172a;
  text-align: left;
}

/* Executive Abstract Box */
.abstract-box {
  background: #f1f5f9;
  border-left: 4px solid #2563eb;
  padding: 10pt 14pt;
  margin: 14pt 0;
  font-size: 9pt;
  page-break-inside: avoid;
}

.abstract-box h3 {
  margin-top: 0;
  color: #1e3a8a;
  font-size: 10.5pt;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 10pt 0 14pt 0;
  font-size: 8pt;
  line-height: 1.35;
  page-break-inside: avoid;
}

th, td {
  padding: 4.5pt 5.5pt;
  border: 1px solid #cbd5e1;
  text-align: left;
}

th {
  background-color: #f1f5f9;
  color: #0f172a;
  font-weight: 600;
  border-bottom: 2px solid #94a3b8;
}

tr:nth-child(even) {
  background-color: #f8fafc;
}

/* Code & Preformatted Blocks */
pre {
  background: #0f172a;
  color: #f8fafc;
  padding: 8pt 10pt;
  border-radius: 4px;
  font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
  font-size: 7.5pt;
  line-height: 1.35;
  overflow-x: hidden;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 8pt 0 12pt 0;
  page-break-inside: avoid;
}

code {
  font-family: 'JetBrains Mono', 'Consolas', monospace;
  font-size: 8.5pt;
  background: #f1f5f9;
  color: #0f172a;
  padding: 1pt 3pt;
  border-radius: 3px;
}

pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}

/* Figures and Images */
figure, .figure-container {
  display: block;
  text-align: center;
  margin: 8pt auto;
  page-break-inside: avoid;
}

img {
  display: block;
  max-width: 88%;
  max-height: 195px;
  margin: 0 auto;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
}

/* Figure 2 (ROC & PR Curves) on Page 14: Enlarge to reduce blank space */
img[src*="roc_pr_curves_test.png"] {
  max-width: 94%;
  max-height: 240px;
}

/* Figure 3 (Confusion Matrices) on Page 15: Enlarge using available vertical space */
img[src*="confusion_matrices_test.png"] {
  max-width: 96%;
  max-height: 305px;
}

/* Figures 4 & 5 (SHAP Feature Importance) on Page 16: Balanced height so both and text fit on single page */
img[src*="shap_importance_mlp.png"],
img[src*="shap_importance_logistic.png"] {
  max-width: 90%;
  max-height: 146px;
}

/* Figures 6 & 7 (SHAP Summary Beeswarm) on Page 17: Enlarge for enhanced readability */
img[src*="shap_summary_mlp.png"],
img[src*="shap_summary_logistic.png"] {
  max-width: 94%;
  max-height: 232px;
}

figcaption, em {
  display: block;
  font-size: 8.5pt;
  color: #475569;
  margin-top: 4pt;
  text-align: center;
}

/* Blockquotes */
blockquote {
  border-left: 3px solid #f59e0b;
  background: #fffbeb;
  margin: 10pt 0;
  padding: 8pt 12pt;
  font-size: 8.5pt;
  color: #78350f;
  page-break-inside: avoid;
}

/* Math elements */
.MathJax {
  font-size: 102% !important;
}

/* Lists */
ul, ol {
  margin-top: 2pt;
  margin-bottom: 4.5pt;
  padding-left: 18pt;
}

li {
  margin-bottom: 3pt;
}

hr {
  display: none;
}
</style>
"""
    # Inject CSS into head
    html = html.replace("</head>", f"{academic_css}\n</head>")

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"[2/3] Styled HTML generated: {OUTPUT_HTML.name} ({len(html)} bytes)")


def render_pdf():
    browser_exe = find_browser()
    print(f"[3/3] Rendering PDF with {Path(browser_exe).name}...")
    
    html_url = OUTPUT_HTML.resolve().as_uri()
    
    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=9000",
        f"--print-to-pdf={OUTPUT_PDF.resolve()}",
        "--no-pdf-header-footer",
        html_url,
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Browser error:", res.stderr)
        sys.exit(1)
        
    if not OUTPUT_PDF.exists():
        print("PDF file was not created!")
        sys.exit(1)
        
    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"SUCCESS! PDF Generated: {OUTPUT_PDF.name} ({size_kb:.1f} KB)")


def inspect_pdf():
    print("\n--- PDF Verification ---")
    data = OUTPUT_PDF.read_bytes()
    
    # Estimate page count by counting Page dictionaries in PDF binary
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    print(f"File Path:   {OUTPUT_PDF}")
    print(f"File Size:   {OUTPUT_PDF.stat().st_size:,} bytes")
    print(f"Page Count:  {pages} pages")
    
    # Verify embedded images exist
    figures = [
        "mlp_training_history.png",
        "roc_pr_curves_test.png",
        "confusion_matrices_test.png",
        "shap_importance_mlp.png",
        "shap_importance_logistic.png",
        "shap_summary_mlp.png",
        "shap_summary_logistic.png",
    ]
    print("\nFigure Verification:")
    for fig in figures:
        p = PROJECT_ROOT / "reports" / "figures" / fig
        status = "EXISTS" if p.exists() else "MISSING"
        size = f"{p.stat().st_size:,} bytes" if p.exists() else "N/A"
        print(f"  ✓ {fig:<32} [{status}] ({size})")


if __name__ == "__main__":
    generate_html()
    render_pdf()
    inspect_pdf()

