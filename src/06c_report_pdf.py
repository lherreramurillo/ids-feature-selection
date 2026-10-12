"""Stage 6: convert the feature selection report to PDF.

What it does
    Renders report/feature_selection_report.md as a PDF with the figures
    embedded. The Markdown file stays the source: this script only changes
    the format, not the content.

Input
    report/feature_selection_report.md (or the _es.md version) and the
    figures it references in report/figures/

Output
    report/feature_selection_report.pdf      English (default)
    report/feature_selection_report_es.pdf   Spanish, with --lang es

How to run (after 06b_report.py with the same language)
    .venv/bin/python src/06c_report_pdf.py
    .venv/bin/python src/06c_report_pdf.py --lang es

Needs the packages `markdown` and `weasyprint` in the venv.
"""
import sys

import markdown
from weasyprint import HTML

from config import (REPORT_DIR, REPORT_FILE, REPORT_FILE_ES, REPORT_PDF_FILE,
                    REPORT_PDF_FILE_ES)

# Print styles. Tables use a smaller font because the cross-validation table
# has nine columns and must fit the width of an A4 page.
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-center { content: counter(page); font-size: 8pt; color: #6b7280; } }
body { font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
       font-size: 9.5pt; line-height: 1.4; color: #1f2328; }
h1 { font-size: 17pt; margin: 0 0 8pt 0; }
h2 { font-size: 13pt; margin: 16pt 0 6pt 0; padding-bottom: 2pt;
     border-bottom: 0.6pt solid #d0d7de; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 12pt 0 4pt 0; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.3pt;
       background: #f3f4f6; padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0;
        font-size: 7.4pt; }
th, td { border: 0.5pt solid #d0d7de; padding: 2.5pt 4pt; text-align: left;
         vertical-align: top; }
th { background: #f3f4f6; }
td code, th code { font-size: 7pt; background: none; padding: 0; }
tr { break-inside: avoid; }
img { max-width: 100%; display: block; margin: 8pt auto; }
p:has(> img) { break-inside: avoid; }
"""


def main():
    """Render the Markdown report to HTML and print it to PDF."""
    lang = "en"
    if "--lang" in sys.argv:
        lang = sys.argv[sys.argv.index("--lang") + 1]
    source_file, pdf_file = {"en": (REPORT_FILE, REPORT_PDF_FILE),
                             "es": (REPORT_FILE_ES, REPORT_PDF_FILE_ES)}[lang]
    source = (REPORT_DIR / source_file).read_text(encoding="utf-8")
    body = markdown.markdown(source, extensions=["tables", "sane_lists"])
    # The lang attribute lets the renderer apply the right hyphenation rules.
    html = (f"<html lang='{lang}'><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    target = REPORT_DIR / pdf_file
    # base_url makes the relative figure paths (figures/...) resolve.
    HTML(string=html, base_url=str(REPORT_DIR)).write_pdf(target)
    print(f"Written: {target} ({target.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
