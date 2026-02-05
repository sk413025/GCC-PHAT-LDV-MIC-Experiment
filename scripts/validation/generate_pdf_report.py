#!/usr/bin/env python
"""Generate PDF report from markdown validation report."""

import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    """Generate PDF from markdown report using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("Installing fpdf2...")
        os.system(f"{sys.executable} -m pip install fpdf2")
        from fpdf import FPDF

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    md_path = base_dir / "results" / "VALIDATION_REPORT_20260205.md"
    pdf_path = base_dir / "results" / "VALIDATION_REPORT_20260205.pdf"

    if not md_path.exists():
        print(f"Error: Markdown report not found at {md_path}")
        return 1

    # Read markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Use built-in font (no external font needed)
    pdf.set_font("Helvetica", size=10)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "LDV Reorientation - Systematic Validation Report", ln=True, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # Process markdown content
    lines = md_content.split("\n")
    in_table = False
    table_lines = []
    in_code_block = False

    for line in lines:
        # Skip the first title line (we already added it)
        if line.startswith("# LDV Reorientation"):
            continue

        # Handle code blocks
        if line.startswith("```"):
            in_code_block = not in_code_block
            if not in_code_block and table_lines:
                # End of code block - render as monospace
                pdf.set_font("Courier", size=8)
                for code_line in table_lines:
                    safe_line = code_line.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.cell(0, 4, safe_line, ln=True)
                table_lines = []
                pdf.set_font("Helvetica", size=10)
            continue

        if in_code_block:
            table_lines.append(line)
            continue

        # Handle tables
        if "|" in line and not line.startswith("```"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            continue
        elif in_table:
            # End of table - render it
            if table_lines:
                pdf.set_font("Courier", size=7)
                for tl in table_lines:
                    safe_line = tl.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.cell(0, 4, safe_line, ln=True)
                pdf.ln(3)
                pdf.set_font("Helvetica", size=10)
            in_table = False
            table_lines = []

        # Handle headers
        if line.startswith("## "):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 14)
            safe_line = line[3:].encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 8, safe_line, ln=True)
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            safe_line = line[4:].encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 7, safe_line, ln=True)
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("#### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            safe_line = line[5:].encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 6, safe_line, ln=True)
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("---"):
            pdf.ln(3)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
        elif line.startswith("- "):
            # Bullet point
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 5, "  " + safe_line, ln=True)
        elif line.startswith("**") and line.endswith("**"):
            # Bold text
            pdf.set_font("Helvetica", "B", 10)
            safe_line = line.strip("*").encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 5, safe_line, ln=True)
            pdf.set_font("Helvetica", size=10)
        elif line.strip():
            # Regular text
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            # Check if we have enough space, otherwise add new page
            if pdf.get_y() > 270:
                pdf.add_page()
            try:
                pdf.multi_cell(0, 5, safe_line)
            except Exception:
                # Fallback to cell if multi_cell fails
                pdf.cell(0, 5, safe_line[:100], ln=True)
        else:
            pdf.ln(2)

    # Handle any remaining table
    if table_lines:
        pdf.set_font("Courier", size=7)
        for tl in table_lines:
            safe_line = tl.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 4, safe_line, ln=True)

    # Save PDF
    pdf.output(str(pdf_path))
    print(f"PDF report saved to: {pdf_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
