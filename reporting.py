from typing import List, Dict
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from models import Finding

def build_report_pdf(path: str,
                     title: str,
                     summary: str,
                     findings: List[Finding]):
    doc = SimpleDocTemplate(path, pagesize=LETTER, topMargin=36, bottomMargin=36, leftMargin=48, rightMargin=48)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal = styles['BodyText']
    bold = ParagraphStyle('bold', parent=normal, fontName='Helvetica-Bold')
    small = ParagraphStyle('small', parent=normal, fontSize=9, leading=11)

    flow = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    flow.append(Paragraph(title or "PDF Severity Phrase Analysis", title_style))
    flow.append(Paragraph(f"Generated: {now}", small))
    flow.append(Spacer(1, 0.2*inch))
    flow.append(Paragraph("Summary", bold))
    flow.append(Paragraph(summary or "", normal))
    flow.append(Spacer(1, 0.2*inch))

    # Group by severity
    groups = {"High": [], "Medium": [], "Low": []}
    for f in findings:
        groups.setdefault(f.severity, []).append(f)

    for sev in ["High", "Medium", "Low"]:
        items = groups.get(sev, [])
        if not items:
            continue
        flow.append(Spacer(1, 0.15*inch))
        flow.append(Paragraph(f"{sev} Severity", bold))
        flow.append(Spacer(1, 0.05*inch))
        for it in items:
            # Add source indicator for better visibility
            source_tag = ""
            if it.source == "LLM-SingleWord-Batch":
                source_tag = " [AI-Enhanced]"
            elif it.source == "LLM-SingleWord":
                source_tag = " [AI-Enhanced]"
            elif it.source == "LLM":
                source_tag = " [AI-Discovered]"
            elif it.source == "CSV":
                source_tag = " [Lexicon]"
            
            # Row 1: Severity
            severity_line = f"<b>Severity:</b> {it.severity}{source_tag}"
            flow.append(Paragraph(severity_line, normal))
            flow.append(Spacer(1, 0.05*inch))
            
            # Row 2: Page
            page_line = f"<b>Page:</b> {it.page}"
            flow.append(Paragraph(page_line, normal))
            flow.append(Spacer(1, 0.05*inch))
            
            # Row 3: Exact Matched Phrase
            phrase_line = f"<b>Exact Matched Phrase:</b> \"{it.phrase}\""
            flow.append(Paragraph(phrase_line, normal))
            flow.append(Spacer(1, 0.05*inch))
            
            # Row 4: Suggested Rewrite
            suggestion_line = f"<b>Suggested Rewrite:</b> \"{it.suggestion}\""
            flow.append(Paragraph(suggestion_line, normal))
            flow.append(Spacer(1, 0.05*inch))
            
            # Row 5: Context
            context_line = f"<b>Context:</b> {it.context}"
            flow.append(Paragraph(context_line, normal))
            
            # Position information (if available) - as additional info
            if it.start_char is not None and it.end_char is not None:
                flow.append(Spacer(1, 0.02*inch))
                position_line = f"<i>Position: [{it.start_char}-{it.end_char}]</i>"
                flow.append(Paragraph(position_line, small))
            
            flow.append(Spacer(1, 0.15*inch))

    doc.build(flow)
def _format_finding_line(f):
    tag = "" if (f.source or "CSV") == "CSV" else " [LLM]"
    return f"Severity: {f.severity}{tag} | Page: {f.page} | Phrase: \"{f.phrase}\""
# Ensure you call this helper when writing each finding row in your PDF.