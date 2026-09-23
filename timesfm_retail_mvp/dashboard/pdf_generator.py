import io
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from forecast.architecture import ForecastResult
from dashboard.executive_summary import generate_evidence_why, generate_action


def _md_to_reportlab_html(text: str) -> str:
    """Safely converts Markdown bold (**text**) to matching ReportLab HTML (<b>text</b>)."""
    if not text:
        return ""
    # Match pairs of ** and replace with <b>...</b>
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Convert double newlines to paragraph breaks
    formatted = formatted.replace("\n\n", "<br/><br/>")
    return formatted


def generate_pdf_report(
    forecast: ForecastResult, 
    company_name: str, 
    historical_trend_pct: float = 9.6, 
    active_events: list = None
) -> bytes:
    """Generates an executive-ready PDF report consuming the ForecastResult object."""
    if active_events is None:
        active_events = []
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        rightMargin=50, 
        leftMargin=50, 
        topMargin=50, 
        bottomMargin=50
    )
    story = []
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    unit_label = "units" if forecast.domain in ["Sales", "Inventory"] else "viewers" if forecast.domain == "Viewership" else "revenue"
    is_viewership = forecast.domain == "Viewership"
    demand_fmt = f"{forecast.total_demand:,.2f}" if is_viewership else f"{forecast.total_demand:,}"
    
    # ---------------------------------------------------------
    # PAGE 1: EXECUTIVE SUMMARY
    # ---------------------------------------------------------
    story.append(Paragraph(f"{company_name.upper()} — DEMAND FORECAST", title_style))
    story.append(Paragraph(f"<b>Target:</b> {forecast.target_name.replace('_', ' ')} &nbsp;|&nbsp; <b>Horizon:</b> {forecast.forecast_start_date} – {forecast.forecast_end_date} ({forecast.horizon_days} Days)", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("EXECUTIVE SUMMARY", h2_style))
    story.append(Paragraph(f"<b>Expected Demand:</b> {demand_fmt} {unit_label}", body_style))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("WHY THIS FORECAST", h2_style))
    why_text = generate_evidence_why(forecast, historical_trend_pct, active_events)
    story.append(Paragraph(_md_to_reportlab_html(why_text), body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("RECOMMENDED ACTION", h2_style))
    action_text = generate_action(forecast)
    story.append(Paragraph(_md_to_reportlab_html(action_text), body_style))
    
    # Inventory Math Breakdown for PDF
    if forecast.domain == "Inventory" and forecast.inventory_metrics:
        shortage = forecast.inventory_metrics['net_shortage']
        if shortage > 0:
            story.append(Spacer(1, 6))
            math_summary = (
                f"<b>Calculation Breakdown:</b><br/>"
                f"Expected Demand ({forecast.total_demand:,}) + "
                f"Safety Buffer ({forecast.inventory_metrics['safety_buffer_units']:,}) - "
                f"Current Stock ({forecast.inventory_metrics['current_stock']:,}) = "
                f"<b>Replenishment Order ({shortage:,} units)</b>"
            )
            story.append(Paragraph(math_summary, body_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("WATCH / UNCERTAINTY RISK", h2_style))
    story.append(Paragraph(
        f"Forecast uncertainty increases toward the end of the selected <b>{forecast.horizon_days}-day period</b>. "
        "Later-period commitments should be reviewed as new actuals become available.", 
        body_style
    ))
    
    story.append(PageBreak())

    # ---------------------------------------------------------
    # PAGE 2: MONTHLY & DETAILED BREAKDOWN
    # ---------------------------------------------------------
    story.append(Paragraph("MONTHLY FORECAST SUMMARY", h2_style))
    
    df_monthly = forecast.monthly_summary
    table_data = [["Month", f"Expected ({unit_label})", "Conservative (P10)", "Optimistic (P90)"]]
    
    for _, row in df_monthly.iterrows():
        exp = f"{row['expected']:,.2f}" if is_viewership else f"{int(row['expected']):,}"
        low = f"{row['lower']:,.2f}" if is_viewership else f"{int(row['lower']):,}"
        upp = f"{row['upper']:,.2f}" if is_viewership else f"{int(row['upper']):,}"
        table_data.append([row['month'], exp, low, upp])

    t_monthly = Table(table_data, colWidths=[120, 130, 130, 130])
    t_monthly.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0"))
    ]))
    story.append(t_monthly)
    story.append(Spacer(1, 16))

    # Scenario Table Overlay (If Active)
    if forecast.has_scenario:
        story.append(Paragraph("WHAT-IF SCENARIO IMPACT", h2_style))
        diff = forecast.scenario_difference
        diff_str = f"+{diff:,.2f}" if is_viewership and diff > 0 else f"{diff:,.2f}" if is_viewership else f"+{diff:,}" if diff > 0 else f"{diff:,}"
        scen_tot_str = f"{forecast.scenario_total:,.2f}" if is_viewership else f"{forecast.scenario_total:,}"
        
        scen_data = [
            ["Metric", "Value"],
            ["Baseline Demand", f"{demand_fmt} {unit_label}"],
            ["Scenario Projection", f"{scen_tot_str} {unit_label}"],
            ["Net Difference", f"{diff_str} {unit_label}"]
        ]
        
        t_scen = Table(scen_data, colWidths=[200, 310])
        t_scen.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0"))
        ]))
        story.append(t_scen)
        story.append(Spacer(1, 16))

    # Governance & Methodology Notes
    story.append(Paragraph("METHODOLOGY & GOVERNANCE NOTES", h2_style))
    notes = (
        "• Forecast generated via Google TimesFM 3.0 Zero-Shot Foundation Model.<br/>"
        f"• Horizon length: {forecast.horizon_days} days.<br/>"
        "• Conservative (P10) and Optimistic (P90) ranges define the model uncertainty distribution.<br/>"
        "• Applied future context variables describe known scenario settings and do not imply proven causal relationship."
    )
    story.append(Paragraph(notes, body_style))

    doc.build(story)
    return buffer.getvalue()