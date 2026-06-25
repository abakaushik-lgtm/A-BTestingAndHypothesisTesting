import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print total page count
    and draw consistent running header and footer layout.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        
        # Suppress header and footer on the cover/first page
        if self._pageNumber == 1:
            self.restoreState()
            return
            
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Running Header
        self.drawString(0.75 * inch, 10.4 * inch, "A/B TESTING & EXPERIMENTATION PLATFORM")
        self.drawRightString(7.75 * inch, 10.4 * inch, f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(0.75 * inch, 10.3 * inch, 7.75 * inch, 10.3 * inch)
        
        # Running Footer
        self.line(0.75 * inch, 0.75 * inch, 7.75 * inch, 0.75 * inch)
        self.drawString(0.75 * inch, 0.55 * inch, "CONFIDENTIAL - FOR INTERNAL EXECUTION ONLY")
        self.drawRightString(7.75 * inch, 0.55 * inch, f"Page {self._pageNumber} of {page_count}")
        
        self.restoreState()

def generate_pdf_report(
    file_path, 
    dataset_summary, 
    cleaning_summary, 
    z_res, 
    t_res_aov, 
    t_res_dur, 
    recs, 
    revenue_impact,
    expected_traffic=1000000,
    aov_input=75.0
):
    """
    Compiles report flowables and saves the PDF.
    """
    # Setup document structure
    # letter is 8.5 x 11 inches. Margins 0.75 in. Printable width = 7 inches.
    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles to look modern and neat
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1E3A8A"), # Deep Navy
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4B5563"), # Cool Grey
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0D9488"), # Teal
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )
    
    bold_body_style = ParagraphStyle(
        "ReportBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=10
    )

    story = []
    
    # ================= PAGE 1: COVER =================
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("A/B Testing & Statistical Experimentation Platform", title_style))
    story.append(Paragraph("Executive Performance & Recommendation Report", subtitle_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Metadata block on cover
    metadata_data = [
        [Paragraph("<b>Date Generated:</b>", body_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), body_style)],
        [Paragraph("<b>Experiment Status:</b>", body_style), Paragraph(recs["decision_text"], bold_body_style)],
        [Paragraph("<b>Primary Metric:</b>", body_style), Paragraph("Conversion Rate (User level)", body_style)],
        [Paragraph("<b>Target Confidence:</b>", body_style), Paragraph("95% (Alpha = 0.05)", body_style)]
    ]
    t_meta = Table(metadata_data, colWidths=[2.0 * inch, 4.5 * inch])
    t_meta.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 1.5 * inch))
    
    # Colored Banner representing the decision
    banner_bg = "#D1FAE5" if recs["decision"] == "DEPLOY_B" else ("#FEE2E2" if recs["decision"] == "RETAIN_A" else "#FEF3C7")
    banner_border = "#10B981" if recs["decision"] == "DEPLOY_B" else ("#EF4444" if recs["decision"] == "RETAIN_A" else "#F59E0B")
    banner_text_color = "#065F46" if recs["decision"] == "DEPLOY_B" else ("#991B1B" if recs["decision"] == "RETAIN_A" else "#92400E")
    
    banner_para = Paragraph(
        f"<b>EXECUTIVE DECISION: {recs['decision_text'].upper()}</b><br/>"
        f"<i>Primary recommendation based on conversion and continuous metrics statistical validations.</i>",
        ParagraphStyle("BannerText", parent=callout_style, textColor=colors.HexColor(banner_text_color))
    )
    t_banner = Table([[banner_para]], colWidths=[7.0 * inch])
    t_banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(banner_bg)),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(banner_border)),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(t_banner)
    
    story.append(PageBreak())
    
    # ================= PAGE 2: EXEC SUMMARY & DATA QUALITY =================
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This report evaluates the performance difference between the current design (Variant A, Control) "
        "and the newly proposed page (Variant B, Treatment). We analyzed user interaction sessions to assess "
        "conversion rate differences, average order size alterations, and user engagement duration adjustments.",
        body_style
    ))
    
    story.append(Paragraph("Core Findings & Decision Rationale", h2_style))
    for rat in recs["rationale"]:
        story.append(Paragraph(f"• {rat}", bullet_style))
        
    if recs["risks"]:
        story.append(Paragraph("Identified Secondary Risks & Trade-Offs", h2_style))
        for risk in recs["risks"]:
            story.append(Paragraph(f"⚠️ {risk}", bullet_style))
            
    story.append(Paragraph("Actionable Next Steps", h2_style))
    for step in recs["next_steps"]:
        story.append(Paragraph(f"➔ {step}", bullet_style))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("2. Data Integrity & Preprocessing", h1_style))
    story.append(Paragraph(
        "A critical phase of A/B test analysis is validating data quality to prevent split-assignment violations, "
        "tracking duplicate sessions, or including corrupt metrics. Below is the preprocessing data quality log:",
        body_style
    ))
    
    cleaning_data = [
        [Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Count / Value</b>", bold_body_style)],
        [Paragraph("Total Loaded Raw Log Records", body_style), Paragraph(f"{cleaning_summary['initial_records_count']:,}", body_style)],
        [Paragraph("Duplicate Clicks / Rows Removed", body_style), Paragraph(f"{cleaning_summary['duplicates_removed']:,}", body_style)],
        [Paragraph("Missing User & Session IDs Removed", body_style), Paragraph(f"{cleaning_summary['missing_users_removed']:,}", body_style)],
        [Paragraph("Corrupted Variant Labels Standardized", body_style), Paragraph(f"{cleaning_summary['corrupted_variants_cleaned']:,}", body_style)],
        [Paragraph("User Variant Leakage (Removed Users / Sessions)", body_style), Paragraph(f"{cleaning_summary['leaked_users_count']:,} users ({cleaning_summary['leaked_records_removed']:,} records)", body_style)],
        [Paragraph("Out-of-Bounds Metrics (e.g. Negative Revenue) Removed", body_style), Paragraph(f"{cleaning_summary['negative_revenue_records_removed']:,}", body_style)],
        [Paragraph("<b>Final Experiment Clean Dataset Size</b>", bold_body_style), Paragraph(f"<b>{cleaning_summary['final_records_count']:,}</b>", bold_body_style)],
    ]
    t_clean = Table(cleaning_data, colWidths=[4.0 * inch, 3.0 * inch])
    t_clean.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_clean)
    
    story.append(PageBreak())
    
    # ================= PAGE 3: STATISTICAL TESTING =================
    story.append(Paragraph("3. Detailed Statistical Hypothesis Testing", h1_style))
    story.append(Paragraph(
        "We ran multiple hypothesis tests to validate the variant differences. The primary KPI is conversion rate, "
        "which compares unique user behaviors. Continuous metrics utilize Welch's T-Test to account for potentially "
        "differing standard deviations between the variants.",
        body_style
    ))
    
    # Conversion Z-Test table
    story.append(Paragraph("Primary Proportion Z-Test (Conversion Rates)", h2_style))
    z_table_data = [
        [Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Variant A (Control)</b>", bold_body_style), Paragraph("<b>Variant B (Treatment)</b>", bold_body_style), Paragraph("<b>Difference / Lift</b>", bold_body_style)],
        [Paragraph("Unique Users", body_style), Paragraph(f"{dataset_summary['users_a']:,}", body_style), Paragraph(f"{dataset_summary['users_b']:,}", body_style), Paragraph("-", body_style)],
        [Paragraph("Total Conversion Count", body_style), Paragraph(f"{dataset_summary['conversions_a']:,}", body_style), Paragraph(f"{dataset_summary['conversions_b']:,}", body_style), Paragraph("-", body_style)],
        [Paragraph("Conversion Rate", body_style), Paragraph(f"{z_res['rate_a']:.4%}", body_style), Paragraph(f"{z_res['rate_b']:.4%}", body_style), Paragraph(f"{z_res['rel_lift']:+.2%} Relative Lift", bold_body_style)],
        [Paragraph("Z-Score Statistic", body_style), Paragraph(f"{z_res['z_stat']:.3f}", body_style), Paragraph("-", body_style), Paragraph("-", body_style)],
        [Paragraph("P-Value", body_style), Paragraph(f"{z_res['p_value']:.4f}", body_style), Paragraph("-", body_style), Paragraph("Significant: <b>" + str(z_res['significant']) + "</b>", body_style)],
        [Paragraph("95% Conf. Interval of Diff", body_style), Paragraph(f"[{z_res['ci_lower']:.3%}, {z_res['ci_upper']:.3%}]", body_style), Paragraph("-", body_style), Paragraph("-", body_style)]
    ]
    t_z = Table(z_table_data, colWidths=[2.2 * inch, 1.6 * inch, 1.6 * inch, 1.6 * inch])
    t_z.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_z)
    
    # Continuous metrics table
    story.append(Spacer(1, 10))
    story.append(Paragraph("Secondary Continuous Metrics (Welch's T-Test)", h2_style))
    
    # Prepare AOV and Duration outputs
    aov_mean_a = f"${t_res_aov['mean_a']:.2f}" if t_res_aov else "N/A"
    aov_mean_b = f"${t_res_aov['mean_b']:.2f}" if t_res_aov else "N/A"
    aov_lift = f"{t_res_aov['rel_lift']:+.2%}" if t_res_aov else "N/A"
    aov_p = f"{t_res_aov['p_value']:.4f}" if t_res_aov else "N/A"
    aov_sig = str(t_res_aov['significant']) if t_res_aov else "N/A"
    
    dur_mean_a = f"{t_res_dur['mean_a']:.1f}s" if t_res_dur else "N/A"
    dur_mean_b = f"{t_res_dur['mean_b']:.1f}s" if t_res_dur else "N/A"
    dur_lift = f"{t_res_dur['rel_lift']:+.2%}" if t_res_dur else "N/A"
    dur_p = f"{t_res_dur['p_value']:.4f}" if t_res_dur else "N/A"
    dur_sig = str(t_res_dur['significant']) if t_res_dur else "N/A"
    
    t_table_data = [
        [Paragraph("<b>Continuous Metric</b>", bold_body_style), Paragraph("<b>Variant A (Mean)</b>", bold_body_style), Paragraph("<b>Variant B (Mean)</b>", bold_body_style), Paragraph("<b>Relative Lift</b>", bold_body_style), Paragraph("<b>P-Value</b>", bold_body_style), Paragraph("<b>Significant</b>", bold_body_style)],
        [Paragraph("Average Order Value (AOV)", body_style), Paragraph(aov_mean_a, body_style), Paragraph(aov_mean_b, body_style), Paragraph(aov_lift, body_style), Paragraph(aov_p, body_style), Paragraph(aov_sig, body_style)],
        [Paragraph("Average Session Duration", body_style), Paragraph(dur_mean_a, body_style), Paragraph(dur_mean_b, body_style), Paragraph(dur_lift, body_style), Paragraph(dur_p, body_style), Paragraph(dur_sig, body_style)]
    ]
    t_t = Table(t_table_data, colWidths=[2.2 * inch, 1.2 * inch, 1.2 * inch, 1.0 * inch, 0.7 * inch, 0.7 * inch])
    t_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_t)
    
    # ================= PAGE 4: BUSINESS IMPACT =================
    if revenue_impact:
        story.append(Spacer(1, 10))
        story.append(Paragraph("4. Business & Financial Impact Projections", h1_style))
        story.append(Paragraph(
            "Translating statistical percentages into dollar figures provides the leadership team with the "
            "necessary commercial justification. Below are projected returns should the treatment version be rolled out:",
            body_style
        ))
        
        impact_data = [
            [Paragraph("<b>Parameter</b>", bold_body_style), Paragraph("<b>Value / Projection</b>", bold_body_style)],
            [Paragraph("Expected Monthly Traffic (Unique Visitors)", body_style), Paragraph(f"{expected_traffic:,}", body_style)],
            [Paragraph("Assumed Average Order Value (AOV)", body_style), Paragraph(f"${aov_input:.2f}", body_style)],
            [Paragraph("Observed Conversion Rate Lift", body_style), Paragraph(f"{z_res['rel_lift']:+.2%} ({z_res['abs_lift']:+.4%} absolute)", body_style)],
            [Paragraph("<b>Additional Monthly Conversions</b>", body_style), Paragraph(f"{revenue_impact['monthly_conversions_gain']:+,.0f}", body_style)],
            [Paragraph("<b>Projected Monthly Revenue Gain</b>", bold_body_style), Paragraph(f"<b>${revenue_impact['monthly_revenue_gain']:+,.2f}</b>", bold_body_style)],
            [Paragraph("<b>Additional Annual Conversions</b>", body_style), Paragraph(f"{revenue_impact['annual_conversions_gain']:+,.0f}", body_style)],
            [Paragraph("<b>Projected Annual Revenue Gain</b>", bold_body_style), Paragraph(f"<b>${revenue_impact['annual_revenue_gain']:+,.2f}</b>", bold_body_style)],
        ]
        t_impact = Table(impact_data, colWidths=[4.0 * inch, 3.0 * inch])
        t_impact.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t_impact)
        
    story.append(Spacer(1, 20))
    
    # Signature area
    story.append(KeepTogether([
        Spacer(1, 15),
        Paragraph("Report Authorization & Sign-off", h2_style),
        Spacer(1, 15),
        Table([
            [Paragraph("_____________________________<br/><b>Lead Data Scientist</b>", body_style), 
             Paragraph("_____________________________<br/><b>VP of Growth & Product</b>", body_style)]
        ], colWidths=[3.5 * inch, 3.5 * inch])
    ]))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
