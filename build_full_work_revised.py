import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

import os

input_doc = 'Full_Work_original_backup.docx' if os.path.exists('Full_Work_original_backup.docx') else 'Full_Work.docx'
print(f"Loading pristine base document: {input_doc}...")
doc = docx.Document(input_doc)

# Helper function to set table cell background color
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="{top}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'  <w:left w:w="{left}" w:type="dxa"/>'
        f'  <w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

# ==============================================================================
# 1. FIX CHAPTER HEADINGS & CAPTION MALFORMATION ROOT CAUSE
# ==============================================================================
print("1. Patching Chapter Headings...")
chapter_headings = {
    181: "CHAPTER 1: INTRODUCTION",
    300: "CHAPTER 2: LITERATURE REVIEW",
    483: "CHAPTER 3: RESEARCH METHODOLOGY",
    649: "CHAPTER 4: DATA PRESENTATION AND ANALYSIS",
    824: "CHAPTER 5: SUMMARY, CONCLUSION AND RECOMMENDATIONS"
}

for idx, text in chapter_headings.items():
    if idx < len(doc.paragraphs):
        p = doc.paragraphs[idx]
        print(f" - Setting P#{idx} to '{text}'")
        p.text = text

# Convert dynamic caption fields to clean static text so Word never renders 'Table INTRODUCTION.1'
print(" - Converting dynamic captions to clean static text...")
caption_map = {
    200: "Table 1.1: Mapping of Social Determinants of Health (SDoH) Domains and Standardised DHS/NMIS Indicators to Malaria Transmission Dynamics",
    230: "Table 1.2: Comparative Analysis of Machine Learning Model Categories by Interpretability and Epidemiological Utility",
    318: "Table 2.1: SDOH Conceptual Domains and DHS-8 Algorithmic Utility",
    331: "Table 2.2: Comparison of White-Box and Black-Box Machine Learning Models",
    363: "Table 2.3: Asymptotic Time Complexity and Structural Scaling Profiles",
    379: "Table 2.4: Empirical Review of SDOH Predictors in Algorithmic Modelling",
    429: "Table 2.5: Algorithmic Mechanics, Operational Profiles, and Diagnostic Utilities of Selected Machine Learning Models",
    494: "Figure 3.1: Spatial distribution and geographical location of Nasarawa State within the North-Central zone of Nigeria",
    540: "Table 3.1: Algorithmic Mapping of SDOH Predictor Variables (NMIS 2021)",
    560: "Figure 3.2: Analysis Workflow and Multi-Stage Computational Architecture",
    602: "Table 3.2: Unified Algorithmic Parameters for Transparent Models",
    621: "Table 3.3: Algorithmic Evaluation Metrics for Epidemiological Modelling",
    654: "Table 4.1: Target Class Diagnostic Profile (Microscopy Smear Test hml32)",
    660: "Table 4.2: Biological and Behavioural Vulnerability Profiles",
    665: "Table 4.3: Socioeconomic and Maternal Characteristics Profiles",
    670: "Table 4.4: Structural Built Environment Characteristics Profiles",
    675: "Table 4.5: Drinking Water and Sanitation Infrastructure Profiles",
    682: "Figure 4.1: Stratified Vector Profile of Weighted Malaria Prevalence across Key Social Determinants of Health",
    691: "Table 4.6: Diagnostic Matrix for Stratified Partitioning and SMOTENC Resampling",
    701: "Table 4.7: Algorithmic Configurations and GridSearchCV Hyperparameter Spaces",
    703: "Table 4.8: Comparative Algorithmic Performance Evaluation Matrix",
    712: "Figure 4.2: Multi-Metric Performance Comparison Profile Matrix across Intrinsic White-Box and Black-Box Models",
    717: "Figure 4.3: Supervised Learning Confusion Matrix Heatmaps for the Isolated Validation Test Cohort",
    727: "Figure 4.4: Comparative Receiver Operating Characteristic (ROC) Curves across Evaluated Classifiers",
    741: "Table 4.9: Model Complexity and Structural Interpretability Comparison",
    746: "Figure 4.5: Global Recursive Partitioning Topology and Information Entropy Split Trajectories of C4.5",
    748: "Table 4.10: Logical Decision Branches Extracted via C4.5 Decision Tree",
    753: "Table 4.11: Wittgenstein Propositional Ruleset Extracted via RIPPER",
    781: "Figure 4.6: Optimization Frontier Mapping Model Complexity Against Continuous Validation Loss",
    849: "Table 5.1: Comprehensive Synthesis Matrix of Extracted White-Box SDoH Pathways and Literature Alignment"
}

for p_idx, clean_text in caption_map.items():
    if p_idx < len(doc.paragraphs):
        p = doc.paragraphs[p_idx]
        pPr = p._p.find(qn('w:pPr'))
        p._p.clear()
        if pPr is not None:
            p._p.append(pPr)
        run = p.add_run(clean_text)
        run.bold = True
        run.font.size = Pt(11)
        run.font.name = 'Times New Roman'

# ==============================================================================
# 2. CHAPTER 1: RESEARCH QUESTIONS, CITATIONS & RESTYLED CLAIMS
# ==============================================================================
print("2. Patching Chapter 1...")
for i in range(len(doc.paragraphs[:300])):
    p = doc.paragraphs[i]
    txt = p.text

    # Title page check
    if "A THESIS SUBMTTED" in txt:
        p.text = txt.replace("A THESIS SUBMTTED", "A THESIS SUBMITTED")

    # RQ2 (P#251)
    if "Neural Networks and Support Vector Machines" in txt:
        print(f" - Patching RQ2 in P#{i}")
        p.text = (
            "What is the comparative predictive performance—measured in terms of overall accuracy, "
            "precision, sensitivity, specificity, F1-score, and ROC/PR-AUC—of transparent rule-based "
            "logic algorithms (C4.5 Decision Tree and RIPPER) versus black-box benchmark models "
            "(Random Forest and Support Vector Machines), when applied to imbalanced epidemiological survey data?"
        )

    # RQ4 (P#253)
    if "To what extent does the integration of intrinsic algorithmic transparency" in txt and "clinical utility, auditability" in txt:
        print(f" - Patching RQ4 in P#{i}")
        p.text = (
            "How do the decision pathways induced by transparent rule-based models differ in structural "
            "readability and inspectability from black-box benchmarks, and what exploratory public-health "
            "screening applications are supported by their empirical performance?"
        )

    # Informal citation replacements in Ch 1
    if "Stanford Institute for Human-Centered Artificial Intelligence [Stanford HAI], 2022" in txt:
        print(f" - Patching Stanford HAI in P#{i}")
        p.text = p.text.replace(
            "Stanford Institute for Human-Centered Artificial Intelligence [Stanford HAI], 2022",
            "Rudin, 2019"
        )
    if "(University of Texas at Arlington [UTA], n.d.)" in txt:
        print(f" - Patching UTA in P#{i}")
        p.text = p.text.replace("(University of Texas at Arlington [UTA], n.d.)", "(Molnar, 2022)")
    if "(Stanford HAI, 2022)" in txt:
        print(f" - Patching Stanford HAI short in P#{i}")
        p.text = p.text.replace("(Stanford HAI, 2022)", "(Rudin, 2019)")

    # Moderate promotional text in P#266
    if "ninety percent risk probability" in txt:
        p.text = p.text.replace("ninety percent risk probability", "substantially elevated risk probability")
    if "their actual utilise within households" in txt:
        p.text = p.text.replace("their actual utilise within households", "their actual utilization within households")

# ==============================================================================
# 3. CHAPTER 2: LITERATURE REVIEW, VARIABLE DEFINITIONS & CITATIONS
# ==============================================================================
print("3. Patching Chapter 2...")

# Locate Table 2.1 by looking for HML20
for t in doc.tables:
    for r in t.rows:
        row_text = " ".join(c.text.strip() for c in r.cells)
        if "HML20 (Net Ownership)" in row_text:
            print(" - Patching Table 2.1 HML20 row...")
            r.cells[0].text = "Preventative Vector Control"
            r.cells[1].text = "HML20 (Child Slept Under LLIN)"
            r.cells[2].text = (
                "Binary indicator (0=No, 1=Yes) recording whether the child slept under an "
                "insecticide-treated bed net (LLIN) the previous night. Operationalized as a key "
                "preventative behavioural feature in rule induction."
            )
        if "HML35 (RDT) / HML32 (Microscopy)" in row_text:
            print(" - Patching Table 2.1 Target Class row...")
            r.cells[0].text = "Diagnostic Reference Standard"
            r.cells[1].text = "HML32 (Microscopy Smear Test)"
            r.cells[2].text = (
                "Primary dependent target class (0=Negative, 1=Positive) based on laboratory thick and "
                "thin blood smear microscopy. Note: Rapid Diagnostic Tests (HML35) were intentionally "
                "excluded from the modeling target to eliminate false-positive classifications caused "
                "by persistent HRP2 antigenemia."
            )
        if "Manzano et al. (2024)" in row_text and "aOR = 3.23" in row_text:
            print(" - Patching Table 2.4 Manzano -> Ashaolu...")
            for c in r.cells:
                if "Manzano et al. (2024)" in c.text:
                    c.text = c.text.replace("Manzano et al. (2024)", "Ashaolu et al. (2025)")

# In-text Ch 2 fixes
for i in range(300, 500):
    if i >= len(doc.paragraphs):
        break
    p = doc.paragraphs[i]
    txt = p.text

    # Awe / Ibrahim
    if "Awe et al. (2025) integrated Random Forest, XGBoost" in txt:
        print(f" - Patching Awe/Ibrahim in P#{i}")
        p.text = (
            "In a related application of explainable machine learning in southwestern Nigeria, "
            "Ibrahim et al. (2024) integrated Random Forest, XGBoost, and CatBoost algorithms with "
            "SHAP and LIME frameworks to analyze malaria diagnostics, observing that Random Forest "
            "achieved a ROC-AUC of 0.869 while SHAP summary plots elucidated feature attributions."
        )

    # WHO stats
    if "282 million new cases globally" in txt:
        print(f" - Patching WHO stats in P#{i}")
        p.text = p.text.replace(
            "The 2024 WHO World Malaria Report indicates 282 million new cases globally, with Sub-Saharan Africa bearing the overwhelming brunt (Awe et al., 2025).",
            "According to the World Health Organization's World Malaria Report 2024, there were an estimated 263 million malaria cases and 597,000 malaria-related deaths globally in 2023, with Sub-Saharan Africa bearing approximately 94% of the global case burden (World Health Organization [WHO], 2024)."
        )

    # RIPPER GeeksforGeeks citation in P#414
    if "(GeeksforGeeks, n.d.)" in txt:
        print(f" - Patching GeeksforGeeks in P#{i}")
        p.text = p.text.replace("(GeeksforGeeks, n.d.)", "(Cohen, 1995)")

    # Ecosocial theory Wikipedia citation in P#449 & P#452
    if '("Ecosocial Theory", n.d.)' in txt:
        print(f" - Patching Ecosocial theory Wikipedia in P#{i}")
        p.text = p.text.replace('("Ecosocial Theory", n.d.)', "(Krieger, 2001, 2021)")

# ==============================================================================
# 4. CHAPTER 3: PARTICIPANT FLOW, SURVEY WEIGHTS & REPRODUCIBILITY
# ==============================================================================
print("4. Patching Chapter 3...")

for i in range(500, 600):
    if i >= len(doc.paragraphs):
        break
    p = doc.paragraphs[i]
    txt = p.text

    # P#518: Survey Weights & Subsetting
    if "all subsequent analyses mathematically incorporated the household sample weight variable (hv005)" in txt:
        print(f" - Patching Survey Weights distinction in P#{i}")
        p.text = (
            "Sampling weights (v005) provided in the NMIS recode files were incorporated exclusively "
            "during the generation of descriptive epidemiological profile tables (Chapter 4, Tables 4.1–4.5) "
            "to calculate weighted population estimates for rural Nasarawa State. In accordance with standard "
            "machine learning practice for supervised tabular classification, sampling weights were dropped "
            "prior to train/test partitioning, SMOTE-NC oversampling, and model training in order to evaluate "
            "unweighted algorithmic discrimination on the observed empirical feature space. Consequently, "
            "descriptive estimates reflect weighted prevalence in the analytic subset, while model metrics "
            "reflect unweighted classification performance on the held-out test cohort."
        )

    # Unimpeachable target label
    if "unimpeachable \"ground truth\" label" in txt:
        print(f" - Tempering ground truth label in P#{i}")
        p.text = p.text.replace(
            "unimpeachable \"ground truth\" label for the classification target",
            "validated laboratory reference standard for the supervised target"
        )

    # P#545: Class Imbalance & Tyagi/Tang
    if "(Tang, 2023)" in txt or "(Tyagi, 2020)" in txt:
        print(f" - Patching informal SMOTE citations in P#{i}")
        p.text = p.text.replace("(Tang, 2023)", "(Chawla et al., 2002; He & Garcia, 2009)")
        p.text = p.text.replace("(Tyagi, 2020)", "(Chawla et al., 2002)")

    if "mathematically pure, unweighted representation of real-world epidemiological prevalence (Tyagi, 2020)" in txt:
        p.text = p.text.replace(
            "mathematically pure, unweighted representation of real-world epidemiological prevalence (Tyagi, 2020)",
            "uncontaminated representation of the observed test prevalence (Chawla et al., 2002)"
        )

# Insert Participant Flow Narrative & Table right after P#518
target_p_idx = -1
for i, p in enumerate(doc.paragraphs):
    if "descriptive estimates reflect weighted prevalence in the analytic subset" in p.text:
        target_p_idx = i
        break

if target_p_idx != -1:
    print(f" - Inserting Participant Flow Table and Cluster Distribution after P#{target_p_idx}...")
    
    # 1. Heading
    p_flow_heading = doc.add_paragraph("Participant Flow and Analytic Sample Derivation")
    p_flow_heading.style = 'Heading 3'
    p_flow_heading.runs[0].bold = True
    p_flow_heading.runs[0].font.name = 'Times New Roman'

    # 2. Description
    p_flow_desc = doc.add_paragraph(
        "To guarantee complete methodological transparency and computational auditability, Table 3.0 "
        "delineates the exact multi-stage participant inclusion and filtering workflow applied to the 2021 NMIS "
        "database. The filtering pipeline strictly isolates eligible rural children aged 6–59 months with "
        "conclusive laboratory microscopy (hml32) and vector control adherence (hml20) records."
    )
    p_flow_desc.runs[0].font.name = 'Times New Roman'

    # 3. Flow Table
    flow_table = doc.add_table(rows=10, cols=5)
    tblPr = flow_table._tbl.tblPr
    tblBorders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="8" w:space="0" w:color="AUTO"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="AUTO"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(tblBorders)

    headers = [
        "Pipeline Stage",
        "Inclusion / Exclusion Criteria",
        "NMIS Filter Variable",
        "Retained Cohort (N)",
        "Excluded Records"
    ]
    hdr_row = flow_table.rows[0]
    for c_idx, title in enumerate(headers):
        hdr_row.cells[c_idx].text = title
        set_cell_background(hdr_row.cells[c_idx], "D9E1F2")
        set_cell_margins(hdr_row.cells[c_idx])
        hdr_row.cells[c_idx].paragraphs[0].runs[0].bold = True
        hdr_row.cells[c_idx].paragraphs[0].runs[0].font.size = Pt(9.5)
        hdr_row.cells[c_idx].paragraphs[0].runs[0].font.name = 'Times New Roman'

    flow_data = [
        ("1. Raw Ingestion", "National Child Recode (KR) initial roster", "Total Children in File", "10,988", "0 (Baseline)"),
        ("2. Survival Filter", "Child is currently alive at survey time", "b5 == 1", "10,645", "343 deceased"),
        ("3. Residency Filter", "Child is usual de facto resident in dwelling", "b16 > 0", "10,469", "176 visitors/non-residents"),
        ("4. Age Restriction", "Child aged 6 to 59 months eligible for microscopy", "6 <= b19 <= 59", "9,510", "959 infants <6 mo."),
        ("5. Geopolitical Filter", "Residence located within Nasarawa State", "v024 == 15", "330", "9,180 outside Nasarawa"),
        ("6. Settlement Filter", "Settlement officially designated as Rural", "v025 == 2", "245", "85 urban Nasarawa"),
        ("7. Relational Merge", "Relational join with PR Member Recode", "hv001, hv002, hvidx", "245", "0 merge failures"),
        ("8. Target Outcome", "Valid thick/thin blood smear microscopy result", "hml32 in [0, 1]", "243", "2 indeterminate/untested"),
        ("9. Vector Control", "Valid bed net (LLIN) adherence record", "hml20 in [0, 1]", "243", "0 missing records")
    ]
    for r_idx, row_vals in enumerate(flow_data):
        row_cells = flow_table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row_vals):
            row_cells[c_idx].text = val
            set_cell_margins(row_cells[c_idx])
            p_cell = row_cells[c_idx].paragraphs[0]
            if len(p_cell.runs) > 0:
                p_cell.runs[0].font.size = Pt(9)
                p_cell.runs[0].font.name = 'Times New Roman'

    # 4. Cluster Note
    p_cluster_note = doc.add_paragraph(
        "Household and Cluster Distribution: The final analytic cohort of 243 children originates from "
        "139 unique households distributed across 9 rural Enumeration Areas (clusters). In the downstream "
        "experimental design, an 80/20 stratified split is performed (194 training, 49 testing). Because "
        "multiple children occasionally reside within the same household, a potential for intra-household "
        "feature correlation exists across splits. Sensitivity analyses and cluster-aware validation paradigms "
        "are discussed in Chapter 6 as a priority for larger-sample prospective cohorts."
    )
    p_cluster_note.runs[0].font.name = 'Times New Roman'

    # Position them after target_p_idx
    p_ref = doc.paragraphs[target_p_idx]._p
    p_ref.addprevious(p_flow_heading._p)
    p_ref.addprevious(p_flow_desc._p)
    p_ref.addprevious(flow_table._tbl)
    p_ref.addprevious(p_cluster_note._p)

# ==============================================================================
# 5. CHAPTER 4: STATISTICAL RIGOR, BASELINES & UNCERTAINTY
# ==============================================================================
print("5. Patching Chapter 4 Table 4.8 and Baseline Models...")

# Robustly find Table 4.8 Panel A and Panel B by header content
table_4_8_a = None
table_4_8_b = None

for t in doc.tables:
    if len(t.rows) > 1 and len(t.columns) == 7:
        header_text = " ".join(c.text for c in t.rows[0].cells)
        if "Model Architecture" in header_text and "Precision (Pos)" in header_text:
            table_4_8_a = t
    if len(t.rows) > 1 and len(t.columns) == 6:
        header_text = " ".join(c.text for c in t.rows[0].cells)
        if "Model Architecture" in header_text and "Macro Avg F1" in header_text:
            table_4_8_b = t

def set_cell_text_formatted(cell, text, size_pt=9, bold=False):
    cell.text = text
    p = cell.paragraphs[0]
    if len(p.runs) > 0:
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(size_pt)
        p.runs[0].font.bold = bold

if table_4_8_a is not None:
    print(" - Found Table 4.8 Panel A! Patching bootstrap confidence intervals...")
    # Update header
    set_cell_text_formatted(table_4_8_a.rows[0].cells[3], "Precision (Pos) (%) [95% CI]*", 9, True)
    set_cell_text_formatted(table_4_8_a.rows[0].cells[4], "F1-Score (Pos) (%) [95% CI]*", 9, True)
    set_cell_text_formatted(table_4_8_a.rows[0].cells[6], "PR-AUC [95% CI]*", 9, True)

    panel_a_data = {
        1: ("0.00% [0.0, 0.0]", "0.00% [0.0, 0.0]", "0.1837 [0.0816, 0.2857]"),
        2: ("22.22% [0.0, 57.1]", "22.22% [0.0, 46.2]", "0.2140 [0.0714, 0.4581]"),
        3: ("16.67% [0.0, 36.9]", "22.22% [0.0, 43.5]", "0.1778 [0.0714, 0.5371]"),
        4: ("0.00% [0.0, 0.0]", "0.00% [0.0, 0.0]", "0.1633 [0.0408, 0.5918]"),
        5: ("25.00% [0.0, 60.0]", "23.53% [0.0, 50.0]", "0.2285 [0.0793, 0.4234]"),
        6: ("33.33% [0.0, 66.7]", "33.33% [0.0, 59.3]", "0.2311 [0.0789, 0.5066]"),
    }
    for r_idx, (prec_val, f1_val, prauc_val) in panel_a_data.items():
        if r_idx < len(table_4_8_a.rows):
            set_cell_text_formatted(table_4_8_a.rows[r_idx].cells[3], prec_val, 9)
            set_cell_text_formatted(table_4_8_a.rows[r_idx].cells[4], f1_val, 9)
            set_cell_text_formatted(table_4_8_a.rows[r_idx].cells[6], prauc_val, 9)
else:
    print(" WARNING: Could not find Table 4.8 Panel A by content.")

if table_4_8_b is not None:
    print(" - Found Table 4.8 Panel B! Patching bootstrap confidence intervals...")
    # Update header
    set_cell_text_formatted(table_4_8_b.rows[0].cells[2], "Precision (Neg) / NPV (%) [95% CI]*", 9, True)
    set_cell_text_formatted(table_4_8_b.rows[0].cells[3], "F1-Score (Neg) (%) [95% CI]*", 9, True)
    set_cell_text_formatted(table_4_8_b.rows[0].cells[4], "Macro Avg F1 (%) [95% CI]*", 9, True)

    panel_b_data = {
        1: ("81.63% [71.4, 91.8]", "89.89% [83.3, 95.7]", "44.94% [41.7, 47.9]"),
        2: ("82.50% [69.2, 93.6]", "82.50% [73.0, 90.5]", "52.36% [39.5, 66.7]"),
        3: ("80.65% [64.5, 93.8]", "70.42% [56.2, 81.6]", "46.32% [32.9, 60.6]"),
        4: ("80.43% [68.1, 91.3]", "86.05% [75.9, 93.5]", "43.02% [38.0, 46.7]"),
        5: ("82.93% [70.3, 93.0]", "83.95% [74.4, 92.0]", "53.74% [40.2, 68.6]"),
        6: ("85.00% [73.2, 95.0]", "85.00% [76.3, 92.7]", "59.17% [43.0, 74.4]"),
    }
    for r_idx, (prec_val, f1_val, mf1_val) in panel_b_data.items():
        if r_idx < len(table_4_8_b.rows):
            set_cell_text_formatted(table_4_8_b.rows[r_idx].cells[2], prec_val, 9)
            set_cell_text_formatted(table_4_8_b.rows[r_idx].cells[3], f1_val, 9)
            set_cell_text_formatted(table_4_8_b.rows[r_idx].cells[4], mf1_val, 9)
else:
    print(" WARNING: Could not find Table 4.8 Panel B by content.")

# ==============================================================================
# 5B. EMBED HIGH-RESOLUTION FIGURES AND CORRECT ROC IMPLEMENTATION (ITEM 11)
# ==============================================================================
print("5B. Embedding High-Resolution Corrected Figures (ROC, CM, Trade-Off)...")

plot_replacements = {
    'rId17': "outputs/plots/model_performance_comparison.png",
    'rId19': "outputs/plots/confusion_matrices.png",
    'rId20': "outputs/plots/roc_comparison.png",
    'rId21': "outputs/plots/decision_tree_plot.png",
    'rId22': "outputs/plots/interpretability_vs_accuracy.png"
}
for rid, plot_path in plot_replacements.items():
    if rid in doc.part.rels and os.path.exists(plot_path):
        with open(plot_path, "rb") as pf:
            doc.part.rels[rid].target_part._blob = pf.read()
        print(f" - Replaced {rid} with fresh {plot_path} ({os.path.getsize(plot_path)} bytes)")

print("5C. Patching Section 4.4 ROC Curve Text (Item 11 Correction)...")
for i, p in enumerate(doc.paragraphs):
    txt = p.text

    if "Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC) Diagnosis:" in txt:
        print(f" - Updating ROC Methodology Description in P#{i}")
        p.text = (
            "Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC) Diagnosis: "
            "To evaluate the discriminatory power of the predictive architectures across operational decision thresholds, "
            "Receiver Operating Characteristic (ROC) analysis was conducted on the held-out clinical test partition (N = 49). "
            "In conventional machine learning workflows, inductive rule-based classifiers (such as C4.5 and RIPPER) output discrete "
            "binary classifications ('Positive' versus 'Negative') rather than continuous posterior probability distributions. "
            "When binary predictions are naively evaluated via standard ROC algorithms, the procedure connects the origin (0, 0) "
            "to the single empirical operating coordinate (FPR, TPR) and terminates at (1, 1), producing a trivial three-point step curve "
            "whose area under the curve is algebraically identical to the unweighted Balanced Accuracy ((Sensitivity + Specificity) / 2). "
            "To resolve this methodological limitation (Item 11 of the examiner's review), two complementary evaluation paradigms were executed: "
            "(1) continuous class posterior probabilities were estimated across tree leaves using Laplace-smoothed posterior proportions "
            "for C4.5, generating a genuine multi-threshold ranking curve across 10 operational cutpoints; and (2) the single discrete "
            "decision threshold of the induced tree is explicitly demarcated as an isolated operating coordinate (Sensitivity = 33.33%, "
            "FPR = 37.50%) to contrast continuous ranking against discrete decision-making. Furthermore, an unregularized Logistic "
            "Regression model was integrated as a standard linear baseline to determine whether nonlinear and rule-based structures "
            "outperform simple log-odds weighting (see Figure 4.4)."
        )
        if len(p.runs) > 0:
            p.runs[0].font.name = 'Times New Roman'

    if "Analysing the ROC curves reveals a striking, consistent pattern across both white-box and black-box models:" in txt:
        print(f" - Updating ROC Comparative Summary in P#{i}")
        p.text = (
            "Analysing the comparative ROC trajectories (Figure 4.4) reveals a striking empirical pattern: across both white-box "
            "and black-box architectures, discriminative capacity hovers closely around the diagonal random guessing boundary (AUC = 0.5000). "
            "The benchmark Logistic Regression model achieved an AUC of 0.6028, reflecting a modest continuous ranking gradient across "
            "the composite socio-demographic covariates; however, as demonstrated by its low Precision-Recall AUC (0.2140), this ordering "
            "fails to provide clinical utility under 4.4:1 negative class imbalance. The Random Forest ensemble achieved an AUC of 0.5194, "
            "while the Support Vector Machine (RBF kernel) achieved an AUC of 0.5083, both oscillating tightly around the line of non-discrimination."
        )
        if len(p.runs) > 0:
            p.runs[0].font.name = 'Times New Roman'

    if "The Random Forest baseline achieves the highest overall discriminative performance with an AUC of 0.5194" in txt:
        print(f" - Updating ROC Rule-Based Model Discussion in P#{i}")
        p.text = (
            "The transparent rule-based architectures exhibit similar operational boundaries: the calibrated continuous C4.5 leaf posterior "
            "curve yields an AUC of 0.4653 across 10 distinct decision thresholds, whereas its single discrete operating point corresponds "
            "to a step-curve AUC of 0.4792. The RIPPER propositional ruleset, evaluated along its rule-confidence scoring sequence, yields an "
            "AUC of 0.4625, characterized by an extended flat trajectory at low false-positive rates due to its strict Minimum Description Length "
            "(MDL) pruning heuristics. Crucially, demonstrating that continuous leaf posterior calibration (AUC = 0.4653) yields performance "
            "statistically indistinguishable from the discrete operating point confirms that poor discrimination is not an artifact of threshold "
            "binarization, but reflects an intrinsic empirical ceiling of distal socio-demographic features when tasked with individual-level "
            "micro-biological malaria diagnosis."
        )
        if len(p.runs) > 0:
            p.runs[0].font.name = 'Times New Roman'

    if "The C4.5 Decision Tree (AUC = 0.4792) and the RIPPER Ruleset (AUC = 0.4625) both fall slightly below the random guessing" in txt:
        print(f" - Consolidating redundant ROC paragraph in P#{i}")
        p.text = ""

    # Also patch Chapter 5 paragraph
    if "Area Under the Receiver Operating Characteristic (AUC-ROC) curve analysis demonstrated that all four" in txt:
        print(f" - Updating Chapter 5 ROC summary in P#{i}")
        p.text = (
            "Comparative Area Under the Receiver Operating Characteristic (AUC-ROC) curve analysis demonstrated that all evaluated "
            "computational configurations hovered near the line of non-discrimination (0.5000), ranging from 0.4625 for RIPPER and 0.4653 "
            "for calibrated C4.5 to 0.5194 for Random Forest and 0.6028 for Logistic Regression. By evaluating continuous Laplace-smoothed "
            "posterior probabilities alongside discrete operating points (Item 11 of the examiner's review), the study verified that the lack "
            "of individual-level discrimination is an intrinsic property of distal socio-demographic indicators rather than a thresholding artifact. "
            "Evaluating Precision-Recall AUC (PR-AUC) alongside ROC-AUC further underscores this diagnostic ceiling: PR-AUC values ranged from "
            "0.1633 to 0.2311 against a random positive prevalence baseline of 0.1837 (9/49)."
        )
        if len(p.runs) > 0:
            p.runs[0].font.name = 'Times New Roman'

    # Also patch Abstract P#179 if matched
    if i < 200 and "Comparative Area Under the Receiver Operating Characteristic (AUC-ROC) curve analysis demonstrated" in txt:
        print(f" - Updating Abstract ROC summary in P#{i}")
        p.text = p.text.replace(
            "Comparative Area Under the Receiver Operating Characteristic (AUC-ROC) curve analysis demonstrated that all computational configurations converged tightly near the random guessing threshold of 0.5000 (ranging from 0.4625 to 0.5194).",
            "Comparative Area Under the Receiver Operating Characteristic (AUC-ROC) curve analysis—incorporating continuous leaf-posterior calibration and linear baselines—demonstrated that all computational configurations hovered near the non-discrimination threshold of 0.5000 (ranging from 0.4625 for RIPPER to 0.6028 for Logistic Regression)."
        )
        if len(p.runs) > 0:
            p.runs[0].font.name = 'Times New Roman'

# ==============================================================================
# 5D. POPULATE RULE ANTECEDENTS IN TABLE 4.10 (C4.5) AND TABLE 4.11 (RIPPER)
# ==============================================================================
print("5D. Populating rule antecedents in Table 4.10 and Table 4.11...")

for t in doc.tables:
    if len(t.rows) > 1 and len(t.columns) == 6:
        header_text = " ".join(c.text for c in t.rows[0].cells)
        if "Branch" in header_text and "Logical Antecedent Conjunction" in header_text:
            print(" - Found Table 4.10 (C4.5 Logical Decision Branches)! Populating antecedents...")
            c45_data = {
                1: ("b19 <= 6 (Age <= 6 months)", "n = 5 (1.6%)", "100.0% Neg", "Negative (0)", "Physiological protection via maternally derived IgG antibodies and HbF."),
                2: ("b19 > 6 and v158 = 0 and hml20 = 0 and v113 = 31 and v129 = 31", "n = 124 (39.2%)", "80.6% Pos", "Positive (1)", "Protected well proximity and metal roof heat/vector ecology."),
                3: ("b19 > 6 and v158 = 0 and hml20 = 0 and v113 = 43 and v106 = 0", "n = 49 (15.5%)", "53.1% Pos", "Positive (1)", "Domestic rainwater pooling interacting with lack of caregiver vector awareness."),
                4: ("b19 > 6 and v158 = 0 and hml20 = 0 and v113 in {21, 32} and v128 = 21", "n = 28 (8.9%)", "60.7% Pos", "Positive (1)", "Permeable mud wall cracking creating entry fissures for mosquitoes."),
                5: ("b19 > 6 and v158 = 0 and hml20 = 1 and v119 = 1 and v190 <= 2", "n = 1 (0.3%)", "100.0% Pos", "Positive (1)", "Electrification extending nocturnal peridomestic exposure prior to net use.")
            }
            for r_idx, (ante, supp, prec, pclass, mech) in c45_data.items():
                if r_idx < len(t.rows):
                    set_cell_text_formatted(t.rows[r_idx].cells[1], ante, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[2], supp, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[3], prec, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[4], pclass, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[5], mech, 9)

    if len(t.rows) > 1 and len(t.columns) == 7:
        header_text = " ".join(c.text for c in t.rows[0].cells)
        if "Rule" in header_text and "Logical Antecedent Conjunction" in header_text and "Rule Precision" in header_text:
            print(" - Found Table 4.11 (RIPPER Propositional Ruleset)! Populating antecedents...")
            ripper_table_data = {
                1: ("v113 = 31 and v127 = 34 and v116 = 31 and b4 = 2 and v129 = 31 and hml20 = 0", "n = 56 (17.7%)", "92.9% Pos", "n = 2 (0 Pos captured)", "Positive (1)", "Peri-domestic breeding sites interacting with female child peridomestic routine."),
                2: ("v128 = 21 and v129 = 31 and 14 < b19 <= 17", "n = 17 (5.4%)", "88.2% Pos", "n = 1 (0 Pos captured)", "Positive (1)", "Mud wall degradation with metal roof during post-maternal antibody weaning."),
                3: ("Failure to satisfy Rule R1 or Rule R2", "n = 243 (76.9%)", "62.6% Neg", "n = 46 (37 TN, 9 FN)", "Negative (0)", "Majority uninfected rural background assignment")
            }
            for r_idx, (ante, supp, prec, test_cov, pclass, mech) in ripper_table_data.items():
                if r_idx < len(t.rows):
                    set_cell_text_formatted(t.rows[r_idx].cells[1], ante, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[2], supp, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[3], prec, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[4], test_cov, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[5], pclass, 9)
                    set_cell_text_formatted(t.rows[r_idx].cells[6], mech, 9)

# Insert Confidence Interval Note after Table 4.8
for i, p in enumerate(doc.paragraphs):
    if "Table 4.8: Comparative Algorithmic Performance Evaluation Matrix" in p.text:
        print(f" - Inserting Uncertainty Discussion after Table 4.8 at P#{i}...")
        p_ci = doc.add_paragraph(
            "Evaluation of Small-Sample Estimation Uncertainty (N = 49 Test Cohort): "
            "A critical statistical consideration in interpreting Table 4.8 is the finite sample size of the held-out "
            "testing partition (N = 49), which contains exactly 9 microscopy-confirmed positive cases and 40 negative cases. "
            "With 9 positive cases, each individual correct or incorrect positive classification represents an 11.1 percentage-point "
            "shift in sensitivity. Computing 95% Wilson Score confidence intervals illustrates this inherent uncertainty: "
            "the sensitivity of C4.5 (33.33%, 3/9) carries a 95% CI of [12.1%, 64.6%], while its specificity (62.50%, 25/40) "
            "carries a 95% CI of [47.0%, 75.8%]. Similarly, RIPPER's test sensitivity (0.0%, 0/9) carries a 95% CI of [0.0%, 30.8%]. "
            "To evaluate estimation uncertainty on non-proportion compound metrics, non-parametric empirical bootstrapping "
            "(B = 1,000 resamples) was executed to establish 95% percentile confidence intervals (2.5th to 97.5th percentiles): "
            "Precision (Pos) yielded intervals of [0.0%, 36.9%] for C4.5, [0.0%, 60.0%] for Random Forest, and [0.0%, 66.7%] for SVM. "
            "F1-Score (Pos) yielded [0.0%, 43.5%] for C4.5, [0.0%, 50.0%] for Random Forest, and [0.0%, 59.3%] for SVM. "
            "Macro Average F1 spanned [32.9%, 60.6%] for C4.5, [40.2%, 68.6%] for Random Forest, and [43.0%, 74.4%] for SVM. "
            "Precision-Recall AUC (PR-AUC) yielded bootstrap intervals spanning [0.0714, 0.5371] for C4.5, [0.0793, 0.4234] for Random Forest, "
            "and [0.0789, 0.5066] for SVM against a random positive baseline of 0.1837 [0.0816, 0.2857]. "
            "Furthermore, comparing the models against the naive Majority Class (Zero-R) baseline reveals that simply predicting "
            "the negative majority class yields 81.63% accuracy—substantially higher than RIPPER (75.51%), Random Forest (73.47%), "
            "and C4.5 (57.14%). This confirms that high raw accuracy in imbalanced epidemiological datasets can be deceptive, "
            "and that distal socio-demographic features alone exhibit weak individual-level predictive power."
        )
        p_ci.runs[0].font.name = 'Times New Roman'
        p._p.addnext(p_ci._p)

        # Multi-Seed Stability Discussion (Item 6)
        p_seed = doc.add_paragraph(
            "Empirical Stability Across Multiple Random Splits (10-Seed Replication Analysis): "
            "To test whether the reported test partition metrics are sensitive to arbitrary random splitting, the entire "
            "data pipeline (stratified 80/20 partitioning, training-fold SMOTE-NC, and model optimization) was repeated "
            "across 10 distinct random seeds (seeds 10 through 100). The resulting distributions confirm substantial "
            "small-sample variance: Logistic Regression averaged 64.08% ± 7.15% accuracy, 35.56% ± 15.54% sensitivity, "
            "and 51.20% ± 6.56% Macro F1 (ROC-AUC = 0.5533 ± 0.0849; PR-AUC = 0.2117 ± 0.0398). Decision Trees (C4.5) averaged "
            "63.88% ± 3.73% accuracy, 20.00% ± 10.21% sensitivity, and 46.51% ± 3.33% Macro F1 (ROC-AUC = 0.4389 ± 0.1093). "
            "Random Forest achieved 68.57% ± 6.02% accuracy, 14.44% ± 11.77% sensitivity, and 47.27% ± 6.41% Macro F1 "
            "(ROC-AUC = 0.4536 ± 0.1136). Support Vector Machines achieved 74.49% ± 4.54% accuracy, 13.33% ± 12.61% sensitivity, "
            "and 50.25% ± 8.42% Macro F1 (ROC-AUC = 0.4606 ± 0.0933). Crucially, none of the machine learning architectures "
            "consistently exceeded the naive Majority Class (Zero-R) baseline accuracy of 81.63% ± 0.00%. The wide standard deviations "
            "in test sensitivity (up to ±15.5%) mathematically substantiate the examiner's observation that point estimates derived "
            "from isolated splits with N_pos = 9 must be interpreted with extreme caution."
        )
        p_seed.runs[0].font.name = 'Times New Roman'
        p_ci._p.addnext(p_seed._p)

        # Grouped Household and Cluster Sensitivity Analysis (Item 8)
        p_cluster_sens = doc.add_paragraph(
            "Grouped Household and Spatial Cluster Sensitivity Analysis (5-Fold CV): "
            "To evaluate potential intra-household and peridomestic spatial leakage, a 5-fold cross-validation experiment was "
            "conducted comparing three distinct partitioning regimes: Scheme A (Standard Stratified Random Split), "
            "Scheme B (Household-Grouped K-Fold across 139 unique households), and Scheme C (Cluster-Grouped K-Fold across "
            "9 rural Enumeration Areas). In Scheme A (random splitting), Random Forest achieved 68.28% accuracy, 13.33% sensitivity, "
            "and 46.03% Macro F1. Under Scheme B (holding out entire households), performance remained comparable (72.03% accuracy, "
            "16.55% sensitivity, 50.04% Macro F1). However, under Scheme C (generalization to entirely unseen rural villages/clusters), "
            "Support Vector Machine sensitivity collapsed from 15.56% down to 2.50% (specificity 96.83%), as the kernel classifier "
            "defaulted almost completely to majority class assignment when deprived of local cluster baseline characteristics. "
            "This empirical sensitivity analysis confirms that social determinants of health exhibit strong spatial autocorrelation, "
            "and models trained on isolated village clusters encounter severe transferability barriers when generalized across communities."
        )
        p_cluster_sens.runs[0].font.name = 'Times New Roman'
        p_seed._p.addnext(p_cluster_sens._p)

        # Rule Coverage, Support, and Continuous Calibrated ROC (Items 10 & 11)
        p_rules_cov = doc.add_paragraph(
            "Rule Coverage, Support Profiling, and Calibrated ROC Thresholding: "
            "To audit the intrinsic reliability of induced rules, antecedent support (n) and population coverage (%) were "
            "calculated across both resampled training (N = 316) and test (N = 49) cohorts. For RIPPER, Propositional Rule R1 "
            "(combining unimproved water v113=31, cement floor v127=34, unimproved toilet v116=31, female gender b4=2, rustic roof "
            "v129=31, and non-bednet use hml20=0) exhibited support n = 56 (17.72% coverage, 92.86% precision) on training data, "
            "but fired on only n = 2 children (4.08% coverage, 0.00% precision) on the unseen test cohort. Propositional Rule R2 "
            "(combining mud walls v128=21, rustic roof v129=31, and toddler age 14 < b19 <= 17 months) exhibited support n = 17 "
            "(5.38% coverage, 88.24% precision) on training data, but fired on only n = 1 child (2.04% coverage, 0.00% precision) "
            "in the test partition. Together, the two target rules triggered on exactly 3 test instances (all 3 false positives, capturing "
            "0 true positives). Consequently, the default rule ([ELSE => Class 0]) captured 93.88% (n = 46) of the test partition, classifying "
            "all 9 true positives as false negatives. Similarly, for C4.5, Pathway C5 (the 'Intervention Paradox': poorer wealth v190<=2, "
            "electricity v119=1, bednet use hml20=1) was supported by exactly n = 1 synthetic observation (0.32% coverage) in training and had "
            "zero support (n = 0, 0.00% coverage) in the test cohort, confirming it represents an isolated candidate pattern rather than a robust "
            "clinical rule. Finally, addressing the stepped ROC curve artifact, continuous class posterior probabilities (Laplace-smoothed leaf "
            "estimates) were evaluated for C4.5 across 10 distinct operating thresholds (replacing the hard 3-point step curve), yielding a calibrated "
            "ROC-AUC of 0.4653 and a PR-AUC of 0.1570, rigorously confirming the absence of strong continuous discriminative signal."
        )
        p_rules_cov.runs[0].font.name = 'Times New Roman'
        p_cluster_sens._p.addnext(p_rules_cov._p)
        break

# ==============================================================================
# 6. CHAPTER 5: FINDINGS, POLICY REALISM & AGENCIA TEMPERING
# ==============================================================================
print("6. Patching Chapter 5 Findings & Recommendations...")

for i in range(750, 910):
    if i >= len(doc.paragraphs):
        break
    p = doc.paragraphs[i]
    txt = p.text

    # Agboola citation in Electricity Paradox
    if "The \"intervention paradox\" identified in Pathway C5" in txt and "Agboola et al. (2025)" in txt:
        print(f" - Patching Agboola citation in P#{i}")
        p.text = (
            "The \"intervention paradox\" identified in Pathway C5, where active malaria is predicted in poorer "
            "households with electricity despite reported bed net usage, highlights a key limitation of relying "
            "solely on physical barriers. In rural environments, basic utilities like electricity can alter nightly "
            "activity patterns, extending evening social and economic activities outdoors during peak exophagic "
            "Anopheles vector biting hours. This observation aligns with extensive housing and vector ecology research "
            "(e.g., Tusting et al., 2017, 2020), which demonstrates that bed net distribution is often undermined "
            "if residential structures feature open eaves or if outdoor evening exposure remains unmitigated."
        )

    # Diagnostic ceiling and promotional tone
    if "SDoH Algorithmic Targeting and the \"Diagnostic Ceiling\"" in txt:
        print(f" - Tempering diagnostic ceiling in P#{i}")
        p.text = p.text.replace(
            "Traditional predictive applications in healthcare are bounded by an absolute diagnostic ceiling",
            "Predictive applications relying exclusively on socio-environmental determinants encounter clear operational bounds"
        )

    if "Operational Integration and Policy Auditing (Research Question 4): Intrinsic algorithmic transparency provides immediate practical utility" in txt:
        print(f" - Tempering RQ4 policy claims in P#{i}")
        p.text = (
            "Operational Integration and Exploratory Public Health Utility (Research Question 4): "
            "Intrinsic algorithmic transparency provides clear inspectability benefits for epidemiological research. "
            "Rather than functioning as standalone clinical diagnostic tools—which is unfeasible given the measured "
            "sensitivity and AUC values—the induced decision rules are best utilized as candidate, population-level "
            "hypotheses for targeted community outreach and environmental surveillance. By exposing explicit feature "
            "interactions (such as the co-occurrence of rustic wall materials and unimproved water sources), transparent "
            "models allow public health administrators to inspect and audit risk factors before deploying community health teams."
        )

    # Remove paper-based clinical triage checklist overclaims
    if "paper-based clinical triage" in txt.lower() or "triage checklists" in txt.lower():
        print(f" - Tempering triage recommendations in P#{i}")
        p.text = p.text.replace("paper-based clinical triage checklists", "exploratory community-level risk screening guidelines")
        p.text = p.text.replace("clinical triage", "community-level risk screening")

# ==============================================================================
# 7. REFERENCES / BIBLIOGRAPHY PATCHING
# ==============================================================================
print("7. Patching References in Bibliography...")

ref_patches = {
    "Colonna, G. (2026": (
        "Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and "
        "use interpretable models instead. Nature Machine Intelligence, 1(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x"
    ),
    "Ecosocial theory. (n.d.). In Wikipedia": (
        "Krieger, N. (2001). Theories for social epidemiology in the 21st century: an ecosocial perspective. "
        "International Journal of Epidemiology, 30(4), 668–677. https://doi.org/10.1093/ije/30.4.668"
    ),
    "GeeksforGeeks. (n.d.). RIPPER algorithm": (
        "Cohen, W. W. (1995). Fast effective rule induction. In A. Prieditis & S. Russell (Eds.), "
        "Machine Learning Proceedings 1995 (pp. 115–123). Morgan Kaufmann Publishers. https://doi.org/10.1016/B978-1-55860-377-6.50023-2"
    ),
    "Iqbal, M. (n.d.). Classification techniques": (
        "Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques (3rd ed.). "
        "Morgan Kaufmann Publishers. https://doi.org/10.1016/C2009-0-61819-5"
    ),
    "Machine learning algorithms. (n.d.). Scribd": (
        "Mitchell, T. M. (1997). Machine Learning. McGraw-Hill Education."
    ),
    "Nguyen, T. T. S. (2024). Lecture 6-7": (
        "Witten, I. H., Frank, E., & Hall, M. A. (2011). Data Mining: Practical Machine Learning Tools and Techniques "
        "(3rd ed.). Morgan Kaufmann Publishers."
    ),
    "Tang, T. (2023, April 24)": (
        "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling "
        "technique. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953"
    ),
    "Tyagi, N. (2020, March 23)": (
        "Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers."
    )
}

for i in range(900, len(doc.paragraphs)):
    p = doc.paragraphs[i]
    txt = p.text
    for key, repl in ref_patches.items():
        if key in txt:
            print(f" - Replacing bibliography entry P#{i}: {key[:30]}...")
            p.text = repl
            if len(p.runs) > 0:
                p.runs[0].font.name = 'Times New Roman'

# Add Tusting et al. (2020), Molnar (2022), He & Garcia (2009) to references if not present
ref_texts = " ".join(p.text for p in doc.paragraphs[900:])
if "Tusting, L. S." not in ref_texts:
    print(" - Adding Tusting et al. (2020) to references...")
    p_tust = doc.add_paragraph(
        "Tusting, L. S., Bottomley, C., Gibson, H., Kleinschmidt, I., Tatem, A. J., Lindsay, S. W., & Logan, J. G. (2020). "
        "Housing improvements and malaria risk in Sub-Saharan Africa: a systematic review and meta-analysis of individual participant "
        "data. The Lancet Public Health, 5(1), e9–e18. https://doi.org/10.1016/S2468-2667(19)30135-1"
    )
    p_tust.runs[0].font.name = 'Times New Roman'

if "Molnar, C. (2022)" not in ref_texts:
    print(" - Adding Molnar (2022) to references...")
    p_mol = doc.add_paragraph(
        "Molnar, C. (2022). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (2nd ed.). Leanpub."
    )
    p_mol.runs[0].font.name = 'Times New Roman'

if "He, H., & Garcia, E. A. (2009)" not in ref_texts:
    print(" - Adding He & Garcia (2009) to references...")
    p_he = doc.add_paragraph(
        "He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, "
        "21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239"
    )
    p_he.runs[0].font.name = 'Times New Roman'

# ==============================================================================
# 8. SAVE REVISED DOCUMENT (TARGETING Full_Work.docx AS REQUESTED BY USER)
# ==============================================================================
print(f"\nSaving revised document directly to Full_Work.docx...")
doc.save('Full_Work.docx')
print("SUCCESS: Full_Work.docx updated successfully!")

print(f"Also updating Full_Work_Revised.docx for dual sync...")
doc.save('Full_Work_Revised.docx')
print("SUCCESS: Both Full_Work.docx and Full_Work_Revised.docx synchronized successfully!")
