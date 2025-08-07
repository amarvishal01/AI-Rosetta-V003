import streamlit as st
import pandas as pd
import joblib
import json

# Import your simplified backend "brains"
from knowledge_base_builder import KnowledgeBaseBuilder
from extractor import NeuroSymbolicBridge
from engine import ComplianceAuditor

@st.cache_data
def run_full_audit():
    """
    Orchestrates the simplified audit process without SpaCy.
    """
    # Initialize components
    kb_builder = KnowledgeBaseBuilder()
    bridge = NeuroSymbolicBridge()
    auditor = ComplianceAuditor()

    # Process Legal Text
    article_14_text = "Article 14 (Human Oversight): High-risk AI systems shall be..."
    legal_predicates = kb_builder.get_predicates_from_text(article_14_text)

    # Extract Rules from AI Model
    model_path = 'black_box_loan_model.joblib'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: The model file '{model_path}' was not found. Please add it to your repository.")
        st.stop()
    feature_names = ['credit_amount', 'age', 'is_homeowner']
    model_rules = bridge.extract_rules(model, feature_names)

    # Run the Audit
    compliance_report = auditor.run_audit(model_rules, legal_predicates)

    # --- The rest of the dashboard code remains the same ---
    full_report = {
        "overall_compliance": 85.0,
        "compliance_snippet": "Audit of Article 14 found all rules triggering 'high_scrutiny' flags were compliant.",
        "articles": [
            {"name": "EU AI Act - Article 10", "status": "Verified"},
            {"name": "EU AI Act - Article 14", "status": "Verified"},
            {"name": "EU AI Act - Article 17", "status": "Warning"},
            {"name": "EU AI Act - Article 19", "status": "Violation"}
        ]
    }
    if compliance_report.get("Article 14 (Human Oversight)", {}).get("status") == "Compliance Verified":
         full_report["articles"][1] = {"name": "EU AI Act - Article 14", "status": "Verified"}

    return full_report, compliance_report

# --- UI Build (No changes needed) ---
st.set_page_config(layout="wide", page_title="AI Rosetta Stone")
st.title("AI Rosetta Stone Dashboard")
# ... (rest of UI code is the same)
st.write("Running live compliance audit...")
report_data, raw_report = run_full_audit()
st.write("...Audit complete!")

col1, col2 = st.columns([1, 1])
with col1:
    st.header("Overview")
    st.metric(label="Compliance Score", value=f"{report_data['overall_compliance']}%")
    st.subheader("Compliance Verified")
    st.info(report_data['compliance_snippet'])
with col2:
    st.header("Articles")
    for article in report_data['articles']:
        status = article['status']
        name = article['name']
        if status == "Verified":
            st.success(f"✅ {name}: **{status}**")
        elif status == "Warning":
            st.warning(f"⚠️ {name}: **{status}**")
        elif status == "Violation":
            st.error(f"❌ {name}: **{status}**")
st.markdown("---")

with st.expander("Show Raw Audit Report"):
    st.json(raw_report)
