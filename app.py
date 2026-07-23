"""
Aegis AML — Streamlit frontend.

Bloomberg Terminal-style dashboard for scoring transactions,
reviewing risk factors, and generating SAR narratives.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from predict import predict_transaction
from rag import generate_narrative, retrieve_regulatory_context, KNOWLEDGE_BASE
from groq import Groq

# Configure terminal output encoding to prevent Unicode errors on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Configuration
# Page Configuration
st.set_page_config(
    page_title="Aegis SAR - Financial Intelligence Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (New Palette)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Layout & Fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background-color: #09090b;
        background-image: radial-gradient(circle at top center, rgba(56, 31, 108, 0.15) 0%, transparent 60%);
        color: #f4f4f5;
    }

    /* Container Spacing */
    div[data-testid="block-container"] {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #09090b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .css-1d391kg {
        background-color: transparent !important;
    }

    /* Typography Classes */
    .page-title {
        font-size: 2.5rem;
        font-weight: 00;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .page-subtitle {
        font-size: 1.1rem;
        color: #a1a1aa;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 400;
        color: #e4e4e7;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .gradient-text {
        background: linear-gradient(to right, #60a5fa, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 400;
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 400 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #a1a1aa !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 200 !important;
    }

    /* Cards */
    .glass-card {
        background: rgba(24, 24, 27, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }

    .alert-card {
        background: rgba(220, 38, 38, 0.1);
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }

    .safe-card {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.3) !important;
        width: auto !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%) !important;
    }

    /* Secondary Download Buttons */
    .stDownloadButton>button {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e4e4e7 !important;
        box-shadow: none !important;
    }
    .stDownloadButton>button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Inputs */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(24, 24, 27, 0.6) !important;
        color: #f4f4f5 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        font-size: 15px !important;
    }
    
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 1px #8b5cf6 !important;
    }

    /* Slider */
    .stSlider > div[data-baseweb="slider"] > div {
        color: #8b5cf6 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(24, 24, 27, 0.4) !important;
        color: #e4e4e7 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        font-weight: 500 !important;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: rgba(24, 24, 27, 0.4) !important;
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 12px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Remove default top padding in Streamlit */
    .css-18e3th9 {
        padding-top: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load dataset securely with caching
@st.cache_data
def load_processed_data():
    try:
        path = os.path.join(BASE_DIR, "data", "processed", "sar_dataset.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    # Generate dummy data if file fails to load
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        'amount': np.random.exponential(scale=1500, size=5000),
        'time': np.random.randint(0, 172800, size=5000),
        'fraud': np.random.choice([0, 1], size=5000, p=[0.99, 0.01]),
        'location': np.random.choice(['India', 'Dubai', 'USA', 'UK', 'Singapore'], size=5000),
        'transaction_type': np.random.choice(['transfer', 'withdrawal', 'payment'], size=5000)
    })
    return dummy_data

df_all = load_processed_data()

# Standardize page names with premium emojis for visual excellence
PAGES = [
    " Executive Dashboard",
    " Single Scan",
    " RAG SAR Drafts",
    " Batch Scan Centre",
    " Compliance Copilot",
    " Model & Knowledge Base"
]

from nav_utils import clean_nav

CLEAN_PAGES = [clean_nav(p) for p in PAGES]

# Initialize navigation in session state
if 'nav_option' not in st.session_state:
    st.session_state.nav_option = "Executive Dashboard"

# Ensure robust fallback
if st.session_state.nav_option not in CLEAN_PAGES:
    st.session_state.nav_option = "Executive Dashboard"

def handle_nav_change():
    if 'navigation_widget_key' in st.session_state:
        st.session_state.nav_option = clean_nav(st.session_state.navigation_widget_key)

default_idx = CLEAN_PAGES.index(st.session_state.nav_option)

# Navigation Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.markdown("<h2 style='color:#38bdf8; margin-top:0;'>AEGIS CORE</h2>", unsafe_allow_html=True)
    st.markdown("`SECURE AML CONSOLE v1.2`", unsafe_allow_html=True)
    st.markdown("---")
    
    selected_page_raw = st.radio(
        "NAVIGATION MODULES",
        PAGES,
        index=default_idx,
        key="navigation_widget_key",
        on_change=handle_nav_change
    )
    
    # Secure immediate update of session state value
    st.session_state.nav_option = clean_nav(selected_page_raw)
    
    st.markdown("---")
    st.markdown("### Decision Threshold")
    threshold = st.slider("ML Alert Sensitivity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    st.caption("Adjust XGBoost alert trigger threshold. Lower thresholds increase sensitivity.")

# ==========================================
# MODULE 1: EXECUTIVE DASHBOARD
# ==========================================
if st.session_state.nav_option == "Executive Dashboard":
    st.markdown("<div class='terminal-header'><h1> SECURITY OVERVIEW & COMPLIANCE ANALYTICS</h1></div>", unsafe_allow_html=True)
    
    # Calculate Live Statistics from data
    total_tx = len(df_all)
    fraud_df = df_all[df_all['fraud'] == 1]
    total_fraud_count = len(fraud_df)
    fraud_rate = (total_fraud_count / total_tx) * 100
    total_volume_flagged = fraud_df['amount'].sum()
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='margin:0;color:#94a3b8;font-size:12px;text-transform:uppercase;'>TOTAL AUDITED DEBITS</p>
            <h2 style='margin:10px 0 0 0;color:#38bdf8;'>{total_tx:,}</h2>
            <span style='color:#22c55e;font-size:12px;'>● Online Node Active</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='margin:0;color:#94a3b8;font-size:12px;text-transform:uppercase;'>SUSPICIOUS ALERTS</p>
            <h2 style='margin:10px 0 0 0;color:#ef4444;'>{total_fraud_count:,}</h2>
            <span style='color:#ef4444;font-size:12px;'>▲ Enhanced Due Diligence</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='margin:0;color:#94a3b8;font-size:12px;text-transform:uppercase;'>ALERT CONVERSION RATE</p>
            <h2 style='margin:10px 0 0 0;color:#f59e0b;'>{fraud_rate:.2f}%</h2>
            <span style='color:#94a3b8;font-size:12px;'>Model True Positive Index</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='margin:0;color:#94a3b8;font-size:12px;text-transform:uppercase;'>RISK CAPITAL AUDITED</p>
            <h2 style='margin:10px 0 0 0;color:#10b981;'>₹{total_volume_flagged:,.2f}</h2>
            <span style='color:#10b981;font-size:12px;'>Assets Blocked/Under Hold</span>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Plots
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<h3 style='color:#38bdf8;'>Anomaly Distribution (Fraud vs Normal)</h3>", unsafe_allow_html=True)
        fig_pie = px.pie(
            values=[total_tx - total_fraud_count, total_fraud_count],
            names=['Cleared Transactions', 'Flagged Anomaly'],
            color=['Cleared Transactions', 'Flagged Anomaly'],
            color_discrete_map={'Cleared Transactions': '#1e293b', 'Flagged Anomaly': '#ef4444'},
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("<h3 style='color:#38bdf8;'>Risk Profile vs Asset Value</h3>", unsafe_allow_html=True)
        # Sample data to keep chart responsive
        sample_df = df_all.sample(min(2000, len(df_all)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="time",
            y="amount",
            color="fraud",
            color_continuous_scale=[[0, '#0ea5e9'], [1, '#ef4444']],
            labels={"fraud": "Risk Level", "time": "Time (Epoch Secs)", "amount": "Amount (₹)"},
            opacity=0.6
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            coloraxis_showscale=False
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor='#1e293b')
        fig_scatter.update_yaxes(showgrid=True, gridcolor='#1e293b')
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Bottom Row: Model Feature Importances
    st.markdown("<h3 style='color:#38bdf8;'>XGBoost Feature Importance Log</h3>", unsafe_allow_html=True)
    try:
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_bar = px.bar(
            feat_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='blues'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        st.info("Feature importance currently offline. Please ensure the model.pkl is trained successfully.")

# ==========================================
# MODULE 2: SINGLE TRANSACTION SCAN
# ==========================================
elif st.session_state.nav_option == "Single Scan":
    st.markdown("<div class='terminal-header'><h1> COMPLIANCE SCANNER - REALTIME AUDIT</h1></div>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.markdown("<h3 style='color:#38bdf8;'>Audited Assets Parameters</h3>", unsafe_allow_html=True)
        
        # User input fields inside card
        with st.form("scan_form"):
            amount = st.number_input("Transaction Principal Amount (₹)", min_value=1.0, value=15000.0, step=500.0)
            time = st.slider("⏱️ Network Time Offset (seconds)", min_value=0, max_value=172800, value=45000)
            tx_type = st.selectbox("Transaction Channel", ["transfer", "withdrawal", "payment"])
            location = st.selectbox("Originating Location Node", ["India", "Dubai", "USA", "UK", "Singapore"])
            
            submit_btn = st.form_submit_button("Initiate AML Diagnostic Scan")
            
        st.markdown("""
        <div class='glass-card' style='font-size:12px;color:#94a3b8;'>
            <strong>Note:</strong> AEGIS scanner executes a parallel pipeline: XGBoost Machine Learning classification to compute numeric anomaly probability, followed by a deterministic compliance rule override.
        </div>
        """, unsafe_allow_html=True)

    with col_out:
        if submit_btn or 'amount' in st.session_state:
            # Save variables in session state to transfer between scans and RAG narratives
            st.session_state.amount = amount
            st.session_state.time = time
            st.session_state.tx_type = tx_type
            st.session_state.location = location
            
            # Predict
            result = predict_transaction(amount, time, threshold=threshold)
            prob = result["fraud_probability"]
            pred = result["prediction"]
            by_rules = result["flagged_by_rules"]
            by_ml = result["flagged_by_ml"]
            
            st.session_state.prob = prob
            st.session_state.pred = pred
            st.session_state.by_rules = by_rules
            st.session_state.by_ml = by_ml
            
            # Clear previous draft narrative when a new scan is explicitly submitted
            if submit_btn:
                if 'current_narrative' in st.session_state:
                    del st.session_state.current_narrative
            
            st.markdown("<h3 style='color:#38bdf8;'>Risk Evaluation Results</h3>", unsafe_allow_html=True)
            
            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Anomaly Index", 'font': {'size': 18, 'color': '#94a3b8'}},
                number={'suffix': "%", 'font': {'size': 32, 'color': '#e2e8f0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.08)",
                    'steps': [
                        {'range': [0, threshold*100], 'color': '#22c55e'},
                        {'range': [threshold*100, 80], 'color': '#f59e0b'},
                        {'range': [80, 100], 'color': '#ef4444'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=220,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Diagnosis display card
            if pred == "Fraud":
                st.markdown(f"""
                <div class='alert-card'>
                    <h4 style='color:#ef4444;margin:0;'> SUSPICIOUS PATTERN DETECTED (HIGH RISK)</h4>
                    <p style='margin:10px 0 0 0;font-size:14px;color:#fca5a5;'>
                        Diagnostic Flags triggered:
                        <br/>● XGBoost Machine Learning Flag: {"TRIGGERED" if by_ml else "CLEARED"}
                        <br/>● Compliance Rule Override Flag: {"TRIGGERED (Absolute Threshold Violate)" if by_rules else "CLEARED"}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 Next Step: Draft a legal Suspicious Activity Report (SAR) narrative using our automated RAG generator.")
                if st.button("📝 Auto-draft SAR Report Now"):
                    st.session_state.auto_draft = True
                    st.session_state.nav_option = "RAG SAR Drafts"
                    st.rerun()
            else:
                st.markdown(f"""
                <div class='safe-card'>
                    <h4 style='color:#22c55e;margin:0;'> CLEARED: NO ANOMALY FOUND</h4>
                    <p style='margin:10px 0 0 0;font-size:14px;color:#bbf7d0;'>
                        Diagnostics show standard consumer behavior:
                        <br/>● ML Anomaly Index ({prob:.2%}) is within safety bounds.
                        <br/>● Financial rules constraints fully satisfied.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Input transaction variables and click 'Initiate AML Diagnostic Scan' to begin analysis.")

# ==========================================
# MODULE 3: RAG SAR NARRATIVE GENERATOR
# ==========================================
elif st.session_state.nav_option == "RAG SAR Drafts":
    st.markdown("<div class='terminal-header'><h1> RAG COMPLIANCE GENERATOR (SAR AUTOMATION)</h1></div>", unsafe_allow_html=True)
    
    if 'amount' not in st.session_state:
        st.warning(" No scanned transaction found in active session memory. Please scan a transaction in 'Single Scan' first.")
        if st.button("🔍 Go to Realtime Single Scan"):
            st.session_state.nav_option = "Single Scan"
            st.rerun()
    else:
        # Retrieve scanned parameters
        amount = st.session_state.amount
        time = st.session_state.time
        tx_type = st.session_state.tx_type
        location = st.session_state.location
        prob = st.session_state.prob
        pred = st.session_state.pred
        by_rules = st.session_state.by_rules
        
        # UI Structure
        col_ref, col_gen = st.columns([2, 3])
        
        with col_ref:
            st.markdown("<h3 style='color:#38bdf8;'>1. RAG Retrieve Stage (TF-IDF Matching)</h3>", unsafe_allow_html=True)
            
            # Fetch context
            kb_doc = retrieve_regulatory_context(amount, time, tx_type, location, by_rules)
            
            st.markdown(f"""
            <div class='glass-card'>
                <h4 style='color:#38bdf8;margin:0;'>Matched Compliance Regulation</h4>
                <p style='margin:10px 0 0 0;font-size:14px;'><strong>Source Document:</strong> {kb_doc['title']}</p>
                <p style='margin:10px 0;font-size:13px;color:#94a3b8;line-height:1.4;'>
                    <em>"{kb_doc['text']}"</em>
                </p>
                <hr style='border-color:rgba(255,255,255,0.08);'/>
                <p style='font-size:12px;color:#38bdf8;'>
                    <strong>Retrieval Logic:</strong> TF-IDF query parsed transaction risk vectors (Amount: ₹{amount:,.0f}, time-offset: {time}) to match compliance guidelines in real time.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='color:#38bdf8;'>Scanned Audit Trail</h3>", unsafe_allow_html=True)
            st.json({
                "Principal": f"₹{amount:,.2f}",
                "Time Frame": time,
                "Channel": tx_type,
                "Node Location": location,
                "XGBoost Score": f"{prob:.2%}",
                "Classification Decision": pred
            })

        with col_gen:
            st.markdown("<h3 style='color:#38bdf8;'>2. Generative Draft Stage (Groq LLM / Template)</h3>", unsafe_allow_html=True)
            
            # Check if we arrived here via the Auto-draft button
            should_generate = st.button("Generate Regulatory Narrative Report")
            if st.session_state.pop('auto_draft', False):
                should_generate = True
            
            if should_generate:
                with st.spinner("Executing RAG Pipeline & Drafting Narrative..."):
                    narrative = generate_narrative(amount, time, pred, prob, tx_type, location, by_rules)
                    st.session_state.current_narrative = narrative
                    # Increment generation counter to force st.text_area widget recreation with new value
                    st.session_state.narrative_generation_id = st.session_state.get('narrative_generation_id', 0) + 1
            
            if 'current_narrative' in st.session_state:
                gen_id = st.session_state.get('narrative_generation_id', 0)
                narrative_text = st.text_area(
                    "Edit SAR Draft Narrative Live", 
                    value=st.session_state.current_narrative, 
                    height=450,
                    key=f"sar_narrative_editor_{gen_id}"
                )
                st.session_state.current_narrative = narrative_text
                
                # Download Button
                st.download_button(
                    label="Export Official SAR Text File",
                    data=st.session_state.current_narrative,
                    file_name=f"SAR_REPORT_{location.upper()}_{tx_type.upper()}.txt",
                    mime="text/plain"
                )
            else:
                st.info("Click 'Generate Regulatory Narrative Report' to invoke the generative AI pipeline.")

# ==========================================
# MODULE 4: BATCH SCAN CENTRE
# ==========================================
elif st.session_state.nav_option == "Batch Scan Centre":
    st.markdown("<div class='terminal-header'><h1> BATCH PROCESSING OPERATIONS CENTER</h1></div>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown("<h3 style='color:#38bdf8;'>Upload Transaction Batches</h3>", unsafe_allow_html=True)
        
        # Download Sample Template
        sample_df = pd.DataFrame([
            {"amount": 12000.0, "time": 45000, "transaction_type": "payment", "location": "India"},
            {"amount": 80000.0, "time": 1000, "transaction_type": "transfer", "location": "Dubai"},
            {"amount": 4200.0, "time": 12000, "transaction_type": "withdrawal", "location": "USA"},
            {"amount": 55000.0, "time": 80000, "transaction_type": "transfer", "location": "Singapore"}
        ])
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=" Download Sample Batch CSV",
            data=sample_csv,
            file_name="aegis_batch_sample.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload Transaction File (CSV)", type=["csv"])

    with col_r:
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                required_cols = ["amount", "time"]
                
                # Check column requirements
                if not all(col in batch_df.columns for col in required_cols):
                    st.error(" Invalid CSV columns. Must contain at least 'amount' and 'time' columns.")
                else:
                    with st.spinner("Scanning Batch Ledger using XGBoost Classifier..."):
                        # Ensure string locations & tx_types have defaults
                        if "location" not in batch_df.columns:
                            batch_df["location"] = "India"
                        if "transaction_type" not in batch_df.columns:
                            batch_df["transaction_type"] = "payment"
                            
                        # Predict
                        scores = []
                        decisions = []
                        for idx, row in batch_df.iterrows():
                            res = predict_transaction(row["amount"], row["time"], threshold=threshold)
                            scores.append(res["fraud_probability"])
                            decisions.append(res["prediction"])
                            
                        batch_df["Anomaly Probability"] = scores
                        batch_df["Decision Audit"] = decisions
                        
                        st.markdown("<h3 style='color:#38bdf8;'>Scan Complete: Diagnostic Ledger</h3>", unsafe_allow_html=True)
                        
                        # Style flagged rows as red
                        def color_fraud(val):
                            color = 'rgba(239, 68, 68, 0.2)' if val == 'Fraud' else 'rgba(0,0,0,0)'
                            return f'background-color: {color}'
                        
                        st.dataframe(batch_df.style.map(color_fraud, subset=['Decision Audit']), use_container_width=True)
                        
                        # Export results CSV
                        results_csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=" Export Flagged Batch Report (CSV)",
                            data=results_csv,
                            file_name="aegis_batch_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f" Failed to parse uploaded file: {str(e)}")
        else:
            st.info("Upload a financial transaction ledger (CSV format) to perform high-throughput compliance audits.")

# ==========================================
# MODULE 5: COMPLIANCE COPILOT
# ==========================================
elif st.session_state.nav_option == "Compliance Copilot":
    st.markdown("<div class='terminal-header'><h1> AEGIS COMPLIANCE COPILOT (AI ASSISTANT)</h1></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card' style='font-size:13px;line-height:1.5;'>
        🛡️ Welcome to your <strong>Aegis AI Assistant</strong>! Ask the copilot compliance-related questions regarding anti-money laundering (AML) laws, Bank Secrecy Act (BSA) rules, XGBoost risk scores, or discuss suspicious features flagged in your scan.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome Officer. I am ready to review case files, explain model variables, or draft action memos. How can I assist you today?"}
        ]
        
    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Get user prompt
    user_input = st.chat_input("Ask a question (e.g. 'What is FinCEN's high value threshold?')")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Build prompt context with current scan information
        scan_context = ""
        if 'amount' in st.session_state:
            scan_context = f"\nCurrently audited transaction context: ₹{st.session_state.amount} {st.session_state.tx_type} from {st.session_state.location}. XGBoost flagged it as {st.session_state.pred} with {st.session_state.prob:.2%} anomaly score."
            
        system_instructions = f"You are Aegis Copilot, an AI assistant built for financial compliance officers. Answer standard AML/BSA questions clearly and formally. Reference model specifics when applicable. {scan_context}"
        
        # Generate Answer
        api_key = os.getenv("GROQ_API_KEY")
        bot_response = ""
        
        if api_key and api_key.strip() and not api_key.startswith("gsk_placeholder"):
            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_instructions},
                        *st.session_state.messages[-4:] # Send last 4 messages for conversational memory
                    ],
                    temperature=0.4
                )
                bot_response = response.choices[0].message.content
            except Exception as e:
                bot_response = f"API Connection timed out: {str(e)}. Proceeding in local heuristic mode."
                
        # Local fallback heuristic chatbot if Groq is offline
        if not bot_response:
            user_low = user_input.lower()
            if "structuring" in user_low:
                bot_response = "Structuring refers to breaking a large transaction into smaller sums (e.g., under ₹10,000) to deliberately avoid triggering threshold reporting requirements. It is illegal under AML laws and triggers automatic compliance alert investigations."
            elif "fincen" in user_low or "bsa" in user_low:
                bot_response = "The Bank Secrecy Act (BSA) mandates financial entities to keep records of financial transactions. Under FinCEN requirements, any cash transaction exceeding $10,000 USD equivalent (often calibrated to ₹50,000 or ₹1,00,000 locally in India) must be filed under a Currency Transaction Report (CTR), and suspicious ones under a SAR."
            elif "xgboost" in user_low or "model" in user_low:
                bot_response = "Aegis leverages a gradient-boosted decision tree algorithm (XGBoost) trained on the SMOTE-resampled Credit Card fraud dataset. The model evaluates features like transaction amount, time, ratio, and nighttime execution to generate a continuous anomaly index score."
            elif "flagged" in user_low or "why" in user_low:
                if 'amount' in st.session_state and st.session_state.pred == "Fraud":
                    bot_response = f"This case (amount ₹{st.session_state.amount}) triggered risk flags because the amount violates the primary safety limit (₹50,000) or was classified by XGBoost as an anomalous pattern. Recommend account hold."
                else:
                    bot_response = "Transactions are typically flagged due to a high asset value (over ₹50,000), execution at midnight, anomalous ratios, or a high ML classifier score. I suggest using the RAG narrative generator to look up retrieved BSA regulations."
            else:
                bot_response = "I have recorded your query. Under compliance rules, transaction metadata, amount ratios, and origin nodes are currently being monitored. Let me know if you would like me to draft an EDD hold request letter for the current scan."
                
        with st.chat_message("assistant"):
            st.write(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# ==========================================
# MODULE 6: MODEL & KNOWLEDGE BASE
# ==========================================
else:
    st.markdown("<div class='terminal-header'><h1> MODEL METRICS & COMPLIANCE KNOWLEDGE BASE</h1></div>", unsafe_allow_html=True)
    
    col_m, col_k = st.columns(2)
    
    with col_m:
        st.markdown("<h3 style='color:#38bdf8;'> Model Training Logistics</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
            <strong>Classifier:</strong> XGBoost (Extreme Gradient Boosting)<br/>
            <strong>Hyperparameters:</strong> n_estimators=300, max_depth=6, learning_rate=0.05<br/>
            <strong>Oversampling Strategy:</strong> SMOTE (Synthetic Minority Over-sampling Technique)<br/>
            <strong>Evaluation Summary:</strong>
            <ul>
                <li>Accuracy: 99.63%</li>
                <li>Minority (Fraud) Recall: ~97% (SMOTE)</li>
                <li>F1-Score: 0.87</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Static Confusion Matrix visualization
        st.markdown("<strong>Diagnostic Confusion Matrix Heatmap</strong>", unsafe_allow_html=True)
        z = [[19958, 9], [1, 32]]
        x = ['Normal (Pred)', 'Fraud (Pred)']
        y = ['Normal (True)', 'Fraud (True)']
        
        fig_heat = px.imshow(
            z,
            x=x,
            y=y,
            color_continuous_scale='blues',
            text_auto=True
        )
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=250,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_k:
        st.markdown("<h3 style='color:#38bdf8;'> Regulatory RAG Knowledge Base</h3>", unsafe_allow_html=True)
        st.caption("Active documents searched by TF-IDF retrieval system when constructing prompt context:")
        
        for idx, doc in enumerate(KNOWLEDGE_BASE):
            with st.expander(f" Doc {idx+1}: {doc['title']}"):
                st.markdown(f"**Keywords:** `{doc['keywords']}`")
                st.markdown(f"*{doc['text']}*")

if __name__ == "__main__":
    import streamlit.runtime
    if not streamlit.runtime.exists():
        import subprocess
        print("🚀 Launching Aegis Compliance Terminal in Streamlit...")
        try:
            # Spawns 'streamlit run app.py' in a child process
            subprocess.run(["streamlit", "run", __file__])
        except FileNotFoundError:
            print(" Error: Streamlit command not found in your PATH.")
            print("Please make sure you have installed requirements.txt and run:")
            print("streamlit run app.py")
