"""
UniPath Korea — The complete AI-powered guide for international students in South Korea.
Single-file Streamlit application with multilingual support (9 languages),
Agentic RAG chatbot, Supabase data layer, email notifications, auth and admin tools.

Run:  streamlit run app.py
"""

# ════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ════════════════════════════════════════════════════════════════════════════
import io
import json
import re
from datetime import datetime, date

import pandas as pd
import streamlit as st

# Plotly is used for the statistics dashboards. Imported defensively so the app
# still boots in minimal environments.
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_READY = True
except Exception:
    PLOTLY_READY = False

# Supabase client (data layer + vector store backend).
try:
    from supabase import create_client
    SUPABASE_LIB = True
except Exception:
    SUPABASE_LIB = False

# LlamaIndex + Google GenAI (RAG + LLM). Imported lazily-guarded so the UI renders
# even when AI dependencies or secrets are missing.
try:
    from llama_index.llms.google_genai import GoogleGenAI as Gemini
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.vector_stores.supabase import SupabaseVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    LLAMA_LIB = True
except Exception:
    LLAMA_LIB = False


# ════════════════════════════════════════════════════════════════════════════
# 2. PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="UniPath Korea — AI Guide for International Students",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",   # was "collapsed",
)


# ════════════════════════════════════════════════════════════════════════════
# 3. PREMIUM DESIGN SYSTEM (CSS)
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Noto+Sans+KR:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --navy: #0D3B8E;
    --navy-2: #1a56c4;
    --emerald: #00C897;
    --orange: #FF6B35;
    --light: #F0F4FF;
    --ink: #1E293B;
    --muted: #64748B;
    --line: #E2E8F0;
}

/* ---------- Base ---------- */
html, body, [class*="css"], .stApp, .main {
    font-family: 'Outfit', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--ink);
}
.stApp {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7F9FF 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 4rem;
    max-width: 1280px;
}
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}
h1, h2, h3, h4, h5 { color: var(--ink); font-weight: 700; letter-spacing: -0.01em; }
a { color: var(--navy); text-decoration: none; }
a:hover { color: var(--emerald); }

/* ---------- Inputs ---------- */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input,
.stPasswordInput input,
input[type="text"], input[type="password"], input[type="email"], input[type="number"] {
    background: #FFFFFF !important;
    border: 1.5px solid var(--navy) !important;
    color: var(--ink) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    transition: all .2s ease;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
input:focus {
    border-color: var(--emerald) !important;
    box-shadow: 0 0 0 3px rgba(0,200,151,0.15) !important;
    outline: none !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder { color: #94A3B8 !important; }

/* ---------- Selectbox / Multiselect ---------- */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--navy) !important;
    border-radius: 12px !important;
}
div[data-baseweb="select"] span { color: var(--ink) !important; }

/* Dropdown popovers / option lists */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
ul[role="listbox"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 32px rgba(13,59,142,0.16) !important;
}
li[role="option"], div[role="option"] {
    background: #FFFFFF !important;
    color: var(--ink) !important;
}
li[role="option"]:hover, div[role="option"]:hover,
li[aria-selected="true"], div[aria-selected="true"] {
    background: #EEF2FF !important;
    color: var(--navy) !important;
}

/* Multiselect chips */
.stMultiSelect span[data-baseweb="tag"] {
    background: var(--navy) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    transition: all .25s ease;
    box-shadow: 0 6px 16px rgba(13,59,142,0.20);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(13,59,142,0.30);
    color: #FFFFFF;
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus { color: #FFFFFF !important; box-shadow: 0 0 0 3px rgba(0,200,151,0.25) !important; }

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--emerald) 0%, #00a87d 100%);
    color: #FFFFFF; border: none; border-radius: 12px; font-weight: 600;
}

/* ---------- Tabs (pill style) ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: none;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background: #FFFFFF;
    border: 1.5px solid var(--line);
    border-radius: 999px;
    padding: 8px 18px;
    color: var(--muted);
    font-weight: 600;
    transition: all .2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    border-color: var(--navy);
    color: var(--navy);
}
.stTabs [aria-selected="true"] {
    background: var(--navy) !important;
    color: #FFFFFF !important;
    border-color: var(--navy) !important;
    box-shadow: 0 6px 16px rgba(13,59,142,0.25);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Expanders ---------- */
.stExpander, details[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1.5px solid var(--navy) !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 20px rgba(13,59,142,0.06) !important;
    overflow: hidden;
}
.streamlit-expanderHeader, summary {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    border-radius: 16px !important;
}
summary:hover { color: var(--navy) !important; }

/* ---------- Sidebar (chat panel) ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, #0a2d6e 100%) !important;
    width: 400px !important;
}
section[data-testid="stSidebar"] * { color: #FFFFFF; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255,255,255,0.95) !important;
    color: var(--ink) !important;
    border: 1.5px solid rgba(255,255,255,0.4) !important;
}
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.95) !important;
    border: 1.5px solid rgba(255,255,255,0.4) !important;
}
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span { color: var(--ink) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--emerald) 0%, #00a87d 100%);
    width: 100%;
}
section[data-testid="stSidebar"] [data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 4px 8px;
}
/* Make the sidebar toggle button highly visible */
button[kind="header"], [data-testid="collapsedControl"] {
    background: var(--navy) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(13,59,142,0.35) !important;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #0D3B8E 0%, #1a56c4 50%, #00C897 100%);
    border-radius: 28px;
    padding: 64px 48px 92px 48px;
    color: #FFFFFF;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(13,59,142,0.25);
}
.hero::after {
    content: "";
    position: absolute;
    right: -80px; top: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.3);
    color: #FFFFFF;
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 22px;
}
.hero h1 {
    color: #FFFFFF !important;
    font-size: 52px;
    line-height: 1.08;
    font-weight: 900;
    margin: 0 0 16px 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: rgba(255,255,255,0.92);
    font-size: 19px;
    max-width: 680px;
    line-height: 1.6;
    margin: 0;
}

/* ---------- KPI cards ---------- */
.kpi-wrap {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-top: -60px;
    position: relative;
    z-index: 10;
    padding: 0 12px;
}
.kpi-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 24px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border-bottom: 4px solid var(--emerald);
    text-align: center;
    transition: transform .25s ease, box-shadow .25s ease;
}
.kpi-card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(13,59,142,0.14); }
.kpi-icon { font-size: 28px; margin-bottom: 6px; }
.kpi-value { font-size: 34px; font-weight: 900; color: var(--navy); line-height: 1; }
.kpi-label { font-size: 13px; color: var(--muted); font-weight: 600; margin-top: 8px; text-transform: uppercase; letter-spacing: .04em; }

/* ---------- Generic cards ---------- */
.upk-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    border: 1px solid var(--line);
    transition: transform .25s ease, box-shadow .25s ease;
    height: 100%;
}
.upk-card:hover { transform: translateY(-4px); box-shadow: 0 16px 36px rgba(13,59,142,0.12); }
.upk-card h3 { margin: 8px 0 8px 0; font-size: 20px; }
.upk-card p { color: var(--muted); font-size: 15px; line-height: 1.6; margin: 0 0 12px 0; }

.section-title {
    font-size: 30px;
    font-weight: 800;
    color: var(--ink);
    margin: 8px 0 4px 0;
    letter-spacing: -0.01em;
}
.section-sub { color: var(--muted); font-size: 16px; margin-bottom: 22px; }

/* ---------- Badges / tags ---------- */
.badge { display:inline-block; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.03em; }
.badge-new { background: #E6FBF4; color: #00966f; border:1px solid #b8f0df; }
.badge-hot { background: #FFEDE5; color: #d94e1f; border:1px solid #ffcab3; }
.badge-ai  { background: #EAF0FF; color: var(--navy); border:1px solid #c3d4ff; }
.tag { display:inline-block; padding:5px 12px; border-radius:8px; font-size:12.5px; font-weight:600; margin:3px 4px 3px 0; }
.tag-navy { background:#EAF0FF; color:var(--navy); }
.tag-green{ background:#E6FBF4; color:#00966f; }
.tag-orange{ background:#FFEDE5; color:#d94e1f; }
.tag-grey { background:#F1F5F9; color:#475569; }

/* ---------- Step flow ---------- */
.flow-wrap { display:flex; align-items:stretch; gap:0; flex-wrap:wrap; margin:18px 0; }
.flow-step {
    flex:1; min-width:180px; background:#FFFFFF; border:1px solid var(--line);
    border-radius:16px; padding:22px 18px; text-align:center;
    box-shadow:0 8px 22px rgba(0,0,0,0.05); transition:transform .2s ease;
}
.flow-step:hover { transform:translateY(-4px); border-color:var(--navy); }
.flow-num { width:42px;height:42px;line-height:42px;border-radius:50%;margin:0 auto 12px auto;
    background:linear-gradient(135deg,var(--navy),var(--emerald)); color:#FFF; font-weight:800; font-size:18px; }
.flow-step h4 { margin:0 0 6px 0; font-size:16px; }
.flow-step p { color:var(--muted); font-size:13px; margin:0; }
.flow-arrow { display:flex; align-items:center; color:var(--emerald); font-size:26px; padding:0 6px; }

/* ---------- Link card ---------- */
.link-card {
    background:#FFFFFF; border:1px solid var(--line); border-radius:16px;
    padding:20px; box-shadow:0 8px 22px rgba(0,0,0,0.05); height:100%;
}
.link-card h4 { margin:0 0 14px 0; font-size:17px; color:var(--navy); }
.link-row { display:block; padding:9px 12px; border-radius:10px; color:var(--ink); font-size:14px;
    font-weight:500; transition:all .18s ease; border:1px solid transparent; }
.link-row:hover { background:#EEF2FF; color:var(--navy); border-color:#c3d4ff; }

/* ---------- Job card ---------- */
.job-card {
    background:#FFFFFF; border:1px solid var(--line); border-radius:16px;
    padding:20px 22px; box-shadow:0 8px 22px rgba(0,0,0,0.05); margin-bottom:16px;
    display:flex; align-items:center; gap:18px; transition:transform .2s ease, box-shadow .2s ease;
}
.job-card:hover { transform:translateY(-3px); box-shadow:0 14px 32px rgba(13,59,142,0.12); }
.job-icon { width:60px;height:60px;border-radius:16px;display:flex;align-items:center;justify-content:center;
    font-size:28px; background:linear-gradient(135deg,var(--navy),var(--emerald)); color:#FFF; flex-shrink:0; }
.job-mid { flex:1; }
.job-mid h4 { margin:0 0 4px 0; font-size:18px; }
.job-mid .job-meta { color:var(--muted); font-size:14px; margin-bottom:8px; }
.job-mid .job-ai { color:var(--navy); font-size:13px; font-style:italic; }
.match-badge { background:linear-gradient(135deg,var(--emerald),#00a87d); color:#FFF; font-weight:800;
    border-radius:12px; padding:10px 14px; text-align:center; min-width:84px; }
.match-badge small { display:block; font-weight:600; font-size:10px; opacity:.9; }

/* ---------- Info / data tables ---------- */
.upk-table { width:100%; border-collapse:separate; border-spacing:0; border-radius:14px; overflow:hidden;
    box-shadow:0 8px 22px rgba(0,0,0,0.05); border:1px solid var(--line); }
.upk-table th { background:var(--navy); color:#FFF; padding:12px 14px; text-align:left; font-size:13px;
    font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
.upk-table td { padding:11px 14px; border-bottom:1px solid var(--line); font-size:14px; color:var(--ink); background:#FFF; }
.upk-table tr:last-child td { border-bottom:none; }
.upk-table tr:hover td { background:#F7F9FF; }

/* ---------- Stat block ---------- */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin:16px 0; }
.stat-box { background:linear-gradient(135deg,#FFFFFF,#F0F4FF); border:1px solid var(--line);
    border-radius:18px; padding:26px 20px; text-align:center; box-shadow:0 8px 22px rgba(0,0,0,0.05); }
.stat-big { font-size:38px; font-weight:900; background:linear-gradient(135deg,var(--navy),var(--emerald));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1; }
.stat-cap { color:var(--muted); font-size:14px; font-weight:600; margin-top:10px; }

/* ---------- Metric boxes ---------- */
.metric-box { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:20px; text-align:center;
    box-shadow:0 8px 22px rgba(0,0,0,0.05); }
.metric-box .mv { font-size:32px; font-weight:900; color:var(--navy); }
.metric-box .ml { font-size:13px; color:var(--muted); font-weight:600; text-transform:uppercase; }

/* ---------- Visa card ---------- */
.visa-card { background:linear-gradient(135deg,#FFFFFF,#F0F4FF); border:1px solid var(--line);
    border-radius:20px; padding:30px; box-shadow:0 12px 30px rgba(13,59,142,0.08); }
.visa-head { display:flex; align-items:center; gap:16px; margin-bottom:8px; }
.visa-head .vi { width:64px;height:64px;border-radius:18px;display:flex;align-items:center;justify-content:center;
    font-size:32px; background:linear-gradient(135deg,var(--navy),var(--emerald)); color:#FFF; }
.info-pill { background:#FFFFFF; border:1px solid var(--line); border-radius:14px; padding:14px 18px;
    box-shadow:0 4px 12px rgba(0,0,0,0.04); }
.info-pill .ip-l { font-size:12px; color:var(--muted); font-weight:600; text-transform:uppercase; }
.info-pill .ip-v { font-size:18px; color:var(--navy); font-weight:800; }
.check-item { display:flex; align-items:flex-start; gap:10px; padding:7px 0; font-size:14.5px; color:var(--ink); }
.check-item .ck { color:var(--emerald); font-weight:900; flex-shrink:0; }

/* ---------- Hero CTA / chat button ---------- */
.chat-cta {
    display:inline-flex; align-items:center; gap:10px;
    background:linear-gradient(135deg,var(--orange),#ff8c5a); color:#FFF;
    padding:14px 26px; border-radius:999px; font-weight:700; font-size:16px;
    box-shadow:0 10px 26px rgba(255,107,53,0.35); animation:pulse 2.4s infinite;
}
@keyframes pulse {
    0% { box-shadow:0 0 0 0 rgba(255,107,53,0.45); }
    70% { box-shadow:0 0 0 18px rgba(255,107,53,0); }
    100% { box-shadow:0 0 0 0 rgba(255,107,53,0); }
}

/* ---------- Navbar ---------- */
.nav-logo { font-size:24px; font-weight:900; color:var(--navy); letter-spacing:-0.02em; }
.nav-logo span { color:var(--emerald); }

/* ---------- Reason / feature grid ---------- */
.reason-card { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:22px;
    box-shadow:0 8px 22px rgba(0,0,0,0.05); transition:transform .2s ease; height:100%; }
.reason-card:hover { transform:translateY(-4px); border-color:var(--emerald); }
.reason-card .ri { font-size:30px; }
.reason-card h4 { margin:10px 0 6px 0; font-size:17px; }
.reason-card p { color:var(--muted); font-size:14px; margin:0; line-height:1.55; }

/* ---------- News card ---------- */
.news-card { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:20px;
    box-shadow:0 8px 22px rgba(0,0,0,0.05); margin-bottom:14px; }
.news-card h4 { margin:8px 0 6px 0; font-size:17px; }
.news-card p { color:var(--muted); font-size:14px; margin:0 0 8px 0; }
.news-meta { font-size:12px; color:#94A3B8; }

/* ---------- Misc ---------- */
.soft-divider { height:1px; background:linear-gradient(90deg,transparent,var(--line),transparent); margin:28px 0; border:none; }
.glass-note { background:rgba(13,59,142,0.04); border:1px solid #c3d4ff; border-radius:14px; padding:16px 20px; color:var(--navy); font-size:14px; }
.portal-card { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:20px; text-align:center;
    box-shadow:0 8px 22px rgba(0,0,0,0.05); transition:transform .2s ease; height:100%; }
.portal-card:hover { transform:translateY(-4px); border-color:var(--navy); }
.portal-card h4 { margin:6px 0; font-size:16px; color:var(--navy); }
.portal-card p { color:var(--muted); font-size:13px; margin:0 0 10px 0; }

@media (max-width: 980px) {
    .kpi-wrap, .stat-grid { grid-template-columns:repeat(2,1fr); }
    .hero h1 { font-size:38px; }
    .flow-arrow { display:none; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─── Design enhancement layer (premium refresh) ──────────────────────────────
ENHANCED_CSS = """
<style>
:root {
    --navy: #0B2A6B;
    --navy-2: #1E50C8;
    --emerald: #06C684;
    --orange: #FF6B35;
    --light: #EEF3FF;
    --ink: #0F172A;
    --muted: #64748B;
    --line: #E6ECF5;
}

/* Ambient page background */
.stApp {
    background:
        radial-gradient(1200px 500px at 100% -10%, rgba(6,198,132,0.10), transparent 60%),
        radial-gradient(1000px 500px at -10% 0%, rgba(30,80,200,0.10), transparent 55%),
        linear-gradient(180deg, #FBFCFF 0%, #F3F6FD 100%);
}
.block-container { padding-top: 0.6rem !important; max-width: 1240px; }

/* Sharper headings */
h1,h2,h3,h4 { letter-spacing: -0.02em; }
.section-title { font-size: 32px; font-weight: 850; letter-spacing:-0.025em;
    background: linear-gradient(90deg, var(--ink) 0%, #21407e 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.section-sub { font-size: 15.5px; color: var(--muted); margin: 6px 0 24px 0; }

/* ===== Sticky glass navbar ===== */
.upk-nav {
    position: sticky; top: 8px; z-index: 999;
    background: rgba(255,255,255,0.72);
    backdrop-filter: saturate(180%) blur(16px);
    -webkit-backdrop-filter: saturate(180%) blur(16px);
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 10px 30px rgba(13,40,100,0.10);
    border-radius: 20px; padding: 6px 10px; margin-bottom: 18px;
}
.nav-logo { font-size: 25px; font-weight: 900; letter-spacing: -0.03em; padding-left: 6px; }
.nav-logo .lp { color: var(--navy); }
.nav-logo .ls { background: linear-gradient(90deg, var(--navy-2), var(--emerald));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Nav buttons → ghost pills with active state via gradient on hover */
.upk-nav .stButton > button {
    background: transparent !important;
    color: var(--muted) !important;
    box-shadow: none !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    padding: 9px 6px !important;
    transition: all .18s ease;
}
.upk-nav .stButton > button:hover {
    background: rgba(30,80,200,0.08) !important;
    color: var(--navy) !important;
    transform: translateY(0) !important;
    border-color: rgba(30,80,200,0.18) !important;
}
.upk-nav-active .stButton > button {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%) !important;
    color: #fff !important;
    box-shadow: 0 8px 18px rgba(13,40,100,0.28) !important;
}

/* ===== Hero refresh ===== */
.hero {
    background:
        radial-gradient(700px 300px at 90% -20%, rgba(255,255,255,0.25), transparent 60%),
        linear-gradient(135deg, #0B2A6B 0%, #1E50C8 52%, #06C684 120%);
    border-radius: 30px;
    padding: 70px 52px 100px 52px;
    box-shadow: 0 30px 70px rgba(11,42,107,0.32);
}
.hero::before {
    content:""; position:absolute; left:-60px; bottom:-80px;
    width:260px; height:260px; border-radius:50%;
    background: radial-gradient(circle, rgba(6,198,132,0.35), transparent 70%);
}
.hero h1 { font-size: 56px; font-weight: 900; }
.hero-badge { background: rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.35);
    box-shadow: 0 6px 18px rgba(0,0,0,0.10); }

/* ===== KPI cards ===== */
.kpi-wrap { gap: 20px; margin-top: -68px; }
.kpi-card {
    border-radius: 22px; padding: 26px 20px;
    border: 1px solid rgba(255,255,255,0.7);
    border-bottom: 4px solid var(--emerald);
    box-shadow: 0 18px 40px rgba(13,40,100,0.10);
    background: linear-gradient(180deg, #FFFFFF, #FBFDFF);
}
.kpi-card:hover { transform: translateY(-8px); box-shadow: 0 26px 55px rgba(13,40,100,0.18); }
.kpi-value { font-size: 38px; font-weight: 900;
    background: linear-gradient(135deg, var(--navy), var(--navy-2));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.kpi-icon { filter: drop-shadow(0 4px 8px rgba(13,40,100,0.18)); }

/* ===== Cards everywhere — softer, deeper ===== */
.upk-card, .link-card, .reason-card, .news-card, .portal-card, .metric-box,
.flow-step, .job-card, .visa-card, .stat-box {
    border-radius: 20px !important;
    border: 1px solid var(--line) !important;
    box-shadow: 0 12px 30px rgba(13,40,100,0.06) !important;
    background: linear-gradient(180deg,#FFFFFF, #FCFDFF) !important;
}
.upk-card:hover, .reason-card:hover, .news-card:hover, .flow-step:hover,
.job-card:hover, .portal-card:hover, .link-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 22px 48px rgba(13,40,100,0.14) !important;
    border-color: rgba(30,80,200,0.25) !important;
}
.upk-card h3 { font-size: 19px; font-weight: 800; }

/* Flow numbers + accent */
.flow-num { box-shadow: 0 10px 22px rgba(6,198,132,0.32); }
.flow-arrow { color: var(--navy-2); font-weight: 800; }

/* ===== Tabs → modern segmented control ===== */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(8px);
    padding: 6px; border-radius: 16px;
    border: 1px solid var(--line);
    box-shadow: 0 8px 22px rgba(13,40,100,0.06);
}
.stTabs [data-baseweb="tab"] {
    border: none !important; background: transparent !important;
    border-radius: 11px !important; padding: 9px 18px !important;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(30,80,200,0.07) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--navy), var(--navy-2)) !important;
    color: #fff !important; box-shadow: 0 8px 18px rgba(13,40,100,0.25) !important;
}

/* ===== Buttons — refined ===== */
.stButton > button {
    border-radius: 14px; font-weight: 700;
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%);
    box-shadow: 0 10px 22px rgba(13,40,100,0.22);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 16px 32px rgba(13,40,100,0.30); }

/* ===== Inputs ===== */
.stTextInput input, .stTextArea textarea, .stDateInput input {
    border: 1.5px solid #D7E0F0 !important; border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(13,40,100,0.04) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--navy-2) !important;
    box-shadow: 0 0 0 4px rgba(30,80,200,0.14) !important;
}
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    border: 1.5px solid #D7E0F0 !important; border-radius: 14px !important;
}

/* ===== Tags / badges — pill polish ===== */
.tag { border-radius: 999px; padding: 5px 13px; font-weight: 700; font-size: 12px; }
.badge { padding: 5px 13px; }

/* Link rows */
.link-row { border-radius: 12px; font-weight: 600; }
.link-row:hover { background: linear-gradient(90deg, rgba(30,80,200,0.08), rgba(6,198,132,0.08)); }

/* Tables */
.upk-table { border-radius: 18px; box-shadow: 0 12px 30px rgba(13,40,100,0.07); }
.upk-table th { background: linear-gradient(135deg, var(--navy), var(--navy-2)); }

/* Stat numbers */
.stat-big { background: linear-gradient(135deg, var(--navy-2), var(--emerald));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2A6B 0%, #082356 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Visa card */
.visa-card { background: linear-gradient(160deg,#FFFFFF 0%, #F4F8FF 100%) !important;
    box-shadow: 0 22px 50px rgba(13,40,100,0.10) !important; }
.visa-head .vi { box-shadow: 0 12px 26px rgba(11,42,107,0.30); }

/* Soft entrance animation for content */
.block-container .element-container { animation: fadeUp .5s ease both; }
@keyframes fadeUp { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform:none; } }

/* Divider */
.soft-divider { margin: 30px 0; background: linear-gradient(90deg,transparent, #D7E0F0, transparent); }

@media (max-width: 980px) {
    .hero h1 { font-size: 36px; }
    .hero { padding: 48px 28px 84px 28px; }
}
</style>
"""
st.markdown(ENHANCED_CSS, unsafe_allow_html=True)

# ─── Final product polish layer ──────────────────────────────────────────────
FINAL_CSS = """
<style>
/* Custom scrollbar */
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #1E50C8, #06C684);
    border-radius: 10px; border: 3px solid #F3F6FD;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #0B2A6B, #06C684); }
::selection { background: rgba(6,198,132,0.28); }

/* Tighten Streamlit's default vertical gaps for a denser, app-like feel */
div[data-testid="stVerticalBlock"] { gap: 0.7rem; }

/* Section title accent bar */
.section-title { position: relative; padding-left: 16px; }
.section-title::before {
    content:""; position:absolute; left:0; top:8px; bottom:8px; width:5px;
    border-radius:6px; background: linear-gradient(180deg, #1E50C8, #06C684);
}
.section-sub { padding-left: 16px; }

/* Hero — decorative dotted grid + crisper type */
.hero {
    background-image:
        radial-gradient(rgba(255,255,255,0.10) 1px, transparent 1px),
        radial-gradient(700px 300px at 90% -20%, rgba(255,255,255,0.25), transparent 60%),
        linear-gradient(135deg, #0B2A6B 0%, #1E50C8 52%, #06C684 120%);
    background-size: 22px 22px, auto, auto;
}
.hero h1 { font-size: 58px; line-height: 1.05; text-shadow: 0 6px 24px rgba(0,0,0,0.18); }
.hero p { font-size: 19.5px; }
.hero-badge { letter-spacing: .02em; }

/* Navbar — active pill gets a soft glow ring */
.upk-nav { padding: 7px 12px; }
.upk-nav-active .stButton > button {
    box-shadow: 0 8px 20px rgba(13,40,100,0.30), 0 0 0 3px rgba(6,198,132,0.18) !important;
}
.upk-nav .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px !important; border: 1px solid #E1E8F4 !important;
    background: rgba(255,255,255,0.9) !important;
}

/* KPI cards — gradient hairline ring */
.kpi-card { position: relative; }
.kpi-card::after {
    content:""; position:absolute; inset:0; border-radius:22px; padding:1px;
    background: linear-gradient(135deg, rgba(30,80,200,0.35), rgba(6,198,132,0.35));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude; pointer-events:none;
}

/* Buttons — sheen sweep on hover */
.stButton > button { position: relative; overflow: hidden; }
.stButton > button::after {
    content:""; position:absolute; top:0; left:-120%; width:60%; height:100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: skewX(-20deg); transition: left .6s ease;
}
.stButton > button:hover::after { left: 140%; }

/* Primary CTA buttons inside content cards a touch larger */
.stButton > button { letter-spacing: .01em; }

/* Inputs — softer, larger touch target */
.stTextInput input, .stTextArea textarea {
    padding: 12px 15px !important; font-size: 15px !important;
}

/* Expander header weight */
summary p { font-weight: 700 !important; font-size: 15.5px !important; }

/* Streamlit alert boxes restyled to the palette */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid #DCE6F7 !important;
    box-shadow: 0 8px 22px rgba(13,40,100,0.05) !important;
}

/* Metric polish */
.metric-box .mv { background: linear-gradient(135deg,#0B2A6B,#1E50C8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* Chat input in sidebar */
section[data-testid="stSidebar"] .stChatInput textarea {
    background: rgba(255,255,255,0.96) !important; color: #0F172A !important;
    border-radius: 14px !important;
}

/* Footer */
.upk-footer {
    margin-top: 48px; padding: 34px 28px; border-radius: 24px;
    background: linear-gradient(135deg, #0B2A6B 0%, #102f78 60%, #0c3f6e 100%);
    color: #fff; box-shadow: 0 20px 50px rgba(11,42,107,0.25);
}
.upk-footer a { color: #9ec5ff; }
.upk-footer a:hover { color: #06C684; }
.upk-foot-logo { font-size: 22px; font-weight: 900; letter-spacing: -0.02em; }
.upk-foot-cols { display:grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 26px; margin-top: 18px; }
.upk-foot-cols h5 { color:#cfe0ff; font-size:13px; text-transform:uppercase; letter-spacing:.06em; margin:0 0 10px 0; }
.upk-foot-cols a { display:block; padding:4px 0; font-size:14px; }
.upk-foot-bottom { margin-top:22px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.14);
    font-size:13px; color:rgba(255,255,255,0.7); display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; }

@media (max-width: 900px) {
    .upk-foot-cols { grid-template-columns: 1fr 1fr; }
    .hero h1 { font-size: 34px; }
}
</style>
"""
st.markdown(FINAL_CSS, unsafe_allow_html=True)

# ─── PROFESSIONAL REDESIGN (final override layer) ────────────────────────────
# A clean, corporate, editorial aesthetic: deep slate-navy + a single refined
# indigo accent, generous whitespace, a solid white top bar with an underline
# active indicator, a confident dark hero, and flat precise cards. This layer
# is loaded last, so it wins the cascade.
PRO_CSS = """
<style>
:root {
    --navy: #16233F;      /* primary deep slate-navy   */
    --navy-2: #2E4B8A;    /* secondary                  */
    --indigo: #3D5AFE;    /* refined accent (sparingly) */
    --emerald: #12A580;   /* success / positive accent  */
    --orange: #E0703A;    /* warm accent                */
    --light: #F5F7FB;
    --ink: #131A2B;
    --muted: #5A6478;
    --line: #E8EBF1;
    --paper: #FFFFFF;
}

/* ===== Page canvas — calm, professional ===== */
.stApp {
    background: #F7F8FB !important;
}
.block-container { max-width: 1440px !important; padding-top: 0.4rem !important; }
html, body, [class*="css"], .stApp { color: var(--ink); }
h1,h2,h3,h4 { color: var(--ink); letter-spacing: -0.018em; }

/* ===== Top navigation — solid white bar, underline active ===== */
.upk-nav {
    position: sticky; top: 0; z-index: 999;
    background: rgba(255,255,255,0.96);
    backdrop-filter: saturate(180%) blur(10px);
    border: none; border-bottom: 1px solid var(--line);
    border-radius: 0; box-shadow: 0 2px 14px rgba(16,35,63,0.05);
    padding: 12px 10px 6px 10px; margin: 0 0 26px 0;
}
.nav-logo { font-size: 22px; font-weight: 800; letter-spacing: -0.03em; }
.nav-logo .lp { color: var(--navy); }
.nav-logo .ls { color: var(--indigo); -webkit-text-fill-color: var(--indigo); background: none; }

.upk-nav .stButton > button {
    background: transparent !important; color: var(--muted) !important;
    border: none !important; border-radius: 0 !important;
    box-shadow: none !important; font-weight: 600 !important;
    font-size: 13px !important; letter-spacing: .02em !important;
    padding: 8px 4px 10px 4px !important;
    border-bottom: 2px solid transparent !important;
    transition: color .15s ease, border-color .15s ease;
}
.upk-nav .stButton > button::after { display: none !important; }
.upk-nav .stButton > button:hover {
    background: transparent !important; color: var(--navy) !important;
    transform: none !important; border-bottom: 2px solid var(--line) !important;
}
.upk-nav-active .stButton > button {
    color: var(--navy) !important; font-weight: 800 !important;
    border-bottom: 2px solid var(--indigo) !important;
    box-shadow: none !important;
}
.upk-nav .stSelectbox div[data-baseweb="select"] > div {
    background: #fff !important; border: 1px solid var(--line) !important;
    border-radius: 9px !important; min-height: 34px;
}

/* ===== Hero — confident dark, editorial ===== */
.hero {
    background:
        radial-gradient(620px 280px at 82% 12%, rgba(61,90,254,0.22), transparent 62%),
        radial-gradient(480px 300px at 8% 95%, rgba(18,165,128,0.16), transparent 60%),
        linear-gradient(160deg, #0E1A33 0%, #16233F 55%, #1B2C50 100%);
    background-size: auto;
    border-radius: 22px;
    padding: 76px 56px 104px 56px;
    box-shadow: 0 24px 60px rgba(14,26,51,0.30);
    border: 1px solid rgba(255,255,255,0.06);
}
.hero::before, .hero::after { display: none; }
.hero-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: #C9D6FF; font-weight: 600; letter-spacing: .03em;
    backdrop-filter: blur(4px); box-shadow: none;
}
.hero h1 {
    color: #FFFFFF !important; font-size: 54px; font-weight: 800;
    line-height: 1.08; letter-spacing: -0.03em; text-shadow: none;
    max-width: 760px; margin: 18px 0 18px 0;
}
.hero p { color: rgba(226,232,245,0.82); font-size: 18.5px; font-weight: 400; max-width: 640px; }

/* ===== KPI cards — flat, precise, divided ===== */
.kpi-wrap { gap: 0; margin-top: -56px; border-radius: 16px; overflow: hidden;
    box-shadow: 0 16px 40px rgba(16,35,63,0.10); }
.kpi-card {
    background: #fff !important; border: 1px solid var(--line) !important;
    border-right: none !important; border-bottom: none !important;
    border-radius: 0 !important; box-shadow: none !important; padding: 26px 22px;
}
.kpi-wrap .kpi-card:first-child { border-top-left-radius: 16px; border-bottom-left-radius: 16px; }
.kpi-wrap .kpi-card:last-child { border-right: 1px solid var(--line) !important;
    border-top-right-radius: 16px; border-bottom-right-radius: 16px; }
.kpi-card::after { display: none !important; }
.kpi-card:hover { transform: none; box-shadow: inset 0 -3px 0 var(--indigo); }
.kpi-value { color: var(--navy); -webkit-text-fill-color: var(--navy); background: none;
    font-size: 34px; font-weight: 800; }
.kpi-label { color: var(--muted); font-weight: 600; letter-spacing: .06em; font-size: 11.5px; }
.kpi-icon { filter: none; opacity: .9; }

/* ===== Section titles — editorial, no gradient text ===== */
.section-title {
    color: var(--ink) !important; -webkit-text-fill-color: var(--ink); background: none;
    font-size: 26px; font-weight: 800; padding-left: 0; letter-spacing: -0.02em;
}
.section-title::before { display: none; }
.section-sub { padding-left: 0; color: var(--muted); font-size: 15px; }

/* ===== Cards — flat, 1px border, hover lift only ===== */
.upk-card, .link-card, .reason-card, .news-card, .portal-card, .metric-box,
.flow-step, .job-card, .visa-card, .stat-box {
    background: #fff !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 2px rgba(16,35,63,0.04) !important;
}
.upk-card:hover, .reason-card:hover, .news-card:hover, .flow-step:hover,
.job-card:hover, .portal-card:hover, .link-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 34px rgba(16,35,63,0.10) !important;
    border-color: #D4DAE6 !important;
}
.upk-card h3 { font-size: 17px; font-weight: 700; }
.upk-card p, .reason-card p, .news-card p { color: var(--muted); }

/* Flow numbers — solid, no glow */
.flow-num { background: var(--navy); box-shadow: none; font-weight: 700; }
.flow-arrow { color: #C4CCDB; }

/* Stat numbers — solid navy */
.stat-big { color: var(--navy); -webkit-text-fill-color: var(--navy); background: none;
    font-weight: 800; }

/* ===== Buttons — solid, restrained ===== */
.stButton > button {
    background: var(--navy) !important;
    color: #fff !important; border: 1px solid var(--navy) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    box-shadow: none !important; padding: 9px 18px !important;
    transition: background .15s ease, transform .12s ease;
}
.stButton > button::after { display: none !important; }
.stButton > button:hover {
    background: #1E3360 !important; border-color: #1E3360 !important;
    color: #fff !important; transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(16,35,63,0.18) !important;
}
.stDownloadButton > button {
    background: #fff !important; color: var(--navy) !important;
    border: 1px solid var(--navy) !important; border-radius: 10px !important; font-weight: 600 !important;
}

/* ===== Tabs — clean underline indicator ===== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent; border: none; border-bottom: 1px solid var(--line);
    border-radius: 0; padding: 0; gap: 4px; box-shadow: none; backdrop-filter: none;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important; border-radius: 0 !important;
    color: var(--muted) !important; font-weight: 600 !important; font-size: 14px !important;
    padding: 11px 16px !important; margin-bottom: -1px;
    border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { background: transparent !important; color: var(--navy) !important; }
.stTabs [aria-selected="true"] {
    background: transparent !important; color: var(--navy) !important;
    border-bottom: 2px solid var(--indigo) !important; box-shadow: none !important;
}

/* ===== Badges / tags — quiet, professional ===== */
.tag { border-radius: 6px; font-weight: 600; font-size: 11.5px; padding: 4px 10px; }
.tag-navy { background: #EEF1F8; color: var(--navy); }
.tag-green { background: #E6F6F1; color: #0B7A5E; }
.tag-orange { background: #FBEEE6; color: #B85426; }
.tag-grey { background: #F1F3F7; color: #57607A; }
.badge-new { background: #E6F6F1; color: #0B7A5E; border: none; }
.badge-hot { background: #FBEEE6; color: #B85426; border: none; }
.badge-ai  { background: #ECEFFE; color: var(--indigo); border: none; }
.match-badge { background: var(--navy); }

/* ===== Inputs ===== */
.stTextInput input, .stTextArea textarea, .stDateInput input {
    border: 1px solid #D6DCE8 !important; border-radius: 10px !important;
    box-shadow: none !important; background: #fff !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--indigo) !important; box-shadow: 0 0 0 3px rgba(61,90,254,0.12) !important;
}
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    border: 1px solid #D6DCE8 !important; border-radius: 10px !important;
}
.stMultiSelect span[data-baseweb="tag"] { background: var(--navy) !important; border-radius: 6px !important; }

/* Expanders — flat */
.stExpander, details[data-testid="stExpander"] {
    border: 1px solid var(--line) !important; border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(16,35,63,0.04) !important;
}

/* Tables */
.upk-table { border: 1px solid var(--line); border-radius: 12px; box-shadow: none; }
.upk-table th { background: var(--navy); font-weight: 600; font-size: 12px; letter-spacing: .04em; }
.upk-table tr:hover td { background: #F7F9FC; }

/* Link rows */
.link-card h4 { color: var(--navy); font-size: 15px; }
.link-row { border-radius: 8px; color: var(--ink); }
.link-row:hover { background: #F1F4FA; color: var(--navy); border-color: transparent; }

/* Metric numbers */
.metric-box .mv { color: var(--navy); -webkit-text-fill-color: var(--navy); background: none; }

/* CTA chip — quieter */
.chat-cta {
    background: #fff; color: var(--navy);
    border: 1px solid var(--line); box-shadow: 0 6px 18px rgba(16,35,63,0.08);
    animation: none; font-weight: 600;
}

/* Sidebar — deep, matches navy */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16233F 0%, #101A30 100%) !important;
}

/* Visa card — flat */
.visa-card { background: #fff !important; box-shadow: 0 1px 2px rgba(16,35,63,0.04) !important; }
.visa-head .vi { background: var(--navy); box-shadow: none; }
.info-pill { border-radius: 10px; }
.info-pill .ip-v { color: var(--navy); }
.check-item .ck { color: var(--emerald); }

/* Footer — refined slate */
.upk-footer { background: linear-gradient(160deg, #16233F 0%, #101A30 100%); border-radius: 20px; }
.upk-footer a { color: #9FB2D8; }

/* Divider */
.soft-divider { background: var(--line); }

/* Stat grid spacing */
.stat-grid { gap: 14px; }
.stat-box { background: #fff !important; }

@media (max-width: 920px) {
    .upk-nav { margin: 0 -1rem 18px -1rem; }
    .hero h1 { font-size: 34px; }
    .hero { padding: 52px 28px 90px 28px; }
}
</style>
"""
st.markdown(PRO_CSS, unsafe_allow_html=True)

# ─── HEADER — polished sticky top bar ────────────────────────────────────────
HEADER_CSS = """
<style>
/* Thin gradient accent strip pinned to the very top of the viewport */
.stApp::before {
    content:""; position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 100000;
    background: linear-gradient(90deg, #16233F 0%, #3D5AFE 50%, #12A580 100%);
}

/* Kill Streamlit's large default top padding so the header sits at the top */
header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] .block-container,
.stMainBlockContainer, .block-container {
    padding-top: 0.5rem !important;
}
[data-testid="stToolbar"] { display: none !important; }

/* The keyed container becomes the real header bar (wraps all widgets) */
.st-key-topnav {
    position: sticky; top: 6px; z-index: 9999;
    background: rgba(255,255,255,0.92);
    backdrop-filter: saturate(180%) blur(14px);
    -webkit-backdrop-filter: saturate(180%) blur(14px);
    border: 1px solid var(--line);
    box-shadow: 0 6px 24px rgba(16,35,63,0.08);
    padding: 10px 22px;
    margin: 0 0 24px 0;
    border-radius: 16px;
}
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items: center; }
/* Vertically center every column's content within the bar */
.st-key-topnav [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
    display: flex; flex-direction: column; justify-content: center;
}
.st-key-topnav [data-testid="stColumn"] [data-testid="stMarkdownContainer"] { width: 100%; }

/* Logo lockup */
.upk-logo { display: flex; align-items: center; gap: 11px; margin: 0; line-height: 1; }
.st-key-topnav [data-testid="stColumn"]:first-child [data-testid="stMarkdownContainer"] { margin: 0; }
.st-key-topnav [data-testid="stColumn"]:first-child p { margin: 0 !important; }
.upk-logo .badge {
    width: 40px; height: 40px; border-radius: 12px; flex-shrink: 0;
    background: linear-gradient(135deg, #16233F 0%, #3D5AFE 100%);
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-size: 21px; box-shadow: 0 8px 18px rgba(61,90,254,0.30);
}
.upk-logo .wm { line-height: 1.05; font-size: 21px; font-weight: 800;
    letter-spacing: -0.035em; color: var(--navy); }
.upk-logo .wm span { color: var(--indigo); }
.upk-logo .wm small { display: block; font-size: 9.5px; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-top: 3px; }

/* Nav links — quiet text, underline indicator */
.st-key-topnav .stButton > button {
    background: transparent !important; color: var(--muted) !important;
    border: none !important; border-radius: 8px !important; box-shadow: none !important;
    font-weight: 600 !important; font-size: 12.5px !important; letter-spacing: 0 !important;
    padding: 9px 2px !important; position: relative; transition: color .15s ease;
    white-space: nowrap !important;
}
.st-key-topnav .stButton > button p { white-space: nowrap !important; }
.st-key-topnav .stButton > button::after { display: none !important; }
.st-key-topnav .stButton > button:hover {
    background: #F3F5FA !important; color: var(--navy) !important; transform: none !important;
}
/* Active nav item — soft tinted pill */
.st-key-nav_active .stButton > button {
    color: var(--navy) !important; font-weight: 800 !important;
    background: linear-gradient(135deg, rgba(61,90,254,0.12), rgba(18,165,128,0.12)) !important;
    border-radius: 9px !important;
}
.st-key-nav_active .stButton > button::before { display: none !important; }

/* Language selector — compact pill */
.st-key-topnav .stSelectbox div[data-baseweb="select"] > div {
    background: #F5F7FB !important; border: 1px solid var(--line) !important;
    border-radius: 10px !important; min-height: 38px;
}

/* Auth CTA — filled, distinct from nav links */
.st-key-nav_cta .stButton > button {
    background: var(--navy) !important; color: #fff !important;
    border: 1px solid var(--navy) !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 13px !important; padding: 9px 14px !important;
    box-shadow: 0 6px 16px rgba(16,35,63,0.20) !important;
}
.st-key-nav_cta .stButton > button:hover {
    background: #1E3360 !important; border-color: #1E3360 !important;
    transform: translateY(-1px); color: #fff !important;
}

@media (max-width: 920px) {
    .st-key-topnav { padding: 10px 12px; }
    .upk-logo .wm small { display: none; }
}
</style>
"""
st.markdown(HEADER_CSS, unsafe_allow_html=True)

# ─── HERO + JOB PORTALS — refined ────────────────────────────────────────────
HERO_CSS = """
<style>
/* Hero — stats integrated inside, no overlapping floating cards */
.hero {
    padding: 60px 54px 52px 54px !important;
    border-radius: 24px !important;
}
.hero h1 { font-size: 52px; max-width: 800px; }
.hero p { margin-bottom: 6px; }

.hero-stats {
    display: grid; grid-template-columns: repeat(4, 1fr);
    margin-top: 38px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 16px;
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    overflow: hidden;
}
.hstat { padding: 22px 18px; text-align: center;
    border-right: 1px solid rgba(255,255,255,0.12);
    transition: background .2s ease; }
.hstat:last-child { border-right: none; }
.hstat:hover { background: rgba(255,255,255,0.06); }
.hstat .hi { font-size: 17px; opacity: .85; margin-bottom: 4px; }
.hstat .hv { font-size: 30px; font-weight: 800; color: #fff; letter-spacing: -0.02em; line-height: 1; }
.hstat .hl { font-size: 11.5px; font-weight: 600; color: rgba(206,219,255,0.82);
    text-transform: uppercase; letter-spacing: .07em; margin-top: 8px; }

/* Popular job portals — premium logo cards */
.portal-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
.pcard {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    background: #fff; border: 1px solid var(--line); border-radius: 16px;
    padding: 22px 14px; text-align: center; transition: all .2s ease; height: 100%;
}
.pcard:hover { transform: translateY(-4px); border-color: #C9D2E6;
    box-shadow: 0 16px 34px rgba(16,35,63,0.12); }
.pcard .pmark {
    width: 46px; height: 46px; border-radius: 13px; display: flex;
    align-items: center; justify-content: center; font-size: 19px; font-weight: 800;
    color: #fff; box-shadow: 0 6px 14px rgba(16,35,63,0.18);
}
.pcard .pname { font-size: 14px; font-weight: 700; color: var(--navy); }
.pcard .pdesc { font-size: 11.5px; color: var(--muted); line-height: 1.4; }
.pcard .pgo { font-size: 11px; font-weight: 700; color: var(--indigo);
    letter-spacing: .03em; margin-top: auto; }

/* Explore the Platform — feature cards */
.feat-card {
    position: relative; overflow: hidden; height: 100%;
    background: #fff; border: 1px solid var(--line);
    border-radius: 18px; border-top: 3px solid var(--fc, #3D5AFE);
    padding: 26px 24px 22px 24px; transition: all .22s ease;
}
.feat-card::before {
    content: ""; position: absolute; right: -40px; top: -40px;
    width: 130px; height: 130px; border-radius: 50%;
    background: radial-gradient(circle, var(--fcglow, rgba(61,90,254,0.10)), transparent 70%);
}
.feat-card:hover { transform: translateY(-5px); border-color: #D4DAE6;
    box-shadow: 0 18px 40px rgba(16,35,63,0.12); }
.feat-tile {
    width: 54px; height: 54px; border-radius: 15px; display: flex;
    align-items: center; justify-content: center; font-size: 26px; color: #fff;
    background: var(--fc, #3D5AFE); box-shadow: 0 10px 22px var(--fcsh, rgba(61,90,254,0.30));
    margin-bottom: 16px;
}
.feat-badge {
    position: absolute; top: 18px; right: 18px; font-size: 11px; font-weight: 800;
    letter-spacing: .06em; padding: 4px 10px; border-radius: 999px;
    background: var(--fcbg, #ECEFFE); color: var(--fc, #3D5AFE);
}
.feat-card h3 { font-size: 19px; font-weight: 800; margin: 0 0 8px 0; color: var(--ink); }
.feat-card p { color: var(--muted); font-size: 14px; line-height: 1.55; margin: 0 0 4px 0; }

@media (max-width: 920px) {
    .hero-stats { grid-template-columns: repeat(2, 1fr); }
    .hstat:nth-child(2) { border-right: none; }
    .portal-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""
st.markdown(HERO_CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 4. MULTILINGUAL SYSTEM — 9 LANGUAGES
# ════════════════════════════════════════════════════════════════════════════
LANGUAGES = [
    "🇺🇸 English", "🇰🇷 한국어", "🇲🇳 Монгол", "🇯🇵 日本語", "🇨🇳 中文",
    "🇻🇳 Tiếng Việt", "🇹🇭 ภาษาไทย", "🇲🇾 Bahasa Melayu", "🇷🇺 Русский",
]

TR = {
    "🇺🇸 English": {
        "app_name": "UniPath", "tagline": "Your AI guide to studying, working & living in Korea",
        "nav_home": "HOME", "nav_university": "UNIVERSITY", "nav_career": "CAREER",
        "nav_job": "JOB", "nav_topik": "TOPIK", "nav_visa": "VISA", "nav_life": "LIFE",
        "login": "Sign In", "logout": "Sign Out", "register": "Create Account",
        "profile": "Profile", "language": "Language", "search": "Search", "apply": "Apply",
        "submit": "Submit", "back": "Back", "save": "Save", "cancel": "Cancel",
        "next": "Next", "skip": "Skip", "download": "Download", "visit": "Visit",
        "loading": "Loading...", "error": "Sorry, something went wrong. Please try again.",
        "empty": "No data available yet. Please check back soon.",
        "home_badge": "🌏 AI-Powered Platform for International Students",
        "home_title": "Your complete path to studying in South Korea",
        "home_subtitle": "Universities, scholarships, TOPIK, visas, jobs and daily life — everything an international student needs, powered by AI and available in 9 languages.",
        "kpi_visa": "Visa Types", "kpi_uni": "Universities", "kpi_topik": "TOPIK Level", "kpi_jobs": "Job Openings",
        "plan_title": "Plan Your Journey", "plan_sub": "Four simple steps from application to your new life in Korea",
        "step1_t": "Choose University", "step1_d": "Find your best-fit school and major",
        "step2_t": "Prepare TOPIK", "step2_d": "Reach the Korean level you need",
        "step3_t": "Apply for Visa", "step3_d": "Get the right visa for your goals",
        "step4_t": "Start Life in Korea", "step4_d": "Housing, banking, health & more",
        "feat_uni_t": "University Search", "feat_uni_d": "Explore 380+ universities, majors, tuition and scholarships in one place.",
        "feat_job_t": "Job Board", "feat_job_d": "AI-matched jobs for international talent, with visa-friendly employers.",
        "feat_visa_t": "Visa Checker", "feat_visa_d": "Understand D-2, D-4, E-7, F-2 and F-5 visas with AI guidance.",
        "official_res": "Official Resources", "official_sub": "Trusted government and university links, all in one place",
        "res_study": "🎓 Study Resources", "res_immig": "🛂 Immigration Guide", "res_tests": "📝 Tests & Employment",
        "job_portals": "Popular Job Portals", "why_korea": "Why Korea", "statistics": "Statistics", "resources": "Resources",
        "overview": "Overview", "latest_news": "Latest News",
        "uni_research": "Research", "uni_school": "School Info", "uni_apply": "Apply",
        "uni_admission": "Admission", "uni_scholar": "Scholarships",
        "filter_keyword": "Keyword", "filter_region": "Region", "filter_gks": "GKS Eligible",
        "kw_ph": "Search universities or majors...", "all": "All", "yes": "Yes", "no": "No",
        "loc_contact": "📍 Location & Contact", "tuition_support": "💰 Tuition & Support",
        "grad_req": "🎓 Graduation Requirements", "avail_majors": "📚 Available Majors",
        "btn_website": "🌐 Official Website", "btn_apply": "📝 Apply Now", "btn_intl": "🌏 International Office",
        "career_cv": "CV Check", "career_interview": "Mock Interview", "career_res": "Resources",
        "cv_title": "AI CV Review", "cv_sub": "Upload your resume and get instant feedback for the Korean job market",
        "cv_upload": "Upload your CV (PDF or DOCX)", "cv_target": "Target job title",
        "cv_target_ph": "e.g. Software Engineer", "cv_analyze": "🔍 Analyze My CV",
        "cv_score": "Overall Score", "cv_grammar": "Grammar", "cv_structure": "Structure",
        "cv_recs": "Recommendations", "cv_culture": "Korean Culture Tips",
        "iv_title": "🎤 AI Mock Interview", "iv_sub": "Practice with realistic questions and get AI feedback",
        "iv_role": "Job role", "iv_role_ph": "e.g. Software Engineer, Marketing Manager",
        "iv_info": "Enter a job role above to begin your mock interview.",
        "iv_start": "🚀 Start Interview", "iv_answer": "Your answer", "iv_answer_ph": "Type your answer (at least 10 characters)...",
        "iv_min": "Please write at least 10 characters.", "iv_feedback": "Generating AI feedback...",
        "iv_overall": "Overall", "iv_confidence": "Confidence", "iv_fit": "Cultural Fit",
        "iv_strengths": "Strengths", "iv_improve": "Improvements", "iv_again": "🔄 Try Again",
        "job_board": "Job Board", "job_matches": "My Matches", "job_portals_tab": "Portals",
        "job_search_ph": "Search role or company...", "job_visa_filter": "Visa Type",
        "job_match": "MATCH", "job_apply": "Apply", "job_upload_cv": "Upload your CV to find best-matching jobs",
        "job_find": "🎯 Find My Matches",
        "topik_schedule": "Schedule", "topik_register": "Register", "topik_levels": "Levels",
        "topik_tips": "Study Tips", "topik_centers": "Test Centers",
        "topik_fee": "Fee Information", "topik_reg_btn": "Register at topik.go.kr →",
        "tc_search": "Search by city...",
        "visa_req": "Requirements", "visa_docs": "Documents", "visa_proc": "Processing Time",
        "visa_for": "For", "visa_fee": "Fee", "visa_dur": "Duration", "visa_apply": "Apply on HiKorea →",
        "life_housing": "Housing", "life_transport": "Transport", "life_health": "Health",
        "life_banking": "Banking", "life_safety": "Safety",
        "auth_welcome": "Welcome to UniPath", "auth_login_sub": "Sign in to save universities, jobs and get alerts",
        "auth_reg_sub": "Create a free account and personalize your journey",
        "full_name": "Full name", "email": "Email", "password": "Password",
        "confirm_pw": "Confirm password", "lang_pref": "Language preference",
        "notif_pref": "Notification preferences",
        "notif_topik": "TOPIK schedule alerts", "notif_visa": "Visa policy updates",
        "notif_job": "New job openings", "notif_uni": "University news", "notif_scholar": "Scholarship announcements",
        "have_account": "Already have an account? Sign In", "no_account": "No account? Create one",
        "pw_mismatch": "Passwords do not match.", "invalid_email": "Please enter a valid email address.",
        "reg_success": "🎉 Account created! Welcome aboard.", "login_fail": "Account not found. Please register first.",
        "login_success": "Welcome back!", "fill_all": "Please fill in all fields.",
        "my_profile": "My Profile", "saved_uni": "🎓 Saved Universities", "saved_jobs": "💼 Saved Jobs",
        "edit_notif": "Edit notification preferences", "profile_saved": "Preferences saved!",
        "none_yet": "Nothing saved yet.",
        "chat_title": "UNI Assistant", "chat_sub": "Ask me anything about life in Korea",
        "placeholder": "Ask me anything about Korea...", "ask_ai": "💬 Ask AI Assistant",
        "subscribe": "Get Email Updates", "email_label": "Your email", "topics": "Topics of interest",
        "sub_success": "✅ Subscribed! You'll get updates in your language.",
        "admin_panel": "⚙️ Admin Panel", "admin_pw": "Admin Password", "admin_ok": "✅ Authorized",
        "admin_pdf": "📄 PDF Upload", "admin_stats": "📊 Stats",
    },
    "🇰🇷 한국어": {
        "app_name": "유니패스", "tagline": "한국 유학·취업·생활을 위한 AI 가이드",
        "nav_home": "홈", "nav_university": "대학", "nav_career": "커리어",
        "nav_job": "취업", "nav_topik": "토픽", "nav_visa": "비자", "nav_life": "생활",
        "login": "로그인", "logout": "로그아웃", "register": "회원가입",
        "profile": "프로필", "language": "언어", "search": "검색", "apply": "지원",
        "submit": "제출", "back": "뒤로", "save": "저장", "cancel": "취소",
        "next": "다음", "skip": "건너뛰기", "download": "다운로드", "visit": "방문",
        "loading": "불러오는 중...", "error": "죄송합니다. 문제가 발생했습니다. 다시 시도해 주세요.",
        "empty": "아직 데이터가 없습니다. 곧 업데이트됩니다.",
        "home_badge": "🌏 외국인 유학생을 위한 AI 플랫폼",
        "home_title": "한국 유학을 위한 완벽한 길잡이",
        "home_subtitle": "대학, 장학금, 토픽, 비자, 취업, 생활까지 — 외국인 유학생에게 필요한 모든 것을 AI가 9개 언어로 안내합니다.",
        "kpi_visa": "비자 종류", "kpi_uni": "대학교", "kpi_topik": "토픽 등급", "kpi_jobs": "채용 공고",
        "plan_title": "여정을 계획하세요", "plan_sub": "지원부터 한국 생활까지 4단계로 안내합니다",
        "step1_t": "대학 선택", "step1_d": "나에게 맞는 학교와 전공 찾기",
        "step2_t": "토픽 준비", "step2_d": "필요한 한국어 등급 달성하기",
        "step3_t": "비자 신청", "step3_d": "목표에 맞는 비자 받기",
        "step4_t": "한국 생활 시작", "step4_d": "주거, 은행, 건강 등",
        "feat_uni_t": "대학 검색", "feat_uni_d": "380개 이상의 대학, 전공, 등록금, 장학금을 한곳에서.",
        "feat_job_t": "채용 게시판", "feat_job_d": "외국인 인재를 위한 AI 매칭 채용, 비자 친화 기업.",
        "feat_visa_t": "비자 확인", "feat_visa_d": "D-2, D-4, E-7, F-2, F-5 비자를 AI와 함께 이해하세요.",
        "official_res": "공식 자료", "official_sub": "신뢰할 수 있는 정부 및 대학 링크를 한곳에",
        "res_study": "🎓 유학 자료", "res_immig": "🛂 출입국 안내", "res_tests": "📝 시험 & 취업",
        "job_portals": "인기 채용 사이트", "why_korea": "왜 한국인가", "statistics": "통계", "resources": "자료",
        "overview": "개요", "latest_news": "최신 뉴스",
        "uni_research": "리서치", "uni_school": "학교 정보", "uni_apply": "지원",
        "uni_admission": "입학", "uni_scholar": "장학금",
        "filter_keyword": "키워드", "filter_region": "지역", "filter_gks": "GKS 대상",
        "kw_ph": "대학 또는 전공 검색...", "all": "전체", "yes": "예", "no": "아니오",
        "loc_contact": "📍 위치 및 연락처", "tuition_support": "💰 등록금 및 지원",
        "grad_req": "🎓 졸업 요건", "avail_majors": "📚 개설 전공",
        "btn_website": "🌐 공식 웹사이트", "btn_apply": "📝 지금 지원", "btn_intl": "🌏 국제처",
        "career_cv": "이력서 점검", "career_interview": "모의 면접", "career_res": "자료",
        "cv_title": "AI 이력서 검토", "cv_sub": "이력서를 업로드하고 한국 취업 시장 맞춤 피드백을 받아보세요",
        "cv_upload": "이력서 업로드 (PDF 또는 DOCX)", "cv_target": "지원 직무",
        "cv_target_ph": "예: 소프트웨어 엔지니어", "cv_analyze": "🔍 이력서 분석",
        "cv_score": "종합 점수", "cv_grammar": "문법", "cv_structure": "구성",
        "cv_recs": "추천 사항", "cv_culture": "한국 문화 팁",
        "iv_title": "🎤 AI 모의 면접", "iv_sub": "실전 질문으로 연습하고 AI 피드백을 받으세요",
        "iv_role": "직무", "iv_role_ph": "예: 소프트웨어 엔지니어, 마케팅 매니저",
        "iv_info": "위에 직무를 입력하면 모의 면접이 시작됩니다.",
        "iv_start": "🚀 면접 시작", "iv_answer": "답변", "iv_answer_ph": "답변을 입력하세요 (최소 10자)...",
        "iv_min": "최소 10자 이상 입력해 주세요.", "iv_feedback": "AI 피드백 생성 중...",
        "iv_overall": "종합", "iv_confidence": "자신감", "iv_fit": "문화 적합도",
        "iv_strengths": "강점", "iv_improve": "개선점", "iv_again": "🔄 다시 시도",
        "job_board": "채용 게시판", "job_matches": "내 매칭", "job_portals_tab": "포털",
        "job_search_ph": "직무 또는 회사 검색...", "job_visa_filter": "비자 종류",
        "job_match": "매칭", "job_apply": "지원", "job_upload_cv": "이력서를 업로드하면 최적의 일자리를 찾아드립니다",
        "job_find": "🎯 내 매칭 찾기",
        "topik_schedule": "일정", "topik_register": "접수", "topik_levels": "등급",
        "topik_tips": "학습 팁", "topik_centers": "시험장",
        "topik_fee": "응시료 안내", "topik_reg_btn": "topik.go.kr에서 접수 →",
        "tc_search": "도시로 검색...",
        "visa_req": "요건", "visa_docs": "서류", "visa_proc": "처리 기간",
        "visa_for": "대상", "visa_fee": "수수료", "visa_dur": "체류 기간", "visa_apply": "하이코리아에서 신청 →",
        "life_housing": "주거", "life_transport": "교통", "life_health": "건강",
        "life_banking": "은행", "life_safety": "안전",
        "auth_welcome": "유니패스에 오신 것을 환영합니다", "auth_login_sub": "로그인하고 대학·일자리 저장 및 알림을 받으세요",
        "auth_reg_sub": "무료 계정을 만들고 나만의 여정을 시작하세요",
        "full_name": "이름", "email": "이메일", "password": "비밀번호",
        "confirm_pw": "비밀번호 확인", "lang_pref": "선호 언어",
        "notif_pref": "알림 설정",
        "notif_topik": "토픽 일정 알림", "notif_visa": "비자 정책 업데이트",
        "notif_job": "신규 채용 공고", "notif_uni": "대학 소식", "notif_scholar": "장학금 공지",
        "have_account": "이미 계정이 있으신가요? 로그인", "no_account": "계정이 없으신가요? 가입하기",
        "pw_mismatch": "비밀번호가 일치하지 않습니다.", "invalid_email": "유효한 이메일을 입력해 주세요.",
        "reg_success": "🎉 계정이 생성되었습니다! 환영합니다.", "login_fail": "계정을 찾을 수 없습니다. 먼저 가입해 주세요.",
        "login_success": "다시 오신 것을 환영합니다!", "fill_all": "모든 항목을 입력해 주세요.",
        "my_profile": "내 프로필", "saved_uni": "🎓 저장한 대학", "saved_jobs": "💼 저장한 일자리",
        "edit_notif": "알림 설정 수정", "profile_saved": "설정이 저장되었습니다!",
        "none_yet": "아직 저장된 항목이 없습니다.",
        "chat_title": "UNI 어시스턴트", "chat_sub": "한국 생활에 대해 무엇이든 물어보세요",
        "placeholder": "한국에 대해 무엇이든 물어보세요...", "ask_ai": "💬 AI 어시스턴트에게 질문",
        "subscribe": "이메일 업데이트 받기", "email_label": "이메일", "topics": "관심 주제",
        "sub_success": "✅ 구독 완료! 선택한 언어로 소식을 보내드립니다.",
        "admin_panel": "⚙️ 관리자 패널", "admin_pw": "관리자 비밀번호", "admin_ok": "✅ 인증됨",
        "admin_pdf": "📄 PDF 업로드", "admin_stats": "📊 통계",
    },
    "🇲🇳 Монгол": {
        "app_name": "ЮниПат", "tagline": "Солонгост суралцах, ажиллах, амьдрах AI хөтөч",
        "nav_home": "НҮҮР", "nav_university": "ИХ СУРГУУЛЬ", "nav_career": "КАРЬЕР",
        "nav_job": "АЖИЛ", "nav_topik": "ТОПИК", "nav_visa": "ВИЗ", "nav_life": "АМЬДРАЛ",
        "login": "Нэвтрэх", "logout": "Гарах", "register": "Бүртгүүлэх",
        "profile": "Профайл", "language": "Хэл", "search": "Хайх", "apply": "Бүртгэл өгөх",
        "submit": "Илгээх", "back": "Буцах", "save": "Хадгалах", "cancel": "Цуцлах",
        "next": "Дараах", "skip": "Алгасах", "download": "Татах", "visit": "Зочлох",
        "loading": "Уншиж байна...", "error": "Уучлаарай, алдаа гарлаа. Дахин оролдоно уу.",
        "empty": "Одоогоор мэдээлэл алга. Удахгүй шинэчлэгдэнэ.",
        "home_badge": "🌏 Гадаад оюутнуудад зориулсан AI платформ",
        "home_title": "Солонгост суралцах таны бүрэн зам",
        "home_subtitle": "Их сургууль, тэтгэлэг, ТОПИК, виз, ажил, амьдрал — гадаад оюутанд хэрэгтэй бүхнийг AI-аар, 9 хэл дээр.",
        "kpi_visa": "Визийн төрөл", "kpi_uni": "Их сургууль", "kpi_topik": "ТОПИК түвшин", "kpi_jobs": "Ажлын байр",
        "plan_title": "Аяллаа төлөвлө", "plan_sub": "Бүртгэлээс эхлээд Солонгос дахь амьдрал хүртэл 4 алхам",
        "step1_t": "Их сургууль сонгох", "step1_d": "Өөрт тохирох сургууль, мэргэжлээ ол",
        "step2_t": "ТОПИК бэлдэх", "step2_d": "Шаардлагатай солонгос хэлний түвшинд хүр",
        "step3_t": "Виз авах", "step3_d": "Зорилгодоо тохирох визээ ав",
        "step4_t": "Амьдралаа эхлүүл", "step4_d": "Орон сууц, банк, эрүүл мэнд гэх мэт",
        "feat_uni_t": "Их сургуулийн хайлт", "feat_uni_d": "380+ их сургууль, мэргэжил, төлбөр, тэтгэлгийг нэг дороос.",
        "feat_job_t": "Ажлын самбар", "feat_job_d": "Гадаад мэргэжилтнүүдэд AI-аар тохирсон, виз дэмждэг ажил олгогчид.",
        "feat_visa_t": "Виз шалгагч", "feat_visa_d": "D-2, D-4, E-7, F-2, F-5 визийг AI-ийн тусламжтай ойлго.",
        "official_res": "Албан ёсны эх сурвалж", "official_sub": "Засгийн газар, их сургуулийн найдвартай холбоосууд нэг дор",
        "res_study": "🎓 Сурлагын эх сурвалж", "res_immig": "🛂 Цагаачлалын заавар", "res_tests": "📝 Шалгалт ба Ажил",
        "job_portals": "Алдартай ажлын сайтууд", "why_korea": "Яагаад Солонгос", "statistics": "Статистик", "resources": "Эх сурвалж",
        "overview": "Тойм", "latest_news": "Сүүлийн мэдээ",
        "uni_research": "Судалгаа", "uni_school": "Сургуулийн мэдээлэл", "uni_apply": "Бүртгэл",
        "uni_admission": "Элсэлт", "uni_scholar": "Тэтгэлэг",
        "filter_keyword": "Түлхүүр үг", "filter_region": "Бүс", "filter_gks": "GKS тэнцэх",
        "kw_ph": "Их сургууль эсвэл мэргэжил хайх...", "all": "Бүгд", "yes": "Тийм", "no": "Үгүй",
        "loc_contact": "📍 Байршил ба холбоо барих", "tuition_support": "💰 Төлбөр ба дэмжлэг",
        "grad_req": "🎓 Төгсөлтийн шаардлага", "avail_majors": "📚 Боломжит мэргэжил",
        "btn_website": "🌐 Албан ёсны вэб", "btn_apply": "📝 Одоо бүртгүүлэх", "btn_intl": "🌏 Олон улсын алба",
        "career_cv": "CV шалгах", "career_interview": "Дадлага ярилцлага", "career_res": "Эх сурвалж",
        "cv_title": "AI CV шинжилгээ", "cv_sub": "CV-гээ оруулж, Солонгосын ажлын зах зээлд тохирсон зөвлөгөө аваарай",
        "cv_upload": "CV оруулах (PDF эсвэл DOCX)", "cv_target": "Зорилтот ажлын байр",
        "cv_target_ph": "ж: Программ хангамжийн инженер", "cv_analyze": "🔍 CV-г шинжлэх",
        "cv_score": "Нийт оноо", "cv_grammar": "Хэл зүй", "cv_structure": "Бүтэц",
        "cv_recs": "Зөвлөмж", "cv_culture": "Солонгос соёлын зөвлөгөө",
        "iv_title": "🎤 AI дадлага ярилцлага", "iv_sub": "Бодит асуултаар дадлага хийж AI зөвлөгөө аваарай",
        "iv_role": "Ажлын байр", "iv_role_ph": "ж: Программист, Маркетингийн менежер",
        "iv_info": "Ярилцлага эхлүүлэхийн тулд дээр ажлын байраа оруулна уу.",
        "iv_start": "🚀 Ярилцлага эхлэх", "iv_answer": "Таны хариулт", "iv_answer_ph": "Хариултаа бичнэ үү (хамгийн багадаа 10 тэмдэгт)...",
        "iv_min": "Дор хаяж 10 тэмдэгт бичнэ үү.", "iv_feedback": "AI зөвлөгөө бэлдэж байна...",
        "iv_overall": "Нийт", "iv_confidence": "Өөртөө итгэх", "iv_fit": "Соёлын нийцэл",
        "iv_strengths": "Давуу тал", "iv_improve": "Сайжруулах зүйл", "iv_again": "🔄 Дахин оролдох",
        "job_board": "Ажлын самбар", "job_matches": "Миний тохирол", "job_portals_tab": "Портал",
        "job_search_ph": "Ажил эсвэл компани хайх...", "job_visa_filter": "Визийн төрөл",
        "job_match": "ТОХИРОЛ", "job_apply": "Бүртгүүлэх", "job_upload_cv": "CV-гээ оруулж хамгийн тохирох ажлаа олоорой",
        "job_find": "🎯 Тохиролоо олох",
        "topik_schedule": "Хуваарь", "topik_register": "Бүртгэл", "topik_levels": "Түвшин",
        "topik_tips": "Сурах зөвлөгөө", "topik_centers": "Шалгалтын төв",
        "topik_fee": "Хураамжийн мэдээлэл", "topik_reg_btn": "topik.go.kr дээр бүртгүүлэх →",
        "tc_search": "Хотоор хайх...",
        "visa_req": "Шаардлага", "visa_docs": "Бичиг баримт", "visa_proc": "Боловсруулах хугацаа",
        "visa_for": "Зориулалт", "visa_fee": "Хураамж", "visa_dur": "Хугацаа", "visa_apply": "HiKorea дээр өргөдөл гаргах →",
        "life_housing": "Орон сууц", "life_transport": "Тээвэр", "life_health": "Эрүүл мэнд",
        "life_banking": "Банк", "life_safety": "Аюулгүй байдал",
        "auth_welcome": "ЮниПат-д тавтай морил", "auth_login_sub": "Нэвтэрч их сургууль, ажлаа хадгалж мэдэгдэл аваарай",
        "auth_reg_sub": "Үнэгүй бүртгэл үүсгэж аяллаа хувийн болгоорой",
        "full_name": "Бүтэн нэр", "email": "Имэйл", "password": "Нууц үг",
        "confirm_pw": "Нууц үг баталгаажуулах", "lang_pref": "Хэлний сонголт",
        "notif_pref": "Мэдэгдлийн тохиргоо",
        "notif_topik": "ТОПИК хуваарийн мэдэгдэл", "notif_visa": "Визийн бодлогын шинэчлэл",
        "notif_job": "Шинэ ажлын байр", "notif_uni": "Их сургуулийн мэдээ", "notif_scholar": "Тэтгэлгийн зар",
        "have_account": "Бүртгэлтэй юу? Нэвтрэх", "no_account": "Бүртгэлгүй юу? Үүсгэх",
        "pw_mismatch": "Нууц үг таарахгүй байна.", "invalid_email": "Зөв имэйл оруулна уу.",
        "reg_success": "🎉 Бүртгэл үүслээ! Тавтай морил.", "login_fail": "Бүртгэл олдсонгүй. Эхлээд бүртгүүлнэ үү.",
        "login_success": "Дахин тавтай морил!", "fill_all": "Бүх талбарыг бөглөнө үү.",
        "my_profile": "Миний профайл", "saved_uni": "🎓 Хадгалсан их сургууль", "saved_jobs": "💼 Хадгалсан ажил",
        "edit_notif": "Мэдэгдлийн тохиргоо засах", "profile_saved": "Тохиргоо хадгалагдлаа!",
        "none_yet": "Одоогоор хадгалсан зүйл алга.",
        "chat_title": "UNI Туслах", "chat_sub": "Солонгос дахь амьдралын талаар юу ч асуу",
        "placeholder": "Солонгосын талаар юу ч асуу...", "ask_ai": "💬 AI Туслахаас асуух",
        "subscribe": "Имэйл шинэчлэл авах", "email_label": "Таны имэйл", "topics": "Сонирхсон сэдэв",
        "sub_success": "✅ Бүртгэгдлээ! Таны хэл дээр мэдээ илгээнэ.",
        "admin_panel": "⚙️ Админ самбар", "admin_pw": "Админ нууц үг", "admin_ok": "✅ Зөвшөөрөгдсөн",
        "admin_pdf": "📄 PDF оруулах", "admin_stats": "📊 Статистик",
    },
    "🇯🇵 日本語": {
        "app_name": "ユニパス", "tagline": "韓国での留学・就職・生活のためのAIガイド",
        "nav_home": "ホーム", "nav_university": "大学", "nav_career": "キャリア",
        "nav_job": "求人", "nav_topik": "TOPIK", "nav_visa": "ビザ", "nav_life": "生活",
        "login": "ログイン", "logout": "ログアウト", "register": "新規登録",
        "profile": "プロフィール", "language": "言語", "search": "検索", "apply": "応募",
        "submit": "送信", "back": "戻る", "save": "保存", "cancel": "キャンセル",
        "next": "次へ", "skip": "スキップ", "download": "ダウンロード", "visit": "訪問",
        "loading": "読み込み中...", "error": "申し訳ありません。問題が発生しました。もう一度お試しください。",
        "empty": "まだデータがありません。後ほどご確認ください。",
        "home_badge": "🌏 外国人留学生のためのAIプラットフォーム",
        "home_title": "韓国留学への完全なロードマップ",
        "home_subtitle": "大学、奨学金、TOPIK、ビザ、就職、生活まで — 留学生に必要なすべてをAIが9言語でご案内します。",
        "kpi_visa": "ビザの種類", "kpi_uni": "大学", "kpi_topik": "TOPIKレベル", "kpi_jobs": "求人",
        "plan_title": "あなたの旅を計画する", "plan_sub": "出願から韓国での生活まで4つのステップ",
        "step1_t": "大学を選ぶ", "step1_d": "自分に合った学校と専攻を見つける",
        "step2_t": "TOPIK準備", "step2_d": "必要な韓国語レベルに到達する",
        "step3_t": "ビザ申請", "step3_d": "目標に合ったビザを取得する",
        "step4_t": "韓国生活開始", "step4_d": "住居、銀行、健康など",
        "feat_uni_t": "大学検索", "feat_uni_d": "380以上の大学、専攻、学費、奨学金を一か所で。",
        "feat_job_t": "求人ボード", "feat_job_d": "外国人材向けのAIマッチング求人、ビザ対応の雇用主。",
        "feat_visa_t": "ビザチェッカー", "feat_visa_d": "D-2、D-4、E-7、F-2、F-5ビザをAIで理解。",
        "official_res": "公式リソース", "official_sub": "信頼できる政府・大学リンクを一か所に",
        "res_study": "🎓 学習リソース", "res_immig": "🛂 出入国ガイド", "res_tests": "📝 試験・就職",
        "job_portals": "人気の求人サイト", "why_korea": "なぜ韓国", "statistics": "統計", "resources": "リソース",
        "overview": "概要", "latest_news": "最新ニュース",
        "uni_research": "リサーチ", "uni_school": "学校情報", "uni_apply": "出願",
        "uni_admission": "入学", "uni_scholar": "奨学金",
        "filter_keyword": "キーワード", "filter_region": "地域", "filter_gks": "GKS対象",
        "kw_ph": "大学や専攻を検索...", "all": "すべて", "yes": "はい", "no": "いいえ",
        "loc_contact": "📍 所在地・連絡先", "tuition_support": "💰 学費・支援",
        "grad_req": "🎓 卒業要件", "avail_majors": "📚 利用可能な専攻",
        "btn_website": "🌐 公式サイト", "btn_apply": "📝 今すぐ出願", "btn_intl": "🌏 国際課",
        "career_cv": "履歴書チェック", "career_interview": "模擬面接", "career_res": "リソース",
        "cv_title": "AI履歴書レビュー", "cv_sub": "履歴書をアップロードして韓国の就職市場向けの即時フィードバックを",
        "cv_upload": "履歴書をアップロード (PDFまたはDOCX)", "cv_target": "希望職種",
        "cv_target_ph": "例：ソフトウェアエンジニア", "cv_analyze": "🔍 履歴書を分析",
        "cv_score": "総合スコア", "cv_grammar": "文法", "cv_structure": "構成",
        "cv_recs": "推奨事項", "cv_culture": "韓国文化のヒント",
        "iv_title": "🎤 AI模擬面接", "iv_sub": "リアルな質問で練習しAIフィードバックを受ける",
        "iv_role": "職種", "iv_role_ph": "例：ソフトウェアエンジニア、マーケティングマネージャー",
        "iv_info": "上に職種を入力すると模擬面接が始まります。",
        "iv_start": "🚀 面接開始", "iv_answer": "あなたの回答", "iv_answer_ph": "回答を入力 (10文字以上)...",
        "iv_min": "10文字以上入力してください。", "iv_feedback": "AIフィードバックを生成中...",
        "iv_overall": "総合", "iv_confidence": "自信", "iv_fit": "文化適合",
        "iv_strengths": "強み", "iv_improve": "改善点", "iv_again": "🔄 もう一度",
        "job_board": "求人ボード", "job_matches": "マイマッチ", "job_portals_tab": "ポータル",
        "job_search_ph": "職種または会社を検索...", "job_visa_filter": "ビザの種類",
        "job_match": "マッチ", "job_apply": "応募", "job_upload_cv": "履歴書をアップロードして最適な求人を見つけよう",
        "job_find": "🎯 マッチを探す",
        "topik_schedule": "日程", "topik_register": "登録", "topik_levels": "レベル",
        "topik_tips": "学習のヒント", "topik_centers": "試験会場",
        "topik_fee": "受験料情報", "topik_reg_btn": "topik.go.krで登録 →",
        "tc_search": "都市で検索...",
        "visa_req": "要件", "visa_docs": "書類", "visa_proc": "処理期間",
        "visa_for": "対象", "visa_fee": "手数料", "visa_dur": "滞在期間", "visa_apply": "HiKoreaで申請 →",
        "life_housing": "住居", "life_transport": "交通", "life_health": "健康",
        "life_banking": "銀行", "life_safety": "安全",
        "auth_welcome": "ユニパスへようこそ", "auth_login_sub": "ログインして大学・求人を保存し通知を受け取る",
        "auth_reg_sub": "無料アカウントを作成して旅をパーソナライズ",
        "full_name": "氏名", "email": "メール", "password": "パスワード",
        "confirm_pw": "パスワード確認", "lang_pref": "言語設定",
        "notif_pref": "通知設定",
        "notif_topik": "TOPIK日程アラート", "notif_visa": "ビザ政策更新",
        "notif_job": "新規求人", "notif_uni": "大学ニュース", "notif_scholar": "奨学金のお知らせ",
        "have_account": "アカウントをお持ちですか？ログイン", "no_account": "アカウントがない？作成する",
        "pw_mismatch": "パスワードが一致しません。", "invalid_email": "有効なメールアドレスを入力してください。",
        "reg_success": "🎉 アカウント作成完了！ようこそ。", "login_fail": "アカウントが見つかりません。先に登録してください。",
        "login_success": "おかえりなさい！", "fill_all": "すべての項目を入力してください。",
        "my_profile": "マイプロフィール", "saved_uni": "🎓 保存した大学", "saved_jobs": "💼 保存した求人",
        "edit_notif": "通知設定を編集", "profile_saved": "設定を保存しました！",
        "none_yet": "まだ何も保存されていません。",
        "chat_title": "UNIアシスタント", "chat_sub": "韓国生活について何でも聞いてください",
        "placeholder": "韓国について何でも聞いてください...", "ask_ai": "💬 AIアシスタントに質問",
        "subscribe": "メール更新を受け取る", "email_label": "あなたのメール", "topics": "興味のあるトピック",
        "sub_success": "✅ 登録完了！あなたの言語で更新をお届けします。",
        "admin_panel": "⚙️ 管理パネル", "admin_pw": "管理者パスワード", "admin_ok": "✅ 認証済み",
        "admin_pdf": "📄 PDFアップロード", "admin_stats": "📊 統計",
    },
    "🇨🇳 中文": {
        "app_name": "优途", "tagline": "您在韩国留学、就业、生活的AI向导",
        "nav_home": "首页", "nav_university": "大学", "nav_career": "职业",
        "nav_job": "求职", "nav_topik": "TOPIK", "nav_visa": "签证", "nav_life": "生活",
        "login": "登录", "logout": "退出", "register": "注册",
        "profile": "个人资料", "language": "语言", "search": "搜索", "apply": "申请",
        "submit": "提交", "back": "返回", "save": "保存", "cancel": "取消",
        "next": "下一步", "skip": "跳过", "download": "下载", "visit": "访问",
        "loading": "加载中...", "error": "抱歉，出现了问题。请重试。",
        "empty": "暂无数据，请稍后再来。",
        "home_badge": "🌏 为国际学生打造的AI平台",
        "home_title": "您赴韩留学的完整路径",
        "home_subtitle": "大学、奖学金、TOPIK、签证、求职、生活——国际学生所需的一切，由AI以9种语言提供。",
        "kpi_visa": "签证类型", "kpi_uni": "大学", "kpi_topik": "TOPIK等级", "kpi_jobs": "招聘职位",
        "plan_title": "规划您的旅程", "plan_sub": "从申请到在韩生活的四个简单步骤",
        "step1_t": "选择大学", "step1_d": "找到最适合的学校和专业",
        "step2_t": "备考TOPIK", "step2_d": "达到所需的韩语水平",
        "step3_t": "申请签证", "step3_d": "获得符合目标的签证",
        "step4_t": "开启韩国生活", "step4_d": "住房、银行、医疗等",
        "feat_uni_t": "大学搜索", "feat_uni_d": "380多所大学、专业、学费、奖学金一站搞定。",
        "feat_job_t": "求职平台", "feat_job_d": "为国际人才AI匹配职位，签证友好的雇主。",
        "feat_visa_t": "签证查询", "feat_visa_d": "借助AI了解D-2、D-4、E-7、F-2、F-5签证。",
        "official_res": "官方资源", "official_sub": "可靠的政府和大学链接，尽在一处",
        "res_study": "🎓 学习资源", "res_immig": "🛂 出入境指南", "res_tests": "📝 考试与就业",
        "job_portals": "热门求职网站", "why_korea": "为何选韩国", "statistics": "数据统计", "resources": "资源",
        "overview": "概览", "latest_news": "最新资讯",
        "uni_research": "调研", "uni_school": "学校信息", "uni_apply": "申请",
        "uni_admission": "招生", "uni_scholar": "奖学金",
        "filter_keyword": "关键词", "filter_region": "地区", "filter_gks": "GKS资格",
        "kw_ph": "搜索大学或专业...", "all": "全部", "yes": "是", "no": "否",
        "loc_contact": "📍 位置与联系", "tuition_support": "💰 学费与资助",
        "grad_req": "🎓 毕业要求", "avail_majors": "📚 开设专业",
        "btn_website": "🌐 官方网站", "btn_apply": "📝 立即申请", "btn_intl": "🌏 国际处",
        "career_cv": "简历检查", "career_interview": "模拟面试", "career_res": "资源",
        "cv_title": "AI简历评估", "cv_sub": "上传简历，获取针对韩国就业市场的即时反馈",
        "cv_upload": "上传简历 (PDF或DOCX)", "cv_target": "目标职位",
        "cv_target_ph": "例如：软件工程师", "cv_analyze": "🔍 分析我的简历",
        "cv_score": "综合评分", "cv_grammar": "语法", "cv_structure": "结构",
        "cv_recs": "建议", "cv_culture": "韩国文化提示",
        "iv_title": "🎤 AI模拟面试", "iv_sub": "用真实问题练习并获得AI反馈",
        "iv_role": "职位", "iv_role_ph": "例如：软件工程师、市场经理",
        "iv_info": "在上方输入职位即可开始模拟面试。",
        "iv_start": "🚀 开始面试", "iv_answer": "您的回答", "iv_answer_ph": "输入您的回答 (至少10个字符)...",
        "iv_min": "请至少输入10个字符。", "iv_feedback": "正在生成AI反馈...",
        "iv_overall": "综合", "iv_confidence": "自信度", "iv_fit": "文化契合",
        "iv_strengths": "优势", "iv_improve": "改进", "iv_again": "🔄 再试一次",
        "job_board": "求职平台", "job_matches": "我的匹配", "job_portals_tab": "门户",
        "job_search_ph": "搜索职位或公司...", "job_visa_filter": "签证类型",
        "job_match": "匹配", "job_apply": "申请", "job_upload_cv": "上传简历，找到最匹配的工作",
        "job_find": "🎯 查找我的匹配",
        "topik_schedule": "日程", "topik_register": "报名", "topik_levels": "等级",
        "topik_tips": "学习技巧", "topik_centers": "考点",
        "topik_fee": "费用信息", "topik_reg_btn": "在topik.go.kr报名 →",
        "tc_search": "按城市搜索...",
        "visa_req": "要求", "visa_docs": "材料", "visa_proc": "办理时间",
        "visa_for": "适用对象", "visa_fee": "费用", "visa_dur": "停留期限", "visa_apply": "在HiKorea申请 →",
        "life_housing": "住房", "life_transport": "交通", "life_health": "医疗",
        "life_banking": "银行", "life_safety": "安全",
        "auth_welcome": "欢迎来到优途", "auth_login_sub": "登录以收藏大学、职位并接收提醒",
        "auth_reg_sub": "创建免费账户，定制您的旅程",
        "full_name": "姓名", "email": "邮箱", "password": "密码",
        "confirm_pw": "确认密码", "lang_pref": "语言偏好",
        "notif_pref": "通知偏好",
        "notif_topik": "TOPIK日程提醒", "notif_visa": "签证政策更新",
        "notif_job": "新职位发布", "notif_uni": "大学资讯", "notif_scholar": "奖学金公告",
        "have_account": "已有账户？登录", "no_account": "没有账户？立即创建",
        "pw_mismatch": "两次密码不一致。", "invalid_email": "请输入有效的邮箱地址。",
        "reg_success": "🎉 账户已创建！欢迎加入。", "login_fail": "未找到账户，请先注册。",
        "login_success": "欢迎回来！", "fill_all": "请填写所有字段。",
        "my_profile": "我的资料", "saved_uni": "🎓 收藏的大学", "saved_jobs": "💼 收藏的职位",
        "edit_notif": "编辑通知偏好", "profile_saved": "偏好已保存！",
        "none_yet": "暂无收藏。",
        "chat_title": "UNI助手", "chat_sub": "韩国生活相关问题随时问我",
        "placeholder": "关于韩国，问我任何问题...", "ask_ai": "💬 询问AI助手",
        "subscribe": "获取邮件更新", "email_label": "您的邮箱", "topics": "感兴趣的主题",
        "sub_success": "✅ 订阅成功！我们将用您的语言发送更新。",
        "admin_panel": "⚙️ 管理面板", "admin_pw": "管理员密码", "admin_ok": "✅ 已授权",
        "admin_pdf": "📄 PDF上传", "admin_stats": "📊 统计",
    },
    "🇻🇳 Tiếng Việt": {
        "app_name": "UniPath", "tagline": "Hướng dẫn AI để du học, làm việc & sinh sống tại Hàn Quốc",
        "nav_home": "TRANG CHỦ", "nav_university": "ĐẠI HỌC", "nav_career": "SỰ NGHIỆP",
        "nav_job": "VIỆC LÀM", "nav_topik": "TOPIK", "nav_visa": "VISA", "nav_life": "ĐỜI SỐNG",
        "login": "Đăng nhập", "logout": "Đăng xuất", "register": "Đăng ký",
        "profile": "Hồ sơ", "language": "Ngôn ngữ", "search": "Tìm kiếm", "apply": "Nộp đơn",
        "submit": "Gửi", "back": "Quay lại", "save": "Lưu", "cancel": "Hủy",
        "next": "Tiếp", "skip": "Bỏ qua", "download": "Tải xuống", "visit": "Truy cập",
        "loading": "Đang tải...", "error": "Xin lỗi, đã xảy ra lỗi. Vui lòng thử lại.",
        "empty": "Chưa có dữ liệu. Vui lòng quay lại sau.",
        "home_badge": "🌏 Nền tảng AI cho sinh viên quốc tế",
        "home_title": "Con đường hoàn chỉnh để du học Hàn Quốc",
        "home_subtitle": "Đại học, học bổng, TOPIK, visa, việc làm và đời sống — mọi thứ sinh viên quốc tế cần, hỗ trợ bởi AI với 9 ngôn ngữ.",
        "kpi_visa": "Loại visa", "kpi_uni": "Đại học", "kpi_topik": "Cấp độ TOPIK", "kpi_jobs": "Tin tuyển dụng",
        "plan_title": "Lên kế hoạch hành trình", "plan_sub": "Bốn bước đơn giản từ nộp đơn đến cuộc sống mới tại Hàn",
        "step1_t": "Chọn trường", "step1_d": "Tìm trường và ngành phù hợp nhất",
        "step2_t": "Ôn TOPIK", "step2_d": "Đạt trình độ tiếng Hàn cần thiết",
        "step3_t": "Xin visa", "step3_d": "Nhận visa phù hợp với mục tiêu",
        "step4_t": "Bắt đầu sống tại Hàn", "step4_d": "Nhà ở, ngân hàng, y tế và hơn thế",
        "feat_uni_t": "Tìm trường", "feat_uni_d": "Khám phá hơn 380 trường, ngành, học phí và học bổng ở một nơi.",
        "feat_job_t": "Bảng việc làm", "feat_job_d": "Việc làm AI gợi ý cho nhân tài quốc tế, nhà tuyển dụng thân thiện visa.",
        "feat_visa_t": "Kiểm tra visa", "feat_visa_d": "Hiểu visa D-2, D-4, E-7, F-2, F-5 với hướng dẫn AI.",
        "official_res": "Nguồn chính thức", "official_sub": "Liên kết chính phủ và đại học đáng tin cậy ở một nơi",
        "res_study": "🎓 Nguồn học tập", "res_immig": "🛂 Hướng dẫn nhập cư", "res_tests": "📝 Thi cử & Việc làm",
        "job_portals": "Trang việc làm phổ biến", "why_korea": "Tại sao Hàn Quốc", "statistics": "Thống kê", "resources": "Tài nguyên",
        "overview": "Tổng quan", "latest_news": "Tin mới nhất",
        "uni_research": "Nghiên cứu", "uni_school": "Thông tin trường", "uni_apply": "Nộp đơn",
        "uni_admission": "Tuyển sinh", "uni_scholar": "Học bổng",
        "filter_keyword": "Từ khóa", "filter_region": "Khu vực", "filter_gks": "Đủ điều kiện GKS",
        "kw_ph": "Tìm trường hoặc ngành...", "all": "Tất cả", "yes": "Có", "no": "Không",
        "loc_contact": "📍 Vị trí & Liên hệ", "tuition_support": "💰 Học phí & Hỗ trợ",
        "grad_req": "🎓 Yêu cầu tốt nghiệp", "avail_majors": "📚 Ngành đào tạo",
        "btn_website": "🌐 Trang chính thức", "btn_apply": "📝 Nộp đơn ngay", "btn_intl": "🌏 Phòng quốc tế",
        "career_cv": "Kiểm tra CV", "career_interview": "Phỏng vấn thử", "career_res": "Tài nguyên",
        "cv_title": "AI đánh giá CV", "cv_sub": "Tải CV lên và nhận phản hồi tức thì cho thị trường việc làm Hàn Quốc",
        "cv_upload": "Tải CV (PDF hoặc DOCX)", "cv_target": "Vị trí mong muốn",
        "cv_target_ph": "ví dụ: Kỹ sư phần mềm", "cv_analyze": "🔍 Phân tích CV",
        "cv_score": "Điểm tổng", "cv_grammar": "Ngữ pháp", "cv_structure": "Cấu trúc",
        "cv_recs": "Khuyến nghị", "cv_culture": "Mẹo văn hóa Hàn",
        "iv_title": "🎤 Phỏng vấn thử AI", "iv_sub": "Luyện tập với câu hỏi thực tế và nhận phản hồi AI",
        "iv_role": "Vị trí", "iv_role_ph": "ví dụ: Kỹ sư phần mềm, Quản lý marketing",
        "iv_info": "Nhập vị trí ở trên để bắt đầu phỏng vấn thử.",
        "iv_start": "🚀 Bắt đầu phỏng vấn", "iv_answer": "Câu trả lời", "iv_answer_ph": "Nhập câu trả lời (ít nhất 10 ký tự)...",
        "iv_min": "Vui lòng viết ít nhất 10 ký tự.", "iv_feedback": "Đang tạo phản hồi AI...",
        "iv_overall": "Tổng thể", "iv_confidence": "Tự tin", "iv_fit": "Phù hợp văn hóa",
        "iv_strengths": "Điểm mạnh", "iv_improve": "Cần cải thiện", "iv_again": "🔄 Thử lại",
        "job_board": "Bảng việc làm", "job_matches": "Phù hợp với tôi", "job_portals_tab": "Cổng",
        "job_search_ph": "Tìm vị trí hoặc công ty...", "job_visa_filter": "Loại visa",
        "job_match": "PHÙ HỢP", "job_apply": "Nộp đơn", "job_upload_cv": "Tải CV để tìm việc phù hợp nhất",
        "job_find": "🎯 Tìm việc phù hợp",
        "topik_schedule": "Lịch thi", "topik_register": "Đăng ký", "topik_levels": "Cấp độ",
        "topik_tips": "Mẹo học", "topik_centers": "Điểm thi",
        "topik_fee": "Thông tin lệ phí", "topik_reg_btn": "Đăng ký tại topik.go.kr →",
        "tc_search": "Tìm theo thành phố...",
        "visa_req": "Yêu cầu", "visa_docs": "Giấy tờ", "visa_proc": "Thời gian xử lý",
        "visa_for": "Dành cho", "visa_fee": "Lệ phí", "visa_dur": "Thời hạn", "visa_apply": "Nộp trên HiKorea →",
        "life_housing": "Nhà ở", "life_transport": "Giao thông", "life_health": "Sức khỏe",
        "life_banking": "Ngân hàng", "life_safety": "An toàn",
        "auth_welcome": "Chào mừng đến UniPath", "auth_login_sub": "Đăng nhập để lưu trường, việc làm và nhận thông báo",
        "auth_reg_sub": "Tạo tài khoản miễn phí và cá nhân hóa hành trình",
        "full_name": "Họ tên", "email": "Email", "password": "Mật khẩu",
        "confirm_pw": "Xác nhận mật khẩu", "lang_pref": "Ngôn ngữ ưa thích",
        "notif_pref": "Tùy chọn thông báo",
        "notif_topik": "Thông báo lịch TOPIK", "notif_visa": "Cập nhật chính sách visa",
        "notif_job": "Việc làm mới", "notif_uni": "Tin tức đại học", "notif_scholar": "Thông báo học bổng",
        "have_account": "Đã có tài khoản? Đăng nhập", "no_account": "Chưa có tài khoản? Tạo ngay",
        "pw_mismatch": "Mật khẩu không khớp.", "invalid_email": "Vui lòng nhập email hợp lệ.",
        "reg_success": "🎉 Đã tạo tài khoản! Chào mừng bạn.", "login_fail": "Không tìm thấy tài khoản. Vui lòng đăng ký trước.",
        "login_success": "Chào mừng trở lại!", "fill_all": "Vui lòng điền tất cả các trường.",
        "my_profile": "Hồ sơ của tôi", "saved_uni": "🎓 Trường đã lưu", "saved_jobs": "💼 Việc đã lưu",
        "edit_notif": "Chỉnh sửa thông báo", "profile_saved": "Đã lưu tùy chọn!",
        "none_yet": "Chưa lưu gì cả.",
        "chat_title": "Trợ lý UNI", "chat_sub": "Hỏi tôi bất cứ điều gì về cuộc sống tại Hàn",
        "placeholder": "Hỏi tôi bất cứ điều gì về Hàn Quốc...", "ask_ai": "💬 Hỏi trợ lý AI",
        "subscribe": "Nhận cập nhật qua email", "email_label": "Email của bạn", "topics": "Chủ đề quan tâm",
        "sub_success": "✅ Đã đăng ký! Bạn sẽ nhận cập nhật bằng ngôn ngữ của mình.",
        "admin_panel": "⚙️ Bảng quản trị", "admin_pw": "Mật khẩu quản trị", "admin_ok": "✅ Đã xác thực",
        "admin_pdf": "📄 Tải PDF", "admin_stats": "📊 Thống kê",
    },
    "🇹🇭 ภาษาไทย": {
        "app_name": "UniPath", "tagline": "ไกด์ AI สำหรับเรียน ทำงาน และใช้ชีวิตในเกาหลี",
        "nav_home": "หน้าแรก", "nav_university": "มหาวิทยาลัย", "nav_career": "อาชีพ",
        "nav_job": "งาน", "nav_topik": "TOPIK", "nav_visa": "วีซ่า", "nav_life": "ชีวิต",
        "login": "เข้าสู่ระบบ", "logout": "ออกจากระบบ", "register": "สมัครสมาชิก",
        "profile": "โปรไฟล์", "language": "ภาษา", "search": "ค้นหา", "apply": "สมัคร",
        "submit": "ส่ง", "back": "กลับ", "save": "บันทึก", "cancel": "ยกเลิก",
        "next": "ถัดไป", "skip": "ข้าม", "download": "ดาวน์โหลด", "visit": "เยี่ยมชม",
        "loading": "กำลังโหลด...", "error": "ขออภัย เกิดข้อผิดพลาด กรุณาลองใหม่",
        "empty": "ยังไม่มีข้อมูล โปรดกลับมาใหม่ภายหลัง",
        "home_badge": "🌏 แพลตฟอร์ม AI สำหรับนักศึกษาต่างชาติ",
        "home_title": "เส้นทางสมบูรณ์สู่การเรียนต่อเกาหลี",
        "home_subtitle": "มหาวิทยาลัย ทุน TOPIK วีซ่า งาน และการใช้ชีวิต — ทุกอย่างที่นักศึกษาต่างชาติต้องการ ขับเคลื่อนด้วย AI ใน 9 ภาษา",
        "kpi_visa": "ประเภทวีซ่า", "kpi_uni": "มหาวิทยาลัย", "kpi_topik": "ระดับ TOPIK", "kpi_jobs": "ตำแหน่งงาน",
        "plan_title": "วางแผนการเดินทางของคุณ", "plan_sub": "สี่ขั้นตอนง่ายๆ ตั้งแต่สมัครจนถึงใช้ชีวิตในเกาหลี",
        "step1_t": "เลือกมหาวิทยาลัย", "step1_d": "หาโรงเรียนและสาขาที่เหมาะกับคุณ",
        "step2_t": "เตรียม TOPIK", "step2_d": "ไปให้ถึงระดับภาษาเกาหลีที่ต้องการ",
        "step3_t": "ขอวีซ่า", "step3_d": "รับวีซ่าที่ตรงกับเป้าหมาย",
        "step4_t": "เริ่มชีวิตในเกาหลี", "step4_d": "ที่พัก ธนาคาร สุขภาพ และอื่นๆ",
        "feat_uni_t": "ค้นหามหาวิทยาลัย", "feat_uni_d": "สำรวจมหาวิทยาลัยกว่า 380 แห่ง สาขา ค่าเล่าเรียน และทุนในที่เดียว",
        "feat_job_t": "บอร์ดงาน", "feat_job_d": "งานจับคู่ด้วย AI สำหรับผู้มีความสามารถต่างชาติ นายจ้างที่เป็นมิตรกับวีซ่า",
        "feat_visa_t": "ตรวจสอบวีซ่า", "feat_visa_d": "เข้าใจวีซ่า D-2, D-4, E-7, F-2, F-5 ด้วยคำแนะนำจาก AI",
        "official_res": "แหล่งข้อมูลทางการ", "official_sub": "ลิงก์รัฐบาลและมหาวิทยาลัยที่เชื่อถือได้ในที่เดียว",
        "res_study": "🎓 แหล่งข้อมูลการเรียน", "res_immig": "🛂 คู่มือตรวจคนเข้าเมือง", "res_tests": "📝 สอบ & งาน",
        "job_portals": "เว็บหางานยอดนิยม", "why_korea": "ทำไมต้องเกาหลี", "statistics": "สถิติ", "resources": "ทรัพยากร",
        "overview": "ภาพรวม", "latest_news": "ข่าวล่าสุด",
        "uni_research": "ค้นคว้า", "uni_school": "ข้อมูลโรงเรียน", "uni_apply": "สมัคร",
        "uni_admission": "การรับเข้า", "uni_scholar": "ทุนการศึกษา",
        "filter_keyword": "คำค้น", "filter_region": "ภูมิภาค", "filter_gks": "มีสิทธิ์ GKS",
        "kw_ph": "ค้นหามหาวิทยาลัยหรือสาขา...", "all": "ทั้งหมด", "yes": "ใช่", "no": "ไม่",
        "loc_contact": "📍 ที่ตั้ง & ติดต่อ", "tuition_support": "💰 ค่าเล่าเรียน & การสนับสนุน",
        "grad_req": "🎓 เงื่อนไขการจบ", "avail_majors": "📚 สาขาที่เปิดสอน",
        "btn_website": "🌐 เว็บไซต์ทางการ", "btn_apply": "📝 สมัครเลย", "btn_intl": "🌏 สำนักงานนานาชาติ",
        "career_cv": "ตรวจ CV", "career_interview": "สัมภาษณ์จำลอง", "career_res": "ทรัพยากร",
        "cv_title": "AI ตรวจ CV", "cv_sub": "อัปโหลดเรซูเม่และรับฟีดแบ็กทันทีสำหรับตลาดงานเกาหลี",
        "cv_upload": "อัปโหลด CV (PDF หรือ DOCX)", "cv_target": "ตำแหน่งเป้าหมาย",
        "cv_target_ph": "เช่น วิศวกรซอฟต์แวร์", "cv_analyze": "🔍 วิเคราะห์ CV",
        "cv_score": "คะแนนรวม", "cv_grammar": "ไวยากรณ์", "cv_structure": "โครงสร้าง",
        "cv_recs": "คำแนะนำ", "cv_culture": "เคล็ดลับวัฒนธรรมเกาหลี",
        "iv_title": "🎤 สัมภาษณ์จำลอง AI", "iv_sub": "ฝึกกับคำถามจริงและรับฟีดแบ็กจาก AI",
        "iv_role": "ตำแหน่งงาน", "iv_role_ph": "เช่น วิศวกรซอฟต์แวร์ ผู้จัดการการตลาด",
        "iv_info": "กรอกตำแหน่งงานด้านบนเพื่อเริ่มสัมภาษณ์จำลอง",
        "iv_start": "🚀 เริ่มสัมภาษณ์", "iv_answer": "คำตอบของคุณ", "iv_answer_ph": "พิมพ์คำตอบ (อย่างน้อย 10 ตัวอักษร)...",
        "iv_min": "กรุณาเขียนอย่างน้อย 10 ตัวอักษร", "iv_feedback": "กำลังสร้างฟีดแบ็ก AI...",
        "iv_overall": "รวม", "iv_confidence": "ความมั่นใจ", "iv_fit": "ความเข้ากับวัฒนธรรม",
        "iv_strengths": "จุดแข็ง", "iv_improve": "จุดที่ควรปรับปรุง", "iv_again": "🔄 ลองอีกครั้ง",
        "job_board": "บอร์ดงาน", "job_matches": "งานที่ตรงกับฉัน", "job_portals_tab": "พอร์ทัล",
        "job_search_ph": "ค้นหาตำแหน่งหรือบริษัท...", "job_visa_filter": "ประเภทวีซ่า",
        "job_match": "ตรงกัน", "job_apply": "สมัคร", "job_upload_cv": "อัปโหลด CV เพื่อหางานที่ตรงที่สุด",
        "job_find": "🎯 หางานที่ตรงกัน",
        "topik_schedule": "กำหนดการ", "topik_register": "ลงทะเบียน", "topik_levels": "ระดับ",
        "topik_tips": "เคล็ดลับการเรียน", "topik_centers": "ศูนย์สอบ",
        "topik_fee": "ข้อมูลค่าธรรมเนียม", "topik_reg_btn": "ลงทะเบียนที่ topik.go.kr →",
        "tc_search": "ค้นหาตามเมือง...",
        "visa_req": "เงื่อนไข", "visa_docs": "เอกสาร", "visa_proc": "ระยะเวลาดำเนินการ",
        "visa_for": "สำหรับ", "visa_fee": "ค่าธรรมเนียม", "visa_dur": "ระยะเวลา", "visa_apply": "สมัครที่ HiKorea →",
        "life_housing": "ที่พัก", "life_transport": "การเดินทาง", "life_health": "สุขภาพ",
        "life_banking": "ธนาคาร", "life_safety": "ความปลอดภัย",
        "auth_welcome": "ยินดีต้อนรับสู่ UniPath", "auth_login_sub": "เข้าสู่ระบบเพื่อบันทึกมหาวิทยาลัย งาน และรับการแจ้งเตือน",
        "auth_reg_sub": "สร้างบัญชีฟรีและปรับแต่งการเดินทางของคุณ",
        "full_name": "ชื่อเต็ม", "email": "อีเมล", "password": "รหัสผ่าน",
        "confirm_pw": "ยืนยันรหัสผ่าน", "lang_pref": "ภาษาที่ต้องการ",
        "notif_pref": "การตั้งค่าแจ้งเตือน",
        "notif_topik": "แจ้งเตือนกำหนดการ TOPIK", "notif_visa": "อัปเดตนโยบายวีซ่า",
        "notif_job": "งานใหม่", "notif_uni": "ข่าวมหาวิทยาลัย", "notif_scholar": "ประกาศทุน",
        "have_account": "มีบัญชีอยู่แล้ว? เข้าสู่ระบบ", "no_account": "ยังไม่มีบัญชี? สร้างเลย",
        "pw_mismatch": "รหัสผ่านไม่ตรงกัน", "invalid_email": "กรุณากรอกอีเมลที่ถูกต้อง",
        "reg_success": "🎉 สร้างบัญชีแล้ว! ยินดีต้อนรับ", "login_fail": "ไม่พบบัญชี กรุณาสมัครก่อน",
        "login_success": "ยินดีต้อนรับกลับมา!", "fill_all": "กรุณากรอกทุกช่อง",
        "my_profile": "โปรไฟล์ของฉัน", "saved_uni": "🎓 มหาวิทยาลัยที่บันทึก", "saved_jobs": "💼 งานที่บันทึก",
        "edit_notif": "แก้ไขการแจ้งเตือน", "profile_saved": "บันทึกการตั้งค่าแล้ว!",
        "none_yet": "ยังไม่มีการบันทึก",
        "chat_title": "ผู้ช่วย UNI", "chat_sub": "ถามฉันได้ทุกเรื่องเกี่ยวกับชีวิตในเกาหลี",
        "placeholder": "ถามฉันได้ทุกเรื่องเกี่ยวกับเกาหลี...", "ask_ai": "💬 ถามผู้ช่วย AI",
        "subscribe": "รับอัปเดตทางอีเมล", "email_label": "อีเมลของคุณ", "topics": "หัวข้อที่สนใจ",
        "sub_success": "✅ สมัครแล้ว! คุณจะได้รับอัปเดตในภาษาของคุณ",
        "admin_panel": "⚙️ แผงผู้ดูแล", "admin_pw": "รหัสผ่านผู้ดูแล", "admin_ok": "✅ ได้รับอนุญาต",
        "admin_pdf": "📄 อัปโหลด PDF", "admin_stats": "📊 สถิติ",
    },
    "🇲🇾 Bahasa Melayu": {
        "app_name": "UniPath", "tagline": "Panduan AI untuk belajar, bekerja & tinggal di Korea",
        "nav_home": "UTAMA", "nav_university": "UNIVERSITI", "nav_career": "KERJAYA",
        "nav_job": "KERJA", "nav_topik": "TOPIK", "nav_visa": "VISA", "nav_life": "KEHIDUPAN",
        "login": "Log Masuk", "logout": "Log Keluar", "register": "Daftar",
        "profile": "Profil", "language": "Bahasa", "search": "Cari", "apply": "Mohon",
        "submit": "Hantar", "back": "Kembali", "save": "Simpan", "cancel": "Batal",
        "next": "Seterusnya", "skip": "Langkau", "download": "Muat turun", "visit": "Lawati",
        "loading": "Memuatkan...", "error": "Maaf, berlaku ralat. Sila cuba lagi.",
        "empty": "Tiada data lagi. Sila semak semula nanti.",
        "home_badge": "🌏 Platform berkuasa AI untuk pelajar antarabangsa",
        "home_title": "Laluan lengkap anda untuk belajar di Korea Selatan",
        "home_subtitle": "Universiti, biasiswa, TOPIK, visa, kerja dan kehidupan harian — semua yang diperlukan pelajar antarabangsa, dikuasakan AI dalam 9 bahasa.",
        "kpi_visa": "Jenis Visa", "kpi_uni": "Universiti", "kpi_topik": "Tahap TOPIK", "kpi_jobs": "Jawatan Kosong",
        "plan_title": "Rancang Perjalanan Anda", "plan_sub": "Empat langkah mudah dari permohonan ke kehidupan di Korea",
        "step1_t": "Pilih Universiti", "step1_d": "Cari sekolah dan jurusan terbaik",
        "step2_t": "Sedia TOPIK", "step2_d": "Capai tahap bahasa Korea yang diperlukan",
        "step3_t": "Mohon Visa", "step3_d": "Dapatkan visa yang sesuai dengan matlamat",
        "step4_t": "Mula Hidup di Korea", "step4_d": "Penginapan, perbankan, kesihatan & lagi",
        "feat_uni_t": "Carian Universiti", "feat_uni_d": "Terokai 380+ universiti, jurusan, yuran dan biasiswa di satu tempat.",
        "feat_job_t": "Papan Kerja", "feat_job_d": "Kerja dipadankan AI untuk bakat antarabangsa, majikan mesra visa.",
        "feat_visa_t": "Penyemak Visa", "feat_visa_d": "Fahami visa D-2, D-4, E-7, F-2, F-5 dengan panduan AI.",
        "official_res": "Sumber Rasmi", "official_sub": "Pautan kerajaan dan universiti dipercayai di satu tempat",
        "res_study": "🎓 Sumber Pengajian", "res_immig": "🛂 Panduan Imigresen", "res_tests": "📝 Peperiksaan & Pekerjaan",
        "job_portals": "Portal Kerja Popular", "why_korea": "Mengapa Korea", "statistics": "Statistik", "resources": "Sumber",
        "overview": "Gambaran Keseluruhan", "latest_news": "Berita Terkini",
        "uni_research": "Penyelidikan", "uni_school": "Maklumat Sekolah", "uni_apply": "Mohon",
        "uni_admission": "Kemasukan", "uni_scholar": "Biasiswa",
        "filter_keyword": "Kata kunci", "filter_region": "Wilayah", "filter_gks": "Layak GKS",
        "kw_ph": "Cari universiti atau jurusan...", "all": "Semua", "yes": "Ya", "no": "Tidak",
        "loc_contact": "📍 Lokasi & Hubungi", "tuition_support": "💰 Yuran & Sokongan",
        "grad_req": "🎓 Syarat Tamat Pengajian", "avail_majors": "📚 Jurusan Tersedia",
        "btn_website": "🌐 Laman Rasmi", "btn_apply": "📝 Mohon Sekarang", "btn_intl": "🌏 Pejabat Antarabangsa",
        "career_cv": "Semak CV", "career_interview": "Temu Duga Olok", "career_res": "Sumber",
        "cv_title": "Semakan CV AI", "cv_sub": "Muat naik resume dan dapatkan maklum balas segera untuk pasaran kerja Korea",
        "cv_upload": "Muat naik CV (PDF atau DOCX)", "cv_target": "Jawatan disasarkan",
        "cv_target_ph": "cth: Jurutera Perisian", "cv_analyze": "🔍 Analisis CV",
        "cv_score": "Skor Keseluruhan", "cv_grammar": "Tatabahasa", "cv_structure": "Struktur",
        "cv_recs": "Cadangan", "cv_culture": "Tip Budaya Korea",
        "iv_title": "🎤 Temu Duga Olok AI", "iv_sub": "Berlatih dengan soalan realistik dan dapatkan maklum balas AI",
        "iv_role": "Jawatan", "iv_role_ph": "cth: Jurutera Perisian, Pengurus Pemasaran",
        "iv_info": "Masukkan jawatan di atas untuk mula temu duga olok.",
        "iv_start": "🚀 Mula Temu Duga", "iv_answer": "Jawapan anda", "iv_answer_ph": "Taip jawapan (sekurang-kurangnya 10 aksara)...",
        "iv_min": "Sila tulis sekurang-kurangnya 10 aksara.", "iv_feedback": "Menjana maklum balas AI...",
        "iv_overall": "Keseluruhan", "iv_confidence": "Keyakinan", "iv_fit": "Kesesuaian Budaya",
        "iv_strengths": "Kekuatan", "iv_improve": "Penambahbaikan", "iv_again": "🔄 Cuba Lagi",
        "job_board": "Papan Kerja", "job_matches": "Padanan Saya", "job_portals_tab": "Portal",
        "job_search_ph": "Cari jawatan atau syarikat...", "job_visa_filter": "Jenis Visa",
        "job_match": "PADANAN", "job_apply": "Mohon", "job_upload_cv": "Muat naik CV untuk cari kerja paling sesuai",
        "job_find": "🎯 Cari Padanan Saya",
        "topik_schedule": "Jadual", "topik_register": "Daftar", "topik_levels": "Tahap",
        "topik_tips": "Tip Belajar", "topik_centers": "Pusat Peperiksaan",
        "topik_fee": "Maklumat Yuran", "topik_reg_btn": "Daftar di topik.go.kr →",
        "tc_search": "Cari mengikut bandar...",
        "visa_req": "Syarat", "visa_docs": "Dokumen", "visa_proc": "Masa Pemprosesan",
        "visa_for": "Untuk", "visa_fee": "Yuran", "visa_dur": "Tempoh", "visa_apply": "Mohon di HiKorea →",
        "life_housing": "Penginapan", "life_transport": "Pengangkutan", "life_health": "Kesihatan",
        "life_banking": "Perbankan", "life_safety": "Keselamatan",
        "auth_welcome": "Selamat datang ke UniPath", "auth_login_sub": "Log masuk untuk simpan universiti, kerja dan terima makluman",
        "auth_reg_sub": "Cipta akaun percuma dan peribadikan perjalanan anda",
        "full_name": "Nama penuh", "email": "E-mel", "password": "Kata laluan",
        "confirm_pw": "Sahkan kata laluan", "lang_pref": "Pilihan bahasa",
        "notif_pref": "Tetapan pemberitahuan",
        "notif_topik": "Makluman jadual TOPIK", "notif_visa": "Kemas kini dasar visa",
        "notif_job": "Jawatan baharu", "notif_uni": "Berita universiti", "notif_scholar": "Pengumuman biasiswa",
        "have_account": "Sudah ada akaun? Log Masuk", "no_account": "Tiada akaun? Cipta satu",
        "pw_mismatch": "Kata laluan tidak sepadan.", "invalid_email": "Sila masukkan e-mel yang sah.",
        "reg_success": "🎉 Akaun dicipta! Selamat datang.", "login_fail": "Akaun tidak ditemui. Sila daftar dahulu.",
        "login_success": "Selamat kembali!", "fill_all": "Sila isi semua medan.",
        "my_profile": "Profil Saya", "saved_uni": "🎓 Universiti Disimpan", "saved_jobs": "💼 Kerja Disimpan",
        "edit_notif": "Edit pemberitahuan", "profile_saved": "Tetapan disimpan!",
        "none_yet": "Tiada apa-apa disimpan lagi.",
        "chat_title": "Pembantu UNI", "chat_sub": "Tanya saya apa sahaja tentang kehidupan di Korea",
        "placeholder": "Tanya saya apa sahaja tentang Korea...", "ask_ai": "💬 Tanya Pembantu AI",
        "subscribe": "Dapatkan Kemas Kini E-mel", "email_label": "E-mel anda", "topics": "Topik diminati",
        "sub_success": "✅ Dilanggan! Anda akan terima kemas kini dalam bahasa anda.",
        "admin_panel": "⚙️ Panel Admin", "admin_pw": "Kata Laluan Admin", "admin_ok": "✅ Dibenarkan",
        "admin_pdf": "📄 Muat Naik PDF", "admin_stats": "📊 Statistik",
    },
    "🇷🇺 Русский": {
        "app_name": "UniPath", "tagline": "AI-гид по учёбе, работе и жизни в Корее",
        "nav_home": "ГЛАВНАЯ", "nav_university": "УНИВЕРСИТЕТ", "nav_career": "КАРЬЕРА",
        "nav_job": "РАБОТА", "nav_topik": "TOPIK", "nav_visa": "ВИЗА", "nav_life": "ЖИЗНЬ",
        "login": "Войти", "logout": "Выйти", "register": "Регистрация",
        "profile": "Профиль", "language": "Язык", "search": "Поиск", "apply": "Подать заявку",
        "submit": "Отправить", "back": "Назад", "save": "Сохранить", "cancel": "Отмена",
        "next": "Далее", "skip": "Пропустить", "download": "Скачать", "visit": "Перейти",
        "loading": "Загрузка...", "error": "Извините, произошла ошибка. Попробуйте снова.",
        "empty": "Данных пока нет. Загляните позже.",
        "home_badge": "🌏 AI-платформа для иностранных студентов",
        "home_title": "Ваш полный путь к учёбе в Южной Корее",
        "home_subtitle": "Университеты, стипендии, TOPIK, визы, работа и быт — всё, что нужно иностранному студенту, на базе AI и на 9 языках.",
        "kpi_visa": "Типы виз", "kpi_uni": "Университеты", "kpi_topik": "Уровень TOPIK", "kpi_jobs": "Вакансии",
        "plan_title": "Спланируйте свой путь", "plan_sub": "Четыре простых шага от подачи заявки до жизни в Корее",
        "step1_t": "Выбрать университет", "step1_d": "Найдите подходящий вуз и специальность",
        "step2_t": "Подготовить TOPIK", "step2_d": "Достигните нужного уровня корейского",
        "step3_t": "Подать на визу", "step3_d": "Получите визу под ваши цели",
        "step4_t": "Начать жизнь в Корее", "step4_d": "Жильё, банк, здоровье и многое другое",
        "feat_uni_t": "Поиск университетов", "feat_uni_d": "380+ вузов, специальностей, плата и стипендии в одном месте.",
        "feat_job_t": "Доска вакансий", "feat_job_d": "AI-подбор вакансий для иностранных талантов, визово-дружелюбные работодатели.",
        "feat_visa_t": "Проверка визы", "feat_visa_d": "Разберитесь в визах D-2, D-4, E-7, F-2, F-5 с помощью AI.",
        "official_res": "Официальные ресурсы", "official_sub": "Надёжные правительственные и вузовские ссылки в одном месте",
        "res_study": "🎓 Учебные ресурсы", "res_immig": "🛂 Иммиграционный гид", "res_tests": "📝 Экзамены и работа",
        "job_portals": "Популярные порталы вакансий", "why_korea": "Почему Корея", "statistics": "Статистика", "resources": "Ресурсы",
        "overview": "Обзор", "latest_news": "Последние новости",
        "uni_research": "Исследование", "uni_school": "Информация о вузе", "uni_apply": "Подача",
        "uni_admission": "Поступление", "uni_scholar": "Стипендии",
        "filter_keyword": "Ключевое слово", "filter_region": "Регион", "filter_gks": "Право на GKS",
        "kw_ph": "Поиск вузов или специальностей...", "all": "Все", "yes": "Да", "no": "Нет",
        "loc_contact": "📍 Расположение и контакты", "tuition_support": "💰 Оплата и поддержка",
        "grad_req": "🎓 Требования к выпуску", "avail_majors": "📚 Доступные специальности",
        "btn_website": "🌐 Официальный сайт", "btn_apply": "📝 Подать сейчас", "btn_intl": "🌏 Международный отдел",
        "career_cv": "Проверка резюме", "career_interview": "Пробное собеседование", "career_res": "Ресурсы",
        "cv_title": "AI-проверка резюме", "cv_sub": "Загрузите резюме и получите мгновенную обратную связь для рынка труда Кореи",
        "cv_upload": "Загрузить резюме (PDF или DOCX)", "cv_target": "Желаемая должность",
        "cv_target_ph": "напр.: Инженер-программист", "cv_analyze": "🔍 Анализировать резюме",
        "cv_score": "Общий балл", "cv_grammar": "Грамматика", "cv_structure": "Структура",
        "cv_recs": "Рекомендации", "cv_culture": "Советы по культуре Кореи",
        "iv_title": "🎤 AI-собеседование", "iv_sub": "Практикуйтесь с реальными вопросами и получайте AI-отзыв",
        "iv_role": "Должность", "iv_role_ph": "напр.: Инженер-программист, Маркетинг-менеджер",
        "iv_info": "Введите должность выше, чтобы начать пробное собеседование.",
        "iv_start": "🚀 Начать собеседование", "iv_answer": "Ваш ответ", "iv_answer_ph": "Введите ответ (минимум 10 символов)...",
        "iv_min": "Пожалуйста, напишите минимум 10 символов.", "iv_feedback": "Генерируем AI-отзыв...",
        "iv_overall": "Итог", "iv_confidence": "Уверенность", "iv_fit": "Культурное соответствие",
        "iv_strengths": "Сильные стороны", "iv_improve": "Улучшения", "iv_again": "🔄 Попробовать снова",
        "job_board": "Доска вакансий", "job_matches": "Мои совпадения", "job_portals_tab": "Порталы",
        "job_search_ph": "Поиск должности или компании...", "job_visa_filter": "Тип визы",
        "job_match": "СОВПАДЕНИЕ", "job_apply": "Подать", "job_upload_cv": "Загрузите резюме, чтобы найти лучшие вакансии",
        "job_find": "🎯 Найти совпадения",
        "topik_schedule": "Расписание", "topik_register": "Регистрация", "topik_levels": "Уровни",
        "topik_tips": "Советы по учёбе", "topik_centers": "Центры тестирования",
        "topik_fee": "Информация о сборах", "topik_reg_btn": "Регистрация на topik.go.kr →",
        "tc_search": "Поиск по городу...",
        "visa_req": "Требования", "visa_docs": "Документы", "visa_proc": "Сроки обработки",
        "visa_for": "Для кого", "visa_fee": "Сбор", "visa_dur": "Срок", "visa_apply": "Подать на HiKorea →",
        "life_housing": "Жильё", "life_transport": "Транспорт", "life_health": "Здоровье",
        "life_banking": "Банк", "life_safety": "Безопасность",
        "auth_welcome": "Добро пожаловать в UniPath", "auth_login_sub": "Войдите, чтобы сохранять вузы, вакансии и получать уведомления",
        "auth_reg_sub": "Создайте бесплатный аккаунт и персонализируйте свой путь",
        "full_name": "Полное имя", "email": "Эл. почта", "password": "Пароль",
        "confirm_pw": "Подтвердите пароль", "lang_pref": "Предпочитаемый язык",
        "notif_pref": "Настройки уведомлений",
        "notif_topik": "Уведомления о расписании TOPIK", "notif_visa": "Обновления визовой политики",
        "notif_job": "Новые вакансии", "notif_uni": "Новости вузов", "notif_scholar": "Объявления о стипендиях",
        "have_account": "Уже есть аккаунт? Войти", "no_account": "Нет аккаунта? Создать",
        "pw_mismatch": "Пароли не совпадают.", "invalid_email": "Введите корректный адрес эл. почты.",
        "reg_success": "🎉 Аккаунт создан! Добро пожаловать.", "login_fail": "Аккаунт не найден. Сначала зарегистрируйтесь.",
        "login_success": "С возвращением!", "fill_all": "Пожалуйста, заполните все поля.",
        "my_profile": "Мой профиль", "saved_uni": "🎓 Сохранённые вузы", "saved_jobs": "💼 Сохранённые вакансии",
        "edit_notif": "Изменить уведомления", "profile_saved": "Настройки сохранены!",
        "none_yet": "Пока ничего не сохранено.",
        "chat_title": "Ассистент UNI", "chat_sub": "Спросите меня о жизни в Корее",
        "placeholder": "Спросите меня о Корее что угодно...", "ask_ai": "💬 Спросить AI-ассистента",
        "subscribe": "Получать обновления на почту", "email_label": "Ваша почта", "topics": "Интересующие темы",
        "sub_success": "✅ Подписка оформлена! Обновления придут на вашем языке.",
        "admin_panel": "⚙️ Панель администратора", "admin_pw": "Пароль администратора", "admin_ok": "✅ Доступ разрешён",
        "admin_pdf": "📄 Загрузка PDF", "admin_stats": "📊 Статистика",
    },
}


def t(key):
    """Translate a key into the currently selected language, falling back to the key."""
    lang = st.session_state.get("lang", "🇺🇸 English")
    table = TR.get(lang, TR["🇺🇸 English"])
    return table.get(key, TR["🇺🇸 English"].get(key, key))


# ════════════════════════════════════════════════════════════════════════════
# 5. SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "lang": "🇺🇸 English",
        "page": "HOME",
        "chat_history": [],
        "user": None,
        "auth_mode": "login",
        "interview_step": 0,
        "interview_qa": [],
        "interview_questions": [],
        "chat_open": False,
        "cv_result": None,
        "iv_result": None,
        "job_match_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ════════════════════════════════════════════════════════════════════════════
# 6. SUPABASE + AI INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════
AI_READY = False
supabase = None


def _secret(key, default=None):
    """Safe secret accessor that never raises even when secrets.toml is missing."""
    try:
        return st.secrets[key]
    except Exception:
        return default


try:
    if LLAMA_LIB and _secret("GEMINI_API_KEY"):
        Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=_secret("GEMINI_API_KEY"))
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="models/text-embedding-004", api_key=_secret("GEMINI_API_KEY")
        )
    if SUPABASE_LIB and _secret("SUPABASE_URL") and _secret("SUPABASE_KEY"):
        supabase = create_client(_secret("SUPABASE_URL"), _secret("SUPABASE_KEY"))
    AI_READY = bool(supabase) and LLAMA_LIB and bool(_secret("GEMINI_API_KEY"))
except Exception:
    supabase = None
    AI_READY = False


def localized(row, base, lang=None):
    """Return the language-appropriate column value.

    Tables may carry `<base>_en` and `<base>_ko` columns. Korean users see the
    Korean text; everyone else sees English. For other languages we fall back to
    English (AI translation hint shown in placeholders where relevant).
    """
    lang = lang or st.session_state.get("lang", "🇺🇸 English")
    if not isinstance(row, dict):
        return ""
    ko = row.get(f"{base}_ko")
    en = row.get(f"{base}_en")
    plain = row.get(base)
    if lang == "🇰🇷 한국어":
        return ko or en or plain or ""
    return en or plain or ko or ""


# ════════════════════════════════════════════════════════════════════════════
# 7. DATA LOADERS (Supabase, cached)
# ════════════════════════════════════════════════════════════════════════════
def _safe_select(table, order_col=None, desc=False, filters=None, limit=None):
    """Run a guarded SELECT and return list of dict rows ([] on any failure)."""
    if supabase is None:
        return []
    try:
        q = supabase.table(table).select("*")
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        if order_col:
            q = q.order(order_col, desc=desc)
        if limit:
            q = q.limit(limit)
        res = q.execute()
        return res.data or []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def load_universities():
    return _safe_select("universities", order_col="name")


@st.cache_data(ttl=300, show_spinner=False)
def load_scholarships():
    return _safe_select("scholarships")


@st.cache_data(ttl=300, show_spinner=False)
def load_jobs():
    rows = _safe_select("jobs", order_col="match_score", desc=True, filters={"is_active": True})
    if not rows:  # some schemas may not have is_active — retry unfiltered
        rows = _safe_select("jobs", order_col="match_score", desc=True)
    return rows


@st.cache_data(ttl=300, show_spinner=False)
def load_topik():
    return _safe_select("topik_schedule", order_col="test_date")


@st.cache_data(ttl=300, show_spinner=False)
def load_topik_info():
    return _safe_select("topik_info", order_col="id")


@st.cache_data(ttl=300, show_spinner=False)
def load_topik_structure():
    return _safe_select("topik_structure")


@st.cache_data(ttl=300, show_spinner=False)
def load_topik_faq():
    return _safe_select("topik_faq")


@st.cache_data(ttl=300, show_spinner=False)
def load_topik_centers():
    return _safe_select("topik_centers", order_col="city")


@st.cache_data(ttl=300, show_spinner=False)
def load_visa():
    return _safe_select("visa_info", order_col="visa_code")


@st.cache_data(ttl=300, show_spinner=False)
def load_news():
    return _safe_select("news", order_col="published_at", desc=True, limit=10)


# ════════════════════════════════════════════════════════════════════════════
# 8. RAG + LLM RESPONSE
# ════════════════════════════════════════════════════════════════════════════
def get_rag_response(query):
    """Answer a user question using the Supabase vector store first, then Gemini."""
    lang = st.session_state.get("lang", "🇺🇸 English")
    lang_name = lang.split(" ", 1)[1] if " " in lang else "English"
    system = (
        f"You are UNI, a helpful AI guide for international students in South Korea. "
        f"Always respond in {lang_name}. Be accurate, friendly, and concise. "
        f"When you are unsure, suggest official sources like hikorea.go.kr or topik.go.kr."
    )

    if not AI_READY:
        return (
            f"{t('error')} (AI service is not configured yet. Please add GEMINI_API_KEY and "
            f"SUPABASE secrets.)",
            "System",
        )

    # 1) Try the knowledge base (RAG).
    try:
        vector_store = SupabaseVectorStore(
            postgres_connection_string=_secret("SUPABASE_DB_CONNECTION"),
            collection_name="documents",
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
        qe = index.as_query_engine(similarity_top_k=3)
        response = qe.query(f"{system}\n\nQuestion: {query}")
        result = str(response).strip()
        low = result.lower()
        if len(result) < 20 or "don't know" in low or "no information" in low or "i do not have" in low:
            raise ValueError("Low quality RAG result")
        return result, "UniPath Knowledge Base"
    except Exception:
        # 2) Fall back to the base LLM.
        try:
            response = Settings.llm.complete(f"{system}\n\nQuestion: {query}")
            return str(response).strip(), "Gemini AI"
        except Exception:
            return t("error"), "System"


def llm_complete(prompt):
    """Thin guarded wrapper around the LLM for CV/interview/job-match features."""
    if not AI_READY:
        return None
    try:
        return str(Settings.llm.complete(prompt)).strip()
    except Exception:
        return None


def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF/DOCX-ish file. Returns string ('' on fail)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        return "".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        try:
            uploaded_file.seek(0)
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""


def parse_json_block(text):
    """Best-effort extraction of a JSON object from an LLM response."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def parse_json_array(text):
    """Best-effort extraction of a JSON array from an LLM response."""
    if not text:
        return None
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else None
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, list) else None
        except Exception:
            return None
    return None


# ════════════════════════════════════════════════════════════════════════════
# CONTENT TRANSLATION ENGINE (AI-cached, batched)
# ────────────────────────────────────────────────────────────────────────────
# UI labels live in the TR dictionary (t()). Free-form descriptive content is
# authored once in English and translated on demand by the LLM, cached per
# (text, language) for 24h so a language switch costs at most one batched call
# per block and is instant on every later render. Fixed values — addresses from
# the database, prices in KRW, phone numbers, brand and app names — are NOT sent
# through here; they are rendered as literals so they never change.
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def _tb_cached(items_tuple, lang):
    items = list(items_tuple)
    # English (or AI unavailable) → identity map, no cost.
    if lang == "🇺🇸 English" or not AI_READY or not items:
        return {s: s for s in items}
    lang_name = lang.split(" ", 1)[1] if " " in lang else "English"
    try:
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(items))
        prompt = (
            f"You are a professional UI localizer. Translate each numbered line into {lang_name}, "
            "natural and fluent for an app aimed at international students in Korea. "
            "Keep unchanged: brand/app/product names, proper nouns, place names, numbers, prices, "
            "currency (KRW), phone numbers, visa codes (e.g. D-2), URLs and emojis. "
            "Preserve any leading symbols/bullets and HTML tags exactly. "
            f"Return ONLY a JSON array of {len(items)} strings, in the same order, no commentary.\n\n"
            + numbered
        )
        raw = str(Settings.llm.complete(prompt)).strip()
        arr = parse_json_array(raw)
        if isinstance(arr, list) and len(arr) == len(items):
            return {s: (str(arr[i]) if arr[i] is not None else s) for i, s in enumerate(items)}
    except Exception:
        pass
    return {s: s for s in items}


def TB(items):
    """Batch-translate a list of English strings into the current language.

    Returns a dict {english: translated}. Cached, so repeated renders are free.
    """
    items = [str(x) for x in items]
    lang = st.session_state.get("lang", "🇺🇸 English")
    return _tb_cached(tuple(items), lang)


def T(text):
    """Translate a single English string into the current language (cached)."""
    if text is None:
        return ""
    return TB([text]).get(str(text), str(text))


def _resp_lang():
    """Human-readable name of the current language, for LLM 'respond in X' prompts."""
    lang = st.session_state.get("lang", "🇺🇸 English")
    return lang.split(" ", 1)[1] if " " in lang else "English"


# ─── small UI helpers ────────────────────────────────────────────────────────
def card_open(extra=""):
    st.markdown(f'<div class="upk-card" style="{extra}">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def section_header(title, sub="", translate_sub=True):
    # Subtitles are usually authored in English here, so auto-translate them.
    if sub and translate_sub:
        sub = T(sub)
    html = f'<div class="section-title">{title}</div>'
    if sub:
        html += f'<div class="section-sub">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="soft-divider"/>', unsafe_allow_html=True)


def goto(page):
    st.session_state.page = page
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# 9. NAVIGATION
# ════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("HOME", "nav_home"), ("UNIVERSITY", "nav_university"), ("CAREER", "nav_career"),
    ("JOB", "nav_job"), ("TOPIK", "nav_topik"), ("VISA", "nav_visa"), ("LIFE", "nav_life"),
]


def render_nav():
    # The whole bar lives inside a keyed container so the sticky white header
    # actually wraps the interactive widgets (a plain markdown <div> would not).
    with st.container(key="topnav"):
        logo_col, nav_col, lang_col, auth_col = st.columns(
            [2.1, 7.2, 1.6, 1.3], vertical_alignment="center")

        with logo_col:
            st.markdown(
                '<div class="upk-logo">'
                '<div class="badge">🎓</div>'
                '<div class="wm">Uni<span>Path</span><small>Korea Guide</small></div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with nav_col:
            ncols = st.columns(len(NAV_ITEMS))
            for i, (page, key) in enumerate(NAV_ITEMS):
                with ncols[i]:
                    active = st.session_state.page == page
                    cont = st.container(key="nav_active") if active else st.container()
                    with cont:
                        if st.button(t(key), key=f"nav_{page}", use_container_width=True):
                            goto(page)

        with lang_col:
            new_lang = st.selectbox(
                t("language"),
                LANGUAGES,
                index=LANGUAGES.index(st.session_state.lang) if st.session_state.lang in LANGUAGES else 0,
                key="lang_select",
                label_visibility="collapsed",
            )
            if new_lang != st.session_state.lang:
                st.session_state.lang = new_lang
                st.rerun()

        with auth_col:
            if st.session_state.user:
                name = st.session_state.user.get("name", "Me")
                short = (name[:7] + "…") if len(name) > 8 else name
                with st.container(key="nav_cta"):
                    if st.button(f"👤 {short}", key="nav_profile", use_container_width=True):
                        goto("PROFILE")
            else:
                with st.container(key="nav_cta"):
                    if st.button(f"🔑 {t('login')}", key="nav_auth", use_container_width=True):
                        goto("AUTH")


# ════════════════════════════════════════════════════════════════════════════
# 10. HOME PAGE
# ════════════════════════════════════════════════════════════════════════════
STUDY_LINKS = [
    ("Korean Government Scholarship (GKS)", "https://www.studyinkorea.go.kr"),
    ("University Search — Study in Korea", "https://www.studyinkorea.go.kr/en/main.do"),
    ("TOPIK Official Information", "https://www.topik.go.kr"),
    ("Visa Guide — HiKorea", "https://www.hikorea.go.kr"),
    ("Life in Korea Guide", "https://www.korea.net"),
    ("Work Guide for Foreigners", "https://www.work.go.kr"),
    ("Korea Immigration News", "https://www.immigration.go.kr"),
    ("Study Flowchart — KGSP", "https://www.studyinkorea.go.kr/en/sub/gks/allnew_invite.do"),
]
IMMIG_LINKS = [
    ("D-2 Student Visa", "https://www.hikorea.go.kr"),
    ("D-4 Language Training Visa", "https://www.hikorea.go.kr"),
    ("E-7 Specialized Work Visa", "https://www.hikorea.go.kr"),
    ("F-2 Residence Visa", "https://www.hikorea.go.kr"),
    ("F-5 Permanent Residency", "https://www.hikorea.go.kr"),
    ("Foreigner Registration (ARC)", "https://www.hikorea.go.kr"),
    ("Change of Address", "https://www.hikorea.go.kr"),
    ("Part-time Work Permit Rules", "https://www.hikorea.go.kr"),
]
TEST_LINKS = [
    ("TOPIK Official Site", "https://www.topik.go.kr"),
    ("TOPIK Exam Schedule", "https://www.topik.go.kr"),
    ("TOPIK Test Centers", "https://www.topik.go.kr"),
    ("Immigration Office Locator", "https://www.hikorea.go.kr"),
    ("Wanted Job Portal", "https://www.wanted.co.kr"),
    ("Saramin Jobs", "https://www.saramin.co.kr"),
    ("Work24 (Employment)", "https://www.work.go.kr"),
]
JOB_PORTALS = [
    ("Wanted", "https://www.wanted.co.kr", "Tech & startup roles with great UX"),
    ("Saramin", "https://www.saramin.co.kr", "Korea's largest job marketplace"),
    ("JobKorea", "https://www.jobkorea.co.kr", "Broad listings across industries"),
    ("Work24", "https://www.work.go.kr", "Government employment portal"),
    ("K-Work", "https://www.k-work.or.kr", "Jobs for foreign residents"),
    ("KoMate", "https://www.komate.kr", "Foreigner-friendly part-time jobs"),
    ("Incruit", "https://www.incruit.com", "Established Korean job board"),
    ("Employment24", "https://www.work.go.kr", "Public employment services"),
    ("KOWORK", "https://www.kowork.kr", "Support for migrant workers"),
]

PORTAL_COLORS = {
    "Wanted": "#3D5AFE", "Saramin": "#16A34A", "JobKorea": "#0EA5E9", "Work24": "#16233F",
    "K-Work": "#7C3AED", "KoMate": "#E0703A", "Incruit": "#DB2777",
    "Employment24": "#0891B2", "KOWORK": "#0B7A5E",
}


def render_portals(portals):
    """Render premium brand-colored portal logo cards in a responsive grid."""
    tx = TB([d for _, _, d in portals])
    cards = ""
    for name, url, desc in portals:
        c = PORTAL_COLORS.get(name, "#16233F")
        initials = name[0].upper()
        cards += (
            f'<a class="pcard" href="{url}" target="_blank">'
            f'<div class="pmark" style="background:linear-gradient(135deg,{c},{c}bb);">{initials}</div>'
            f'<div class="pname">{name}</div>'
            f'<div class="pdesc">{tx.get(desc, desc)}</div>'
            f'<div class="pgo">{t("visit")} →</div></a>'
        )
    st.markdown(f'<div class="portal-grid">{cards}</div>', unsafe_allow_html=True)


def _link_card(title, links):
    # Translate the link labels (descriptive text); URLs and brand names are kept.
    tx = TB([name for name, _ in links])
    rows = "".join(
        f'<a class="link-row" href="{url}" target="_blank">🔗 {tx.get(name, name)}</a>'
        for name, url in links
    )
    st.markdown(f'<div class="link-card"><h4>{title}</h4>{rows}</div>', unsafe_allow_html=True)


def page_home():
    # ---- Hero (stats integrated, no overlap) ----
    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-badge">{t('home_badge')}</span>
            <h1>{t('home_title')}</h1>
            <p>{t('home_subtitle')}</p>
            <div class="hero-stats">
                <div class="hstat"><div class="hi">🛂</div><div class="hv">24</div>
                    <div class="hl">{t('kpi_visa')}</div></div>
                <div class="hstat"><div class="hi">🏛️</div><div class="hv">386</div>
                    <div class="hl">{t('kpi_uni')}</div></div>
                <div class="hstat"><div class="hi">📊</div><div class="hv">LV.1~6</div>
                    <div class="hl">{t('kpi_topik')}</div></div>
                <div class="hstat"><div class="hi">💼</div><div class="hv">1,240</div>
                    <div class="hl">{t('kpi_jobs')}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    tabs = st.tabs([t("overview"), t("why_korea"), t("statistics"), t("resources")])

    # ───────────────────────── TAB 1: OVERVIEW ─────────────────────────
    with tabs[0]:
        # Plan your journey
        section_header(t("plan_title"), t("plan_sub"))
        st.markdown(
            f"""
            <div class="flow-wrap">
                <div class="flow-step"><div class="flow-num">1</div>
                    <h4>{t('step1_t')}</h4><p>{t('step1_d')}</p></div>
                <div class="flow-arrow">→</div>
                <div class="flow-step"><div class="flow-num">2</div>
                    <h4>{t('step2_t')}</h4><p>{t('step2_d')}</p></div>
                <div class="flow-arrow">→</div>
                <div class="flow-step"><div class="flow-num">3</div>
                    <h4>{t('step3_t')}</h4><p>{t('step3_d')}</p></div>
                <div class="flow-arrow">→</div>
                <div class="flow-step"><div class="flow-num">4</div>
                    <h4>{t('step4_t')}</h4><p>{t('step4_d')}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        # AI assistant CTA
        st.markdown(
            f'<div style="text-align:center;margin:10px 0 26px 0;">'
            f'<span class="chat-cta">{t("ask_ai")} — {T("open the sidebar")} →</span></div>',
            unsafe_allow_html=True,
        )

        divider()

        # Feature cards
        section_header(T("Explore the Platform"), t("plan_sub"))
        feats = [
            # (icon, accent, glow, shadow, badge_bg, badge, title_key, desc_key, btn, key, page)
            ("🏛️", "#3D5AFE", "rgba(61,90,254,0.12)", "rgba(61,90,254,0.30)", "#ECEFFE",
             "NEW", "feat_uni_t", "feat_uni_d", f"🔎 {t('search')}", "home_uni_btn", "UNIVERSITY"),
            ("💼", "#12A580", "rgba(18,165,128,0.12)", "rgba(18,165,128,0.30)", "#E6F6F1",
             "HOT", "feat_job_t", "feat_job_d", f"📋 {t('job_board')}", "home_job_btn", "JOB"),
            ("🛂", "#E0703A", "rgba(224,112,58,0.12)", "rgba(224,112,58,0.30)", "#FBEEE6",
             "AI", "feat_visa_t", "feat_visa_d", f"✅ {t('nav_visa')}", "home_visa_btn", "VISA"),
        ]
        fcols = st.columns(3)
        for col, (icon, accent, glow, sh, bbg, badge, tk, dk, btn, bkey, page) in zip(fcols, feats):
            with col:
                st.markdown(
                    f'<div class="feat-card" style="--fc:{accent};--fcglow:{glow};--fcsh:{sh};--fcbg:{bbg};">'
                    f'<span class="feat-badge">{badge}</span>'
                    f'<div class="feat-tile">{icon}</div>'
                    f'<h3>{t(tk)}</h3><p>{t(dk)}</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button(btn, key=bkey, use_container_width=True):
                    goto(page)

        divider()

        # Official resources
        section_header(t("official_res"), t("official_sub"))
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            _link_card(t("res_study"), STUDY_LINKS)
        with rc2:
            _link_card(t("res_immig"), IMMIG_LINKS)
        with rc3:
            _link_card(t("res_tests"), TEST_LINKS)

        divider()

        # Job portal logos row
        section_header(t("job_portals"))
        render_portals(JOB_PORTALS[:5])

    # ───────────────────────── TAB 2: WHY KOREA ─────────────────────────
    with tabs[1]:
        section_header(t("why_korea"), "A safe, world-class destination for your education")
        caps = ["International Students in Korea", "Korean Universities in World Top 100",
                "Fastest Internet Speed Globally", "Countries Reached by K-Culture"]
        tc = TB(caps)
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-box"><div class="stat-big">160,000+</div>
                    <div class="stat-cap">{tc.get(caps[0], caps[0])}</div></div>
                <div class="stat-box"><div class="stat-big">3</div>
                    <div class="stat-cap">{tc.get(caps[1], caps[1])}</div></div>
                <div class="stat-box"><div class="stat-big">#1</div>
                    <div class="stat-cap">{tc.get(caps[2], caps[2])}</div></div>
                <div class="stat-box"><div class="stat-big">100+</div>
                    <div class="stat-cap">{tc.get(caps[3], caps[3])}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        divider()
        section_header(T("Six Reasons to Choose Korea"), translate_sub=False)
        reasons = [
            ("🛡️", "Safety", "Among the safest countries in the world, with low crime and 24/7 convenience."),
            ("🎓", "Education Quality", "Globally ranked universities with strong STEM, business and arts programs."),
            ("📈", "Career Opportunities", "A booming tech, manufacturing and creative economy hungry for talent."),
            ("🎬", "K-Culture", "Live at the heart of music, film, food and fashion the world loves."),
            ("💰", "Affordable Living", "Generous scholarships and reasonable costs compared to the US/EU."),
            ("🚀", "Advanced Technology", "5G everywhere, smart cities, and cutting-edge research facilities."),
        ]
        rtx = TB([r[1] for r in reasons] + [r[2] for r in reasons])
        r1 = st.columns(3)
        r2 = st.columns(3)
        for i, (icon, title, desc) in enumerate(reasons):
            col = (r1 if i < 3 else r2)[i % 3]
            with col:
                st.markdown(
                    f'<div class="reason-card"><div class="ri">{icon}</div>'
                    f'<h4>{rtx.get(title, title)}</h4><p>{rtx.get(desc, desc)}</p></div>',
                    unsafe_allow_html=True,
                )
                st.write("")

    # ───────────────────────── TAB 3: STATISTICS ─────────────────────────
    with tabs[2]:
        section_header(t("statistics"), "International student trends in South Korea")
        if not PLOTLY_READY:
            st.info("Charts require plotly. Install with: pip install plotly")
        else:
            c1, c2 = st.columns(2)
            with c1:
                years = ["2019", "2020", "2021", "2022", "2023", "2024"]
                counts = [160165, 153695, 152281, 166892, 181842, 208962]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years, y=counts, mode="lines+markers",
                    line=dict(color="#16233F", width=3),
                    marker=dict(size=9, color="#3D5AFE"),
                    fill="tozeroy", fillcolor="rgba(61,90,254,0.07)",
                ))
                fig.update_layout(
                    title="International Students in Korea (Trend)",
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="Outfit"), height=360, margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                nats = ["China", "Vietnam", "Mongolia", "Uzbekistan", "Japan", "Others"]
                share = [33, 23, 6, 5, 4, 29]
                fig2 = px.pie(
                    names=nats, values=share, hole=0.5,
                    color_discrete_sequence=["#16233F", "#3D5AFE", "#12A580", "#2E4B8A", "#8AA0E8", "#CBD2E0"],
                )
                fig2.update_layout(
                    title="Top Nationalities", paper_bgcolor="white",
                    font=dict(family="Outfit"), height=360, margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                majors = ["Korean Language", "Business", "Engineering", "Arts", "Social Sci."]
                vals = [38, 22, 18, 12, 10]
                fig3 = go.Figure(go.Bar(
                    x=vals, y=majors, orientation="h",
                    marker=dict(color="#3D5AFE"),
                ))
                fig3.update_layout(
                    title="Popular Fields of Study (%)", paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="Outfit"), height=340, margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig3, use_container_width=True)
            with c4:
                yrs = ["1yr", "2yr", "3yr", "4yr", "5yr"]
                emp = [42, 58, 67, 74, 81]
                fig4 = go.Figure(go.Bar(x=yrs, y=emp, marker=dict(color="#12A580")))
                fig4.update_layout(
                    title="Graduate Employment Rate (%)", paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="Outfit"), height=340, margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig4, use_container_width=True)
        st.markdown(
            '<div class="glass-note">📌 Figures are illustrative trends compiled from public sources '
            '(Korea Immigration Service, MOE). For official statistics, visit moe.go.kr.</div>',
            unsafe_allow_html=True,
        )

    # ───────────────────────── TAB 4: RESOURCES ─────────────────────────
    with tabs[3]:
        section_header(t("resources"), "Curated links and the latest news")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            _link_card(t("res_study"), STUDY_LINKS)
        with rcol2:
            _link_card(t("res_immig"), IMMIG_LINKS)
        with rcol3:
            _link_card(t("res_tests"), TEST_LINKS)

        divider()
        section_header(t("latest_news"))
        news = load_news()
        if not news:
            st.info(t("empty"))
        else:
            for n in news:
                title_txt = localized(n, "title") or n.get("title", "—")
                summary = localized(n, "summary") or n.get("summary", "")
                category = n.get("category", "News")
                source = n.get("source", "")
                pub = n.get("published_at", "")
                link = n.get("link") or n.get("url", "#")
                st.markdown(
                    f"""
                    <div class="news-card">
                        <span class="tag tag-navy">{category}</span>
                        <h4>{title_txt}</h4>
                        <p>{summary}</p>
                        <div class="news-meta">📰 {source} &nbsp;•&nbsp; 🗓️ {pub} &nbsp;•&nbsp;
                            <a href="{link}" target="_blank">{t('visit')} →</a></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# 11. UNIVERSITY PAGE
# ════════════════════════════════════════════════════════════════════════════
def _uni_field(row, *keys, default="—"):
    for k in keys:
        v = row.get(k)
        if v:
            return v
    return default


def page_university():
    section_header(f"🏛️ {t('nav_university')}", t("feat_uni_d"))
    tabs = st.tabs([
        t("uni_research"), t("uni_school"), t("uni_apply"),
        t("uni_admission"), t("uni_scholar"),
    ])

    unis = load_universities()

    # ───────────── TAB 1: RESEARCH ─────────────
    with tabs[0]:
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            keyword = st.text_input(t("filter_keyword"), placeholder=t("kw_ph"), key="uni_kw")
        with f2:
            regions = sorted({_uni_field(u, "region", default="") for u in unis if u.get("region")})
            region = st.selectbox(t("filter_region"), [t("all")] + regions, key="uni_region")
        with f3:
            gks = st.selectbox(t("filter_gks"), [t("all"), t("yes"), t("no")], key="uni_gks")

        # Apply filters
        filtered = []
        for u in unis:
            name = _uni_field(u, "name", default="")
            majors = _uni_field(u, "majors", default="")
            if keyword and keyword.lower() not in (name + " " + majors).lower():
                continue
            if region != t("all") and _uni_field(u, "region", default="") != region:
                continue
            gks_val = u.get("gks_eligible") or u.get("gks")
            is_gks = bool(gks_val) and str(gks_val).lower() not in ("false", "0", "no", "")
            if gks == t("yes") and not is_gks:
                continue
            if gks == t("no") and is_gks:
                continue
            filtered.append(u)

        if not unis:
            st.info(t("empty"))
        elif not filtered:
            st.warning("No universities match your filters. Try widening your search.")
        else:
            st.caption(f"{len(filtered)} / {len(unis)} {t('nav_university').lower()}")
            for u in filtered:
                name = _uni_field(u, "name", default="University")
                region_v = _uni_field(u, "region", default="—")
                rank = _uni_field(u, "rank", "ranking", default="—")
                expanded = "sookmyung" in name.lower()
                header = f"⭐ {name}  |  📍 {region_v}  |  🏆 {rank}"
                with st.expander(header, expanded=expanded):
                    gks_val = u.get("gks_eligible") or u.get("gks")
                    is_gks = bool(gks_val) and str(gks_val).lower() not in ("false", "0", "no", "")
                    topik_req = _uni_field(u, "topik_req", "topik_requirement", default="TOPIK 3+")
                    uni_type = _uni_field(u, "uni_type", "type", default="—")
                    founded = _uni_field(u, "founded", "established", default="—")
                    tags = (
                        (f'<span class="tag tag-green">GKS ✓</span>' if is_gks else
                         f'<span class="tag tag-grey">GKS ✗</span>')
                        + f'<span class="tag tag-navy">TOPIK: {topik_req}</span>'
                        + f'<span class="tag tag-orange">{uni_type}</span>'
                        + f'<span class="tag tag-grey">Est. {founded}</span>'
                    )
                    st.markdown(f'<div style="margin-bottom:10px;">{tags}</div>', unsafe_allow_html=True)

                    lc, rc = st.columns(2)
                    with lc:
                        st.markdown(
                            f"""
                            <div class="upk-card"><h3>{t('loc_contact')}</h3>
                            <p>📍 {_uni_field(u, 'location_detail', 'address')}<br>
                            ☎️ {_uni_field(u, 'phone')}<br>
                            ✉️ {_uni_field(u, 'email')}<br>
                            🕘 {_uni_field(u, 'office_hours', default='Mon–Fri 09:00–18:00')}<br>
                            👥 {_uni_field(u, 'students', 'student_count', default='—')} students</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"""
                            <div class="upk-card" style="margin-top:14px;"><h3>{t('tuition_support')}</h3>
                            <p>💵 Tuition: {_uni_field(u, 'tuition', default='—')}<br>
                            🏠 Dorm: {_uni_field(u, 'dorm', 'dorm_info', default='Available')}<br>
                            🎁 Scholarship: {_uni_field(u, 'scholarship', 'scholarship_info', default='See admissions')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with rc:
                        st.markdown(
                            f"""
                            <div class="upk-card"><h3>{t('grad_req')}</h3>
                            <p>{_uni_field(u, 'grad_req', 'graduation_requirements', default='Standard credit & thesis requirements apply.')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"""
                            <div class="upk-card" style="margin-top:14px;"><h3>{t('avail_majors')}</h3>
                            <p>{_uni_field(u, 'majors', default='Wide range of undergraduate & graduate programs.')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    b1, b2, b3 = st.columns(3)
                    website = _uni_field(u, "website", "url", default="https://www.studyinkorea.go.kr")
                    apply_url = _uni_field(u, "apply_url", default=website)
                    intl_url = _uni_field(u, "intl_url", "international_office", default=website)
                    with b1:
                        st.markdown(f'<a class="link-row" style="text-align:center;background:#EEF2FF;" '
                                    f'href="{website}" target="_blank">{t("btn_website")}</a>',
                                    unsafe_allow_html=True)
                    with b2:
                        st.markdown(f'<a class="link-row" style="text-align:center;background:#E6FBF4;" '
                                    f'href="{apply_url}" target="_blank">{t("btn_apply")}</a>',
                                    unsafe_allow_html=True)
                    with b3:
                        st.markdown(f'<a class="link-row" style="text-align:center;background:#FFEDE5;" '
                                    f'href="{intl_url}" target="_blank">{t("btn_intl")}</a>',
                                    unsafe_allow_html=True)

    # ───────────── TAB 2: SCHOOL INFO ─────────────
    with tabs[1]:
        section_header(t("uni_school"), "Graduation requirements, documents and deadlines")
        L = [
            "Graduation Requirements (typical)",
            "• Complete required credits (usually 130+ for bachelor's)",
            "• Maintain minimum GPA (often 2.0/4.5 or higher)",
            "• Pass a graduation thesis or capstone project",
            "• Meet Korean language milestones (TOPIK level often required)",
            "• Fulfill mandatory liberal arts / major core courses",
            "Required Documents Checklist",
            "☑ Completed application form", "☑ Passport copy & passport photos",
            "☑ High school / bachelor's diploma & transcripts (apostilled)",
            "☑ Proof of language ability (TOPIK / IELTS / TOEFL)",
            "☑ Statement of purpose & study plan", "☑ Recommendation letters (1–2)",
            "☑ Financial proof (bank statement)", "☑ Family relationship certificate",
            "Important Deadlines (general)", "Intake", "Application Period", "Notes",
            "Spring (March)", "September – November", "Most popular intake",
            "Fall (September)", "March – May", "Smaller programs",
            "GKS (Government)", "February – March (embassy)", "One round per year",
        ]
        tx = TB(L)
        x = lambda s: tx.get(s, s)
        c1, c2 = st.columns(2)
        grad = "<br>".join(x(s) for s in L[1:6])
        c1.markdown(f'<div class="upk-card"><h3>🎓 {x(L[0])}</h3><p>{grad}</p></div>',
                    unsafe_allow_html=True)
        docs = "<br>".join(x(s) for s in L[7:15])
        c2.markdown(f'<div class="upk-card"><h3>📋 {x(L[6])}</h3><p>{docs}</p></div>',
                    unsafe_allow_html=True)
        divider()
        st.markdown(
            f"""
            <div class="upk-card"><h3>🗓️ {x("Important Deadlines (general)")}</h3>
            <table class="upk-table">
                <tr><th>{x("Intake")}</th><th>{x("Application Period")}</th><th>{x("Notes")}</th></tr>
                <tr><td>{x("Spring (March)")}</td><td>{x("September – November")}</td><td>{x("Most popular intake")}</td></tr>
                <tr><td>{x("Fall (September)")}</td><td>{x("March – May")}</td><td>{x("Smaller programs")}</td></tr>
                <tr><td>{x("GKS (Government)")}</td><td>{x("February – March (embassy)")}</td><td>{x("One round per year")}</td></tr>
            </table></div>
            """,
            unsafe_allow_html=True,
        )

    # ───────────── TAB 3: APPLY ─────────────
    with tabs[2]:
        section_header(t("uni_apply"), "Your application timeline at a glance")
        L = [
            "Research", "Shortlist 3–5 universities", "Prepare Docs", "Apostille & translate",
            "Submit", "Online + courier", "Interview", "Online / on-site",
            "Offer & Visa", "CoA → D-2 visa", "Ask AI for guidance", "Download Checklist",
        ]
        x = (lambda m: (lambda s: m.get(s, s)))(TB(L))
        st.markdown(
            f"""
            <div class="flow-wrap">
                <div class="flow-step"><div class="flow-num">1</div><h4>{x("Research")}</h4>
                    <p>{x("Shortlist 3–5 universities")}</p></div>
                <div class="flow-arrow">→</div>
                <div class="flow-step"><div class="flow-num">2</div><h4>{x("Prepare Docs")}</h4>
                    <p>{x("Apostille & translate")}</p></div>
                <div class="flow-arrow">→</div>
                <div class="flow-step"><div class="flow-num">3</div><h4>{x("Submit")}</h4>
                    <p>{x("Online + courier")}</p></div>
                <div class="flow-arrow">→</div>
                <div class="flow-step"><div class="flow-num">4</div><h4>{x("Interview")}</h4>
                    <p>{x("Online / on-site")}</p></div>
                <div class="flow-arrow">→</div>
                <div class="flow-step"><div class="flow-num">5</div><h4>{x("Offer & Visa")}</h4>
                    <p>{x("CoA → D-2 visa")}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            if st.button(f"💬 {x('Ask AI for guidance')}", key="uni_ask_ai", use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user", "content": "How do I apply to a Korean university step by step?"})
                with st.spinner(t("loading")):
                    ans, src = get_rag_response(
                        "How do I apply to a Korean university step by step?")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ans, "source": src})
                st.toast("Answer added to the sidebar chat 👉")
        with cc2:
            checklist = (
                "UNIPATH KOREA — UNIVERSITY APPLICATION CHECKLIST\n\n"
                "[ ] Completed application form\n[ ] Passport copy & photos\n"
                "[ ] Diploma & transcripts (apostilled)\n[ ] Language proof (TOPIK/IELTS/TOEFL)\n"
                "[ ] Statement of purpose\n[ ] Study plan\n[ ] Recommendation letters\n"
                "[ ] Financial proof\n[ ] Family relationship certificate\n[ ] Application fee paid\n"
            )
            st.download_button(f"⬇️ {x('Download Checklist')}", checklist,
                               "unipath_checklist.txt", use_container_width=True)

    # ───────────── TAB 4: ADMISSION ─────────────
    with tabs[3]:
        section_header(t("uni_admission"), "GKS and university scholarships")
        L = [
            "Global Korea Scholarship (GKS / KGSP)",
            "The flagship Korean government scholarship fully funds international students.",
            "✅ Full tuition coverage", "✅ Monthly stipend (~900,000 KRW graduate)",
            "✅ Round-trip airfare", "✅ One-year Korean language training",
            "✅ Settlement & research allowance", "✅ Medical insurance",
            "<b>Two tracks:</b> Embassy track (apply via Korean embassy in your country) and "
            "University track (apply directly to a designated university).",
            "GKS Application Portal",
        ]
        x = (lambda m: (lambda s: m.get(s, s)))(TB(L))
        perks = "<br>".join(x(s) for s in L[2:8])
        st.markdown(
            f'<div class="upk-card"><h3>🌟 {x(L[0])}</h3>'
            f'<p>{x(L[1])}</p><p>{perks}</p><p>{x(L[8])}</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<a class="link-row" style="text-align:center;background:#EEF2FF;margin-top:12px;" '
                    f'href="https://www.studyinkorea.go.kr" target="_blank">🌐 {x("GKS Application Portal")} →</a>',
                    unsafe_allow_html=True)

    # ───────────── TAB 5: SCHOLARSHIPS ─────────────
    with tabs[4]:
        section_header(t("uni_scholar"), "Funding opportunities for international students")
        schols = load_scholarships()
        if not schols:
            st.info(t("empty"))
        else:
            for s in schols:
                provider = _uni_field(s, "provider", "name", default="Scholarship")
                stype = _uni_field(s, "type", default="Merit")
                coverage = _uni_field(s, "coverage", default="—")
                monthly = _uni_field(s, "monthly", "monthly_amount", default="—")
                topik_req = _uni_field(s, "topik_req", "topik_requirement", default="—")
                with st.expander(f"🎁 {provider}  —  {coverage}"):
                    st.markdown(
                        f'<span class="tag tag-navy">{stype}</span>'
                        f'<span class="tag tag-green">Monthly: {monthly}</span>'
                        f'<span class="tag tag-orange">TOPIK: {topik_req}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Eligibility:** {_uni_field(s, 'eligibility', default='See provider details.')}")
                    st.markdown(
                        f"**Application deadline:** {_uni_field(s, 'deadline', 'application_deadline', default='Varies')}")
                    apply_url = _uni_field(s, "apply_url", "url", default="https://www.studyinkorea.go.kr")
                    st.markdown(f'<a class="link-row" style="text-align:center;background:#E6FBF4;" '
                                f'href="{apply_url}" target="_blank">{t("apply")} →</a>',
                                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 12. CAREER PAGE
# ════════════════════════════════════════════════════════════════════════════
INTERVIEW_QUESTIONS = [
    "Tell me about yourself and why you want to work in Korea.",
    "What are your greatest strengths, and how do they fit this role?",
    "Describe a challenge you faced and how you overcame it.",
    "How do you adapt to a new culture and work environment?",
    "Where do you see yourself in five years, and why this company?",
]


def page_career():
    section_header(f"💼 {t('nav_career')}", "Build a standout profile for the Korean job market")
    tabs = st.tabs([t("career_cv"), t("career_interview"), t("career_res")])

    # ───────────── TAB 1: CV CHECK ─────────────
    with tabs[0]:
        section_header(t("cv_title"), t("cv_sub"))
        col1, col2 = st.columns([1, 1])
        with col1:
            cv_file = st.file_uploader(t("cv_upload"), type=["pdf", "docx", "txt"], key="cv_upload")
        with col2:
            target = st.text_input(t("cv_target"), placeholder=t("cv_target_ph"), key="cv_target")

        if st.button(t("cv_analyze"), key="cv_btn", use_container_width=True):
            if not cv_file:
                st.warning(t("cv_upload"))
            else:
                with st.spinner(t("loading")):
                    text = extract_pdf_text(cv_file)
                    if not text.strip():
                        st.error("Could not read the file. Please try a text-based PDF.")
                    else:
                        prompt = (
                            "You are an expert career coach for the South Korean job market. "
                            f"Analyze this CV for the target role: '{target or 'general'}'. "
                            f"Write all string values (recommendations, korean_tips) in {_resp_lang()}. "
                            "Return ONLY valid JSON with keys: overall (0-100 int), grammar (0-100 int), "
                            "structure (0-100 int), recommendations (list of 4-6 short strings), "
                            "korean_tips (list of 3-4 short strings about Korean workplace expectations).\n\n"
                            f"CV TEXT:\n{text[:6000]}"
                        )
                        raw = llm_complete(prompt)
                        data = parse_json_block(raw) if raw else None
                        if not data:
                            data = {
                                "overall": 72, "grammar": 80, "structure": 68,
                                "recommendations": [
                                    "Add measurable achievements with numbers and impact.",
                                    "Tailor your summary to the specific Korean company.",
                                    "Keep the CV to 1–2 pages with clean formatting.",
                                    "Include a professional photo (common in Korea).",
                                ],
                                "korean_tips": [
                                    "Korean CVs often include a photo, age and visa status.",
                                    "Highlight any Korean language ability (TOPIK level).",
                                    "Show respect for hierarchy and teamwork in your summary.",
                                ],
                            }
                            if not AI_READY:
                                st.info("Showing a sample analysis (AI not configured).")
                        st.session_state.cv_result = data

        data = st.session_state.cv_result
        if data:
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-box"><div class="mv">{data.get("overall","—")}</div>'
                        f'<div class="ml">{t("cv_score")}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="mv">{data.get("grammar","—")}</div>'
                        f'<div class="ml">{t("cv_grammar")}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="mv">{data.get("structure","—")}</div>'
                        f'<div class="ml">{t("cv_structure")}</div></div>', unsafe_allow_html=True)
            st.write("")
            cc1, cc2 = st.columns(2)
            with cc1:
                recs = "".join(f'<div class="check-item"><span class="ck">✓</span>{r}</div>'
                               for r in data.get("recommendations", []))
                st.markdown(f'<div class="upk-card"><h3>💡 {t("cv_recs")}</h3>{recs}</div>',
                            unsafe_allow_html=True)
            with cc2:
                tips = "".join(f'<div class="check-item"><span class="ck">🇰🇷</span>{r}</div>'
                               for r in data.get("korean_tips", []))
                st.markdown(f'<div class="upk-card"><h3>🏢 {t("cv_culture")}</h3>{tips}</div>',
                            unsafe_allow_html=True)

    # ───────────── TAB 2: MOCK INTERVIEW ─────────────
    with tabs[1]:
        section_header(t("iv_title"), t("iv_sub"))
        role = st.text_input(t("iv_role"), placeholder=t("iv_role_ph"), key="iv_role")

        if st.session_state.interview_step == 0:
            if not role.strip():
                st.info(t("iv_info"))
            if st.button(t("iv_start"), key="iv_start_btn", disabled=not role.strip(),
                         use_container_width=True):
                st.session_state.interview_questions = INTERVIEW_QUESTIONS
                st.session_state.interview_qa = []
                st.session_state.interview_step = 1
                st.session_state.iv_result = None
                st.rerun()

        step = st.session_state.interview_step
        questions = st.session_state.interview_questions or INTERVIEW_QUESTIONS

        if 0 < step <= len(questions):
            qword = T("Question")
            st.progress(step / len(questions), text=f"{qword} {step} / {len(questions)}")
            q = questions[step - 1]
            st.markdown(
                f'<div class="upk-card" style="border-left:5px solid var(--navy);">'
                f'<h3>❓ {qword} {step}</h3>'
                f'<p style="font-size:16px;color:var(--ink);">{T(q)}</p></div>',
                unsafe_allow_html=True,
            )
            answer = st.text_area(t("iv_answer"), placeholder=t("iv_answer_ph"),
                                  key=f"iv_ans_{step}", height=140)
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button(t("next"), key=f"iv_next_{step}", use_container_width=True):
                    if len(answer.strip()) < 10:
                        st.warning(t("iv_min"))
                    else:
                        st.session_state.interview_qa.append({"q": q, "a": answer.strip()})
                        st.session_state.interview_step += 1
                        st.rerun()
            with b2:
                if st.button(t("skip"), key=f"iv_skip_{step}", use_container_width=True):
                    st.session_state.interview_qa.append({"q": q, "a": "(skipped)"})
                    st.session_state.interview_step += 1
                    st.rerun()

        elif step > len(questions):
            if st.session_state.iv_result is None:
                with st.spinner(t("iv_feedback")):
                    qa_text = "\n".join(
                        f"Q{i+1}: {x['q']}\nA{i+1}: {x['a']}"
                        for i, x in enumerate(st.session_state.interview_qa))
                    prompt = (
                        f"You are an interview coach for the Korean job market. The candidate applied for "
                        f"'{role or 'a role'}'. Evaluate their mock interview answers. "
                        f"Write all string values (strengths, improvements, encouragement) in {_resp_lang()}. "
                        "Return ONLY valid JSON "
                        "with keys: overall (0-100 int), confidence (0-100 int), cultural_fit (0-100 int), "
                        "strengths (list of 3-4 strings), improvements (list of 3-4 strings), "
                        "encouragement (one warm sentence).\n\n" + qa_text
                    )
                    raw = llm_complete(prompt)
                    res = parse_json_block(raw) if raw else None
                    if not res:
                        res = {
                            "overall": 78, "confidence": 74, "cultural_fit": 81,
                            "strengths": [
                                "Clear motivation for working in Korea.",
                                "Good structure in your answers (STAR-like).",
                                "Positive, respectful tone.",
                            ],
                            "improvements": [
                                "Add concrete examples with measurable results.",
                                "Practice concise answers (60–90 seconds).",
                                "Research the company's products before interviews.",
                            ],
                            "encouragement": "You're on the right track — keep practicing and you'll shine! 🌟",
                        }
                        if not AI_READY:
                            st.info("Showing sample feedback (AI not configured).")
                    st.session_state.iv_result = res

            res = st.session_state.iv_result
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-box"><div class="mv">{res.get("overall","—")}</div>'
                        f'<div class="ml">{t("iv_overall")}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="mv">{res.get("confidence","—")}</div>'
                        f'<div class="ml">{t("iv_confidence")}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="mv">{res.get("cultural_fit","—")}</div>'
                        f'<div class="ml">{t("iv_fit")}</div></div>', unsafe_allow_html=True)
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                items = "".join(f'<div class="check-item"><span class="ck" style="color:var(--emerald);">✓</span>{s}</div>'
                                for s in res.get("strengths", []))
                st.markdown(f'<div class="upk-card"><h3 style="color:#00966f;">💪 {t("iv_strengths")}</h3>{items}</div>',
                            unsafe_allow_html=True)
            with c2:
                items = "".join(f'<div class="check-item"><span class="ck" style="color:var(--orange);">▲</span>{s}</div>'
                                for s in res.get("improvements", []))
                st.markdown(f'<div class="upk-card"><h3 style="color:#d94e1f;">🔧 {t("iv_improve")}</h3>{items}</div>',
                            unsafe_allow_html=True)
            st.markdown(
                f'<div class="upk-card" style="background:#E6FBF4;border-color:#b8f0df;margin-top:14px;">'
                f'<h3>🌟</h3><p style="font-size:16px;color:#00795b;">{res.get("encouragement","")}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(t("iv_again"), key="iv_again_btn", use_container_width=True):
                st.session_state.interview_step = 0
                st.session_state.interview_qa = []
                st.session_state.iv_result = None
                st.rerun()

    # ───────────── TAB 3: RESOURCES ─────────────
    with tabs[2]:
        section_header(t("career_res"), "Tools, culture tips and salary guidance")
        L = [
            "Career Tools", "Wanted — Tech Jobs", "LinkedIn Korea",
            "Rocketpunch — Startups", "Korean Resume Templates", "Work24 Career Services",
            "Korean Work Culture Tips",
            "• <b>Hierarchy (직급):</b> Respect seniority and use formal titles.",
            "• <b>Nunchi (눈치):</b> Read the room and social cues.",
            "• <b>Teamwork:</b> Group harmony is valued over individual spotlight.",
            "• <b>Punctuality:</b> Always arrive early; lateness is frowned upon.",
            "• <b>After-work (회식):</b> Team dinners help build relationships.",
            "Salary Negotiation Guide",
            "• Research market rates on Wanted, JobPlanet and Blind.",
            "• Entry-level office roles often start around 30–40M KRW/year.",
            "• Negotiate total package: base, bonus (성과급), severance (퇴직금), housing.",
            "• For E-7 visas, salary must meet the legal minimum threshold.",
            "• Be polite and data-driven — aggressive negotiation is uncommon.",
        ]
        tx = TB(L)
        x = lambda s: tx.get(s, s)
        rc1, rc2 = st.columns(2)
        with rc1:
            _link_card(f"🧰 {x('Career Tools')}", [
                (x("Wanted — Tech Jobs"), "https://www.wanted.co.kr"),
                (x("LinkedIn Korea"), "https://www.linkedin.com"),
                (x("Rocketpunch — Startups"), "https://www.rocketpunch.com"),
                (x("Korean Resume Templates"), "https://www.saramin.co.kr"),
                (x("Work24 Career Services"), "https://www.work.go.kr"),
            ])
        with rc2:
            tips = "<br>".join(x(s) for s in L[7:12])
            st.markdown(
                f'<div class="upk-card"><h3>🏢 {x("Korean Work Culture Tips")}</h3><p>{tips}</p></div>',
                unsafe_allow_html=True,
            )
        divider()
        sal = "<br>".join(x(s) for s in L[13:18])
        st.markdown(
            f'<div class="upk-card"><h3>💵 {x("Salary Negotiation Guide")}</h3><p>{sal}</p></div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# 13. JOB PAGE
# ════════════════════════════════════════════════════════════════════════════
JOB_ICONS = ["💻", "📈", "🎨", "🔬", "🏭", "📊", "🩺", "🌐", "📱", "🛠️"]


def _job_field(j, *keys, default="—"):
    for k in keys:
        v = j.get(k)
        if v:
            return v
    return default


def page_job():
    section_header(f"💼 {t('nav_job')}", t("feat_job_d"))
    tabs = st.tabs([t("job_board"), t("job_matches"), t("job_portals_tab")])

    jobs = load_jobs()

    # ───────────── TAB 1: JOB BOARD ─────────────
    with tabs[0]:
        f1, f2 = st.columns([2, 1])
        with f1:
            search = st.text_input(t("search"), placeholder=t("job_search_ph"), key="job_search")
        with f2:
            visa_types = sorted({_job_field(j, "visa_type", default="") for j in jobs if j.get("visa_type")})
            visa_filter = st.selectbox(t("job_visa_filter"), [t("all")] + visa_types, key="job_visa")

        filtered = []
        for j in jobs:
            role = _job_field(j, "role", "title", default="")
            company = _job_field(j, "company", default="")
            if search and search.lower() not in (role + " " + company).lower():
                continue
            if visa_filter != t("all") and _job_field(j, "visa_type", default="") != visa_filter:
                continue
            filtered.append(j)

        if not jobs:
            st.info(t("empty"))
        elif not filtered:
            st.warning("No jobs match your search. Try different keywords.")
        else:
            st.caption(f"{len(filtered)} / {len(jobs)} {t('nav_job').lower()}")
            for idx, j in enumerate(filtered):
                role = _job_field(j, "role", "title", default="Role")
                company = _job_field(j, "company", default="Company")
                location = _job_field(j, "location", default="Korea")
                salary = _job_field(j, "salary", default="Competitive")
                visa = _job_field(j, "visa_type", default="—")
                summary = localized(j, "summary") or _job_field(j, "summary", "ai_summary", default="")
                match = _job_field(j, "match_score", "match", default="—")
                apply_url = _job_field(j, "apply_url", "url", default="https://www.wanted.co.kr")
                icon = JOB_ICONS[idx % len(JOB_ICONS)]
                st.markdown(
                    f"""
                    <div class="job-card">
                        <div class="job-icon">{icon}</div>
                        <div class="job-mid">
                            <h4>{role}</h4>
                            <div class="job-meta">🏢 {company} &nbsp;•&nbsp; 📍 {location}</div>
                            <span class="tag tag-green">💰 {salary}</span>
                            <span class="tag tag-navy">🛂 {visa}</span>
                            <div class="job-ai">🤖 {summary}</div>
                        </div>
                        <div style="text-align:center;">
                            <div class="match-badge">{match}%<small>{t('job_match')}</small></div>
                            <a class="link-row" style="text-align:center;background:#EEF2FF;margin-top:8px;"
                               href="{apply_url}" target="_blank">{t('job_apply')} →</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ───────────── TAB 2: MY MATCHES ─────────────
    with tabs[1]:
        section_header(t("job_matches"), t("job_upload_cv"))
        st.markdown(
            f'<div class="upk-card"><h3>🎯 {t("job_upload_cv")}</h3>'
            f'<p>Our AI compares your skills with open roles and ranks the best fits.</p></div>',
            unsafe_allow_html=True,
        )
        cv_file = st.file_uploader(t("cv_upload"), type=["pdf", "docx", "txt"], key="match_cv")
        if st.button(t("job_find"), key="match_btn", use_container_width=True):
            if not cv_file:
                st.warning(t("cv_upload"))
            elif not jobs:
                st.info(t("empty"))
            else:
                with st.spinner(t("loading")):
                    text = extract_pdf_text(cv_file)
                    job_titles = [
                        f"{_job_field(j, 'role', 'title')} @ {_job_field(j, 'company')} "
                        f"({_job_field(j, 'visa_type', default='—')})"
                        for j in jobs[:20]
                    ]
                    prompt = (
                        "Given this candidate CV and the list of jobs, return ONLY valid JSON with key "
                        "'matches' = list of up to 5 objects {title, score (0-100), reason (short)}. "
                        "Pick the best fits.\n\nCV:\n" + text[:4000] + "\n\nJOBS:\n" + "\n".join(job_titles)
                    )
                    raw = llm_complete(prompt)
                    parsed = parse_json_block(raw) if raw else None
                    if parsed and isinstance(parsed.get("matches"), list):
                        st.session_state.job_match_result = parsed["matches"]
                    else:
                        # Heuristic fallback using existing match scores
                        st.session_state.job_match_result = [
                            {"title": f"{_job_field(j, 'role', 'title')} @ {_job_field(j, 'company')}",
                             "score": _job_field(j, "match_score", "match", default=70),
                             "reason": "Strong overlap with your profile."}
                            for j in sorted(jobs, key=lambda x: x.get("match_score", 0), reverse=True)[:5]
                        ]
                        if not AI_READY:
                            st.info("Showing heuristic matches (AI not configured).")

        matches = st.session_state.job_match_result
        if matches:
            st.write("")
            for m in matches:
                st.markdown(
                    f"""
                    <div class="job-card">
                        <div class="job-icon">⭐</div>
                        <div class="job-mid">
                            <h4>{m.get('title','Role')}</h4>
                            <div class="job-ai">🤖 {m.get('reason','')}</div>
                        </div>
                        <div class="match-badge">{m.get('score','—')}%<small>{t('job_match')}</small></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ───────────── TAB 3: PORTALS ─────────────
    with tabs[2]:
        section_header(t("job_portals_tab"), t("job_portals"), translate_sub=False)
        render_portals(JOB_PORTALS)


# ════════════════════════════════════════════════════════════════════════════
# 14. TOPIK PAGE
# ════════════════════════════════════════════════════════════════════════════
def _gf(row, *keys, default="—"):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return default


def page_topik():
    section_header(f"📊 {t('nav_topik')}", "Test of Proficiency in Korean — your full guide")
    tabs = st.tabs([
        t("topik_schedule"), t("topik_register"), t("topik_levels"),
        t("topik_tips"), t("topik_centers"),
    ])

    # ───────────── TAB 1: SCHEDULE ─────────────
    with tabs[0]:
        section_header(t("topik_schedule"), "Official test rounds and key dates")
        rows = load_topik()
        if not rows:
            st.info(t("empty"))
        else:
            body = ""
            for r in rows:
                ttype = str(_gf(r, "type", "test_type", default="PBT")).upper()
                if "IBT" in ttype:
                    tag = '<span class="tag tag-green">IBT</span>'
                elif "SPEAK" in ttype:
                    tag = '<span class="tag tag-orange">Speaking</span>'
                else:
                    tag = '<span class="tag tag-navy">PBT</span>'
                body += (
                    f"<tr><td>{_gf(r, 'session', 'round')}</td>"
                    f"<td>{_gf(r, 'test_date')}</td>"
                    f"<td>{_gf(r, 'registration', 'reg_period')}</td>"
                    f"<td>{_gf(r, 'results', 'result_date')}</td>"
                    f"<td>{tag}</td></tr>"
                )
            st.markdown(
                f"""
                <table class="upk-table">
                    <tr><th>Session</th><th>Test Date</th><th>Registration</th><th>Results</th><th>Type</th></tr>
                    {body}
                </table>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="glass-note" style="margin-top:14px;">📌 '
                    f'{T("Source: topik.go.kr official schedule. Always confirm dates on the official website.")}'
                    '</div>', unsafe_allow_html=True)

    # ───────────── TAB 2: REGISTER ─────────────
    with tabs[1]:
        section_header(t("topik_register"), "Step-by-step registration guide")
        steps = [
            ("Visit topik.go.kr", "Go to the official TOPIK website and switch to your language."),
            ("Create an account", "Register using your passport number and personal details."),
            ("Select round & level", "Choose the test round and either TOPIK I or TOPIK II."),
            ("Choose test location", "Pick a test center near you (seats fill quickly!)."),
            ("Pay the registration fee", "Pay online with a card or supported method."),
            ("Download admission ticket", "Print or save your admission ticket (수험표)."),
            ("Bring passport + ticket", "On exam day, bring your passport and admission ticket."),
        ]
        stx = TB([s[0] for s in steps] + [s[1] for s in steps] + ["Format"])
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(
                f'<div class="upk-card" style="margin-bottom:10px;display:flex;gap:16px;align-items:center;">'
                f'<div class="flow-num" style="margin:0;flex-shrink:0;">{i}</div>'
                f'<div><h4 style="margin:0 0 4px 0;">{stx.get(title, title)}</h4>'
                f'<p style="margin:0;color:var(--muted);">{stx.get(desc, desc)}</p></div></div>',
                unsafe_allow_html=True,
            )
        divider()
        st.markdown(
            f"""
            <div class="upk-card"><h3>💳 {t('topik_fee')}</h3>
            <table class="upk-table">
                <tr><th>{stx.get("Format", "Format")}</th><th>TOPIK I</th><th>TOPIK II</th></tr>
                <tr><td>PBT (Paper)</td><td>40,000 KRW</td><td>55,000 KRW</td></tr>
                <tr><td>IBT (Internet)</td><td>70,000 KRW</td><td>95,000 KRW</td></tr>
            </table></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<a class="link-row" style="text-align:center;background:#EEF2FF;margin-top:12px;" '
                    f'href="https://www.topik.go.kr" target="_blank">{t("topik_reg_btn")}</a>',
                    unsafe_allow_html=True)

    # ───────────── TAB 3: LEVELS ─────────────
    with tabs[2]:
        section_header(t("topik_levels"), "Understand the six TOPIK levels")
        infos = load_topik_info()
        if not infos:
            st.info(t("empty"))
        else:
            cols = st.columns(2)
            for i, lv in enumerate(infos):
                level_name = _gf(lv, "level", "level_name", default="TOPIK")
                score = _gf(lv, "score_range", "score", default="—")
                desc = localized(lv, "description") or _gf(lv, "description", "description_en", default="")
                is_topik2 = "II" in str(level_name) or any(x in str(level_name) for x in ["3", "4", "5", "6"])
                grad = ("linear-gradient(135deg,#3D5AFE,#12A580)" if is_topik2
                        else "linear-gradient(135deg,#16233F,#2E4B8A)")
                with cols[i % 2]:
                    st.markdown(
                        f'<div class="upk-card" style="border-top:5px solid transparent;'
                        f'background:linear-gradient(#fff,#fff) padding-box,{grad} border-box;'
                        f'border:2px solid transparent;">'
                        f'<span class="tag" style="background:{grad};color:#fff;">{level_name}</span>'
                        f'<h3 style="margin-top:10px;">📊 {score}</h3>'
                        f'<p>{desc}</p></div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")

        divider()
        section_header(T("Test Format"), translate_sub=False)
        struct = load_topik_structure()
        if not struct:
            st.info(t("empty"))
        else:
            th = TB(["Type", "Section", "Duration", "Questions", "Max Score"])
            body = "".join(
                f"<tr><td>{_gf(s, 'type', 'test_type')}</td><td>{_gf(s, 'section')}</td>"
                f"<td>{_gf(s, 'duration')}</td><td>{_gf(s, 'questions')}</td>"
                f"<td>{_gf(s, 'max_score')}</td></tr>"
                for s in struct
            )
            st.markdown(
                f'<table class="upk-table"><tr><th>{th.get("Type","Type")}</th>'
                f'<th>{th.get("Section","Section")}</th><th>{th.get("Duration","Duration")}</th>'
                f'<th>{th.get("Questions","Questions")}</th><th>{th.get("Max Score","Max Score")}</th></tr>'
                f'{body}</table>',
                unsafe_allow_html=True,
            )

        divider()
        section_header(T("FAQ"), translate_sub=False)
        faqs = load_topik_faq()
        if not faqs:
            st.info(t("empty"))
        else:
            for f in faqs:
                q = localized(f, "question") or _gf(f, "question", "question_en")
                a = localized(f, "answer") or _gf(f, "answer", "answer_en")
                with st.expander(f"❓ {q}"):
                    st.write(a)

    # ───────────── TAB 4: STUDY TIPS ─────────────
    with tabs[3]:
        section_header(t("topik_tips"), "Resources and a study timeline")
        rc1, rc2 = st.columns(2)
        with rc1:
            _link_card(f"📚 {T('Free Resources')}", [
                ("Free Past Papers (TOPIK Guide)", "https://www.topikguide.com"),
                ("Official TOPIK Samples", "https://www.topik.go.kr"),
                ("Korean News (Easy)", "https://www.kbs.co.kr"),
                ("Naver Korean Dictionary", "https://dict.naver.com"),
            ])
        with rc2:
            _link_card(f"📱 {T('Mobile Apps')}", [
                ("TOPIK ONE — Practice", "https://play.google.com"),
                ("Memrise — Vocabulary", "https://www.memrise.com"),
                ("Anki — Flashcards", "https://apps.ankiweb.net"),
                ("HelloTalk — Language Exchange", "https://www.hellotalk.com"),
            ])
        divider()
        TL = [
            "Recommended Study Timeline",
            "• <b>Months 1–2:</b> Master Hangul, basic grammar & 800 core words (→ TOPIK I).",
            "• <b>Months 3–4:</b> Intermediate grammar, reading speed & 1,500 words.",
            "• <b>Months 5–6:</b> Past papers, listening drills & timed mock tests.",
            "• <b>Final 2 weeks:</b> Full mock exams under timed conditions, review weak areas.",
            "• <b>Daily habit:</b> 30 minutes of Korean news + 20 new words.",
        ]
        ttx = TB(TL)
        timeline = "<br>".join(ttx.get(s, s) for s in TL[1:])
        st.markdown(
            f'<div class="upk-card"><h3>🗓️ {ttx.get(TL[0], TL[0])}</h3><p>{timeline}</p></div>',
            unsafe_allow_html=True,
        )

    # ───────────── TAB 5: TEST CENTERS ─────────────
    with tabs[4]:
        section_header(t("topik_centers"), "Find a test center near you")
        centers = load_topik_centers()
        city_q = st.text_input(t("search"), placeholder=t("tc_search"), key="tc_search")
        if not centers:
            st.info(t("empty"))
        else:
            shown = [
                c for c in centers
                if not city_q or city_q.lower() in str(_gf(c, "city", "city_en", default="")).lower()
            ]
            if not shown:
                st.warning("No centers found for that city.")
            cols = st.columns(2)
            for i, c in enumerate(shown):
                city = _gf(c, "city_en", "city")
                center = _gf(c, "test_center_en", "test_center", "center")
                address = _gf(c, "address")
                contact = _gf(c, "contact", "phone")
                with cols[i % 2]:
                    st.markdown(
                        f'<div class="upk-card"><span class="tag tag-navy">📍 {city}</span>'
                        f'<h4 style="margin:10px 0 6px 0;">{center}</h4>'
                        f'<p>🏠 {address}<br>☎️ {contact}</p></div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")


# ════════════════════════════════════════════════════════════════════════════
# 15. VISA PAGE
# ════════════════════════════════════════════════════════════════════════════
VISA_META = {
    "D-2": ("🎓", "Student Visa", "Degree-seeking students at Korean universities"),
    "D-4": ("📖", "Language Training Visa", "Korean language program students"),
    "E-7": ("💼", "Specialized Work Visa", "Skilled professionals with a job offer"),
    "F-2": ("🏡", "Residence Visa", "Long-term residents (points-based)"),
    "F-5": ("⭐", "Permanent Residency", "Permanent residents of Korea"),
}
VISA_ORDER = ["D-2", "D-4", "E-7", "F-2", "F-5"]


def _visa_lookup(rows, code):
    for r in rows:
        if str(r.get("visa_code", "")).upper().replace(" ", "") == code.replace(" ", ""):
            return r
    return {}


def _split_list(text, sep="|"):
    if not text:
        return []
    return [x.strip() for x in str(text).split(sep) if x.strip()]


def page_visa():
    section_header(f"🛂 {t('nav_visa')}", t("feat_visa_d"))
    rows = load_visa()
    tabs = st.tabs(VISA_ORDER)

    for i, code in enumerate(VISA_ORDER):
        with tabs[i]:
            icon, title, for_who = VISA_META[code]
            row = _visa_lookup(rows, code)

            # Fields with sensible defaults so the page is useful even pre-data.
            # DB-provided values are shown as-is; English defaults are translated.
            db_for = localized(row, "for_who") or row.get("for_who")
            fee = row.get("fee") or "60,000 KRW (single)"
            duration = row.get("duration") or "1–2 years (renewable)"
            proc = row.get("processing_time") or "2–4 weeks"
            req_from_db = _split_list(row.get("requirements"))
            doc_from_db = _split_list(row.get("documents"))
            requirements = req_from_db or [
                "Valid passport (6+ months)",
                "Certificate of Admission / employment contract",
                "Proof of financial means",
                "Completed visa application form",
            ]
            documents = doc_from_db or [
                "Passport & passport photo",
                "Application form",
                "Certificate of Admission (for students)",
                "Bank statement / financial proof",
                "Health & background documents (if required)",
            ]

            # Translate English content (title, defaults). Keep DB text & numbers fixed.
            to_translate = [title]
            if not db_for:
                to_translate.append(for_who)
            if not row.get("fee"):
                to_translate.append(fee)
            if not row.get("duration"):
                to_translate.append(duration)
            if not row.get("processing_time"):
                to_translate.append(proc)
            if not req_from_db:
                to_translate += requirements
            if not doc_from_db:
                to_translate += documents
            tx = TB(to_translate)
            x = lambda s: tx.get(s, s)

            title = x(title)
            db_for = db_for or x(for_who)
            if not row.get("fee"):
                fee = x(fee)
            if not row.get("duration"):
                duration = x(duration)
            if not row.get("processing_time"):
                proc = x(proc)
            if not req_from_db:
                requirements = [x(r) for r in requirements]
            if not doc_from_db:
                documents = [x(d) for d in documents]

            req_html = "".join(
                f'<div class="check-item"><span class="ck">✓</span>{r}</div>' for r in requirements)
            doc_html = "".join(
                f'<div class="check-item"><span class="ck">📄</span>{d}</div>' for d in documents)

            st.markdown(
                f"""
                <div class="visa-card">
                    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:16px;">
                        <div class="visa-head">
                            <div class="vi">{icon}</div>
                            <div><h2 style="margin:0;">{code} — {title}</h2>
                            <p style="color:var(--muted);margin:4px 0 0 0;">👤 {db_for}</p></div>
                        </div>
                        <div style="display:flex;gap:12px;">
                            <div class="info-pill"><div class="ip-l">{t('visa_fee')}</div>
                                <div class="ip-v">{fee}</div></div>
                            <div class="info-pill"><div class="ip-l">{t('visa_dur')}</div>
                                <div class="ip-v">{duration}</div></div>
                        </div>
                    </div>
                    <hr class="soft-divider"/>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
                        <div><h3>✅ {t('visa_req')}</h3>{req_html}</div>
                        <div><h3>📋 {t('visa_docs')}</h3>{doc_html}</div>
                    </div>
                    <hr class="soft-divider"/>
                    <div class="glass-note">⏱️ <b>{t('visa_proc')}:</b> {proc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                '<a class="link-row" style="text-align:center;background:#16233F;color:#fff;'
                'margin-top:14px;font-weight:700;" href="https://www.hikorea.go.kr" target="_blank">'
                f'{t("visa_apply")}</a>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# 16. LIFE PAGE
# ════════════════════════════════════════════════════════════════════════════
def _life_card(x, icon, title, lines, tag=None, tag_class="tag-navy", mt=False):
    """Render an info card whose text is translated via the map `x`."""
    body = "<br>".join(x(l) for l in lines)
    tag_html = (f'<span class="tag {tag_class}">{tag}</span>' if tag else "")
    style = 'style="margin-top:14px;"' if mt else ""
    return (
        f'<div class="upk-card" {style}><h3>{icon} {x(title)}</h3>{tag_html}'
        f'<p style="margin-top:10px;">{body}</p></div>'
    )


def page_life():
    section_header(f"🏠 {t('nav_life')}", "Everything you need for daily life in Korea")
    tabs = st.tabs([
        t("life_housing"), t("life_transport"), t("life_health"),
        t("life_banking"), t("life_safety"),
    ])

    # ───────────── HOUSING ─────────────
    with tabs[0]:
        section_header(f"🏠 {t('life_housing')}", "Find the right place to live")
        L = [
            "University Dormitory", "<b>Pros:</b> Cheap, on campus, easy to make friends, furnished.",
            "<b>Cons:</b> Curfews, roommates, limited privacy, competitive spots.",
            "One-room (원룸)", "<b>Pros:</b> Privacy, own kitchen & bathroom, flexible.",
            "<b>Cons:</b> Deposit (보증금) required, utility bills, contracts in Korean.",
            "Goshiwon (고시원)", "<b>Pros:</b> Cheapest, no deposit, short-term friendly.",
            "<b>Cons:</b> Very small rooms, shared facilities, thin walls.",
            "Housing Tips",
            "• Use apps: <b>Naver Real Estate (네이버 부동산)</b>, <b>Zigbang (직방)</b>, <b>Dabang (다방)</b>.",
            "• <b>전세 (Jeonse):</b> Large lump-sum deposit, no monthly rent — returned when you leave.",
            "• <b>월세 (Wolse):</b> Smaller deposit + monthly rent — most common for students.",
            "• Always verify the landlord and contract; bring a Korean-speaking friend if possible.",
            "• Check distance to subway, supermarket and your campus before signing.",
        ]
        tx = TB(L)
        x = lambda s: tx.get(s, s)
        c1, c2, c3 = st.columns(3)
        c1.markdown(_life_card(x, "🏫", "University Dormitory",
                    [L[1], L[2]], "~300,000–600,000 KRW/mo", "tag-green"), unsafe_allow_html=True)
        c2.markdown(_life_card(x, "🚪", "One-room (원룸)",
                    [L[4], L[5]], "~400,000–800,000 KRW/mo", "tag-navy"), unsafe_allow_html=True)
        c3.markdown(_life_card(x, "🛏️", "Goshiwon (고시원)",
                    [L[7], L[8]], "~300,000–500,000 KRW/mo", "tag-orange"), unsafe_allow_html=True)
        divider()
        st.markdown(_life_card(x, "💡", "Housing Tips", L[10:15]), unsafe_allow_html=True)

    # ───────────── TRANSPORT ─────────────
    with tabs[1]:
        section_header(f"🚇 {t('life_transport')}", "Get around Korea like a local")
        L = [
            "T-money Card",
            "• Buy at any convenience store (GS25, CU, 7-Eleven) for ~2,500 KRW.",
            "• Charge (충전) with cash at stores or station machines.",
            "• Tap on entry and exit for subway, bus and even taxis.",
            "• Works nationwide — one card for the whole country.",
            "Bus Guide",
            "• Blue = main routes, Green = local, Red = express, Yellow = circular.",
            "• Tap T-money on entry AND exit to get transfer discounts.",
            "• Use <b>Naver Map</b> or <b>Kakao Bus</b> for live arrival times.",
            "Subway Guide",
            "• Use <b>Kakao Metro</b> or <b>Naver Maps</b> for the fastest routes.",
            "• Lines are color-coded and signs are in English too.",
            "• Avoid rush hours (08:00–09:00, 18:00–19:00) when possible.",
            "Taxi",
            "• Use the <b>Kakao T</b> app to call and pay easily.",
            "• Pay with T-money, card or cash.",
            "• Base fare ~4,800 KRW in Seoul; late-night surcharge applies.",
            "Typical subway/bus fare:",
        ]
        tx = TB(L)
        x = lambda s: tx.get(s, s)
        c1, c2 = st.columns(2)
        c1.markdown(_life_card(x, "💳", "T-money Card", L[1:5]), unsafe_allow_html=True)
        c1.markdown(_life_card(x, "🚌", "Bus Guide", L[6:9], mt=True), unsafe_allow_html=True)
        c2.markdown(_life_card(x, "🚇", "Subway Guide", L[10:13]), unsafe_allow_html=True)
        c2.markdown(_life_card(x, "🚕", "Taxi", L[14:17], mt=True), unsafe_allow_html=True)
        st.markdown(f'<div class="glass-note" style="margin-top:14px;">💰 {x(L[17])} '
                    '<b>~1,400 KRW</b>.</div>', unsafe_allow_html=True)

    # ───────────── HEALTH ─────────────
    with tabs[2]:
        section_header(f"🏥 {t('life_health')}", "Stay healthy and insured in Korea")
        L = [
            "National Health Insurance (NHIS)",
            "• <b>Mandatory</b> for international students after 6 months of stay.",
            "• Monthly cost: <b>~130,000 KRW</b> for international students.",
            "• Covers a large portion of hospital and clinic costs.",
            "• Download the <b>건강보험 (NHIS)</b> app to manage your account.",
            "How to See a Doctor",
            "• Find hospitals/clinics (병원/의원) with <b>Naver Maps</b>.",
            "• Bring your <b>ARC card</b> for insurance billing.",
            "• Many big hospitals have international clinics with English support.",
            "• Pharmacies (약국) are everywhere for prescriptions.",
            "Emergency Numbers", "Number", "Service",
            "Ambulance & Fire", "Police", "Medical Consultation (24/7)",
        ]
        tx = TB(L)
        x = lambda s: tx.get(s, s)
        c1, c2 = st.columns(2)
        c1.markdown(_life_card(x, "🩺", "National Health Insurance (NHIS)", L[1:5]), unsafe_allow_html=True)
        c2.markdown(_life_card(x, "🏨", "How to See a Doctor", L[6:10]), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="upk-card" style="margin-top:14px;"><h3>🚨 {x("Emergency Numbers")}</h3>
            <table class="upk-table">
                <tr><th>{x("Number")}</th><th>{x("Service")}</th></tr>
                <tr><td>119</td><td>{x("Ambulance & Fire")}</td></tr>
                <tr><td>112</td><td>{x("Police")}</td></tr>
                <tr><td>1339</td><td>{x("Medical Consultation (24/7)")}</td></tr>
            </table></div>
            """,
            unsafe_allow_html=True,
        )

    # ───────────── BANKING ─────────────
    with tabs[3]:
        section_header(f"🏦 {t('life_banking')}", "Open an account and manage money")
        L = [
            "Required Documents", "ARC (Alien Registration Card)", "Passport",
            "University enrollment certificate", "Korean phone number",
            "Foreigner-friendly Banks",
            "• <b>KEB Hana Bank</b> — strong foreign-customer support.",
            "• <b>IBK</b> (Industrial Bank of Korea).",
            "• <b>Woori Bank</b> and <b>Shinhan Bank</b>.",
            "Steps to Open an Account",
            "1. Visit a branch with your documents.",
            "2. Fill out the application (staff often help in English).",
            "3. Receive your debit card & bankbook (통장).",
            "4. Set up mobile banking with your card & ID.",
            "5. Register for online transfers (may need extra verification).",
            "Sending Money Abroad",
            "• <b>Wise</b> for low-fee international transfers.",
            "• <b>WeChat Pay / Alipay</b> popular for Chinese students.",
            "• Banks offer remittance but with higher fees.",
        ]
        tx = TB(L)
        x = lambda s: tx.get(s, s)
        c1, c2 = st.columns(2)
        docs = "".join(f'<div class="check-item"><span class="ck">✓</span>{x(d)}</div>' for d in L[1:5])
        c1.markdown(f'<div class="upk-card"><h3>📋 {x("Required Documents")}</h3>{docs}</div>',
                    unsafe_allow_html=True)
        c1.markdown(_life_card(x, "🏦", "Foreigner-friendly Banks", L[6:9], mt=True), unsafe_allow_html=True)
        c2.markdown(_life_card(x, "🪪", "Steps to Open an Account", L[10:15]), unsafe_allow_html=True)
        c2.markdown(_life_card(x, "🌍", "Sending Money Abroad", L[16:19], mt=True), unsafe_allow_html=True)

    # ───────────── SAFETY ─────────────
    with tabs[4]:
        section_header(f"🛟 {t('life_safety')}", "Stay safe and know who to call")
        L = [
            "Emergency & Support Numbers", "Number", "Service",
            "Ambulance & Fire", "Police", "Seoul City Helpline (multilingual)",
            "Tourism Hotline (24/7, multilingual)", "Immigration Contact Center",
            "Mental Health / Suicide Prevention",
            "Daily Safety Tips",
            "• Korea is very safe, but stay alert late at night.",
            "• Keep your ARC and passport copies in a safe place.",
            "• Beware of phone/voice phishing (보이스피싱) scams.",
            "• Save your address in Korean for taxis and emergencies.",
            "Useful Apps & Centers",
            "• <b>Emergency Ready App (안전디딤돌)</b> — disaster alerts in English.",
            "• <b>112/119 apps</b> for silent emergency reporting.",
            "• Visit your city's <b>Global / Foreigner Support Center</b> for free help.",
            "• Seoul Global Center offers free legal & life counseling.",
        ]
        tx = TB(L)
        x = lambda s: tx.get(s, s)
        st.markdown(
            f"""
            <div class="upk-card"><h3>📞 {x("Emergency & Support Numbers")}</h3>
            <table class="upk-table">
                <tr><th>{x("Number")}</th><th>{x("Service")}</th></tr>
                <tr><td>119</td><td>{x("Ambulance & Fire")}</td></tr>
                <tr><td>112</td><td>{x("Police")}</td></tr>
                <tr><td>120</td><td>{x("Seoul City Helpline (multilingual)")}</td></tr>
                <tr><td>1330</td><td>{x("Tourism Hotline (24/7, multilingual)")}</td></tr>
                <tr><td>1345</td><td>{x("Immigration Contact Center")}</td></tr>
                <tr><td>1393</td><td>{x("Mental Health / Suicide Prevention")}</td></tr>
            </table></div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.markdown(_life_card(x, "🛡️", "Daily Safety Tips", L[10:14], mt=True), unsafe_allow_html=True)
        c2.markdown(_life_card(x, "📱", "Useful Apps & Centers", L[15:19], mt=True), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 17. FLOATING CHATBOT (sidebar)
# ════════════════════════════════════════════════════════════════════════════
def floating_chat():
    t_val = TR.get(st.session_state.get("lang", "🇺🇸 English"), TR["🇺🇸 English"])
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                <div style="width:48px;height:48px;border-radius:50%;
                    background:linear-gradient(135deg,#3D5AFE,#16233F);display:flex;
                    align-items:center;justify-content:center;font-size:24px;">🤖</div>
                <div><div style="font-size:18px;font-weight:800;color:#fff;">{t_val.get('chat_title','UNI Assistant')}</div>
                <div style="font-size:12px;color:rgba(255,255,255,0.8);">{t_val.get('chat_sub','')}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chat_container = st.container(height=400)
        if not st.session_state.chat_history:
            with chat_container.chat_message("assistant"):
                st.write("👋 " + t_val.get("placeholder", "Ask me anything about Korea..."))
        for msg in st.session_state.chat_history:
            with chat_container.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("source"):
                    st.caption(f"📍 {msg['source']}")

        if prompt := st.chat_input(t_val.get("placeholder", "Ask me anything...")):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("..."):
                ans, source = get_rag_response(prompt)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": ans, "source": source})
            try:
                if supabase is not None:
                    supabase.table("chat_history").insert({
                        "question": prompt,
                        "answer": ans,
                        "source": source,
                        "lang": st.session_state.lang,
                        "created_at": datetime.utcnow().isoformat(),
                    }).execute()
            except Exception:
                pass
            st.rerun()

        st.divider()

        # ── Email notification subscription ──
        st.markdown(f"#### 🔔 {t_val.get('subscribe','Get Email Updates')}")
        email = st.text_input(t_val.get("email_label", "Your email"), key="notif_email")
        topics = st.multiselect(
            t_val.get("topics", "Topics of interest"),
            ["TOPIK Updates", "Visa News", "Job Alerts", "Scholarship Info", "Life Tips"],
            key="notif_topics",
        )
        if st.button(t_val.get("submit", "Submit"), key="sub_btn"):
            if email and "@" in email and "." in email:
                try:
                    if supabase is not None:
                        supabase.table("notifications").upsert({
                            "email": email,
                            "topics": topics,
                            "lang": st.session_state.lang,
                            "created_at": datetime.utcnow().isoformat(),
                        }).execute()
                    st.success(t_val.get("sub_success", "✅ Subscribed!"))
                except Exception:
                    st.success(t_val.get("sub_success", "✅ Subscribed!"))
            else:
                st.error(t_val.get("invalid_email", "Please enter a valid email"))


# ════════════════════════════════════════════════════════════════════════════
# 18. ADMIN PANEL
# ════════════════════════════════════════════════════════════════════════════
def admin_panel():
    with st.sidebar.expander(t("admin_panel")):
        password = st.text_input(t("admin_pw"), type="password", key="admin_pw_input")
        if not password:
            return
        if password != _secret("ADMIN_PASSWORD", "admin"):
            st.error("❌ Incorrect password")
            return
        st.success(t("admin_ok"))

        tab1, tab2 = st.tabs([t("admin_pdf"), t("admin_stats")])

        # ── PDF upload → vector store ──
        with tab1:
            st.info("Upload PDFs to build the AI knowledge base")
            files = st.file_uploader("Upload PDFs", type=["pdf"],
                                     accept_multiple_files=True, key="admin_pdfs")
            chunk_size = st.slider("Chunk Size (words)", 100, 800, 400, 50, key="admin_chunk")
            overlap = st.slider("Overlap (words)", 0, 200, 50, 10, key="admin_overlap")

            if files and st.button("🚀 Vectorize & Upload", key="admin_vec"):
                if supabase is None or not AI_READY:
                    st.error("Supabase / AI not configured. Add secrets first.")
                else:
                    import pypdf
                    total_chunks = 0
                    for f in files:
                        with st.spinner(f"Processing {f.name}..."):
                            try:
                                reader = pypdf.PdfReader(io.BytesIO(f.read()))
                                text = "".join([p.extract_text() or "" for p in reader.pages])
                                words = text.split()
                                step = max(1, chunk_size - overlap)
                                chunks = [
                                    " ".join(words[i:i + chunk_size])
                                    for i in range(0, len(words), step)
                                    if words[i:i + chunk_size]
                                ]
                                progress = st.progress(0, f"Uploading {f.name}...")
                                for idx, chunk in enumerate(chunks):
                                    embedding = Settings.embed_model.get_text_embedding(chunk)
                                    supabase.table("documents").insert({
                                        "content": chunk,
                                        "metadata": {"source": f.name, "chunk": idx},
                                        "embedding": embedding,
                                    }).execute()
                                    progress.progress((idx + 1) / len(chunks))
                                    total_chunks += 1
                                st.success(f"✅ {f.name} → {len(chunks)} chunks")
                            except Exception as e:
                                st.error(f"❌ {f.name}: {e}")
                    if total_chunks > 0:
                        st.balloons()
                        st.toast(f"🎉 {total_chunks} chunks uploaded!")

            if st.button("📊 View Document Stats", key="admin_docstats"):
                try:
                    res = supabase.table("documents").select("metadata").execute()
                    sources = {}
                    for r in res.data:
                        src = (r.get("metadata") or {}).get("source", "Unknown")
                        sources[src] = sources.get(src, 0) + 1
                    st.dataframe(pd.DataFrame(
                        [{"File": k, "Chunks": v} for k, v in sources.items()]))
                except Exception:
                    st.error("Could not load stats")

        # ── Stats & subscriber export ──
        with tab2:
            col1, col2, col3 = st.columns(3)
            try:
                chats = supabase.table("chat_history").select("id", count="exact").execute()
                col1.metric("Total Chats", chats.count or 0)
            except Exception:
                col1.metric("Total Chats", "—")
            try:
                docs = supabase.table("documents").select("id", count="exact").execute()
                col2.metric("KB Documents", docs.count or 0)
            except Exception:
                col2.metric("KB Documents", "—")
            try:
                subs = supabase.table("notifications").select("id", count="exact").execute()
                col3.metric("Subscribers", subs.count or 0)
            except Exception:
                col3.metric("Subscribers", "—")

            col1.metric("Active Language", st.session_state.lang[:6])

            # Subscriber breakdowns
            try:
                res = supabase.table("notifications").select("*").execute()
                if res.data:
                    df = pd.DataFrame(res.data)
                    st.markdown("**Breakdown by language**")
                    if "lang" in df.columns:
                        st.dataframe(df["lang"].value_counts().rename_axis("Language")
                                     .reset_index(name="Subscribers"), use_container_width=True)
                    topic_counts = {}
                    for tlist in df.get("topics", []):
                        for tp in (tlist or []):
                            topic_counts[tp] = topic_counts.get(tp, 0) + 1
                    if topic_counts:
                        st.markdown("**Breakdown by topic**")
                        st.dataframe(pd.DataFrame(
                            [{"Topic": k, "Subscribers": v} for k, v in topic_counts.items()]),
                            use_container_width=True)
                    st.download_button("📥 Export Subscribers CSV",
                                       df.to_csv(index=False), "subscribers.csv",
                                       key="admin_export")
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════
# 19. AUTH
# ════════════════════════════════════════════════════════════════════════════
def render_auth():
    """Ensure auth-related session keys exist (defensive; init_state already does)."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"


def page_auth():
    section_header(t("auth_welcome"),
                   t("auth_login_sub") if st.session_state.auth_mode == "login" else t("auth_reg_sub"))

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.session_state.auth_mode == "register":
            _register_form()
        else:
            _login_form()


def _register_form():
    card_open()
    st.markdown(f"### 📝 {t('register')}")
    name = st.text_input(t("full_name"), key="reg_name")
    email = st.text_input(t("email"), key="reg_email")
    lang_pref = st.selectbox(t("lang_pref"), LANGUAGES,
                             index=LANGUAGES.index(st.session_state.lang), key="reg_lang")
    pw = st.text_input(t("password"), type="password", key="reg_pw")
    pw2 = st.text_input(t("confirm_pw"), type="password", key="reg_pw2")

    st.markdown(f"**{t('notif_pref')}**")
    n_topik = st.checkbox(t("notif_topik"), value=True, key="reg_n_topik")
    n_visa = st.checkbox(t("notif_visa"), value=True, key="reg_n_visa")
    n_job = st.checkbox(t("notif_job"), value=True, key="reg_n_job")
    n_uni = st.checkbox(t("notif_uni"), value=False, key="reg_n_uni")
    n_schol = st.checkbox(t("notif_scholar"), value=False, key="reg_n_schol")

    if st.button(t("register"), key="reg_submit", use_container_width=True):
        if not name or not email or not pw:
            st.error(t("fill_all"))
        elif "@" not in email or "." not in email:
            st.error(t("invalid_email"))
        elif pw != pw2:
            st.error(t("pw_mismatch"))
        else:
            topics = []
            if n_topik: topics.append("TOPIK Updates")
            if n_visa: topics.append("Visa News")
            if n_job: topics.append("Job Alerts")
            if n_uni: topics.append("University News")
            if n_schol: topics.append("Scholarship Info")
            record = {
                "name": name, "email": email, "lang": lang_pref,
                "topics": topics, "created_at": datetime.utcnow().isoformat(),
            }
            try:
                if supabase is not None:
                    supabase.table("users").upsert(record).execute()
                    # Also seed the notifications table for this subscriber.
                    supabase.table("notifications").upsert({
                        "email": email, "topics": topics, "lang": lang_pref,
                        "created_at": datetime.utcnow().isoformat(),
                    }).execute()
            except Exception:
                pass
            st.session_state.user = record
            st.session_state.lang = lang_pref
            st.success(t("reg_success"))
            st.balloons()
            goto("HOME")

    st.write("")
    if st.button(t("have_account"), key="to_login", use_container_width=True):
        st.session_state.auth_mode = "login"
        st.rerun()
    card_close()


def _login_form():
    card_open()
    st.markdown(f"### 🔑 {t('login')}")
    email = st.text_input(t("email"), key="login_email")
    pw = st.text_input(t("password"), type="password", key="login_pw")

    if st.button(t("login"), key="login_submit", use_container_width=True):
        if not email:
            st.error(t("fill_all"))
        else:
            user = None
            try:
                if supabase is not None:
                    res = supabase.table("users").select("*").eq("email", email).execute()
                    if res.data:
                        user = res.data[0]
            except Exception:
                user = None
            if user:
                st.session_state.user = user
                if user.get("lang") in LANGUAGES:
                    st.session_state.lang = user["lang"]
                st.success(t("login_success"))
                goto("HOME")
            else:
                st.error(t("login_fail"))

    st.write("")
    if st.button(t("no_account"), key="to_register", use_container_width=True):
        st.session_state.auth_mode = "register"
        st.rerun()
    card_close()


def page_profile():
    user = st.session_state.user
    if not user:
        st.info("Please sign in first.")
        if st.button(t("login"), key="profile_login"):
            goto("AUTH")
        return

    section_header(t("my_profile"))
    card_open()
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:18px;">
            <div style="width:70px;height:70px;border-radius:50%;
                background:linear-gradient(135deg,#16233F,#3D5AFE);display:flex;
                align-items:center;justify-content:center;font-size:32px;color:#fff;">👤</div>
            <div><h3 style="margin:0;">{user.get('name','Student')}</h3>
            <p style="margin:4px 0;color:var(--muted);">✉️ {user.get('email','')}</p>
            <p style="margin:0;color:var(--muted);">🌐 {user.get('lang','')}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_close()

    divider()
    st.markdown(f"#### 🔔 {t('edit_notif')}")
    existing = user.get("topics", []) or []
    n_topik = st.checkbox(t("notif_topik"), value="TOPIK Updates" in existing, key="p_topik")
    n_visa = st.checkbox(t("notif_visa"), value="Visa News" in existing, key="p_visa")
    n_job = st.checkbox(t("notif_job"), value="Job Alerts" in existing, key="p_job")
    n_uni = st.checkbox(t("notif_uni"), value="University News" in existing, key="p_uni")
    n_schol = st.checkbox(t("notif_scholar"), value="Scholarship Info" in existing, key="p_schol")
    if st.button(t("save"), key="profile_save"):
        topics = []
        if n_topik: topics.append("TOPIK Updates")
        if n_visa: topics.append("Visa News")
        if n_job: topics.append("Job Alerts")
        if n_uni: topics.append("University News")
        if n_schol: topics.append("Scholarship Info")
        st.session_state.user["topics"] = topics
        try:
            if supabase is not None:
                supabase.table("users").update({"topics": topics}).eq(
                    "email", user.get("email")).execute()
        except Exception:
            pass
        st.success(t("profile_saved"))

    divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {t('saved_uni')}")
        st.markdown(f'<div class="glass-note">{t("none_yet")}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f"#### {t('saved_jobs')}")
        st.markdown(f'<div class="glass-note">{t("none_yet")}</div>', unsafe_allow_html=True)

    divider()
    if st.button(f"🚪 {t('logout')}", key="profile_logout", type="primary"):
        st.session_state.user = None
        st.session_state.chat_history = []
        goto("HOME")


def render_footer():
    L = [
        "Your complete AI guide to studying, working and living in South Korea.",
        "Built with care for international students, in 9 languages.",
        "Explore", "Universities", "TOPIK", "Visas", "Jobs",
        "Official", "Study in Korea", "HiKorea", "TOPIK Official", "Work24",
        "Support", "AI Assistant", "Email Updates", "Sign In",
        "Made with ❤️ for international students.", "Not affiliated with the Korean government.",
    ]
    tx = TB(L)
    x = lambda s: tx.get(s, s)
    st.markdown(
        f"""
        <div class="upk-footer">
            <div class="upk-foot-logo">🎓 UniPath <span style="color:#06C684;">Korea</span></div>
            <div style="max-width:520px;color:rgba(255,255,255,0.82);font-size:14px;margin-top:8px;">
                {x(L[0])} {x(L[1])}
            </div>
            <div class="upk-foot-cols">
                <div>
                    <h5>{x("Explore")}</h5>
                    <a href="#">🏛️ {x("Universities")}</a>
                    <a href="#">📊 {x("TOPIK")}</a>
                    <a href="#">🛂 {x("Visas")}</a>
                    <a href="#">💼 {x("Jobs")}</a>
                </div>
                <div>
                    <h5>{x("Official")}</h5>
                    <a href="https://www.studyinkorea.go.kr" target="_blank">{x("Study in Korea")}</a>
                    <a href="https://www.hikorea.go.kr" target="_blank">HiKorea</a>
                    <a href="https://www.topik.go.kr" target="_blank">{x("TOPIK Official")}</a>
                    <a href="https://www.work.go.kr" target="_blank">Work24</a>
                </div>
                <div>
                    <h5>{x("Support")}</h5>
                    <a href="#">💬 {x("AI Assistant")}</a>
                    <a href="#">🔔 {x("Email Updates")}</a>
                    <a href="#">🔑 {x("Sign In")}</a>
                </div>
                <div>
                    <h5>9 Languages</h5>
                    <div style="font-size:22px;line-height:1.6;">🇺🇸 🇰🇷 🇲🇳 🇯🇵 🇨🇳<br>🇻🇳 🇹🇭 🇲🇾 🇷🇺</div>
                </div>
            </div>
            <div class="upk-foot-bottom">
                <span>© 2026 UniPath Korea · {x("Made with ❤️ for international students.")}</span>
                <span>{x("Not affiliated with the Korean government.")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 20. MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════
render_nav()
render_auth()

_page = st.session_state.page
if _page == "HOME":
    page_home()
elif _page == "UNIVERSITY":
    page_university()
elif _page == "CAREER":
    page_career()
elif _page == "JOB":
    page_job()
elif _page == "TOPIK":
    page_topik()
elif _page == "VISA":
    page_visa()
elif _page == "LIFE":
    page_life()
elif _page == "AUTH":
    page_auth()
elif _page == "PROFILE":
    page_profile()
else:
    page_home()

render_footer()
floating_chat()
admin_panel()










