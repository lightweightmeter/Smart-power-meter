import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from PIL import Image
from io import BytesIO
import base64

# ------------------- Config -------------------
REFRESH_SECONDS = 5
API_BASE = "http://127.0.0.1:5000"
st_autorefresh(interval=REFRESH_SECONDS * 1000, limit=None, key="auto_refresh")

st.set_page_config(
    page_title="Smart Power Meter — RL Prototype",
    layout="wide",
    page_icon="⚡",
)

# ------------------- Modern 3D UI CSS + Animations -------------------
st.markdown(
    """
    <style>
    :root{
        --navy:#0B2D4F;
        --orange:#FF4C00;
        --cream:#f3f1ee;
        --white:#ffffff;
    }

    body, .stApp{
        background:linear-gradient(180deg,#f3f1ee 0%,#f8f6f4 100%);
        color:var(--navy);
        font-family:"Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif;
    }

    /* ---- Logo ---- */
    .logo-container{
        display:flex;
        justify-content:center;
        align-items:center;
        margin-top:5px;
        margin-bottom:25px;
    }
    .logo-container img{
        width:260px;
        filter:drop-shadow(0 0 8px rgba(11,45,79,.35));
        transition:transform .25s ease,filter .25s ease;
    }
    .logo-container img:hover{
        transform:scale(1.03);
        filter:drop-shadow(0 0 14px rgba(11,45,79,.55));
    }

    /* ---- 3D Tabs ---- */
    .stTabs [role="tablist"]{
        display:flex;
        justify-content:center;
        gap:14px;
        padding:12px 0;
    }
    .stTabs [role="tab"]{
        background:linear-gradient(180deg,#0B2D4F 0%,#071e35 100%)!important;
        color:#ffffff!important;
        border:none!important;
        border-radius:10px!important;
        padding:10px 26px!important;
        font-weight:600!important;
        letter-spacing:.2px;
        box-shadow:
            inset 0 2px 1px rgba(255,255,255,.25),
            0 3px 5px rgba(0,0,0,.35),
            0 6px 12px rgba(11,45,79,.2);
        transition:all .25s ease;
    }
    .stTabs [role="tab"]:hover{
        transform:translateY(-2px);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,.4),
            0 4px 8px rgba(0,0,0,.3),
            0 8px 16px rgba(11,45,79,.25);
    }
    .stTabs [role="tab"][aria-selected="true"]{
        background:linear-gradient(180deg,#FF4C00 0%,#c73d00 100%)!important;
        color:#ffffff!important;
        box-shadow:
            inset 0 3px 2px rgba(255,255,255,.3),
            inset 0 -2px 4px rgba(0,0,0,.25),
            0 3px 6px rgba(11,45,79,.3);
        transform:none;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [role="tablist"]::after {
        display:none!important;
    }

    /* ---- Cards ---- */
    .card{
        background:var(--white);
        box-shadow:0 8px 20px rgba(11,45,79,.12),0 2px 6px rgba(0,0,0,.05);
        border-radius:12px;
        padding:22px 28px;
        margin-bottom:25px;
        transition:transform .25s ease,box-shadow .25s ease;
    }
    .card:hover{
        transform:translateY(-3px);
        box-shadow:0 10px 24px rgba(11,45,79,.18);
    }

    /* ---- Animation Keyframes ---- */
    @keyframes fadeInSlide {
        0% {opacity:0; transform:translateY(20px);}
        100% {opacity:1; transform:translateY(0);}
    }

    /* ---- Animated 3D Alerts ---- */
    .alert{
        border-radius:12px;
        padding:18px 22px;
        font-weight:600;
        margin-bottom:14px;
        animation: fadeInSlide 0.6s ease forwards;
        box-shadow:0 4px 10px rgba(0,0,0,.08);
        transition:all .25s ease;
    }

    .alert:hover{
        transform:translateY(-2px);
        box-shadow:
            inset 0 2px 1px rgba(255,255,255,0.5),
            0 5px 10px rgba(0,0,0,0.2),
            0 8px 16px rgba(11,45,79,0.18);
    }

    .alert-critical{
        background:linear-gradient(180deg,rgba(255,76,0,.12) 0%,rgba(255,76,0,.18) 100%);
        border-left:6px solid var(--orange);
        color:var(--orange);
    }
    .alert-warning{
        background:linear-gradient(180deg,rgba(255,187,0,.12),rgba(255,150,0,.18));
        border-left:6px solid #e68a00;
        color:#b35a00;
    }
    .alert-success{
        background:linear-gradient(180deg,#e9edf1 0%,#dfe3e8 100%);
        border-left:6px solid var(--navy);
        color:var(--navy);
        box-shadow:
            inset 0 1px 1px rgba(255,255,255,0.4),
            0 3px 6px rgba(0,0,0,0.15),
            0 6px 12px rgba(11,45,79,0.15);
        text-shadow:0 1px 0 rgba(255,255,255,0.6);
    }

    /* ---- Metrics ---- */
    .stMetric{
        background:#ffffffb0;
        border-radius:10px;
        box-shadow:0 3px 8px rgba(11,45,79,.08);
        padding:10px;
    }
    div[data-testid="stMetricValue"]{ color:var(--orange); }

    /* ---- Tables & Charts ---- */
    .stDataFrame thead tr th{
        background:var(--navy)!important;
        color:white!important;
        font-weight:600!important;
    }
    .chart-card{
        background:var(--white);
        border-radius:12px;
        box-shadow:0 4px 10px rgba(11,45,79,.1);
        padding:20px;
        margin-top:10px;
    }
    canvas{
        background:var(--white)!important;
        border-radius:10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- Logo -------------------
def get_base64(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

try:
    # IMPORTANT: Ensure 'logo.png' exists in the script directory
    logo = Image.open("logo.png") 
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{get_base64(logo)}" alt="Smart Power Meter Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.warning("⚠️ Logo not found. Please place 'logo.png' in the same directory.")

# ------------------- Helper (Using st.cache_data for 5s TTL) -------------------
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_json_cached(path):
    """Fetches data from the API and caches it for REFRESH_SECONDS to reduce flicker."""
    try:
        r = requests.get(API_BASE + path, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        # Return a sensible default structure (empty list or dictionary) on error
        if path in ["/get_data", "/get_alerts", "/get_forecast", "/get_summary", "/get_rl_rewards"]:
            return []
        return {}

# ------------------- Fetch Data -------------------
# Data is fetched once and cached for 5 seconds.
data      = fetch_json_cached("/get_data")
billing   = fetch_json_cached("/get_billing")
alerts    = fetch_json_cached("/get_alerts")
forecast  = fetch_json_cached("/get_forecast")
summary   = fetch_json_cached("/get_summary")
rl_action = fetch_json_cached("/get_rl_action")
rl_rewards= fetch_json_cached("/get_rl_rewards")
rl_costs  = fetch_json_cached("/get_rl_costs")

# ------------------- Tabs -------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Real-time Data", "💰 Billing & Alerts", "🔮 Forecast", "📅 Daily Summary", "🤖 RL Agent"]
)

# --- Tab 1 (Using st.fragment for non-flickering 5s update) ---
# NOTE: The decorator is changed from st.experimental_fragment to st.fragment
@st.fragment(run_every=f'{REFRESH_SECONDS}s')
def display_realtime_data(data):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Real-time Power (W)")
    df = pd.DataFrame(data)
    if not df.empty and "timestamp" in df:
        df = df.sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_display = df.set_index("timestamp")
        
        # Use an empty container to hold the chart to prevent it from jumping
        chart_container = st.container()
        with chart_container:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.line_chart(df_display[["power"]])
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Recent Readings")
        # Use fixed height for dataframe to prevent layout shifting
        st.dataframe(df_display.tail(20), use_container_width=True, height=500)
    else:
        st.info("No data yet.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab1:
    # Pass the data fetched from the cached function to the fragment
    display_realtime_data(data)

# --- Tab 2 ---
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Billing Summary")
    if isinstance(billing, dict) and "total_kwh" in billing:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total kWh", f"{billing.get('total_kwh', 0):.2f}")
        with col2:
            st.metric("Estimated Bill (₹)", f"₹{billing.get('bill_estimate', 0):,.2f}")
    else:
        st.write("Billing data not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Alerts")
    if isinstance(alerts, list) and alerts:
        for a in alerts:
            level = a.get("level", "success")
            # Ensure the level maps to the CSS class
            css_level = level.lower().replace('critical', 'critical').replace('warning', 'warning').replace('info', 'success').replace('success', 'success')
            st.markdown(f"<div class='alert alert-{css_level}'>{a['msg']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert alert-success'>No alerts detected ✅</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 3 ---
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Forecast (next 5 minutes)")
    if isinstance(forecast, list) and forecast:
        df_fore = pd.DataFrame(forecast)
        df_fore["timestamp"] = pd.to_datetime(df_fore["timestamp"])
        df_fore = df_fore.set_index("timestamp")
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.line_chart(df_fore[["predicted_power"]])
        st.markdown("</div>", unsafe_allow_html=True)
        st.dataframe(df_fore)
    else:
        st.write("Forecast not available yet")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 4 ---
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Daily Energy Summary (kWh)")
    if isinstance(summary, list) and summary:
        df_sum = pd.DataFrame(summary)
        df_sum["date"] = pd.to_datetime(df_sum["date"])
        df_sum = df_sum.set_index("date")
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.bar_chart(df_sum[["energy"]])
        st.markdown("</div>", unsafe_allow_html=True)
        st.dataframe(df_sum)
    else:
        st.write("Summary not available")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 5 ---
with tab5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("RL Agent Suggestion 🤖")
    if isinstance(rl_action, dict) and "action" in rl_action:
        st.markdown(f"<div class='alert alert-success'>Agent recommends: **{rl_action['action']}**</div>", unsafe_allow_html=True)
    else:
        st.write("RL agent not available")

    st.subheader("RL Training Performance")
    if isinstance(rl_rewards, list) and rl_rewards:
        df_rewards = pd.DataFrame({"Episode": range(len(rl_rewards)), "Reward": rl_rewards})
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.line_chart(df_rewards.set_index("Episode"))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("No reward data available")

    st.subheader("Cost Comparison: RL vs Baseline")
    if isinstance(rl_costs, dict) and "baseline" in rl_costs and "rl" in rl_costs and rl_costs["baseline"] and rl_costs["rl"]:
        
        # Ensure lists are of equal length before creating DataFrame
        min_len = min(len(rl_costs["baseline"]), len(rl_costs["rl"]))
        df_costs = pd.DataFrame({
            "Episode": range(min_len),
            "Baseline Cost": rl_costs["baseline"][:min_len],
            "RL Cost": rl_costs["rl"][:min_len]
        }).set_index("Episode")
        
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.line_chart(df_costs)
        st.markdown("</div>", unsafe_allow_html=True)
        
        baseline_total = sum(rl_costs["baseline"][:min_len])
        rl_total = sum(rl_costs["rl"][:min_len])
        savings = baseline_total - rl_total
        
        st.markdown(f"<div class='alert alert-success'>💰 Total Savings with RL: **₹{savings:.2f}**</div>", unsafe_allow_html=True)
    else:
        st.write("No cost data available")
    st.markdown("</div>", unsafe_allow_html=True)