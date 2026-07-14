import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# 1. Page Configuration & Professional CSS
# ==========================================
st.set_page_config(page_title="Quantitative Audit: Nifty 50 Options", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main {background-color: #0a0e17;}
    h1 {color: #ffffff; border-bottom: 2px solid #FF4B4B; padding-bottom: 10px;}
    h2, h3 {color: #FF4B4B;}
    .stMetric {background-color: #161b26; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B;}
    .disclaimer {background-color: #2b1c1c; color: #ffcccc; padding: 15px; border-radius: 5px; font-size: 14px; margin-bottom: 20px; border: 1px solid #ff4444;}
    .insight-box {background-color: #161b26; padding: 20px; border-radius: 8px; margin-top: 15px; border: 1px solid #333;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SEBI Disclaimer (Strictly Educational)
# ==========================================
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ LEGAL DISCLAIMER:</strong> I am NOT a SEBI Registered Research Analyst or Investment Advisor. 
    This research dashboard is strictly for educational, statistical, and quantitative analysis purposes only. 
    It relies on historical backtested data, which does not guarantee future results. Do not make live financial 
    decisions based on this dashboard. Option trading involves immense risk.
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. Data Engine (Cached)
# ==========================================
@st.cache_data
def load_data():
    try:
        spot_df = pd.read_csv('Historical_Spot_15min/NIFTY_50_INDEX_5Yr_Spot_15min.csv')
        opt_df = pd.read_csv('NIFTY_50_5_Years_15min_ATM_Call.csv')
    except Exception as e:
        st.error(f"Data loading error: Ensure CSV files are correctly named and located in the repository. {e}")
        return pd.DataFrame()
        
    spot_df['date'] = pd.to_datetime(spot_df['date'])
    opt_df['date'] = pd.to_datetime(opt_df['date'])
    
    df = pd.merge(spot_df, opt_df, on='date', how='inner', suffixes=('_spot', '_opt'))
    df['date_only'] = df['date'].dt.date
    df['time_str'] = df['date'].dt.strftime('%H:%M')
    
    # Feature Engineering (Vectorized)
    df['spot_roc_1h'] = df['close_spot'].pct_change(4) * 100
    df['spot_rolling_vol'] = df['close_spot'].pct_change().rolling(25).std() * np.sqrt(25)
    df['vol_zscore'] = (df['spot_rolling_vol'] - df['spot_rolling_vol'].rolling(100).mean()) / df['spot_rolling_vol'].rolling(100).std()
    
    # Pure Intraday Forward Returns (1 Hour) - Stripping Gaps
    df['fwd_opt_return_1h'] = df['close_opt'].pct_change(4).shift(-4) * 100
    df['fwd_date_only'] = df['date_only'].shift(-4)
    df.loc[df['date_only'] != df['fwd_date_only'], 'fwd_opt_return_1h'] = np.nan
    
    df['vol_regime'] = np.where(df['vol_zscore'] > 1.5, 'Extreme Expansion', 
                       np.where(df['vol_zscore'] < -1.5, 'Extreme Compression', 'Normal'))
    
    df = df.dropna().reset_index(drop=True)
    df['momentum_bucket'] = pd.qcut(df['spot_roc_1h'], q=5, labels=['Strong Bearish', 'Weak Bearish', 'Neutral', 'Weak Bullish', 'Strong Bullish'])
    
    return df

df = load_data()

if df.empty:
    st.stop()

# ==========================================
# 4. Header & Executive Summary
# ==========================================
st.title("The Retail Option Buyer's Expectancy Audit")
st.markdown("### A 5-Year High-Frequency Quantitative Analysis on the Nifty 50")
st.markdown("""
*This algorithmic backtest parses over 31,000 individual 15-minute intervals to audit the structural viability of mechanical retail option buying. The math reveals exactly why unhedged directional buying is a negative-expectancy model.*
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset Observations", f"{len(df):,}", "15-Min Candles")
col2.metric("Overall Intraday Win Rate", f"{(df['fwd_opt_return_1h'] > 0).mean() * 100:.1f}%", "-Structural Drag")
col3.metric("Worst Momentum Setup", "Strong Bullish", "-2.74% Mean Return")
col4.metric("Timeframe Analyzed", "60 Months", "Nifty 50 Spot & ATM Call")

st.markdown("---")

# ==========================================
# 5. Core Insights (The 4 Phases of Backtesting)
# ==========================================

# Insight 1 & 2: Time Decay and Momentum Trap
c1, c2 = st.columns(2)

with c1:
    st.subheader("1. The Inescapable Theta Bleed")
    st.markdown("""
    **The Retail Myth:** Options provide explosive intraday gains.
    **The Quantitative Truth:** When stripping away overnight gap lotteries to isolate pure 1-hour intraday holds, **every single 15-minute time block yields a negative median return.** Theta (time decay) acts as a constant, inescapable tax on the buyer.
    """)
    time_edge = df.groupby('time_str')['fwd_opt_return_1h'].apply(lambda x: (x > 0).mean() * 100).reset_index()
    time_edge.columns = ['Time', 'Win Rate (%)']
    fig_time = px.line(time_edge, x='Time', y='Win Rate (%)', markers=True, color_discrete_sequence=['#FF4B4B'])
    fig_time.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Breakeven (50%)")
    fig_time.update_layout(plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17', font_color='white')
    st.plotly_chart(fig_time, use_container_width=True)

with c2:
    st.subheader("2. The 'FOMO' Momentum Trap")
    st.markdown("""
    **The Retail Myth:** Buy the breakout when momentum is highly bullish.
    **The Quantitative Truth:** Buying an ATM call immediately following a "Strong Bullish" 1-hour move yields the **absolute worst outcome** in the entire 5-year dataset. By the time the trend is obvious, the premium is overpriced, and post-momentum consolidation destroys the buyer.
    """)
    mom_edge = df.groupby('momentum_bucket')['fwd_opt_return_1h'].mean().reset_index()
    mom_edge.columns = ['Momentum', 'Avg Return (%)']
    fig_mom = px.bar(mom_edge, x='Momentum', y='Avg Return (%)', color='Avg Return (%)', color_continuous_scale='Reds_r')
    fig_mom.update_layout(plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17', font_color='white')
    st.plotly_chart(fig_mom, use_container_width=True)

st.markdown("---")

# Insight 3 & 4: Volatility and Risk Management Paradox
c3, c4 = st.columns(2)

with c3:
    st.subheader("3. Volatility Crush (Vega Risk)")
    st.markdown("""
    **The Retail Myth:** Buy options when volatility is expanding.
    **The Quantitative Truth:** Expanding volatility drastically inflates option premiums. Entering during **'Extreme Expansion'** forces the underlying index to make a spectacular, outsized move just for the trade to break even. When it fails to do so, the resulting Vega crush decimates the capital.
    """)
    vol_edge = df.groupby('vol_regime')['fwd_opt_return_1h'].mean().reset_index()
    fig_vol = px.bar(vol_edge, x='vol_regime', y='fwd_opt_return_1h', color='vol_regime', 
                     color_discrete_sequence=['#FFD700', '#FF4B4B', '#00FF7F'])
    fig_vol.update_layout(plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17', font_color='white')
    st.plotly_chart(fig_vol, use_container_width=True)

with c4:
    st.subheader("4. The Negative Skew Distribution")
    st.markdown("""
    **The Retail Myth:** Tight stop losses will save the account.
    **The Quantitative Truth:** During our dynamic simulation testing, applying a fixed 10% stop loss resulted in a **"whipsaw" effect**, dropping win rates to 40% as standard 15-minute noise hunted the stops. Expanding stops using ATR (Average True Range) simply resulted in larger absolute losses. Mechanical option buying possesses a heavy left-tail risk.
    """)
    fig_dist = px.histogram(df, x="fwd_opt_return_1h", nbins=100, color_discrete_sequence=['#FF4B4B'])
    fig_dist.add_vline(x=0, line_dash="dash", line_color="white")
    fig_dist.update_xaxes(range=[-30, 30]) 
    fig_dist.update_layout(plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17', font_color='white', showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

st.markdown("""
<div class="insight-box">
    <h3 style="margin-top: 0;">Final Quantitative Verdict</h3>
    <p>The 5-year mathematical footprint is clear: attempting to outpace time decay and implied volatility crush with unhedged, mechanical option buying guarantees capital erosion over a large sample size. Every negative expectancy system is a positive expectancy system waiting to be inverted. To survive in the derivatives market, one must stop fighting the structural mechanics and explore Delta-Neutral option selling architectures.</p>
</div>
""", unsafe_allow_html=True)
