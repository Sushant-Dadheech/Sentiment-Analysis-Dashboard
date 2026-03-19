import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import os, re, io

# ─── VADER Analyzer ───
vader = SentimentIntensityAnalyzer()

# ─── Page Config ───
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Light base ── */
.main, .stApp {
    background: #f8f9fc !important;
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Hero banner ── */
.hero-banner {
    position: relative;
    background: linear-gradient(135deg, #ff9a56 0%, #e86cba 35%, #a855f7 65%, #6c63ff 100%);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    overflow: hidden;
    color: white;
    box-shadow: 0 8px 40px rgba(168, 85, 247, 0.25);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.12);
    border-radius: 50%;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 40%;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}

.hero-overline {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.85;
    margin-bottom: 0.4rem;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.4rem;
}

.hero-sub {
    font-size: 1rem;
    opacity: 0.85;
    margin-bottom: 1.2rem;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}

.card {
    background: #fff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #eee;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.1);
}

.card-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8892a4;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 2rem;
    font-weight: 700;
}

.card-delta {
    font-size: 0.82rem;
    margin-top: 0.3rem;
}

.clr-blue   { color: #3b82f6; }
.clr-green  { color: #22c55e; }
.clr-red    { color: #ef4444; }
.clr-purple { color: #a855f7; }
.clr-amber  { color: #f59e0b; }
.clr-gray   { color: #64748b; }

/* ── Alert cards ── */
.alert-stack {
    background: #fff;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #eee;
}

.alert-item {
    padding: 0.8rem 0;
    border-bottom: 1px solid #f1f1f4;
}

.alert-item:last-child { border-bottom: none; }

.alert-title {
    font-weight: 600;
    font-size: 0.92rem;
    color: #1e293b;
}

.alert-time {
    font-size: 0.75rem;
    color: #94a3b8;
    float: right;
}

.alert-action {
    display: inline-block;
    margin-top: 6px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
}

.action-green { background: #ecfdf5; color: #16a34a; border: 1px solid #bbf7d0; }
.action-orange { background: #fff7ed; color: #ea580c; border: 1px solid #fed7aa; }
.action-blue { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }

/* ── Section headers ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.25rem;
}

.section-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}

/* ── Data table ── */
.table-wrap {
    background: #fff;
    border-radius: 16px;
    padding: 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #eee;
}

.tool-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
}

.badge-pos  { background: #ecfdf5; color: #16a34a; border: 1px solid #bbf7d0; }
.badge-neg  { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.badge-neu  { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }

/* ── Analysis result ── */
.analysis-result {
    background: #fff;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #eee;
    margin-top: 1rem;
}

.polarity-bar-bg {
    width: 100%;
    height: 12px;
    background: #f1f5f9;
    border-radius: 6px;
    position: relative;
    margin: 1rem 0;
}

.polarity-bar-fill {
    height: 12px;
    border-radius: 6px;
    transition: width 0.6s ease;
}

.confidence-meter {
    display: flex;
    gap: 4px;
    margin-top: 0.5rem;
}

.conf-segment {
    height: 6px;
    border-radius: 3px;
    flex: 1;
    transition: background 0.3s;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid #eee;
}

div[data-testid="stSidebar"] .stRadio label {
    color: #334155 !important;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: #f8f9fc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    font-size: 0.95rem !important;
}

.stTextArea textarea:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #a855f7, #6c63ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(168,85,247,0.35) !important;
}

.stDataFrame, .stTable { border-radius: 12px; overflow: hidden; }

/* ── Word cloud container ── */
.wc-container img {
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}

/* Responsive */
@media (max-width: 768px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 1.6rem; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════
#  DUAL-ENGINE SENTIMENT ANALYSIS
# ════════════════════════════════════════

def analyze_sentiment_advanced(text):
    """
    Dual-engine sentiment: combines VADER (rule-based, great for social media/informal text)
    with TextBlob (pattern-based) for higher accuracy.
    Returns: combined_score, vader_scores, textblob_polarity, label, confidence
    """
    # VADER
    vs = vader.polarity_scores(text)
    vader_compound = vs['compound']   # -1 to 1

    # TextBlob
    tb = TextBlob(text)
    tb_polarity = tb.sentiment.polarity  # -1 to 1

    # Weighted combination: VADER 65%, TextBlob 35%
    # VADER is better for social media / informal text
    combined = 0.65 * vader_compound + 0.35 * tb_polarity

    # Confidence: agreement between the two engines
    agreement = 1 - abs(vader_compound - tb_polarity) / 2
    confidence = round(agreement * 100, 1)

    # Label with nuanced thresholds
    if combined >= 0.3:
        label = "Positive"
    elif combined >= 0.05:
        label = "Slightly Positive"
    elif combined <= -0.3:
        label = "Negative"
    elif combined <= -0.05:
        label = "Slightly Negative"
    else:
        label = "Neutral"

    return {
        'combined_score': round(combined, 4),
        'vader_compound': round(vader_compound, 4),
        'vader_pos': round(vs['pos'], 4),
        'vader_neg': round(vs['neg'], 4),
        'vader_neu': round(vs['neu'], 4),
        'textblob_polarity': round(tb_polarity, 4),
        'textblob_subjectivity': round(tb.sentiment.subjectivity, 4),
        'label': label,
        'confidence': confidence,
    }


def simple_label(label):
    """Map nuanced labels to simple ones for grouping."""
    if 'Positive' in label:
        return 'Positive'
    elif 'Negative' in label:
        return 'Negative'
    return 'Neutral'


# ════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════

@st.cache_data(show_spinner="Analyzing 40,000 texts with dual-engine AI…")
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "Data", "New Data.csv")
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    # Apply dual-engine analysis
    results = df['text'].apply(analyze_sentiment_advanced)
    res_df = pd.DataFrame(results.tolist())

    df = pd.concat([df, res_df], axis=1)
    df['Simple_Label'] = df['label'].apply(simple_label)
    return df


# ════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Sentiment AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview Dashboard", "✍️ Live Analysis", "🔬 Deep Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.78rem;'>"
        "Dual-engine: <strong>VADER</strong> + <strong>TextBlob</strong><br>"
        "Built by <strong>Sushant Dadheech</strong></p>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════
#  PAGE 1: OVERVIEW DASHBOARD
# ════════════════════════════════════════

if page == "📊 Overview Dashboard":

    # ── Hero ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-overline">Your overview</div>
        <div class="hero-title">Sentiment Analysis<br>Dashboard</div>
        <div class="hero-sub">Dual-engine AI analysis across 40,000 texts — VADER + TextBlob combined for superior accuracy.</div>
    </div>
    """, unsafe_allow_html=True)

    data = load_data()

    if data is None:
        st.error("❌ CSV file not found in the Data folder.")
        st.stop()

    total = len(data)
    counts = data['Simple_Label'].value_counts()
    pos = counts.get('Positive', 0)
    neg = counts.get('Negative', 0)
    neu = counts.get('Neutral', 0)
    avg_score = data['combined_score'].mean()
    avg_conf = data['confidence'].mean()

    # ── Metric cards ──
    st.markdown(f"""
    <div class="metric-grid">
        <div class="card">
            <div class="card-label">Total Texts</div>
            <div class="card-value clr-blue">{total:,}</div>
            <div class="card-delta clr-gray">Full dataset</div>
        </div>
        <div class="card">
            <div class="card-label">Positive</div>
            <div class="card-value clr-green">{pos:,}</div>
            <div class="card-delta clr-green">▲ {pos/total*100:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-label">Negative</div>
            <div class="card-value clr-red">{neg:,}</div>
            <div class="card-delta clr-red">▼ {neg/total*100:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-label">Avg Confidence</div>
            <div class="card-value clr-purple">{avg_conf:.1f}%</div>
            <div class="card-delta clr-gray">Engine agreement</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Insight alerts ──
    top_pos = data.nlargest(1, 'combined_score').iloc[0]
    top_neg = data.nsmallest(1, 'combined_score').iloc[0]

    col_hero_l, col_hero_r = st.columns([3, 2])

    with col_hero_l:
        st.markdown('<div class="section-title">📈 Sentiment Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Pie chart of overall dataset sentiment</div>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        colors_pie = ['#22c55e', '#ef4444', '#3b82f6']
        labels_order = ['Positive', 'Negative', 'Neutral']
        values = [counts.get(l, 0) for l in labels_order]
        wedges, texts, autotexts = ax1.pie(
            values, labels=labels_order, autopct='%1.1f%%',
            colors=colors_pie, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            pctdistance=0.8, wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
        )
        ax1.set_facecolor('#f8f9fc')
        fig1.patch.set_facecolor('#f8f9fc')
        st.pyplot(fig1, use_container_width=True)

    with col_hero_r:
        st.markdown("""
        <div class="alert-stack">
            <div style="font-weight:700; font-size:0.9rem; color:#1e293b; margin-bottom:0.8rem;">
                🔔 Key Insights
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="alert-item">
                <span class="alert-title">Most Positive Text</span>
                <span class="alert-time">Score: {top_pos['combined_score']:.3f}</span><br>
                <span class="alert-action action-green">✓ {top_pos['combined_score']:.3f} combined</span>
            </div>
            <div class="alert-item">
                <span class="alert-title">Most Negative Text</span>
                <span class="alert-time">Score: {top_neg['combined_score']:.3f}</span><br>
                <span class="alert-action action-orange">✗ {top_neg['combined_score']:.3f} combined</span>
            </div>
            <div class="alert-item">
                <span class="alert-title">Avg Polarity</span>
                <span class="alert-time">{avg_score:+.4f}</span><br>
                <span class="alert-action action-blue">{'Leans positive' if avg_score > 0 else 'Leans negative' if avg_score < 0 else 'Balanced'}</span>
            </div>
            <div class="alert-item">
                <span class="alert-title">Neutral Texts</span>
                <span class="alert-time">{neu:,}</span><br>
                <span class="alert-action action-blue">{neu/total*100:.1f}% of dataset</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Polarity Histogram ──
    st.markdown('<div class="section-title">📊 Polarity Score Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Combined VADER + TextBlob scores across all texts</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    n, bins, patches = ax2.hist(data['combined_score'], bins=50, edgecolor='white', linewidth=0.5)
    for i, p in enumerate(patches):
        center = (bins[i] + bins[i+1]) / 2
        if center > 0.05:
            p.set_facecolor('#22c55e')
        elif center < -0.05:
            p.set_facecolor('#ef4444')
        else:
            p.set_facecolor('#3b82f6')
    ax2.set_xlabel('Combined Polarity Score', fontsize=11, color='#334155')
    ax2.set_ylabel('Frequency', fontsize=11, color='#334155')
    ax2.tick_params(colors='#64748b')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#e2e8f0')
    ax2.spines['bottom'].set_color('#e2e8f0')
    ax2.set_facecolor('#fdfdfe')
    fig2.patch.set_facecolor('#f8f9fc')
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    # ── Data table ──
    st.markdown('<div class="section-title">📋 Sample Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Showing first 25 rows with dual-engine scores</div>', unsafe_allow_html=True)

    display_cols = ['text', 'combined_score', 'vader_compound', 'textblob_polarity', 'confidence', 'label']
    st.dataframe(
        data[display_cols].head(25),
        use_container_width=True,
        height=500,
        column_config={
            'text': st.column_config.TextColumn('Text', width='large'),
            'combined_score': st.column_config.NumberColumn('Combined', format='%.3f'),
            'vader_compound': st.column_config.NumberColumn('VADER', format='%.3f'),
            'textblob_polarity': st.column_config.NumberColumn('TextBlob', format='%.3f'),
            'confidence': st.column_config.ProgressColumn('Confidence', min_value=0, max_value=100, format='%.1f%%'),
            'label': st.column_config.TextColumn('Label'),
        }
    )


# ════════════════════════════════════════
#  PAGE 2: LIVE ANALYSIS
# ════════════════════════════════════════

elif page == "✍️ Live Analysis":

    st.markdown("""
    <div class="hero-banner" style="padding:2rem 2.5rem;">
        <div class="hero-overline">Live analysis</div>
        <div class="hero-title">Analyze Any Text</div>
        <div class="hero-sub">Type or paste text below for instant dual-engine sentiment analysis with confidence scoring.</div>
    </div>
    """, unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter your text:",
        height=140,
        placeholder="e.g. I absolutely loved the new movie! The acting was incredible. 🎬",
    )

    analyze_btn = st.button("🔍 Analyze Sentiment")

    if analyze_btn and user_text.strip():
        result = analyze_sentiment_advanced(user_text)

        lbl = result['label']
        simple = simple_label(lbl)
        emoji = '😊' if simple == 'Positive' else '😔' if simple == 'Negative' else '😐'
        badge_cls = 'badge-pos' if simple == 'Positive' else 'badge-neg' if simple == 'Negative' else 'badge-neu'
        clr = '#22c55e' if simple == 'Positive' else '#ef4444' if simple == 'Negative' else '#3b82f6'

        # Polarity bar position (map -1..1 to 0..100%)
        bar_pct = (result['combined_score'] + 1) / 2 * 100

        conf = result['confidence']
        conf_segs = int(conf / 10)

        # Build confidence meter segments
        conf_segments = ""
        for i in range(10):
            seg_color = clr if i < conf_segs else "#e2e8f0"
            conf_segments += f'<div style="height:8px;border-radius:4px;flex:1;background:{seg_color};"></div>'

        st.markdown(f"""<div class="analysis-result">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
<div>
<span style="font-size:2.2rem;">{emoji}</span>
<span class="tool-badge {badge_cls}" style="font-size:1.1rem;padding:6px 18px;margin-left:8px;">{lbl}</span>
</div>
<div style="text-align:right;">
<div style="font-size:0.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Confidence</div>
<div style="font-size:1.8rem;font-weight:700;color:{clr};">{conf}%</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="background:#fff;border-radius:0 0 16px 16px;padding:0 2rem 2rem 2rem;margin-top:-1rem;border:1px solid #eee;border-top:none;">
<div style="font-size:0.82rem;color:#64748b;margin-bottom:4px;">Combined Polarity Score</div>
<div style="width:100%;height:12px;background:#f1f5f9;border-radius:6px;position:relative;margin:0.6rem 0;">
<div style="height:12px;border-radius:6px;width:{bar_pct:.1f}%;background:{clr};"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;">
<span>-1 (Very Negative)</span>
<span style="font-weight:600;color:{clr};">{result['combined_score']:+.4f}</span>
<span>+1 (Very Positive)</span>
</div>
<div style="margin-top:1.2rem;">
<div style="font-size:0.82rem;color:#64748b;margin-bottom:6px;">Confidence Meter (engine agreement)</div>
<div style="display:flex;gap:4px;margin-top:0.5rem;">
{conf_segments}
</div>
</div>
</div>""", unsafe_allow_html=True)

        # ── Engine Comparison Table ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔬 Engine Breakdown</div>', unsafe_allow_html=True)

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown(f"""
            <div class="card">
                <div style="font-weight:700; color:#1e293b; margin-bottom:0.8rem;">🔴 VADER Analysis</div>
                <table style="width:100%; font-size:0.88rem; color:#334155;">
                <tr><td>Compound</td><td style="text-align:right; font-weight:600;">{result['vader_compound']:+.4f}</td></tr>
                <tr><td>Positive</td><td style="text-align:right; color:#22c55e;">{result['vader_pos']:.1%}</td></tr>
                <tr><td>Negative</td><td style="text-align:right; color:#ef4444;">{result['vader_neg']:.1%}</td></tr>
                <tr><td>Neutral</td><td style="text-align:right; color:#3b82f6;">{result['vader_neu']:.1%}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with comp_col2:
            st.markdown(f"""
            <div class="card">
                <div style="font-weight:700; color:#1e293b; margin-bottom:0.8rem;">🟣 TextBlob Analysis</div>
                <table style="width:100%; font-size:0.88rem; color:#334155;">
                <tr><td>Polarity</td><td style="text-align:right; font-weight:600;">{result['textblob_polarity']:+.4f}</td></tr>
                <tr><td>Subjectivity</td><td style="text-align:right;">{result['textblob_subjectivity']:.1%}</td></tr>
                <tr><td colspan="2" style="padding-top:10px; font-size:0.78rem; color:#94a3b8;">
                    Polarity: opinion direction<br>Subjectivity: fact vs opinion
                </td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("⚠️ Please enter some text to analyze.")


# ════════════════════════════════════════
#  PAGE 3: DEEP INSIGHTS
# ════════════════════════════════════════

elif page == "🔬 Deep Insights":

    st.markdown("""
    <div class="hero-banner" style="padding:2rem 2.5rem; background: linear-gradient(135deg, #6c63ff 0%, #a855f7 50%, #e86cba 100%);">
        <div class="hero-overline">Deep insights</div>
        <div class="hero-title">Advanced Analytics</div>
        <div class="hero-sub">Word clouds, engine comparison, subjectivity analysis, and more.</div>
    </div>
    """, unsafe_allow_html=True)

    data = load_data()

    if data is None:
        st.error("❌ CSV file not found.")
        st.stop()

    # ── VADER vs TextBlob scatter ──
    st.markdown('<div class="section-title">⚔️ VADER vs TextBlob Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the two engines compare on every text in the dataset</div>', unsafe_allow_html=True)

    sample = data.sample(min(2000, len(data)), random_state=42)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    colors_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#3b82f6'}
    for label, color in colors_map.items():
        mask = sample['Simple_Label'] == label
        ax3.scatter(
            sample.loc[mask, 'vader_compound'],
            sample.loc[mask, 'textblob_polarity'],
            c=color, alpha=0.35, s=15, label=label
        )
    ax3.axhline(0, color='#e2e8f0', linewidth=0.8)
    ax3.axvline(0, color='#e2e8f0', linewidth=0.8)
    ax3.plot([-1, 1], [-1, 1], color='#94a3b8', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('VADER Compound', fontsize=11, color='#334155')
    ax3.set_ylabel('TextBlob Polarity', fontsize=11, color='#334155')
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#e2e8f0')
    ax3.spines['bottom'].set_color('#e2e8f0')
    ax3.set_facecolor('#fdfdfe')
    fig3.patch.set_facecolor('#f8f9fc')
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    # ── Word Clouds ──
    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        st.markdown('<div class="section-title">☁️ Positive Words</div>', unsafe_allow_html=True)
        pos_text = ' '.join(data[data['Simple_Label'] == 'Positive']['text'].dropna().head(5000))
        if pos_text.strip():
            wc_pos = WordCloud(
                width=600, height=350, background_color='#f8f9fc',
                colormap='Greens', max_words=120, contour_width=0
            ).generate(pos_text)
            fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 3.5))
            ax_wc1.imshow(wc_pos, interpolation='bilinear')
            ax_wc1.axis('off')
            fig_wc1.patch.set_facecolor('#f8f9fc')
            fig_wc1.tight_layout(pad=0)
            st.pyplot(fig_wc1, use_container_width=True)

    with wc_col2:
        st.markdown('<div class="section-title">☁️ Negative Words</div>', unsafe_allow_html=True)
        neg_text = ' '.join(data[data['Simple_Label'] == 'Negative']['text'].dropna().head(5000))
        if neg_text.strip():
            wc_neg = WordCloud(
                width=600, height=350, background_color='#f8f9fc',
                colormap='Reds', max_words=120, contour_width=0
            ).generate(neg_text)
            fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 3.5))
            ax_wc2.imshow(wc_neg, interpolation='bilinear')
            ax_wc2.axis('off')
            fig_wc2.patch.set_facecolor('#f8f9fc')
            fig_wc2.tight_layout(pad=0)
            st.pyplot(fig_wc2, use_container_width=True)

    # ── Subjectivity Distribution ──
    st.markdown('<div class="section-title">🎯 Subjectivity Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How opinionated vs factual are the texts?</div>', unsafe_allow_html=True)

    fig4, ax4 = plt.subplots(figsize=(12, 3.5))
    ax4.hist(data['textblob_subjectivity'], bins=50, color='#a855f7', edgecolor='white', linewidth=0.5)
    ax4.set_xlabel('Subjectivity (0 = Objective, 1 = Subjective)', fontsize=11, color='#334155')
    ax4.set_ylabel('Frequency', fontsize=11, color='#334155')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color('#e2e8f0')
    ax4.spines['bottom'].set_color('#e2e8f0')
    ax4.set_facecolor('#fdfdfe')
    fig4.patch.set_facecolor('#f8f9fc')
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)

    # ── Confidence Distribution ──
    st.markdown('<div class="section-title">🛡️ Confidence Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How well do VADER and TextBlob agree on each text?</div>', unsafe_allow_html=True)

    fig5, ax5 = plt.subplots(figsize=(12, 3.5))
    n5, bins5, patches5 = ax5.hist(data['confidence'], bins=40, edgecolor='white', linewidth=0.5)
    for i, p in enumerate(patches5):
        center = (bins5[i] + bins5[i+1]) / 2
        if center >= 80:
            p.set_facecolor('#22c55e')
        elif center >= 60:
            p.set_facecolor('#f59e0b')
        else:
            p.set_facecolor('#ef4444')
    ax5.set_xlabel('Confidence %', fontsize=11, color='#334155')
    ax5.set_ylabel('Frequency', fontsize=11, color='#334155')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['left'].set_color('#e2e8f0')
    ax5.spines['bottom'].set_color('#e2e8f0')
    ax5.set_facecolor('#fdfdfe')
    fig5.patch.set_facecolor('#f8f9fc')
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)

    # ── Detailed Label Breakdown ──
    st.markdown('<div class="section-title">📊 Nuanced Sentiment Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Five-level sentiment classification</div>', unsafe_allow_html=True)

    nuanced = data['label'].value_counts()
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    bar_colors = {
        'Positive': '#22c55e', 'Slightly Positive': '#86efac',
        'Neutral': '#3b82f6',
        'Slightly Negative': '#fca5a5', 'Negative': '#ef4444'
    }
    order = ['Positive', 'Slightly Positive', 'Neutral', 'Slightly Negative', 'Negative']
    ordered = [nuanced.get(o, 0) for o in order]
    cols_bar = [bar_colors.get(o, '#94a3b8') for o in order]
    bars = ax6.bar(order, ordered, color=cols_bar, edgecolor='white', linewidth=0.5, width=0.65)
    for bar, val in zip(bars, ordered):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ordered)*0.015,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold', color='#334155')
    ax6.set_ylabel('Count', fontsize=11, color='#334155')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_color('#e2e8f0')
    ax6.spines['bottom'].set_color('#e2e8f0')
    ax6.tick_params(colors='#64748b')
    ax6.set_facecolor('#fdfdfe')
    fig6.patch.set_facecolor('#f8f9fc')
    fig6.tight_layout()
    st.pyplot(fig6, use_container_width=True)
