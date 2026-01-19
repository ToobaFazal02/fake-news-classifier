
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🔍 Fake News Detector - AI Powered",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS 
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');
    
    /* Dark Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,69,0,0.1));
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 2px solid rgba(255,215,0,0.3);
        box-shadow: 0 0 40px rgba(255,215,0,0.2);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(255,215,0,0.2); }
        to { box-shadow: 0 0 40px rgba(255,215,0,0.4); }
    }
    
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #FFD700, #FF6B35, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(255,215,0,0.3);
    }
    
    .hero-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.5rem;
        color: #00D9FF;
        font-weight: 600;
        letter-spacing: 2px;
    }
    
    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00D9FF, #0099FF);
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1rem;
        box-shadow: 0 5px 25px rgba(0,217,255,0.4);
    }
    
    /* Analyze Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF6B35, #F7931E) !important;
        color: white !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        padding: 1rem 2rem !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0 10px 30px rgba(255,107,53,0.4) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 2px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(255,107,53,0.6) !important;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background-color: rgba(26,26,46,0.8) !important;
        border: 2px solid rgba(255,215,0,0.3) !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, rgba(26,26,46,0.9), rgba(22,33,62,0.9));
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: 2px solid rgba(255,215,0,0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .fake-result {
        border-left: 5px solid #FF4444;
        background: linear-gradient(135deg, rgba(255,68,68,0.1), rgba(255,107,53,0.1));
    }
    
    .real-result {
        border-left: 5px solid #00FF88;
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,217,255,0.1));
    }
    
    /* Metrics */
    .metric-container {
        text-align: center;
        padding: 1.5rem;
        background: rgba(26,26,46,0.6);
        border-radius: 15px;
        border: 1px solid rgba(255,215,0,0.2);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #00D9FF;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Pulsing Animation */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_URL = "http://127.0.0.1:8000"

def check_api():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_news(text):
    """Call API for prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_gauge_chart(probability, label):
    """Create a circular gauge for probability"""
    color = "#FF4444" if label == "FAKE" else "#00FF88"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{label} PROBABILITY", 'font': {'size': 24, 'color': color}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': color},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0,255,136,0.2)'},
                {'range': [50, 100], 'color': 'rgba(255,68,68,0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Rajdhani"},
        height=300
    )
    
    return fig

def create_confidence_bar(fake_prob, real_prob):
    """Create confidence comparison bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=[real_prob * 100, fake_prob * 100],
            y=['REAL', 'FAKE'],
            orientation='h',
            marker=dict(
                color=['#00FF88', '#FF4444'],
                line=dict(color='white', width=2)
            ),
            text=[f"{real_prob*100:.1f}%", f"{fake_prob*100:.1f}%"],
            textposition='auto',
            textfont=dict(size=18, color='white', family='Orbitron')
        )
    ])
    
    fig.update_layout(
        title={
            'text': "CONFIDENCE BREAKDOWN",
            'font': {'size': 20, 'color': '#FFD700', 'family': 'Orbitron'}
        },
        paper_bgcolor="rgba(26,26,46,0.5)",
        plot_bgcolor="rgba(0,0,0,0.3)",
        font={'color': "white"},
        xaxis={'range': [0, 100], 'showgrid': False},
        yaxis={'showgrid': False},
        height=250
    )
    
    return fig

def create_wordcloud(words):
    """Generate word cloud from trigger words"""
    if not words:
        return None
    
    text = ' '.join(words * 3)  # Repeat for visibility
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='rgba(15,15,30,0.8)',
        colormap='RdYlGn_r',
        max_words=50,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_alpha(0)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    # Hero Section
    st.markdown("""
    <div class="hero-container pulse">
        <h1 class="hero-title">FAKE NEWS DETECTOR</h1>
        <p class="hero-subtitle">AI-POWERED TRUTH VERIFICATION</p>
        <div class="accuracy-badge"> 97% ACCURACY </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown("###  SYSTEM STATUS")
        
        if check_api():
            st.success("AI MODEL ONLINE")
        else:
            st.error(" API OFFLINE")
            st.warning("Start API: `uvicorn app.main:app --reload`")
        
        st.markdown("---")
        st.markdown("###  STATISTICS")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "97%", delta="↑ 2%")
        with col2:
            st.metric("Speed", "0.3s", delta="Fast")
        
        st.markdown("---")
        st.markdown("###  AI TECHNOLOGY")
        st.info("""
        **Powered by:**
        - TF-IDF Vectorization
        - Logistic Regression
        - NLP Preprocessing
        - 10,000+ Features
        """)
        
        st.markdown("---")
        st.markdown("###  FAKE NEWS INDICATORS")
        st.error("""
        - Sensational headlines
        - ALL CAPS text
        - "Click here" phrases
        - Excessive punctuation!!!
        - No credible sources
        """)
        
        st.markdown("---")
        st.markdown("###  REAL NEWS SIGNS")
        st.success("""
        - Professional writing
        - Cited sources
        - Balanced reporting
        - Clear authorship
        - Verifiable facts
        """)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Input Section
    st.markdown("###  ENTER NEWS ARTICLE")
    
    article_text = st.text_area(
        "",
        height=200,
        placeholder="Paste your news article here (minimum 50 characters)...",
        label_visibility="collapsed"
    )
    
    # Example Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(" Try Real Example"):
            article_text = "The Federal Reserve announced today that it will maintain current interest rates for the third consecutive quarter. The decision comes after careful analysis of inflation data, employment figures, and economic growth indicators. Federal Reserve Chairman stated that the policy aims to support sustainable economic growth while managing inflation expectations. Economists from major financial institutions have provided mixed reactions, with some supporting the decision while others suggest rate adjustments may be needed in coming months."
            st.rerun()
    
    with col2:
        if st.button(" Try Fake Example"):
            article_text = "SHOCKING DISCOVERY! Scientists STUNNED by this ONE WEIRD TRICK that DOCTORS don't want you to KNOW! This MIRACLE cure will CHANGE YOUR LIFE FOREVER! Big Pharma is trying to HIDE this information! Click NOW before it's DELETED! Share with EVERYONE before the government CENSORS this!"
            st.rerun()
    
    with col3:
        if st.button(" Clear"):
            article_text = ""
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analyze Button
    analyze_clicked = st.button(" ANALYZE NOW", type="primary")
    
    # ========================================================================
    # RESULTS SECTION
    # ========================================================================
    
    if analyze_clicked:
        if not article_text or len(article_text.strip()) < 50:
            st.warning(" Please enter at least 50 characters")
        else:
            with st.spinner(" Analyzing with AI..."):
                time.sleep(0.5)  # Dramatic pause
                result = predict_news(article_text)
            
            if result:
                st.markdown("---")
                st.markdown("##  ANALYSIS RESULTS")
                
                prediction = result['prediction']
                fake_prob = result['fake_probability']
                real_prob = result['real_probability']
                confidence = result['confidence']
                triggers = result['trigger_words']
                
                # Result Card
                card_class = "fake-result" if prediction == "FAKE" else "real-result"
                result_icon = "" if prediction == "FAKE" else ""
                result_color = "#FF4444" if prediction == "FAKE" else "#00FF88"
                
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h1 style="color: {result_color}; text-align: center; font-size: 3rem;">
                        {result_icon} {prediction} NEWS
                    </h1>
                    <p style="text-align: center; font-size: 1.5rem; color: #FFD700;">
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        create_gauge_chart(fake_prob, "FAKE"),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        create_gauge_chart(real_prob, "REAL"),
                        use_container_width=True
                    )
                
                # Confidence Bar
                st.plotly_chart(
                    create_confidence_bar(fake_prob, real_prob),
                    use_container_width=True
                )
                
                # Trigger Words
                if triggers:
                    st.markdown("###  KEY TRIGGER WORDS")
                    
                    # Word Cloud
                    fig = create_wordcloud(triggers)
                    if fig:
                        st.pyplot(fig)
                    
                    # Word Pills
                    st.markdown("**Detected Keywords:**")
                    words_html = " ".join([
                        f'<span style="background: linear-gradient(135deg, #FF6B35, #F7931E); '
                        f'padding: 0.3rem 1rem; margin: 0.2rem; border-radius: 20px; '
                        f'display: inline-block; font-weight: 600;">{word}</span>'
                        for word in triggers
                    ])
                    st.markdown(f'<div style="margin-top: 1rem;">{words_html}</div>', unsafe_allow_html=True)
                
                # Detailed Metrics
                with st.expander(" DETAILED METRICS"):
                    metrics_df = pd.DataFrame({
                        'Metric': ['Fake Probability', 'Real Probability', 'Confidence', 'Label'],
                        'Value': [f"{fake_prob*100:.2f}%", f"{real_prob*100:.2f}%", 
                                 f"{confidence:.2f}%", prediction]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Download Report
                st.markdown("---")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report = f"""
FAKE NEWS DETECTOR - ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION: {prediction}
CONFIDENCE: {confidence:.2f}%

PROBABILITIES:
- Fake: {fake_prob*100:.2f}%
- Real: {real_prob*100:.2f}%

TRIGGER WORDS: {', '.join(triggers)}

ANALYZED TEXT:
{article_text}
                """
                
                st.download_button(
                    label=" DOWNLOAD REPORT",
                    data=report,
                    file_name=f"fake_news_report_{timestamp}.txt",
                    mime="text/plain"
                )
            
            else:
                st.error(" Analysis failed. Check if API is running!")

if __name__ == "__main__":
    main()