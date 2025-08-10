import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spotify Hit Predictor | AI-Powered Music Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INITIALIZE SESSION STATE ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.pop_probability = 0.0
    st.session_state.prediction = None

# --- CONSTANTS ---
FEATURE_ORDER = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness',
    'tempo', 'valence'
]

DEFAULT_VALUES = {
    'danceability': 0.7, 'energy': 0.8, 'loudness': -5.0, 'speechiness': 0.1,
    'acousticness': 0.5, 'instrumentalness': 0.0, 'liveness': 0.15, 'valence': 0.5,
    'tempo': 120.0, 'duration_ms': 200000
}

ZEROED_VALUES = {
    'danceability': 0.0, 'energy': 0.0, 'loudness': -60.0, 'speechiness': 0.0,
    'acousticness': 0.0, 'instrumentalness': 0.0, 'liveness': 0.0, 'valence': 0.0,
    'tempo': 0.0, 'duration_ms': 20000
}

FEATURE_DESCRIPTIONS = {
    'danceability': 'How suitable a track is for dancing (0.0 = least danceable, 1.0 = most danceable)',
    'energy': 'Perceptual measure of intensity and power (0.0 = low energy, 1.0 = high energy)',
    'loudness': 'Overall loudness of a track in decibels (dB)',
    'speechiness': 'Presence of spoken words (0.0 = music, 1.0 = speech)',
    'acousticness': 'Confidence measure of whether the track is acoustic',
    'instrumentalness': 'Predicts whether a track contains no vocals',
    'liveness': 'Detects the presence of an audience in the recording',
    'valence': 'Musical positiveness (0.0 = sad/angry, 1.0 = happy/euphoric)',
    'tempo': 'Overall estimated tempo in beats per minute (BPM)',
    'duration_ms': 'Duration of the track in milliseconds'
}

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model with error handling."""
    try:
        model = joblib.load('spotify_model.joblib')
        return model, True
    except (FileNotFoundError, Exception):
        return None, False

model, model_loaded = load_model()

# --- ENHANCED UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .block-container {
        padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .result-wrapper { animation: fadeIn 0.5s ease-out; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        font-size: 3.5rem; font-weight: 700; text-align: center;
        margin-bottom: 0.5rem; letter-spacing: -0.02em;
    }
    .subtitle {
        text-align: center; color: #B3B3B3; font-size: 1.2rem;
        font-weight: 400; margin-bottom: 2rem;
    }
    .result-container {
        text-align: center; padding: 2rem; border-radius: 20px;
        margin-top: 1.5rem; position: relative; overflow: hidden;
    }
    .result-hit {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.15) 0%, rgba(30, 215, 96, 0.15) 100%);
        border: 2px solid #1DB954;
    }
    .result-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .result-subtitle { font-size: 1.1rem; opacity: 0.8; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- ERROR HANDLING FOR MODEL ---
if not model_loaded:
    st.error("⚠️ Model Not Found. Please ensure 'spotify_model.joblib' is in the correct directory.", icon="🔥")
    st.stop()

# --- HEADER SECTION ---
st.markdown('<h1 class="main-header">Spotify Hit Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Harness the power of AI to predict your song\'s chart potential</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎛️ Audio Features Control Panel")
    st.markdown("---")
    
    def create_feature_input(feature_name, min_val, max_val, default_val, step, emoji, unit=""):
        col1, col2 = st.columns([0.1, 0.9])
        with col1: st.write(emoji)
        with col2:
            return st.slider(f"**{feature_name.title()}**{unit}", min_val, max_val, default_val, step, help=FEATURE_DESCRIPTIONS.get(feature_name, ""), key=feature_name)

    danceability = create_feature_input('danceability', 0.0, 1.0, DEFAULT_VALUES['danceability'], 0.01, '🕺')
    energy = create_feature_input('energy', 0.0, 1.0, DEFAULT_VALUES['energy'], 0.01, '⚡')
    loudness = create_feature_input('loudness', -60.0, 0.0, DEFAULT_VALUES['loudness'], 0.1, '🔊', ' (dB)')
    speechiness = create_feature_input('speechiness', 0.0, 1.0, DEFAULT_VALUES['speechiness'], 0.01, '💬')
    acousticness = create_feature_input('acousticness', 0.0, 1.0, DEFAULT_VALUES['acousticness'], 0.01, '🎻')
    instrumentalness = create_feature_input('instrumentalness', 0.0, 1.0, DEFAULT_VALUES['instrumentalness'], 0.01, '🎸')
    liveness = create_feature_input('liveness', 0.0, 1.0, DEFAULT_VALUES['liveness'], 0.01, '🏟️')
    valence = create_feature_input('valence', 0.0, 1.0, DEFAULT_VALUES['valence'], 0.01, '😊')
    tempo = create_feature_input('tempo', 0.0, 250.0, DEFAULT_VALUES['tempo'], 1.0, '🥁', ' (BPM)')
    duration_ms = create_feature_input('duration_ms', 20000, 500000, DEFAULT_VALUES['duration_ms'], 1000, '⏱️', ' (ms)')
    st.markdown("---")
    
    def reset_features_to_zero():
        """Sets all feature values in session_state to their zeroed/minimum value."""
        for feature, value in ZEROED_VALUES.items():
            if feature == 'duration_ms':
                st.session_state[feature] = int(value)
            else:
                st.session_state[feature] = float(value)

    st.button(
        "🔄 Reset to Zero",
        use_container_width=True,
        on_click=reset_features_to_zero
    )
        
    st.markdown("---")
    st.markdown(f'<div style="text-align: center; color: #888;">App Last Updated: {datetime.now().strftime("%B %Y")}</div>', unsafe_allow_html=True)

# Prepare input data for prediction
input_data = {
    'danceability': danceability, 'energy': energy, 'loudness': loudness, 
    'speechiness': speechiness, 'acousticness': acousticness, 
    'instrumentalness': instrumentalness, 'liveness': liveness, 
    'valence': valence, 'tempo': tempo, 'duration_ms': duration_ms
}
input_df = pd.DataFrame(input_data, index=[0])


# --- Main content area
with st.container(border=True):
    col1, col2 = st.columns([0.55, 0.45], gap="large")
    with col1:
        st.markdown("<h4 style='text-align: center;'>Audio Feature Profile</h4>", unsafe_allow_html=True)    
        st.markdown("<br>", unsafe_allow_html=True)
        radar_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
        radar_values = input_df[radar_features].iloc[0].values.tolist()
        fig_radar = go.Figure(go.Scatterpolar(r=radar_values, theta=[feat.capitalize() for feat in radar_features], fill='toself', name='Your Track', line=dict(color='#1DB954', width=3), hovertemplate='%{theta}: %{r:.2f}<extra></extra>'))
        fig_radar.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.15)')), showlegend=False, height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col2:
        st.markdown("<h3 style='text-align: center;'>🚀 Hit Probability</h3>", unsafe_allow_html=True)
        def create_gauge_chart(probability):
            bar_color = "#1DB954" if probability >= 50 else "#E53935" if probability > 0 else "#B3B3B3"
            fig = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=probability, 
                domain={'x': [0, 1], 'y': [0, 1]}, 
                number={'suffix': "%", 'font': {'size': 36, 'color': bar_color}}, 
                gauge={
                    'axis': {'range': [None, 100]}, 
                    'bar': {'color': bar_color, 'thickness': 0.3}, 
                    'bgcolor': "rgba(255, 255, 255, 0.05)", 
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(229, 57, 53, 0.2)'}, 
                        {'range': [50, 100], 'color': 'rgba(29, 185, 84, 0.2)'}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"}, height=230, margin=dict(l=20, r=20, t=20, b=10))
            return fig
            
        st.plotly_chart(create_gauge_chart(st.session_state.pop_probability), use_container_width=True)

        if st.button('🎵 **ANALYZE HIT POTENTIAL**', use_container_width=True, type="primary"):
            with st.spinner('🎵 Analyzing your track...'):
                time.sleep(0.5)
                prediction = model.predict(input_df[FEATURE_ORDER])
                prediction_proba = model.predict_proba(input_df[FEATURE_ORDER])
                st.session_state.pop_probability = prediction_proba[0][1] * 100
                st.session_state.prediction = prediction[0]
                st.session_state.prediction_made = True
            st.rerun()

        if st.session_state.prediction_made:
            if st.session_state.prediction == 1:
                st.markdown('<div class="result-wrapper">', unsafe_allow_html=True)
                st.markdown('<div class="result-container result-hit"><div class="result-title" style="color: #1DB954;">🎉 CHART BUSTER!</div><div class="result-subtitle">This track has strong hit potential</div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.toast("📈 Needs Work: Consider adjusting key features.", icon="🤔")

# --- FOOTER & MODEL INFO ---
st.markdown("---")
# CHANGED: The content in the expander columns is now wrapped in styled HTML for center alignment.
with st.expander("🧠 **About the AI Model & Methodology**", expanded=False):
   col1, col2, col3 = st.columns(3)
  
   with col1:
       st.markdown("""
       <div style="text-align: center;">
           <h4>🤖 Model Architecture</h4>
           <ul style="display: inline-block; text-align: left;">
               <li><strong>Algorithm</strong>: Random Forest Classifier</li>
               <li><strong>Trees</strong>: 100 estimators</li>
               <li><strong>Training Data</strong>: 30,000+ Spotify tracks</li>
               <li><strong>Accuracy</strong>: ~85% on test set</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)
  
   with col2:
       st.markdown("""
       <div style="text-align: center;">
           <h4>📊 Feature Importance</h4>
           <ol style="display: inline-block; text-align: left;">
               <li><strong>Danceability</strong> - Most predictive</li>
               <li><strong>Energy</strong> - High correlation with hits</li>
               <li><strong>Valence</strong> - Positive songs perform better</li>
               <li><strong>Loudness</strong> - Commercial tracks are louder</li>
           </ol>
       </div>
       """, unsafe_allow_html=True)
  
   with col3:
       st.markdown("""
       <div style="text-align: center;">
           <h4>🎯 Classification Logic</h4>
           <ul style="display: inline-block; text-align: left;">
                <li><strong>Hit Threshold</strong>: Popularity ≥ 60</li>
                <li><strong>Data Source</strong>: Spotify Web API</li>
                <li><strong>Time Period</strong>: 2010-2023</li>
                <li><strong>Genres</strong>: All major genres included</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
   st.markdown(f"""
   <div style="text-align: center; padding: 2rem;">
       <p style="color: #B3B3B3; margin-bottom: 1rem;">Created with ❤️ by <strong>Muhammad Waseem Sabir</strong></p>
       <p style="color: #666; font-size: 0.9rem;">
           Powered by AI • Last updated: {datetime.now().strftime('%B %Y')}
       </p>
       <a href="https://github.com/waseemkathia/spotify-songs-popularity"
          style="color: #1DB954; text-decoration: none; font-weight: 500;">
           🔗 View Source Code on GitHub
       </a>
   </div>
   """, unsafe_allow_html=True)