# src/live_scanner.py - Live RF Scanner with Modern UI
# RUN THIS AS streamlit run c:\Users\GeethD\Documents\Projects\Spectrum_Analyser_AI\src\live_scanner.py 

import streamlit as st
import numpy as np
import time
import threading
import json
import joblib
import tensorflow as tf
from rtlsdr import RtlSdr
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import os
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(
    page_title="RF Signal Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLING
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ff88;
        background: linear-gradient(90deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #00ff88;
    }
    .status-card {
        background: #1e1e2f;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: #0f0c29;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #302b63;
        text-align: center;
    }
    .signal-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00ff88;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #1e1e2f;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #00ff88, #00ccff);
        transition: width 0.5s;
    }
    .freq-display {
        font-size: 2.2rem;
        font-weight: 600;
        color: #00ccff;
        text-align: center;
        padding: 1rem;
        background: #0f0c29;
        border-radius: 10px;
        border: 1px solid #302b63;
    }
    .ensemble-result {
        background: linear-gradient(135deg, #0f0c29, #302b63);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #ffd700;
        text-align: center;
        margin: 1rem 0;
    }
    .ensemble-label {
        font-size: 2rem;
        font-weight: 700;
        color: #ffd700;
    }
</style>
""", unsafe_allow_html=True)

 
# LOAD MODELS
 
@st.cache_resource
def load_models():
    """Load all trained models and configurations."""
    models = {}
    
    try:
        # Load Random Forest
        models['rf'] = joblib.load('models/rf_radioml.pkl')
        models['rf_scaler'] = joblib.load('models/scaler_radioml.pkl')
        st.sidebar.success(" Random Forest loaded")
    except:
        st.sidebar.error(" Random Forest not found")
        models['rf'] = None
    
    try:
        # Load CNN
        models['cnn'] = tf.keras.models.load_model('models/cnn_radioml_final.keras')
        st.sidebar.success(" CNN loaded")
    except:
        st.sidebar.error(" CNN not found")
        models['cnn'] = None
    
    try:
        # Load class names
        with open('data/modulations.json', 'r') as f:
            models['class_names'] = json.load(f)
        
    except:
        st.sidebar.error(" Class names not found")
        models['class_names'] = ['Unknown'] * 11
    
    return models

 
# SIGNAL PROCESSING
 
def iq_to_spectrogram(iq_samples, fs=2.4e6, nperseg=256):
    """Convert IQ samples to spectrogram for display."""
    try:
       
        f, t, Sxx = signal.spectrogram(
            np.real(iq_samples),
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg//2,
            window='hann'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        return f, t, Sxx_db
    except:
        return None, None, None

def preprocess_for_cnn(iq_samples):
    """Preprocess IQ samples for CNN input."""
    iq_samples = iq_samples[:128]
    i = np.real(iq_samples)
    q = np.imag(iq_samples)
    iq_reshaped = np.stack([i, q], axis=0)
    iq_reshaped = iq_reshaped.reshape(1, 2, 128, 1)
    return iq_reshaped.astype(np.float32)

def preprocess_for_rf(iq_samples, scaler):
    """Preprocess IQ samples for Random Forest."""
    iq_samples = iq_samples[:128]
    i = np.real(iq_samples)
    q = np.imag(iq_samples)
    iq_flat = np.concatenate([i, q]).reshape(1, -1)
    return scaler.transform(iq_flat)

 
# CLASSIFICATION FUNCTIONS
 
def classify_signal(iq_samples, models):
    """Classify signal using both models and ensemble."""
    results = {}
    
    # Random Forest
    if models['rf'] is not None:
        try:
            X_rf = preprocess_for_rf(iq_samples, models['rf_scaler'])
            rf_probs = models['rf'].predict_proba(X_rf)[0]
            rf_pred = np.argmax(rf_probs)
            results['rf'] = {
                'prediction': rf_pred,
                'confidence': np.max(rf_probs) * 100,
                'probabilities': rf_probs
            }
        except:
            results['rf'] = None
    else:
        results['rf'] = None
    
    # CNN
    if models['cnn'] is not None:
        try:
            X_cnn = preprocess_for_cnn(iq_samples)
            cnn_probs = models['cnn'].predict(X_cnn, verbose=0)[0]
            cnn_pred = np.argmax(cnn_probs)
            results['cnn'] = {
                'prediction': cnn_pred,
                'confidence': np.max(cnn_probs) * 100,
                'probabilities': cnn_probs
            }
        except:
            results['cnn'] = None
    else:
        results['cnn'] = None
    
    # Ensemble (Average Probabilities)
    if results['rf'] is not None and results['cnn'] is not None:
        ensemble_probs = (results['rf']['probabilities'] + results['cnn']['probabilities']) / 2
        ensemble_pred = np.argmax(ensemble_probs)
        results['ensemble'] = {
            'prediction': ensemble_pred,
            'confidence': np.max(ensemble_probs) * 100,
            'probabilities': ensemble_probs
        }
    elif results['cnn'] is not None:
        results['ensemble'] = results['cnn']
    elif results['rf'] is not None:
        results['ensemble'] = results['rf']
    else:
        results['ensemble'] = None
    
    return results

 
# SDR SCANNER
 
class SDRScanner:
    def __init__(self, models):
        self.models = models
        self.sdr = None
        self.is_scanning = False
        self.current_freq = 0
        self.results = []
        self.spectrogram_data = []
        self.all_power = []
        self.all_frequencies = []
        
    def init_sdr(self):
        """Initialize RTL-SDR."""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = 2.4e6
            self.sdr.gain = 30
            return True
        except Exception as e:
            st.error(f"Failed to initialize SDR: {e}")
            return False
    
    def scan_frequency(self, freq, duration=0.3):
        """Scan a single frequency and classify."""
        if self.sdr is None:
            return None
        
        try:
            self.sdr.center_freq = freq
            self.current_freq = freq
            
            # Read samples
            samples = self.sdr.read_samples(262144)
            
            # Compute power
            power = np.mean(np.abs(samples) ** 2)
            
            # Get spectrogram
            f, t, Sxx = iq_to_spectrogram(samples)
            
            # Classify
            results = classify_signal(samples, self.models)
            
            if results and results['ensemble'] is not None:
                class_name = self.models['class_names'][results['ensemble']['prediction']]
                confidence = results['ensemble']['confidence']
                
                return {
                    'frequency': freq,
                    'power': power,
                    'prediction': class_name,
                    'confidence': confidence,
                    'rf_result': results.get('rf'),
                    'cnn_result': results.get('cnn'),
                    'ensemble_result': results['ensemble'],
                    'spectrogram': (f, t, Sxx),
                    'samples': samples
                }
            return None
            
        except Exception as e:
            return None
    
    def scan_range(self, start_freq, end_freq, step_size, progress_callback=None, stop_event=None):
        """Scan a range of frequencies."""
        self.is_scanning = True
        self.results = []
        self.spectrogram_data = []
        self.all_power = []
        self.all_frequencies = []
        
        frequencies = np.arange(start_freq, end_freq + step_size, step_size)
        total_steps = len(frequencies)
        
        for i, freq in enumerate(frequencies):
            if stop_event and stop_event.is_set():
                break
            
            result = self.scan_frequency(freq)
            
            if result:
                self.results.append(result)
                self.all_power.append(result['power'])
                self.all_frequencies.append(freq)
                
                if result['spectrogram'][0] is not None:
                    self.spectrogram_data.append({
                        'freq': freq,
                        'f': result['spectrogram'][0],
                        't': result['spectrogram'][1],
                        'Sxx': result['spectrogram'][2]
                    })
            
            if progress_callback:
                progress_callback((i + 1) / total_steps, freq)
        
        self.is_scanning = False
        return self.results
    
    def close(self):
        if self.sdr:
            self.sdr.close()
            self.sdr = None

 
# UI COMPONENTS
 
def render_header():
    st.markdown('<div class="main-header"> Real-Time RF Spectrum Analysis & Modulation Recognition Using <b> Hybrid CNN-Random Forest Ensemble</div>', unsafe_allow_html=True)

def render_controls():
    """Render control panel."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_freq = st.number_input(
            "Start Frequency (MHz)",
            min_value=24.0,
            max_value=1700.0,
            value=88.0,
            step=1.0,
            format="%.1f"
        )
    
    with col2:
        end_freq = st.number_input(
            "End Frequency (MHz)",
            min_value=24.0,
            max_value=1700.0,
            value=108.0,
            step=1.0,
            format="%.1f"
        )
    
    with col3:
        step_size = st.selectbox(
            "Step Size (MHz)",
            options=[0.1, 0.25, 0.5, 1.0, 2.0],
            index=2
        )
    
    return start_freq, end_freq, step_size

def render_results_dashboard(results, class_names):
    """Render classification results dashboard with proper type handling."""
    
    if not results:
        st.info("No signals detected yet. Start scanning!")
        return
    
    # Latest result
    latest = results[-1]
    
    # Convert numpy types to Python types
    freq_mhz = float(latest["frequency"] / 1e6)
    
    # Frequency display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'<div class="freq-display">{freq_mhz:.3f} MHz</div>', unsafe_allow_html=True)
    
    # Three-column layout for model outputs
    st.markdown("### Model Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("**🌲 Random Forest**")
        if latest.get('rf_result'):
            rf = latest['rf_result']
            pred_idx = int(rf['prediction'])
            pred_name = class_names[pred_idx]
            conf = float(rf['confidence'])
            st.markdown(f'<div class="signal-label" style="color:#00ff88">{pred_name}</div>', unsafe_allow_html=True)
            st.progress(conf/100)
            st.caption(f"Confidence: {conf:.1f}%")
        else:
            st.caption("N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("**🧠 CNN**")
        if latest.get('cnn_result'):
            cnn = latest['cnn_result']
            pred_idx = int(cnn['prediction'])
            pred_name = class_names[pred_idx]
            conf = float(cnn['confidence'])
            st.markdown(f'<div class="signal-label" style="color:#00ccff">{pred_name}</div>', unsafe_allow_html=True)
            st.progress(conf/100)
            st.caption(f"Confidence: {conf:.1f}%")
        else:
            st.caption("N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="ensemble-result">', unsafe_allow_html=True)
        st.markdown("**⭐ Ensemble (Average)**")
        if latest.get('ensemble_result'):
            ensemble = latest['ensemble_result']
            pred_idx = int(ensemble['prediction'])
            pred_name = class_names[pred_idx]
            conf = float(ensemble['confidence'])
            st.markdown(f'<div class="ensemble-label">{pred_name}</div>', unsafe_allow_html=True)
            st.progress(conf/100)
            st.caption(f"Confidence: {conf:.1f}%")
        else:
            st.caption("N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show all probabilities as a bar chart (optional)
    if latest.get('ensemble_result'):
        st.markdown("###  Class Probabilities")
        probs = latest['ensemble_result']['probabilities']
        
        # Convert to Python list and flatten
        if hasattr(probs, 'flatten'):
            probs = probs.flatten()
        
        # Create a simple bar chart
        import pandas as pd
        prob_df = pd.DataFrame({
            'Class': class_names,
            'Probability': [float(p) * 100 for p in probs[:len(class_names)]]
        })
        prob_df = prob_df.sort_values('Probability', ascending=False)
        
        st.dataframe(
            prob_df,
            use_container_width=True,
            hide_index=True
        )

def render_spectrogram(spectrogram_data, class_names):
    """Render 2D spectrogram using Plotly."""
    if not spectrogram_data:
        st.info("Collecting spectrogram data...")
        return
    
    # Use latest spectrogram
    latest = spectrogram_data[-1]
    f = latest['f']
    t = latest['t']
    Sxx = latest['Sxx']
    freq_mhz = latest['freq'] / 1e6
    
    if f is None:
        st.info("Spectrogram data unavailable")
        return
    
    fig = go.Figure(data=go.Heatmap(
        z=Sxx,
        x=t,
        y=f / 1e6,
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)'),
        hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.3f} MHz<br>Power: %{z:.1f} dB<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Spectrogram at {freq_mhz:.3f} MHz",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (MHz)",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_waterfall(results, class_names):
    """Render waterfall plot of signal power and classifications."""
    if not results:
        return
    
    frequencies = [float(r['frequency'] / 1e6) for r in results]
    powers = [float(r['power']) for r in results]
    
    # Get classifications and confidences
    classifications = []
    confidences = []
    for r in results:
        if r.get('ensemble_result'):
            pred_idx = int(r['ensemble_result']['prediction'])
            classifications.append(class_names[pred_idx])
            confidences.append(float(r['ensemble_result']['confidence']))
        else:
            classifications.append('Unknown')
            confidences.append(0)
    
    # Color map for signal types
    color_map = {
        'FM_broadcast': '#00ff88',
        'APRS': '#ff6b6b',
        'NOAA_weather': '#5f27cd',
        'BPSK': '#00d2d3',
        'QPSK': '#54a0ff',
        '8PSK': '#5f27cd',
        'QAM16': '#ff6b6b',
        'QAM64': '#feca57',
        'PAM4': '#1dd1a1',
        'GFSK': '#48dbfb',
        'CPFSK': '#ff9f43',
        'AM-DSB': '#00d2d3',
        'AM-SSB': '#54a0ff',
        'WBFM': '#ff6b6b',
        'ISM_sensors': '#1dd1a1',
        'pager': '#ff9f43',
        'noise': '#8395a7',
        'FRS_GMRS': '#48dbfb',
        'Unknown': '#ffffff'
    }
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(" Power Spectrum", " - Signal Classifications")
    )
    
    # Power spectrum (Row 1)
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=powers,
            mode='lines+markers',
            name='Power',
            line=dict(color='#00ff88', width=2),
            marker=dict(size=4, color='#00ff88'),
            hovertemplate='Frequency: %{x:.3f} MHz<br>Power: %{y:.3e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Classifications (Row 2) - stacked as colored markers
    # Group by classification for better visualization
    unique_classes = list(set(classifications))
    
    # Create a y-position for each classification
    y_positions = {}
    for i, cls in enumerate(unique_classes):
        y_positions[cls] = i * 0.1 + 0.05
    
    for cls in unique_classes:
        indices = [i for i, c in enumerate(classifications) if c == cls]
        if not indices:
            continue
        
        x_vals = [frequencies[i] for i in indices]
        y_vals = [y_positions[cls]] * len(indices)
        conf_vals = [confidences[i] for i in indices]
        
        color = color_map.get(cls, '#ffffff')
        
        # Create hover text
        hover_text = [f"<b>{cls}</b><br>Confidence: {conf:.1f}%" for conf in conf_vals]
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=cls,
                marker=dict(
                    size=12,
                    color=color,
                    symbol='square',
                    line=dict(width=1, color='#ffffff')
                ),
                text=hover_text,
                hoverinfo='text+x',
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10)
        ),
        hovermode='x unified'
    )
    
    # Update axes (use update_yaxes for subplots)
    fig.update_yaxes(
        title_text="Power (linear)",
        row=1, col=1,
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False
    )
    
    fig.update_yaxes(
        title_text="Classification",
        row=2, col=1,
        showticklabels=False,
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False
    )
    
    fig.update_xaxes(
        title_text="Frequency (MHz)",
        row=2, col=1,
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False
    )
    
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False,
        row=1, col=1
    )
    
    st.plotly_chart(fig, use_container_width=True, key="waterfall_plot")

 
# MAIN APPLICATION
 
def main():
    render_header()
    
    # Initialize session state
    if 'scanner' not in st.session_state:
        st.session_state.scanner = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'spectrogram_data' not in st.session_state:
        st.session_state.spectrogram_data = []
    if 'is_scanning' not in st.session_state:
        st.session_state.is_scanning = False
    if 'stop_scan' not in st.session_state:
        st.session_state.stop_scan = False
    if 'sdr_connected' not in st.session_state:
        st.session_state.sdr_connected = False
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    if models['rf'] is None and models['cnn'] is None:
        st.error("No models loaded! Please train models first.")
        return
    
    st.sidebar.title("📡 Controls")
    
    # SDR Connection Status
    st.sidebar.markdown("### SDR Status")
    
    if st.session_state.sdr_connected and st.session_state.scanner:
        st.sidebar.success(" SDR Connected")
    else:
        st.sidebar.error(" SDR Not Connected")
        st.sidebar.caption("Click 'Connect SDR' to start")
    
    # Connect SDR Button
    if st.sidebar.button(" Connect SDR", use_container_width=True):
        with st.spinner("Connecting to RTL-SDR..."):
            scanner = SDRScanner(models)
            if scanner.init_sdr():
                st.session_state.scanner = scanner
                st.session_state.sdr_connected = True
                st.sidebar.success(" SDR Connected!")
                st.rerun()
            else:
                st.sidebar.error(" Failed to connect SDR")
                st.session_state.sdr_connected = False
                st.rerun()
    
    # Disconnect SDR Button
    if st.session_state.sdr_connected:
        if st.sidebar.button(" Disconnect SDR", use_container_width=True):
            if st.session_state.scanner:
                st.session_state.scanner.close()
            st.session_state.scanner = None
            st.session_state.sdr_connected = False
            st.session_state.results = []
            st.session_state.spectrogram_data = []
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Scan Settings")
    
    start_freq, end_freq, step_size = render_controls()
    
    # Convert to Hz
    start_hz = start_freq * 1e6
    end_hz = end_freq * 1e6
    step_hz = step_size * 1e6
    
    st.sidebar.caption(f"Scanning {start_freq:.1f} to {end_freq:.1f} MHz")
    total_steps = int((end_hz - start_hz) / step_hz) + 1
    st.sidebar.caption(f"Total steps: {total_steps}")
    
    # Scan button
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        scan_button = st.button("▶️ Start Scan", use_container_width=True, type="primary")
    
    with col2:
        stop_button = st.button("⏹️ Stop", use_container_width=True)
    
    # Progress
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    
    if scan_button:
        # Check if SDR is connected
        if not st.session_state.sdr_connected or st.session_state.scanner is None:
            st.sidebar.error(" SDR not connected! Click 'Connect SDR' first.")
            st.error(" Cannot scan: RTL-SDR not connected. Please connect your SDR first.")
        elif st.session_state.is_scanning:
            st.warning("Scan already in progress!")
        else:
            st.session_state.is_scanning = True
            st.session_state.stop_scan = False
            st.session_state.results = []
            st.session_state.spectrogram_data = []
            
            scanner = st.session_state.scanner
            
            # Progress callback
            def update_progress(progress, freq):
                progress_bar.progress(progress)
                status_text.text(f"Scanning: {freq/1e6:.3f} MHz")
            
            # Run scan
            results = scanner.scan_range(
                start_hz, end_hz, step_hz,
                progress_callback=update_progress,
                stop_event=None
            )
            
            if results:
                st.session_state.results = results
                for r in results:
                    if r.get('spectrogram') and r['spectrogram'][0] is not None:
                        st.session_state.spectrogram_data.append({
                            'freq': r['frequency'],
                            'f': r['spectrogram'][0],
                            't': r['spectrogram'][1],
                            'Sxx': r['spectrogram'][2]
                        })
            
            st.session_state.is_scanning = False
            progress_bar.progress(1.0)
            status_text.text("Scan complete!")
            st.rerun()
    
    if stop_button:
        st.session_state.stop_scan = True
        if st.session_state.scanner:
            st.session_state.scanner.is_scanning = False
        st.session_state.is_scanning = False
        st.warning("Scan stopped!")
    
     
    # DISPLAY RESULTS
     
    st.markdown("---")
    
    # Show SDR status banner in main area
    if not st.session_state.sdr_connected:
        st.warning("""
         **SDR Not Connected**
        
        Please connect your RTL-SDR device and click **"Connect SDR"** in the sidebar to start scanning.
        """)
        st.info("📡 **Tip**: Make sure your RTL-SDR is plugged in and the drivers are installed correctly.")
        return
    
    if st.session_state.results:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            render_results_dashboard(st.session_state.results, models['class_names'])
            st.markdown("###  Waterfall View")
            render_waterfall(st.session_state.results, models['class_names'])
        
        with col_right:
            st.markdown("###  Spectrogram")
            render_spectrogram(st.session_state.spectrogram_data, models['class_names'])
            
            # Summary stats
            st.markdown("###  Scan Summary")
            
            if st.session_state.results:
                classes_found = {}
                for r in st.session_state.results:
                    if r.get('ensemble_result'):
                        cls = models['class_names'][int(r['ensemble_result']['prediction'])]
                        classes_found[cls] = classes_found.get(cls, 0) + 1
                
                if classes_found:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame({
                            'Signal Type': list(classes_found.keys()),
                            'Count': list(classes_found.values())
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
    else:
        st.info(" No results to display. Start a scan to see signals!")

if __name__ == "__main__":
    import pandas as pd
    main()