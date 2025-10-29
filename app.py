import time
from collections import deque
from dataclasses import dataclass
import av
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks
import streamlit as st
import requests

# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================
st.set_page_config(
    page_title="PPG Cardíaco em Tempo Real",
    page_icon="❤️",
    layout="wide",
)

# ==========================================================
# CSS / ESTILO
# ==========================================================
st.markdown(
    """
<style>
.main-header {
    background: linear-gradient(90deg,#ff6b6b,#ee5a24);
    padding:1rem;
    border-radius:12px;
    color:white;
    margin-bottom:1rem;
    text-align:center;
}
.metric-card {
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    padding:1rem;
    border-radius:14px;
    color:white;
    text-align:center;
}
.status {
    padding:.8rem;
    border-radius:10px;
    border-left:4px solid #00d4aa;
    background:rgba(0,212,170,.10);
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# FUNÇÃO DE FILTRO (Butterworth)
# ==========================================================
def bandpass_filter(sig, low=0.8, high=3.0, fs=30.0):
    sig = np.asarray(sig, dtype=float)
    if sig.size < 8:
        return sig - np.mean(sig)
    nyq = 0.5 * fs
    low = max(0.1, low)
    high = min(high, 0.95 * nyq)
    if nyq <= 0 or high <= low:
        return sig - np.mean(sig)
    wn = [low / nyq, high / nyq]
    b, a = butter(2, wn, btype="band")
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig - np.mean(sig)

# ==========================================================
# TURNAUTOMÁTICO (Metered) — força TCP/443
# ==========================================================
@st.cache_resource
def get_rtc_configuration():
    """
    Busca TURN automático via Metered e força relay (TCP/443)
    para funcionar no Render (sem UDP).
    """
    try:
        resp = requests.get("https://apppy.metered.live/api/v1/turn/credentials", timeout=5)
        data = resp.json()
        ice_servers = data.get("iceServers", [])
        if not ice_servers:
            raise RuntimeError("Nenhum servidor ICE retornado")
        return {
            "iceServers": ice_servers,
            "iceTransportPolicy": "relay",  # força uso de TURN via TCP
        }
    except Exception as e:
        st.warning(f"Falha ao buscar TURN ({e}), usando STUN Google.")
        return {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
            "iceTransportPolicy": "all",
        }

# ==========================================================
# ESTADO COMPARTILHADO
# ==========================================================
@dataclass
class SharedState:
    buffer_G: deque
    last_filtered: np.ndarray
    last_peaks: np.ndarray
    fps: float
    bpm_fixed: float | None
    bpm_collect: list
    last_bpm: float | None
    last_signal_ready: bool

if "ppg_state" not in st.session_state:
    st.session_state.ppg_state = SharedState(
        buffer_G=deque(maxlen=100),
        last_filtered=np.zeros(100),
        last_peaks=np.array([], dtype=int),
        fps=30.0,
        bpm_fixed=None,
        bpm_collect=[],
        last_bpm=None,
        last_signal_ready=False,
    )

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Nome", "BPM", "Data", "Hora"])

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### Configurações")
    cam_choice = st.radio(
        "Câmera", ["Traseira (environment)", "Frontal (user)"], index=0
    )
    facing_mode = "environment" if cam_choice.startswith("Traseira") else "user"
    if st.button("Nova medição"):
        s = st.session_state.ppg_state
        s.bpm_fixed = None
        s.bpm_collect = []
        s.last_bpm = None

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
<div class="main-header">
  <h1>PPG Cardíaco em Tempo Real</h1>
  <p>Detecção do pulso via câmera do celular (com flash automático)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# WEBRTC IMPORT
# ==========================================================
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    WEBRTC_AVAILABLE = True
except Exception as e:
    webrtc_streamer = None
    WebRtcMode = None
    VideoProcessorBase = object
    WEBRTC_AVAILABLE = False
    st.error("Dependências WebRTC ausentes: instale streamlit-webrtc, aiortc, av")

# ==========================================================
# PROCESSADOR DE VÍDEO
# ==========================================================
class PPGProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_time = time.time()
        self.frame_counter = 0
        self.fps = 30.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # ROI central
        side = int(min(h, w) * 0.5)
        y0, x0 = (h - side) // 2, (w - side) // 2
        roi = img[y0:y0+side, x0:x0+side]

        self.frame_counter += 1
        if self.frame_counter >= 30:
            now = time.time()
            elapsed = now - self.prev_time
            if elapsed > 0:
                self.fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.prev_time = now

        mean_bgr = np.mean(roi, axis=(0, 1))
        B, G, R = mean_bgr
        s: SharedState = st.session_state.ppg_state
        s.fps = float(self.fps)
        s.buffer_G.append(G)

        dedo = False
        if len(s.buffer_G) >= 30:
            dc = np.mean(s.buffer_G)
            ac = np.std(list(s.buffer_G)[-10:])
            snr_like = ac / (dc + 1e-9)
            vermelho_min = 60.0
            dedo = (R > G + vermelho_min) and (R > B + vermelho_min) and (snr_like > 0.005)

        if len(s.buffer_G) == s.buffer_G.maxlen and dedo:
            g = np.asarray(s.buffer_G, dtype=float)
            g = g / (np.mean(g) + 1e-9)
            g_ac = g - np.mean(g)
            g_f = bandpass_filter(g_ac, low=0.8, high=3.0, fs=max(s.fps, 1.0))
            min_dist = max(1, int(max(s.fps, 1.0) * 60.0 / 220.0))
            peaks, _ = find_peaks(g_f, distance=min_dist)
            s.last_filtered, s.last_peaks, s.last_signal_ready = g_f, peaks, True

            if len(peaks) > 1:
                intervals = np.diff(peaks) / max(s.fps, 1.0)
                bpm_now = 60.0 / np.mean(intervals)
                if 40.0 < bpm_now < 180.0:
                    if s.bpm_fixed is None:
                        s.bpm_collect.append(bpm_now)
                        if len(s.bpm_collect) >= 3:
                            s.bpm_fixed = np.mean(s.bpm_collect)
                            s.bpm_collect = []
                    s.last_bpm = s.bpm_fixed or bpm_now
                else:
                    s.last_bpm = None
        else:
            s.last_signal_ready = False

        cv2.putText(img, f"BPM: {s.last_bpm:.1f}" if s.last_bpm else "BPM: --",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(img, f"FPS: {s.fps:.1f}", (w - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================================
# INICIA STREAM
# ==========================================================
if WEBRTC_AVAILABLE:
    webrtc_streamer(
        key="ppg-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=get_rtc_configuration(),
        media_stream_constraints={
            "video": {
                "facingMode": facing_mode,
                "torch": True,  # ativa flash no Android
            },
            "audio": False,
        },
        video_processor_factory=PPGProcessor,
        async_processing=True,
    )
else:
    st.error("streamlit-webrtc não disponível.")

# ==========================================================
# LAYOUT PRINCIPAL
# ==========================================================
col1, col2 = st.columns([2, 1])
s = st.session_state.ppg_state

with col2:
    st.markdown("### Métricas")
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("BPM", f"{s.last_bpm:.1f}" if s.last_bpm else "--")
    st.metric("FPS", f"{s.fps:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Salvar Medição")
    nome = st.text_input("Nome", "")
    if st.button("Salvar"):
        if s.bpm_fixed:
            now = time.localtime()
            row = {
                "Nome": nome or "Sem nome",
                "BPM": round(s.bpm_fixed, 2),
                "Data": time.strftime("%Y-%m-%d", now),
                "Hora": time.strftime("%H:%M:%S", now),
            }
            st.session_state.history = pd.concat(
                [st.session_state.history, pd.DataFrame([row])], ignore_index=True
            )
            st.success(f"Salvo: {row['Nome']} - {row['BPM']} BPM")
        else:
            st.warning("Aguardando estabilizar o sinal (~3 amostras).")

    st.markdown("---")
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history, hide_index=True, use_container_width=True)
        csv = st.session_state.history.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV", csv, "historico_bpm.csv", "text/csv")
    else:
        st.info("Nenhuma medição salva.")

with col1:
    st.markdown("### Sinal PPG (tempo real)")
    if s.last_signal_ready:
        x = np.arange(len(s.last_filtered))
        y = s.last_filtered
        peaks = s.last_peaks
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="PPG"))
        if peaks.size > 0:
            fig.add_trace(go.Scatter(x=x[peaks], y=y[peaks], mode="markers", name="Picos"))
        fig.update_layout(height=320, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="status">Posicione o dedo e aguarde estabilizar.</div>', unsafe_allow_html=True)
