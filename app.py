# ==========================================
# PPG Cardíaco em Tempo Real (Android + iOS)
# - TURN automático (Metered)
# - Cálculo de BPM correto por picos no canal G
# - Flash automático: torch (Android) / tela branca (iOS fallback)
# - Layout em 2 colunas + histórico + CSV
# ==========================================

import time
from collections import deque
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from scipy.signal import butter, filtfilt, find_peaks
import streamlit as st
import streamlit.components.v1 as components

# ---------- Config da página ----------
st.set_page_config(
    page_title="PPG Cardíaco em Tempo Real",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Estilo rápido ----------
st.markdown(
    """
    <style>
    .main-header {background: linear-gradient(90deg,#ff6b6b,#ee5a24);padding:1rem;border-radius:12px;color:white;margin-bottom:1rem;text-align:center}
    .metric-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:14px;color:white;text-align:center}
    .status {padding:.8rem;border-radius:10px;border-left:4px solid #00d4aa;background:rgba(0,212,170,.10)}
    .warning {padding:.8rem;border-radius:10px;border-left:4px solid #ff9f43;background:rgba(255,159,67,.10)}
    .error {padding:.8rem;border-radius:10px;border-left:4px solid #ff6b6b;background:rgba(255,107,107,.10)}
    .screen-torch{position:fixed;inset:0;background:#ffffff;z-index:9999;display:none;align-items:center;justify-content:center}
    .screen-torch span{font-family:system-ui, Arial; color:#000; opacity:.35; font-weight:700}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utilidades ----------
def bandpass_filter(sig, low=0.8, high=3.0, fs=30.0):
    """
    Butterworth passa-faixa (~48-180 bpm) no canal G.
    Robusto a variações pequenas de FPS.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size < 8:
        return sig - np.mean(sig) if sig.size else sig

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


# ---------- Estado de sessão ----------
if "ppg_state" not in st.session_state:
    st.session_state.ppg_state = SharedState(
        buffer_G=deque(maxlen=120),  # ~4s @30fps
        last_filtered=np.zeros(120),
        last_peaks=np.array([], dtype=int),
        fps=30.0,
        bpm_fixed=None,
        bpm_collect=[],
        last_bpm=None,
        last_signal_ready=False,
    )

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Nome", "BPM", "Data", "Hora"])

if "camera_on" not in st.session_state:
    st.session_state.camera_on = True  # precisa de gesto no iOS, toggle ajuda

# ---------- Header ----------
st.markdown(
    """
    <div class="main-header">
      <h1>PPG Cardíaco em Tempo Real</h1>
      <p>Medição via câmera (WebRTC) + gráfico ao vivo</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Configurações")
    cam_choice = st.radio(
        "Câmera do celular", ["Traseira (environment)", "Frontal (user)"], index=0
    )
    facing_mode = "environment" if cam_choice.startswith("Traseira") else "user"

    st.markdown("---")
    st.markdown("### Câmera")
    st.toggle("Iniciar câmera", key="camera_on")

    st.markdown("---")
    st.markdown("### Dicas")
    st.markdown(
        "- Posicione **o dedo** cobrindo totalmente a câmera.\n"
        "- Mantenha o celular **parado**.\n"
        "- Aguardem ~5–10 s para estabilizar e fixar o BPM."
    )
    st.markdown("---")
    if st.button("Nova medição"):
        s = st.session_state.ppg_state
        s.bpm_fixed = None
        s.bpm_collect = []
        s.last_bpm = None

# ---------- TURN automático (Metered) ----------
# Se mudar o subdomínio, troque a linha abaixo:
METERED_DOMAIN = "apppy.metered.live"  # <--- seu subdomínio (pelo print)

@st.cache_resource
def get_turn_rtc_configuration():
    """
    Busca iceServers (STUN+TURN) efêmeros na API pública do Metered.
    Funciona no plano free. Se falhar, cai para STUN padrão.
    """
    url = f"https://{METERED_DOMAIN}/api/v1/turn/credentials"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        ice = data.get("iceServers", [])
        if not ice:
            raise RuntimeError("Resposta sem iceServers.")
        return {
            "iceServers": ice,
            "iceTransportPolicy": "all",  # mude para "relay" se quiser forçar TURN
        }
    except Exception as e:
        st.info("Usando STUN padrão (não foi possível obter TURN automático).")
        return {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"]}
            ],
            "iceTransportPolicy": "all",
        }

rtc_configuration = get_turn_rtc_configuration()

# ---------- Vídeo processor ----------
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    WEBRTC_AVAILABLE = True
except Exception as e:
    webrtc_streamer = None
    WebRtcMode = None
    VideoProcessorBase = object
    WEBRTC_AVAILABLE = False
    st.error(
        "Dependências do WebRTC ausentes. Instale: "
        "`pip install streamlit-webrtc aiortc av pyopenssl cryptography`"
    )

class PPGProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_time = time.time()
        self.frame_counter = 0
        self.fps = 30.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # ROI central (50% do menor lado)
        side = int(min(h, w) * 0.5)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        y1 = y0 + side
        x1 = x0 + side
        roi = img[y0:y1, x0:x1]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # FPS móvel
        self.frame_counter += 1
        if self.frame_counter >= 30:
            now = time.time()
            elapsed = now - self.prev_time
            if elapsed > 0:
                self.fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.prev_time = now

        # Média de cor
        mean_bgr = np.mean(roi, axis=(0, 1))
        B, G, R = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])

        s: SharedState = st.session_state.ppg_state
        s.fps = float(self.fps)
        s.buffer_G.append(G)

        # Heurística: dedo presente e qualidade mínima
        dedo = False
        if len(s.buffer_G) >= 30:
            dc = float(np.mean(s.buffer_G))
            ac = float(np.std(list(s.buffer_G)[-10:]))
            snr_like = ac / (dc + 1e-9)
            vermelho_min = 60.0
            dedo = (R > G + vermelho_min) and (R > B + vermelho_min) and (snr_like > 0.005)

        # Processo somente com janela cheia e dedo presente
        if len(s.buffer_G) == s.buffer_G.maxlen and dedo:
            g = np.asarray(s.buffer_G, dtype=float)
            g = g / (np.mean(g) + 1e-9)
            g_ac = g - np.mean(g)
            g_f = bandpass_filter(g_ac, low=0.8, high=3.0, fs=max(s.fps, 1.0))

            # Distância mínima entre picos para ~220 bpm máx
            min_dist = max(1, int(max(s.fps, 1.0) * 60.0 / 220.0))
            peaks, _ = find_peaks(g_f, distance=min_dist, prominence=np.std(g_f)*0.2)

            s.last_filtered = g_f
            s.last_peaks = peaks
            s.last_signal_ready = True

            if len(peaks) > 1:
                intervals = np.diff(peaks) / max(s.fps, 1.0)  # segundos entre batidas
                bpm_now = 60.0 / np.mean(intervals)
                if 40.0 < bpm_now < 180.0:
                    if s.bpm_fixed is None:
                        s.bpm_collect.append(bpm_now)
                        if len(s.bpm_collect) >= 3:  # fixa após 3 estimativas
                            s.bpm_fixed = float(np.mean(s.bpm_collect))
                            s.bpm_collect = []
                    s.last_bpm = s.bpm_fixed if s.bpm_fixed is not None else bpm_now
                else:
                    s.last_bpm = None
            else:
                s.last_bpm = None
        else:
            s.last_signal_ready = False

        # Overlays
        txt_bpm = f"BPM: {s.last_bpm:.1f}" if s.last_bpm else "BPM: --"
        cv2.putText(img, txt_bpm, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(img, f"FPS: {s.fps:.1f}", (w - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- Start WebRTC ----------
if not WEBRTC_AVAILABLE:
    st.stop()

webrtc_ctx = webrtc_streamer(
    key="ppg-webrtc",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": {"facingMode": {"ideal": "environment"}},  # leve p/ iOS
        "audio": False,
    },
    rtc_configuration=rtc_configuration,
    video_processor_factory=PPGProcessor,
    async_processing=True,
    # iOS precisa disso para não travar em "tap to play"
    video_html_attrs={"playsinline": True, "autoplay": True, "muted": True},
    # gesto do usuário (toggle da sidebar)
    desired_playing_state=st.session_state.get("camera_on", True),
)

# ---------- Flash automático ----------
# - Android/Chrome: tenta ligar torch no mesmo track
# - iOS/Safari: não expõe torch -> liga "tela branca" (flood light)
components.html(
    """
    <div id="screenTorch" class="screen-torch"><span>💡 Iluminação de tela ativa</span></div>
    <script>
    (function(){
      const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
      let tries = 0;

      function getActiveVideo(){
        const vids = Array.from(document.querySelectorAll("video"));
        return vids.find(v => v.srcObject && v.srcObject.getVideoTracks().length>0) || null;
      }

      async function enableTorchIfPossible(){
        const v = getActiveVideo();
        if(!v){ if(tries++ < 40) return setTimeout(enableTorchIfPossible, 250); return; }

        const track = v.srcObject.getVideoTracks()[0];
        const caps = track.getCapabilities ? track.getCapabilities() : {};
        const torchSupported = !!caps.torch;

        if(torchSupported){
          try{
            await track.applyConstraints({ advanced: [{ torch: true }] });
          }catch(e){
            console.log("Falha ao ligar torch:", e);
          }
        }else{
          // iOS fallback: tela branca como "lanterna" de tela
          if(isIOS){
            const div = document.getElementById("screenTorch");
            if(div) div.style.display = "flex";
            // Ao tocar na tela branca, desliga
            if(div) div.addEventListener("click", ()=>{ div.style.display="none"; });
          }
        }
      }

      // tenta automát., várias vezes até o vídeo estar pronto
      enableTorchIfPossible();
    })();
    </script>
    """,
    height=0,
)

# ---------- Layout principal ----------
col_left, col_right = st.columns([2, 1])

with col_right:
    s = st.session_state.ppg_state
    st.markdown("### Métricas")
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("BPM", f"{s.last_bpm:.1f}" if s.last_bpm else "--")
    st.metric("FPS", f"{s.fps:.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Salvar medição")
    nome = st.text_input("Nome", "")
    if st.button("Salvar"):
        if s.bpm_fixed:
            now = time.localtime()
            row = {
                "Nome": nome if nome else "Sem nome",
                "BPM": round(s.bpm_fixed, 2),
                "Data": time.strftime("%Y-%m-%d", now),
                "Hora": time.strftime("%H:%M:%S", now),
            }
            st.session_state.history = pd.concat(
                [st.session_state.history, pd.DataFrame([row])], ignore_index=True
            )
            st.success(f"Medição salva: {row['Nome']} - {row['BPM']} BPM")
        else:
            st.warning("Aguarde estabilizar (~3 estimativas) para fixar o BPM.")

    st.markdown("---")
    st.markdown("### Histórico")
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history, hide_index=True, use_container_width=True)
        csv = st.session_state.history.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV", data=csv, file_name="historico_bpm.csv", mime="text/csv")
    else:
        st.info("Nenhuma medição salva ainda.")

with col_left:
    st.markdown("### Sinal PPG (tempo real)")
    s = st.session_state.ppg_state
    if s.last_signal_ready:
        x = np.arange(len(s.last_filtered))
        y = s.last_filtered
        peaks = s.last_peaks
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="PPG", line=dict(width=2)))
        if peaks.size > 0:
            fig.add_trace(go.Scatter(x=x[peaks], y=y[peaks], mode="markers", name="Picos"))
        fig.update_layout(
            height=360,
            margin=dict(l=30, r=30, t=30, b=30),
            xaxis_title="Frames",
            yaxis_title="Amplitude (normalizada)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            '<div class="status">Posicione o dedo sobre a câmera e aguarde o sinal estabilizar.</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption(
    "Compatível com Android (Chrome) e iOS (Safari). Em redes móveis/firewalls, o app usa TURN automático (Metered). "
    "No iOS o torch não é exposto via WebRTC: usamos tela branca como fallback de iluminação."
)
