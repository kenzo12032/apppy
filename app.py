import streamlit as st
import cv2
import numpy as np
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq


# Configuração Streamlit
st.set_page_config(page_title="PPG Real-Time + Flash", layout="centered")
st.title("💡 Monitor Cardíaco em Tempo Real")


# CSS (coloquei para tentar tirar os botões, mas falhei miseravelmente)
st.markdown(
    """
    <style>
    /* Esconde a linha "Running" / "Stopped" */
    div[data-testid="stWebRTCStatus"] { 
        display: none !important; 
    }

    /* Nova regra para esconder botões e seletor */
    div[data-key="camera_flash"] {
        .toolbar { display: none !important; }
        button[title="Start"], button[title="Stop"] { display: none !important; }
        select[title="Select device"] { display: none !important; }
        label { display: none !important; }
    }

    /* Rotação do vídeo */
    video {
        transform: rotate(90deg);
        object-fit: cover;
        width: 100%;
        height: 80vh;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Botão Ligar/Desligar Câmera
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

st.toggle("Ligar/Desligar Câmera", key="camera_on")


# Processador de vídeo
class MeuProcessadorDeVideo(VideoProcessorBase):
    def __init__(self):
        self.buffer_R = []
        self.buffer_G = []
        self.buffer_B = []
        self.buffer_size = 150
        self.lock = threading.Lock()
        self.bpm = 0.0
        self.fps = 0.0
        self.contador_frames = 0
        self.tempo_espera = time.time()
        self.alpha_suavizacao = 0.1

    def recv(self, frame):
        with self.lock:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            # ROI (Região de Interesse)
            roi_size = min(h, w) // 3
            x_i = (w - roi_size) // 2
            y_i = (h - roi_size) // 2
            roi = img[y_i : y_i + roi_size, x_i : x_i + roi_size]

            # Lógica de detecção de dedo
            if roi.size > 0:
                mean_bgr = np.mean(roi, axis=(0, 1))
                mean_red_value = mean_bgr[2]  # Canal Vermelho
            else:
                mean_red_value = 0

            RED_THRESHOLD = 220
            if mean_red_value > RED_THRESHOLD:
                # Dedo detectado
                self.buffer_B.append(mean_bgr[0])
                self.buffer_G.append(mean_bgr[1])
                self.buffer_R.append(mean_bgr[2])

                # Limita o tamanho do buffer
                if len(self.buffer_R) > self.buffer_size:
                    self.buffer_R.pop(0)
                    self.buffer_G.pop(0)
                    self.buffer_B.pop(0)

                self.contador_frames += 1
                if self.contador_frames % 10 == 0:
                    bpm_bruto = self.calcula_bpm()
                    if bpm_bruto > 40:
                        self.bpm = (
                            bpm_bruto
                            if self.bpm == 0
                            else (
                                bpm_bruto * self.alpha_suavizacao
                                + self.bpm * (1.0 - self.alpha_suavizacao)
                            )
                        )

                # Calcula FPS
                agora = time.time()
                delta_t = agora - self.tempo_espera
                if delta_t > 0:
                    self.fps = 10 / delta_t
                self.tempo_espera = agora

                # Mostra BPM
                cv2.putText(
                    img,
                    f"BPM: {self.bpm:.1f}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
            else:

                # Dedo não detectado
                if len(self.buffer_R) > 0:
                    self.buffer_R.clear()
                    self.buffer_G.clear()
                    self.buffer_B.clear()
                    self.bpm = 0.0
                    self.contador_frames = 0
                    self.tempo_espera = time.time()

                # Mostra instrução
                cv2.putText(
                    img,
                    "Posicione o dedo",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            # Desenha ROI + FPS
            cv2.rectangle(
                img, (x_i, y_i), (x_i + roi_size, y_i + roi_size), (0, 255, 0), 2
            )
            cv2.putText(
                img,
                f"FPS: {self.fps:.1f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Função de cálculo de BPM (Filtros e FFT)
    def calcula_bpm(self):
        if len(self.buffer_R) < 30:
            return 0

        Gnorm = np.array(self.buffer_G) / (np.mean(self.buffer_G) + 1e-9)
        sinal_ac = -(Gnorm - np.mean(Gnorm))

        try:
            fs = self.fps
            if fs < 1.0:
                return 0

            nyq = 0.5 * fs
            b, a = butter(2, [0.7 / nyq, 3.5 / nyq], btype="band")
            filtered = filtfilt(b, a, sinal_ac)

            N = len(filtered)
            yf = np.abs(fft(filtered))
            xf = fftfreq(N, d=1 / fs)
            bpm_vals = xf * 60
            idx = (bpm_vals > 40) & (bpm_vals < 200)

            if np.sum(idx) == 0:
                return 0

            peak_idx = np.argmax(yf[idx])
            return bpm_vals[idx][peak_idx]

        except:
            return 0


# Configuração WebRTC (Flash ligado)
video_constraints = {
    "video": {
        "facingMode": {"ideal": "environment"},
        "width": {"ideal": 1920},
        "height": {"ideal": 1080},
        "frameRate": {"ideal": 30},
        "torch": True,
    }
}


# Inicia o Streamer (CONTROLADO PELO BOTÃO TOGGLE)
ctx = webrtc_streamer(
    key="camera_flash",
    video_processor_factory=MeuProcessadorDeVideo,
    media_stream_constraints=video_constraints,
    async_processing=True,
    desired_playing_state=st.session_state.camera_on,
)
