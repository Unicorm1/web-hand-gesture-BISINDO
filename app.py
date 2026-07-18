import cv2
import numpy as np
import pickle
import av
import streamlit as st
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tensorflow.keras.models import load_model
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Penerjemah BISINDO", page_icon="🤟")
st.title("🤟 Sistem Penerjemah BISINDO Real-Time")
st.markdown("Aplikasi berbasis Web ini mendeteksi gestur tangan dan menerjemahkannya ke dalam alfabet BISINDO secara real-time.")

# Muat Model AI (Menggunakan cache agar tidak membebani server berulang kali)
@st.cache_resource
def load_ai_model():
    model = load_model('model_bisindo.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_ai_model()

# Kelas Pemroses Video untuk Server Cloud
class BisindoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.buffer_prediksi = deque(maxlen=10)
        self.tebakan_final = "Menunggu Isyarat..."

    def recv(self, frame):
        # Konversi frame dari browser ke format OpenCV
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(image_rgb)
        tebakan_sementara = "Tidak yakin"

        if results.multi_hand_landmarks:
            row_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                pusat_x = hand_landmarks.landmark[0].x
                pusat_y = hand_landmarks.landmark[0].y
                pusat_z = hand_landmarks.landmark[0].z
                
                for landmark in hand_landmarks.landmark:
                    row_data.extend([
                        landmark.x - pusat_x,
                        landmark.y - pusat_y,
                        landmark.z - pusat_z
                    ])
            
            while len(row_data) < 126:
                row_data.append(0.0)
            row_data = row_data[:126]
            
            input_data = np.array([row_data]) 
            prediksi = model.predict(input_data, verbose=0)
            indeks_kelas = np.argmax(prediksi)
            akurasi_prediksi = np.max(prediksi)
            
            if akurasi_prediksi > 0.75:
                tebakan_sementara = encoder.inverse_transform([indeks_kelas])[0]

        # Logika Buffer Stabilizer untuk mengurangi flicker prediksi
        if tebakan_sementara != "Tidak yakin":
            self.buffer_prediksi.append(tebakan_sementara)
        
        if len(self.buffer_prediksi) == 10:
            huruf_terbanyak = Counter(self.buffer_prediksi).most_common(1)[0][0]
            self.tebakan_final = huruf_terbanyak

        # Visualisasi Hasil di Layar
        cv2.rectangle(image, (0, 0), (300, 60), (245, 117, 16), -1)
        cv2.putText(image, f'Prediksi: {self.tebakan_final}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Kembalikan frame yang sudah digambar ke browser pengguna
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Modul Kamera WebRTC (Menggunakan server STUN Google agar bisa diakses publik)
webrtc_streamer(
    key="bisindo-kamera", 
    video_processor_factory=BisindoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
