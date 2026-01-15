import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
import threading
import queue
from collections import Counter
import time

# --- High-Accuracy Gender Vision System ---
class GenderVision:
    def __init__(self):
        # Pre-load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # Warm up DeepFace (Pre-loads model once)
        try:
            DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=['gender'], enforce_detection=False, silent=True)
        except: pass

    def align_face(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda x: x[0])
                le_c = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
                re_c = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
                dy, dx = re_c[1] - le_c[1], re_c[0] - le_c[0]
                angle = np.degrees(np.arctan2(dy, dx))
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
            return img
        except: return img

    def optimize_crop(self, frame, x, y, w, h):
        h_orig, w_orig = frame.shape[:2]
        pad_w, pad_h = int(w * 0.35), int(h * 0.35)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(w_orig, x + w + pad_w), min(h_orig, y + h + pad_h)
        return self.align_face(frame[y1:y2, x1:x2])

# --- AI Worker: Single Persistent Instance ---
def ai_worker(in_q, out_q):
    while True:
        try:
            img = in_q.get(timeout=1)
            if img is None: break
            try:
                res = DeepFace.analyze(img, actions=['gender'], enforce_detection=False, silent=True)
                g_probs = res[0]['gender']
                dominant = res[0]['dominant_gender']
                if g_probs[dominant] > 68:
                    out_q.put({"gender": dominant, "conf": g_probs[dominant]})
            except: pass
        except queue.Empty: continue

# --- Cached Resource Loading ---
@st.cache_resource
def load_vision_system():
    return GenderVision()

# --- Page Setup & Style ---
st.set_page_config(page_title="Gender AI Pro", layout="wide")
st.markdown("""
<style>
    .stApp {background: #0d1117; color: #e1e4e8;}
    h1 {color: #58a6ff; font-weight: 500;}
    .stButton>button {background: #238636; color: white; border-radius: 6px; padding: 0.8rem;}
    [data-testid="stMetricValue"] {color: #7ee787; font-size: 2.5rem;}
    .info-box {background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; margin-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("Real-Time Gender Analysis")

# Persistent State Management
if 'iq' not in st.session_state:
    st.session_state.iq = queue.Queue(maxsize=1)
    st.session_state.oq = queue.Queue(maxsize=1)
if 'worker_thread' not in st.session_state or not st.session_state.worker_thread.is_alive():
    st.session_state.worker_thread = threading.Thread(target=ai_worker, args=(st.session_state.iq, st.session_state.oq), daemon=True)
    st.session_state.worker_thread.start()

if 'g_buffer' not in st.session_state: st.session_state.g_buffer = []
if 'stable_res' not in st.session_state: st.session_state.stable_res = {"label": "Detecting...", "conf": 0}

vision = load_vision_system()

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### 🔌 Control Center")
    scanning = st.toggle("Activate Lens", value=False)
    st.divider()
    st.caption("Deep Engine v2.1 (Optimized)")
    st.caption("Developed by Shabir Ahmad")
    st.markdown("</div>", unsafe_allow_html=True)

with col1:
    v_ph = st.empty()
    m_ph = st.empty()
    
    if scanning:
        cam = cv2.VideoCapture(0)
        try:
            while scanning:
                ret, frame = cam.read()
                if not ret: break
                
                # Fast detection path
                h, w = frame.shape[:2]
                low = cv2.resize(frame, (w//2, h//2))
                gray = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
                objs = vision.face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(60, 60))
                
                if len(objs) > 0:
                    (ox, oy, ow, oh) = max(objs, key=lambda f: f[2]*f[3])
                    x, y, w, h = ox*2, oy*2, ow*2, oh*2
                    
                    # UI Logic
                    label_color = (88, 166, 255) if st.session_state.stable_res['label'] == 'Man' else (255, 126, 187)
                    if st.session_state.stable_res['label'] == 'Detecting...': label_color = (200, 200, 200)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), label_color, 2)
                    
                    # Only queue if worker is ready to avoid "kept loading" lag
                    if st.session_state.iq.empty():
                        st.session_state.iq.put(vision.optimize_crop(frame, x, y, w, h))
                    
                    # Update metrics
                    try:
                        pkg = st.session_state.oq.get_nowait()
                        st.session_state.g_buffer.append(pkg['gender'])
                        if len(st.session_state.g_buffer) > 10: st.session_state.g_buffer.pop(0)
                        
                        vote = Counter(st.session_state.g_buffer).most_common(1)[0][0]
                        st.session_state.stable_res = {"label": vote, "conf": pkg['conf']}
                    except queue.Empty: pass
                    
                    res = st.session_state.stable_res
                    cv2.putText(frame, f"{res['label']} ({int(res['conf'])}%)", (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    
                    with m_ph.container():
                        st.metric("Subject Gender", res['label'])
                        st.progress(float(res['conf'])/100, "Confidence Score")
                else:
                    st.session_state.g_buffer = []
                    m_ph.empty()
                
                v_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.01) # Keep system responsive
                
        finally:
            cam.release()
    else:
        v_ph.info("Lens Offline. Activate to begin Scanning.")
