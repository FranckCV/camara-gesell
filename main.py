import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
import os
import json
from datetime import datetime
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import dlib
from imutils import face_utils
import mediapipe as mp
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import joblib

# Configurar el tema de customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

modelo_posturas = joblib.load("modelo_posturas.pkl")

# ================= CONFIGURACIÓN ===================
fer = HSEmotionRecognizer(model_name='enet_b2_7')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

CAM01 = 'cam1'
CAM02 = 'cam3'

EMOCIONES = {
    "neutral": "Neutral",
    "happiness": "Feliz",
    "sadness": "Triste",
    "surprise": "Sorpresa",
    "fear": "Miedo",
    "disgust": "Disgusto",
    "anger": "Enojo",
    "contempt": "Desdén",
    "valence_positive": "Valencia Positiva",
    "valence_negative": "Valencia Negativa",
    "engagement": "Comprometido",
    "sleepy": "Somnoliento",
    "excited": "Emocionado",
    "bored": "Aburrido",
    "anxious": "Ansioso",
    "tired": "Cansado"
}

try:
    with open("camaras.json", "r") as f:
        camera_sources = json.load(f)
except FileNotFoundError:
    camera_sources = {CAM01: 0, CAM02: 0}

captures = {}
recording_flags = {}
recorders = {}
emotion_counts = {CAM01: {v: 0 for v in EMOCIONES.values()},
                  CAM02: {}}
posture_counts = {CAM02: {}}

# ================= FUNCIONES PRINCIPALES ===================
def detectar_emociones(frame, cam_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray)
    for rostro in rostros:
        x, y, w, h = rostro.left(), rostro.top(), rostro.width(), rostro.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        shape = predictor(gray, rostro)
        shape_np = face_utils.shape_to_np(shape)
        for (sx, sy) in shape_np:
            cv2.circle(frame, (sx, sy), 1, (0, 255, 0), -1)
        rostro_recortado = frame[y:y + h, x:x + w]
        if rostro_recortado.size != 0:
            try:
                emociones, scores = fer.predict_emotions(rostro_recortado, logits=False)
                clave = emociones.strip().lower()
                emocion_nombre = EMOCIONES.get(clave, emociones)
                porcentaje = scores[0] * 100
                cv2.putText(frame, f"{emocion_nombre}: {porcentaje:.1f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if emocion_nombre in emotion_counts[cam_name]:
                    emotion_counts[cam_name][emocion_nombre] += 1
            except:
                pass
    return frame

def detectar_postura(frame):
    estado = "Desconocido"
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extraer landmarks
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        flattened_landmarks = [coord for landmark in landmarks for coord in landmark]
        
        # Normalizar landmarks
        modelo = joblib.load("modelo_posturas.pkl")
        scaler = modelo["scaler"]
        knn = modelo["model"]
        labels = modelo["labels"]
        normalized_landmarks = scaler.transform([flattened_landmarks])
        
        # Predecir postura
        distances, indices = knn.kneighbors(normalized_landmarks)
        if distances[0][0] < 0.5:  # Umbral para considerar una postura válida
            estado = labels[indices[0][0]]
        else:
            estado = "Desconocido"
        
        # Mostrar el estado en el frame
        cv2.putText(frame, f"Postura: {estado}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame

def actualizar_grafica():
    # Configurar estilo de matplotlib para tema oscuro
    plt.style.use('dark_background')
    
    ax.clear()
    for idx, cam in enumerate([CAM01, CAM02]):
        top_emotions = sorted(emotion_counts[cam].items(), key=lambda x: x[1], reverse=True)[:5]
        if top_emotions:
            emociones, cantidades = zip(*top_emotions)
            posiciones = np.arange(len(emociones)) + idx * (len(emociones) + 1)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            ax.bar(posiciones, cantidades, label=f"{cam} - Emociones", color=colors[:len(emociones)])
            ax.set_xticks(posiciones)
            ax.set_xticklabels(emociones, rotation=45, fontsize=7)
    ax.set_ylabel("Frecuencia", fontsize=8)
    ax.set_title("Emociones Detectadas", fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    canvas.draw()

    ax_posturas.clear()
    top_postures = sorted(posture_counts[CAM02].items(), key=lambda x: x[1], reverse=True)[:5]
    if top_postures:
        posturas, cantidades = zip(*top_postures)
        posiciones = np.arange(len(posturas))
        ax_posturas.bar(posiciones, cantidades, color=["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"][:len(posturas)], label="Posturas")
        ax_posturas.set_xticks(posiciones)
        ax_posturas.set_xticklabels(posturas, rotation=45, fontsize=7)
    ax_posturas.set_ylabel("Frecuencia", fontsize=8)
    ax_posturas.set_title("Posturas Detectadas", fontsize=9, fontweight='bold')
    ax_posturas.legend(fontsize=7)
    ax_posturas.grid(True, alpha=0.3)
    ax_posturas.tick_params(labelsize=7)
    canvas_posturas.draw()

    root.after(1500, actualizar_grafica)

def toggle_recording(cam_name):
    if not recording_flags.get(cam_name, False):
        os.makedirs(f"grabaciones/{cam_name}", exist_ok=True)
        path = f"grabaciones/{cam_name}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        recorders[cam_name] = cv2.VideoWriter(path, fourcc, 20, (400, 300))
        recording_flags[cam_name] = True
        
        # Actualizar texto del botón
        if cam_name == CAM01:
            btn_rec1.configure(text="🔴 REC", fg_color="#ff4757")
        else:
            btn_rec2.configure(text="🔴 REC", fg_color="#ff4757")
        
        messagebox.showinfo("✅ Grabación", f"Grabando en {path}")
    else:
        recording_flags[cam_name] = False
        recorders[cam_name].release()
        
        # Restaurar texto del botón
        if cam_name == CAM01:
            btn_rec1.configure(text="📹 REC", fg_color="#3498db")
        else:
            btn_rec2.configure(text="📹 REC", fg_color="#3498db")
        
        messagebox.showinfo("⏹️ Grabación", f"Grabación detenida para {cam_name}")

def iniciar_camara(cam_name, label):
    def loop():
        cap = cv2.VideoCapture(camera_sources[cam_name])
        captures[cam_name] = cap
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (400, 300))
                
                if cam_name == CAM01:
                    frame = detectar_emociones(frame, cam_name)
                
                if cam_name == CAM02:
                    frame = detectar_postura(frame)
                
                if recording_flags.get(cam_name, False):
                    recorders[cam_name].write(frame)
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(img)
                label.imgtk = imgtk
                label.configure(image=imgtk)
            
            if not root.winfo_exists():
                break
    threading.Thread(target=loop, daemon=True).start()

# ================= GUI MODERNA COMPACTA ===================
root = ctk.CTk()
root.title("🎥 Cámara Gesell - Sistema de Análisis")
root.geometry("1200x800")

# Header compacto
header_frame = ctk.CTkFrame(root, height=50, corner_radius=10)
header_frame.pack(fill="x", padx=10, pady=5)

title_label = ctk.CTkLabel(
    header_frame, 
    text="🎥 CÁMARA GESELL - ANÁLISIS EMOCIONAL Y POSTURAL", 
    font=ctk.CTkFont(size=16, weight="bold")
)
title_label.pack(pady=10)

# Contenedor principal
main_container = ctk.CTkFrame(root, corner_radius=10)
main_container.pack(fill="both", expand=True, padx=10, pady=5)

# Sección superior - Cámaras
cameras_section = ctk.CTkFrame(main_container, corner_radius=8)
cameras_section.pack(fill="x", padx=8, pady=5)

# Cámara 1
cam1_container = ctk.CTkFrame(cameras_section, corner_radius=8)
cam1_container.pack(side="left", padx=5, pady=5, fill="both", expand=True)

# Header cámara 1 con botón
cam1_header = ctk.CTkFrame(cam1_container, height=35, corner_radius=5)
cam1_header.pack(fill="x", padx=5, pady=2)

cam1_title = ctk.CTkLabel(cam1_header, text="📷 CÁMARA 1 - EMOCIONES", 
                         font=ctk.CTkFont(size=11, weight="bold"))
cam1_title.pack(side="left", padx=10, pady=5)

btn_rec1 = ctk.CTkButton(
    cam1_header, 
    text="📹 REC", 
    command=lambda: toggle_recording(CAM01),
    font=ctk.CTkFont(size=10, weight="bold"),
    width=60,
    height=25,
    corner_radius=5,
    fg_color="#3498db",
    hover_color="#2980b9"
)
btn_rec1.pack(side="right", padx=10, pady=5)

frame1 = ctk.CTkLabel(cam1_container, text="", width=400, height=300, corner_radius=5)
frame1.pack(padx=5, pady=2)

# Cámara 2
cam2_container = ctk.CTkFrame(cameras_section, corner_radius=8)
cam2_container.pack(side="right", padx=5, pady=5, fill="both", expand=True)

# Header cámara 2 con botón
cam2_header = ctk.CTkFrame(cam2_container, height=35, corner_radius=5)
cam2_header.pack(fill="x", padx=5, pady=2)

cam2_title = ctk.CTkLabel(cam2_header, text="🧍 CÁMARA 2 - POSTURAS", 
                         font=ctk.CTkFont(size=11, weight="bold"))
cam2_title.pack(side="left", padx=10, pady=5)

btn_rec2 = ctk.CTkButton(
    cam2_header, 
    text="📹 REC", 
    command=lambda: toggle_recording(CAM02),
    font=ctk.CTkFont(size=10, weight="bold"),
    width=60,
    height=25,
    corner_radius=5,
    fg_color="#3498db",
    hover_color="#2980b9"
)
btn_rec2.pack(side="right", padx=10, pady=5)

frame2 = ctk.CTkLabel(cam2_container, text="", width=400, height=300, corner_radius=5)
frame2.pack(padx=5, pady=2)

# Sección inferior - Gráficas
graphs_section = ctk.CTkFrame(main_container, corner_radius=8)
graphs_section.pack(fill="both", expand=True, padx=8, pady=5)

# Gráfica de emociones
emotions_container = ctk.CTkFrame(graphs_section, corner_radius=8)
emotions_container.pack(side="left", padx=5, pady=5, fill="both", expand=True)

emotions_title = ctk.CTkLabel(emotions_container, text="📊 ESTADÍSTICAS EMOCIONALES", 
                             font=ctk.CTkFont(size=10, weight="bold"))
emotions_title.pack(pady=2)

fig, ax = plt.subplots(figsize=(5, 2.5))
fig.patch.set_facecolor('#212121')
fig.tight_layout(pad=1.0)
canvas = FigureCanvasTkAgg(fig, master=emotions_container)
canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=2)

# Gráfica de posturas
postures_container = ctk.CTkFrame(graphs_section, corner_radius=8)
postures_container.pack(side="right", padx=5, pady=5, fill="both", expand=True)

postures_title = ctk.CTkLabel(postures_container, text="🏃 ESTADÍSTICAS POSTURALES", 
                             font=ctk.CTkFont(size=10, weight="bold"))
postures_title.pack(pady=2)

fig_posturas, ax_posturas = plt.subplots(figsize=(5, 2.5))
fig_posturas.patch.set_facecolor('#212121')
fig_posturas.tight_layout(pad=1.0)
canvas_posturas = FigureCanvasTkAgg(fig_posturas, master=postures_container)
canvas_posturas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=2)

# Footer con botón de salida y status
footer_frame = ctk.CTkFrame(root, height=40, corner_radius=10)
footer_frame.pack(fill="x", padx=10, pady=5)

btn_exit = ctk.CTkButton(
    footer_frame, 
    text="❌ SALIR", 
    command=root.destroy, 
    font=ctk.CTkFont(size=12, weight="bold"),
    width=100,
    height=30,
    corner_radius=8,
    fg_color="#e74c3c",
    hover_color="#c0392b"
)
btn_exit.pack(side="right", padx=10, pady=5)

status_label = ctk.CTkLabel(
    footer_frame, 
    text="🟢 Sistema activo - Monitoreando en tiempo real", 
    font=ctk.CTkFont(size=10)
)
status_label.pack(side="left", padx=10, pady=5)

iniciar_camara(CAM01, frame1)
iniciar_camara(CAM02, frame2)
actualizar_grafica()

root.mainloop()
