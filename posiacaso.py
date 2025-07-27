import tkinter as tk
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
                  CAM02: {v: 0 for v in EMOCIONES.values()}}

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
        l_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        if l_wrist.y < l_shoulder.y and r_wrist.y < r_shoulder.y:
            estado = "Exaltado"
        elif nose.y > (l_shoulder.y + r_shoulder.y) / 2:
            estado = "Decaido"
        else:
            estado = "Normal"
        cv2.putText(frame, f"Postura: {estado}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame

def actualizar_grafica():
    ax.clear()
    for idx, cam in enumerate([CAM01, CAM02]):
        top_emotions = sorted(emotion_counts[cam].items(), key=lambda x: x[1], reverse=True)[:5]
        if top_emotions:
            emociones, cantidades = zip(*top_emotions)
            posiciones = np.arange(len(emociones)) + idx * (len(emociones) + 1)
            ax.bar(posiciones, cantidades, label=cam)
            ax.set_xticks(posiciones)
            ax.set_xticklabels(emociones, rotation=20)
    ax.set_ylabel("Frecuencia")
    ax.set_title("Top emociones detectadas por cámara")
    ax.legend()
    canvas.draw()
    root.after(1500, actualizar_grafica)

def toggle_recording(cam_name):
    if not recording_flags.get(cam_name, False):
        os.makedirs(f"grabaciones/{cam_name}", exist_ok=True)
        path = f"grabaciones/{cam_name}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        recorders[cam_name] = cv2.VideoWriter(path, fourcc, 20, (480, 360))
        recording_flags[cam_name] = True
        messagebox.showinfo("Grabación", f"Grabando en {path}")
    else:
        recording_flags[cam_name] = False
        recorders[cam_name].release()
        messagebox.showinfo("Grabación", f"Grabación detenida para {cam_name}")

def iniciar_camara(cam_name, label):
    def loop():
        cap = cv2.VideoCapture(camera_sources[cam_name])
        captures[cam_name] = cap
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (480, 360))
                
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
                label.config(image=imgtk)
            
            if not root.winfo_exists():
                break
    threading.Thread(target=loop, daemon=True).start()

# ================= GUI ===================
root = tk.Tk()
root.title("Cámara Gesell - Emociones, Postura y Gráficas en vivo")

frame1 = tk.Label(root)
frame1.grid(row=0, column=0, padx=10, pady=10)
frame2 = tk.Label(root)
frame2.grid(row=0, column=1, padx=10, pady=10)

btn_rec1 = tk.Button(root, text=f"Grabar/Detener {CAM01}", command=lambda: toggle_recording(CAM01))
btn_rec1.grid(row=1, column=0, pady=5)
btn_rec2 = tk.Button(root, text=f"Grabar/Detener {CAM02}", command=lambda: toggle_recording(CAM02))
btn_rec2.grid(row=1, column=1, pady=5)

fig, ax = plt.subplots(figsize=(6, 3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, pady=10)

btn_exit = tk.Button(root, text="Salir", command=root.destroy, bg="red", fg="white")
btn_exit.grid(row=3, column=0, columnspan=2, pady=10)

iniciar_camara(CAM01, frame1)
iniciar_camara(CAM02, frame2)
actualizar_grafica()

root.mainloop()
