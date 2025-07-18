import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
import json
from datetime import datetime
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import numpy as np
import dlib
from imutils import face_utils

# ========== Configuración de emoción ==========
fer = HSEmotionRecognizer(model_name='enet_b2_7')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

CAM01 = 'cam2'
CAM02 = "cam3"

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

# ========== Configuración de cámaras ==========
try:
    with open("camaras.json", "r") as f:
        camera_sources = json.load(f)
except FileNotFoundError:
    camera_sources = {"cam1": 0, "cam2": 0}

captures = {}
recording_flags = {}
recorders = {}

# ========== Funciones de cámaras ==========
def detectar_y_procesar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray)

    for rostro in rostros:
        x, y, w, h = rostro.left(), rostro.top(), rostro.width(), rostro.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        shape = predictor(gray, rostro)
        shape_np = face_utils.shape_to_np(shape)
        for (sx, sy) in shape_np:
            cv2.circle(frame, (sx, sy), 1, (0, 255, 0), -1)

        # Simulación de detección de lágrimas: intensidad baja en región ocular
        left_eye = shape_np[36:42]
        right_eye = shape_np[42:48]
        eye_region = np.concatenate([left_eye, right_eye])
        if np.mean(gray[eye_region[:, 1], eye_region[:, 0]]) < 50:
            cv2.putText(frame, "Lagrimas Detectadas", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        rostro_recortado = frame[y:y + h, x:x + w]
        if rostro_recortado.size != 0:
            try:
                emociones, scores = fer.predict_emotions(rostro_recortado, logits=False)
                clave = emociones.strip().lower()
                emocion_nombre = EMOCIONES.get(clave, emociones)
                porcentaje = scores[0] * 100
                texto = f"{emocion_nombre}: {porcentaje:.2f}%"
                cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception:
                pass
    return frame

def update_frame(cam_name, label):
    global captures, recording_flags, recorders
    source = camera_sources[cam_name]
    cap = cv2.VideoCapture(source if not str(source).isdigit() else int(source))
    captures[cam_name] = cap
    recording_flags[cam_name] = False
    recorders[cam_name] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (480, 360))
        frame = detectar_y_procesar(frame)

        if recording_flags[cam_name]:
            recorders[cam_name].write(frame)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        time.sleep(0.03)

def toggle_recording(cam_name):
    if not recording_flags[cam_name]:
        os.makedirs(f"grabaciones/{cam_name}", exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"grabaciones/{cam_name}/{now}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        recorders[cam_name] = cv2.VideoWriter(path, fourcc, 20.0, (480, 360))
        recording_flags[cam_name] = True
        messagebox.showinfo("Grabación", f"Iniciada grabación en {path}")
    else:
        recording_flags[cam_name] = False
        recorders[cam_name].release()
        messagebox.showinfo("Grabación", f"Grabación detenida para {cam_name}")

# ========== Interfaz Tkinter ==========
root = tk.Tk()
root.title("Cámara Gesell con Reconocimiento de Emociones y Grabación")

frame1 = tk.Label(root)
frame1.grid(row=0, column=0, padx=10, pady=10)

frame2 = tk.Label(root)
frame2.grid(row=0, column=1, padx=10, pady=10)

btn_rec1 = tk.Button(root, text=f"Grabar / Detener {CAM01}", command=lambda: toggle_recording(CAM01))
btn_rec1.grid(row=1, column=0, pady=5)

btn_rec2 = tk.Button(root, text=f"Grabar / Detener {CAM02}", command=lambda: toggle_recording(CAM02))
btn_rec2.grid(row=1, column=1, pady=5)

btn_exit = tk.Button(root, text="Salir", command=root.destroy, bg="red", fg="white")
btn_exit.grid(row=2, column=0, columnspan=2, pady=10)

# Iniciar threads de cada cámara
t1 = threading.Thread(target=update_frame, args=(CAM01, frame1), daemon=True)
t2 = threading.Thread(target=update_frame, args=(CAM02, frame2), daemon=True)
t1.start()
t2.start()

root.mainloop()
