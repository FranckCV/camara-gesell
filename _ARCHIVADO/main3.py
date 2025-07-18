import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
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

# ========== Manejo de cámaras ==========
try:
    with open("camaras.json", "r") as f:
        camera_sources = json.load(f)
except FileNotFoundError:
    camera_sources = {"cam1": 0, "cam2": "http://192.168.0.101:8080/video"}

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

        # Simulación de detección de lágrimas: revisa si hay puntos cerca del ojo (68 landmarks)
        # Puedes refinar si deseas usando regiones específicas
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

    cap = cv2.VideoCapture(camera_sources[cam_name] if not str(camera_sources[cam_name]).isdigit() else int(camera_sources[cam_name]))
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

def add_camera():
    name = simpledialog.askstring("Agregar cámara", "Nombre de la cámara:")
    url = simpledialog.askstring("Agregar cámara", "URL o índice de cámara:")
    if name and url:
        camera_sources[name] = url if not url.isdigit() else int(url)
        with open("camaras.json", "w") as f:
            json.dump(camera_sources, f, indent=4)
        messagebox.showinfo("Cámara añadida", f"Cámara '{name}' añadida correctamente.\nReinicia para verla en la interfaz.")

# ========== Interfaz Tkinter ==========

root = tk.Tk()
root.title("Cámara Gesell con Reconocimiento de Emociones y Grabación")

frame1 = tk.Label(root)
frame1.grid(row=0, column=0, padx=10, pady=10)

frame2 = tk.Label(root)
frame2.grid(row=0, column=1, padx=10, pady=10)

btn_rec1 = tk.Button(root, text="Grabar / Detener Cam 1", command=lambda: toggle_recording("cam1"))
btn_rec1.grid(row=1, column=0, pady=5)

btn_rec2 = tk.Button(root, text="Grabar / Detener Cam 2", command=lambda: toggle_recording("cam2"))
btn_rec2.grid(row=1, column=1, pady=5)

btn_add = tk.Button(root, text="Agregar Cámara IP", command=add_camera, bg="lightgreen")
btn_add.grid(row=2, column=0, columnspan=2, pady=5)

btn_exit = tk.Button(root, text="Salir", command=root.destroy, bg="red", fg="white")
btn_exit.grid(row=3, column=0, columnspan=2, pady=5)

# Iniciar threads de cada cámara
t1 = threading.Thread(target=update_frame, args=("cam1", frame1), daemon=True)
t1.start()

t2 = threading.Thread(target=update_frame, args=("cam2", frame2), daemon=True)
t2.start()

root.mainloop()
