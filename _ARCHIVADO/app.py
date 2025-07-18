import cv2
import dlib
import numpy as np
import threading
from imutils import face_utils
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer 

# Inicializar modelo de emociones
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

# Inicializar dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Mapeo de etiquetas a nombres completos
EMOCIONES = {
    "A": "Enojo",
    "D": "Disgusto",
    "F": "Miedo",
    "H": "Feliz",
    "S": "Triste",
    "N": "Neutral",
    "C": "Desdén",
    "U": "Sorpresa"
}

# Variables globales
camaras_fuentes = []
frames_actuales = []
bloqueo = threading.Lock()
grabando = False
video_writer = None

def detectar_emocion(face_img):
    try:
        emociones, scores = fer.predict_emotions(face_img, logits=False)
        emocion_letra = emociones[0]
        emocion = EMOCIONES.get(emocion_letra, emocion_letra)
        return emocion, float(scores[0])
    except Exception as e:
        return "Error", 0.0

def procesar_camara(idx, fuente):
    cap = cv2.VideoCapture(fuente if not fuente.isdigit() else int(fuente))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara: {fuente}")
        return

    global frames_actuales
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = detector(gray)

        for rostro in rostros:
            shape = predictor(gray, rostro)
            shape_np = face_utils.shape_to_np(shape)

            for (x, y) in shape_np:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            x, y, w, h = rostro.left(), rostro.top(), rostro.width(), rostro.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            rostro_recortado = frame[y:y+h, x:x+w]
            if rostro_recortado.size != 0:
                emocion, score = detectar_emocion(rostro_recortado)
                cv2.putText(frame, f"{emocion} ({score:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        with bloqueo:
            frames_actuales[idx] = frame.copy()

    cap.release()

def unificar_frames():
    with bloqueo:
        frames_validos = [cv2.resize(f, (320, 240)) for f in frames_actuales if f is not None]
        if not frames_validos:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        filas = []
        for i in range(0, len(frames_validos), 2):
            fila = np.hstack(frames_validos[i:i+2])
            filas.append(fila)

        return np.vstack(filas) if filas else np.zeros((240, 320, 3), dtype=np.uint8)

def actualizar_video():
    global grabando, video_writer

    frame_unificado = unificar_frames()
    frame_rgb = cv2.cvtColor(frame_unificado, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=im)

    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor=NW, image=imgtk)

    if grabando and video_writer is not None:
        video_writer.write(frame_unificado)

    ventana.after(30, actualizar_video)

def agregar_camara():
    fuente = entrada.get()
    if fuente == "":
        return

    idx = len(camaras_fuentes)
    camaras_fuentes.append(fuente)
    frames_actuales.append(None)
    lista.insert(END, f"Cámara {idx}: {fuente}")

    hilo = threading.Thread(target=procesar_camara, args=(idx, fuente))
    hilo.daemon = True
    hilo.start()

def toggle_grabacion():
    global grabando, video_writer

    if not grabando:
        frame_test = unificar_frames()
        height, width = frame_test.shape[:2]

        ruta = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            title="Guardar grabación como..."
        )
        if not ruta:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(ruta, fourcc, 20.0, (width, height))

        grabando = True
        btn_grabar.config(text="Detener Grabación")
        print(f"[INFO] Grabando en: {ruta} ({width}x{height})")
    else:
        grabando = False
        if video_writer:
            video_writer.release()
            video_writer = None
        btn_grabar.config(text="Iniciar Grabación")
        print("[INFO] Grabación detenida")

# Interfaz gráfica
ventana = Tk()
ventana.title("Reconocimiento de Emociones con HSEmotionONNX")
ventana.geometry("700x600")

entrada = StringVar()
Label(ventana, text="Índice/IP de cámara:").pack(pady=5)
Entry(ventana, textvariable=entrada, width=50).pack()
Button(ventana, text="Agregar Cámara", command=agregar_camara).pack(pady=5)

Label(ventana, text="Cámaras activas:").pack()
frame_listado = Frame(ventana)
scroll = Scrollbar(frame_listado)
scroll.pack(side=RIGHT, fill=Y)
lista = Listbox(frame_listado, yscrollcommand=scroll.set, width=60)
lista.pack()
scroll.config(command=lista.yview)
frame_listado.pack(pady=5)

btn_grabar = Button(ventana, text="Iniciar Grabación", command=toggle_grabacion)
btn_grabar.pack(pady=10)

canvas = Canvas(ventana, width=640, height=480)
canvas.pack()

actualizar_video()
ventana.mainloop()
