import cv2
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import numpy as np
import dlib
from imutils import face_utils

# Inicializar modelo
fer = HSEmotionRecognizer(model_name='enet_b2_7')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Diccionario de emociones
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

# Configuración de cámaras
camera_sources = [
    0,  # laptop
    "http://192.168.69.95:8080/video"  # IP Webcam
]

caps = [cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src, int) else 0) for src in camera_sources]

# Ajustar resolución (si soporta)
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Configuración de grabación
record = True
fps = 15
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('grabacion_combinada.mp4', fourcc, fps, (640, 240))

def procesar_frame(frame):
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
                texto = f"{emocion_nombre}: {porcentaje:.1f}%"
                cv2.putText(frame, texto, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except Exception as e:
                cv2.putText(frame, "Error", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, "No signal", (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame = cv2.resize(frame, (320, 240))
            frame = procesar_frame(frame)
        frames.append(frame)

    combined = np.hstack(frames)

    cv2.imshow("Cámara Gesell - Emociones", combined)

    if record:
        out.write(combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
if record:
    out.release()
cv2.destroyAllWindows()
