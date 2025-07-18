from flask import Flask, render_template, Response
import cv2
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import numpy as np
import dlib
from imutils import face_utils

app = Flask(__name__)

# Inicializar modelo de emociones
fer = HSEmotionRecognizer(model_name='enet_b2_7')

# Inicializar dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Cámara
cap = cv2.VideoCapture(0)

# Mapeo de etiquetas a nombres en español
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

def generar_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

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

                    texto = f"{emocion_nombre}: {porcentaje:.2f}%"
                    cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
