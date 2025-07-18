from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import numpy as np
import dlib
from imutils import face_utils

app = Flask(__name__)

fer = HSEmotionRecognizer(model_name='enet_b2_7')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

camera_sources = {
    "cam1": 0,
    "cam2": "http://192.168.69.95:8080/video"
}

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

def procesar_camara(source):
    cap = cv2.VideoCapture(source if isinstance(source, int) else source)
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    while True:
        if not cap.isOpened():
            cap.open(source)
        success, frame = cap.read()
        if not success:
            continue

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
                except Exception:
                    cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<cam_name>')
def video(cam_name):
    source = camera_sources.get(cam_name, 0)
    return Response(procesar_camara(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera/<cam_name>', methods=['POST'])
def set_camera(cam_name):
    url = request.form.get('source')
    camera_sources[cam_name] = url if not url.isdigit() else int(url)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)