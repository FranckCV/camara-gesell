import os
import csv
import joblib
import mediapipe as mp
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Crear archivo CSV para guardar datos si no existe
if not os.path.exists("posturas.csv"):
    with open("posturas.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["landmarks", "etiqueta"])  # Columnas: landmarks y etiqueta

def capturar_postura(video_source, etiqueta):
    """
    Captura una postura desde la cámara y guarda los landmarks con la etiqueta proporcionada.
    """
    cap = cv2.VideoCapture(video_source)
    print(f"Presiona 'c' para capturar la postura '{etiqueta}' o 'q' para salir.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(frame, f"Etiqueta: {etiqueta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Captura de Postura", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capturar landmarks
            if results.pose_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                with open("posturas.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([landmarks, etiqueta])
                print(f"Postura '{etiqueta}' capturada y guardada.")
            else:
                print("No se detectaron landmarks. Intenta nuevamente.")
        elif key == ord('q'):  # Salir
            break
    
    cap.release()
    cv2.destroyAllWindows()

def entrenar_modelo():
    """
    Entrena un modelo de clasificación basado en los datos etiquetados y lo guarda en un archivo.
    """
    if not os.path.exists("posturas.csv"):
        print("No se encontraron datos etiquetados. Captura posturas primero.")
        return
    
    # Cargar datos etiquetados
    data = []
    labels = []
    with open("posturas.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Aplanar los landmarks (de 3D a 1D)
            flattened_landmarks = [coord for landmark in eval(row["landmarks"]) for coord in landmark]
            data.append(flattened_landmarks)
            labels.append(row["etiqueta"])
    
    # Convertir a numpy arrays
    X = np.array(data)
    y = np.array(labels)
    
    # Normalizar los datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Entrenar modelo k-NN
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(X, y)
    
    # Evaluar modelo
    y_pred = knn.predict(X)
    print("\nReporte de clasificación:")
    print(classification_report(y, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y, y_pred))
    
    # Guardar modelo entrenado y etiquetas
    joblib.dump({"model": knn, "scaler": scaler, "labels": np.unique(y)}, "modelo_posturas.pkl")
    print("Modelo entrenado guardado como 'modelo_posturas.pkl'.")

if __name__ == "__main__":
    print("Opciones:")
    print("1. Capturar postura")
    print("2. Entrenar modelo")
    opcion = input("Selecciona una opción (1/2): ")
    
    if opcion == "1":
        etiqueta = input("Ingresa la etiqueta para la postura: ")
        video_source = input("Ingresa la fuente de video (0 para cámara local o URL): ")
        video_source = int(video_source) if video_source.isdigit() else video_source
        capturar_postura(video_source, etiqueta)
    elif opcion == "2":
        entrenar_modelo()
    else:
        print("Opción no válida.")