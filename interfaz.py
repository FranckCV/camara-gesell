import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def crear_interfaz(root, iniciar_camara, toggle_recording, actualizar_grafica, CAM01, CAM02):
    """
    Crea la interfaz gráfica principal.
    """
    root.title("Cámara Gesell - Emociones, Postura y Gráficas en vivo")
    root.geometry("800x600")
    root.configure(bg="#f0f0f0")

    # Estilo de los botones
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 10), padding=5)
    style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))

    # Frames para las cámaras
    frame1 = ttk.Label(root, text="Cámara 1", anchor="center")
    frame1.grid(row=0, column=0, padx=10, pady=10)
    frame2 = ttk.Label(root, text="Cámara 2", anchor="center")
    frame2.grid(row=0, column=1, padx=10, pady=10)

    # Botones para grabar/detener
    btn_rec1 = ttk.Button(root, text=f"Grabar/Detener {CAM01}", command=lambda: toggle_recording(CAM01))
    btn_rec1.grid(row=1, column=0, pady=5)
    btn_rec2 = ttk.Button(root, text=f"Grabar/Detener {CAM02}", command=lambda: toggle_recording(CAM02))
    btn_rec2.grid(row=1, column=1, pady=5)

    # Gráfica de emociones
    fig, ax = plt.subplots(figsize=(6, 3))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, pady=10)

    # Botón para salir
    btn_exit = ttk.Button(root, text="Salir", command=root.destroy)
    btn_exit.grid(row=3, column=0, columnspan=2, pady=10)

    # Iniciar cámaras
    iniciar_camara(CAM01, frame1)
    iniciar_camara(CAM02, frame2)

    # Actualizar gráfica periódicamente
    actualizar_grafica()

    return root