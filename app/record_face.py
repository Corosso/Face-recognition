import cv2
import csv
import os
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)


cap = cv2.VideoCapture(0)
# Cargar el clasificador de rostros pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def guardar_registro(numero_identificacion, ruta_imagen):
    with open('records/records.csv', 'a', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([numero_identificacion, ruta_imagen])

def registrar_cara(numero_identificacion, cap, face_cascade):
    ruta_imagen = 'records/'  # Ruta base para guardar las imágenes
    
    # Crear la carpeta "records" si no existe
    if not os.path.exists('records'):
        os.makedirs('records')

    # Capturar frame por frame
    ret, frame = cap.read()
    
    # Convertir el frame a escala de grises para el reconocimiento facial
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Dibujar un rectángulo alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Guardar la imagen del rostro con el número de identificación
    nombre_archivo = f"rostro_{numero_identificacion}.jpg"
    cv2.imwrite(os.path.join(ruta_imagen, nombre_archivo), gray[y:y+h, x:x+w])
    
    ruta_imagen = os.path.join(ruta_imagen, nombre_archivo)  # Ruta completa de la imagen
    guardar_registro(numero_identificacion, ruta_imagen)

    # Liberar la captura de video y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

@app.route('/record_face', methods=['POST'])
def registrar_rostro_route(numero_identificacion, cap, face_cascade):
    
    numero_identificacion = request.form['numero_identificacion']
    registrar_cara(numero_identificacion, cap, face_cascade)
    return render_template("registrar.html")
    
@app.route('/registrar')
def registrar():
    return render_template('registrar.html')

if __name__ == "__main__":
    app.run(debug=True)
