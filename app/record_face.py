from flask import request


import cv2

import csv

def guardar_registro(numero_identificacion, ruta_imagen):
    with open('records.csv', 'a', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([numero_identificacion, ruta_imagen])

def registrar_cara(numero_identificacion):
    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    # Cargar el clasificador de rostros pre-entrenado
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        
        # Convertir el frame a escala de grises para el reconocimiento facial
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros en el frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Dibujar un rectángulo alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Mostrar el frame capturado en una ventana
        cv2.imshow('Registro de Caras', frame)
        
        # Esperar la entrada del usuario para registrar la cara
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Guardar la imagen del rostro con el número de identificación
            nombre_archivo = f"rostro_{numero_identificacion}.jpg"
            cv2.imwrite(nombre_archivo, gray[y:y+h, x:x+w])
            break
    guardar_registro(numero_identificacion, ruta_imagen)

    # Liberar la captura de video y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    numero_identificacion = input("Ingrese el número de identificación para esta cara: ")
    registrar_cara(numero_identificacion)

@app.route('/registrar_rostro', methods=['POST'])
def registrar_rostro():
    numero_identificacion = request.form['numero_identificacion']
    # Llama a la función para registrar el rostro con el número de identificación
    registrar_cara(numero_identificacion)
    return 'Rostro registrado correctamente.'
