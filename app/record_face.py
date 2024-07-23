import cv2
import csv
import os

def guardar_registro(numero_identificacion, ruta_imagen):
    with open('records/records.csv', 'a', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([numero_identificacion, ruta_imagen])

def registrar_rostro_route(numero_identificacion, face_cascade):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if not ret:
        print("Error: no se pudo capturar la imagen")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No se detectaron rostros")
        return

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        nombre_archivo = f"rostro_{numero_identificacion}.jpg"
        ruta_imagen = os.path.join('records', nombre_archivo)
        cv2.imwrite(ruta_imagen, gray[y:y+h, x:x+w])
        guardar_registro(numero_identificacion, ruta_imagen)
        break

    cap.release()
    cv2.destroyAllWindows()
