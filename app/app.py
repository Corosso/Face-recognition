from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
from record_face import registrar_rostro_route

app = Flask(__name__)

# Cargar el clasificador de rostros pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registrar')
def registrar():
    return render_template('registrar.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registrar_rostro', methods=['POST'])
def registrar_rostro():
    numero_identificacion = request.form['numero_identificacion']
    registrar_rostro_route(numero_identificacion, face_cascade)
    print('Rostro registrado correctamente.')
    return redirect(url_for('registrar'))

def generar_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

if __name__ == '__main__':
    if not os.path.exists('records'):
        os.makedirs('records')
    app.run(debug=True)
