from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Cargar el clasificador de rostros pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def reconocimiento_facial(cap, face_cascade):
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
        
        # Codificar la imagen como JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_route')
def video_feed_route(cap, face_cascade):
    return Response(reconocimiento_facial(cap, face_cascade), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
