from flask import Flask, render_template, request
from video_feed import video_feed
from record_face import registrar_rostro

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed_route():
    return video_feed()

@app.route('/registrar_rostro', methods=['POST'])
def registrar_rostro_route():
    numero_identificacion = request.form['numero_identificacion']
    registrar_rostro(numero_identificacion)
    return 'Rostro registrado correctamente.'

if __name__ == '__main__':
    app.run(debug=True)
