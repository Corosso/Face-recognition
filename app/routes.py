from app import app
from flask import render_template

@app.route('/')
@app.route('/index')
@app.route('/video_feed')
@app.route('/record_face')
def index():
    return render_template('index.html')
