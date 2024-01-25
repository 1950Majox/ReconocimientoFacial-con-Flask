from flask import Flask, render_template, Response, request
import cv2
import os
from simple_facerec import SimpleFacerec

app = Flask(__name__)
sfr = SimpleFacerec()
sfr.load_encoding_images("imagenes/")  # Asegúrate de que la ruta sea correcta

# Intentar abrir varias cámaras
cameras = [cv2.VideoCapture(i) for i in range(4)]  # Puedes ajustar el rango según la cantidad de cámaras que quieras probar

current_cam = 0  # Índice de la cámara actual
recognizing = False

UPLOAD_FOLDER = 'imagenes'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    while True:
        ret, frame = cameras[current_cam].read()  # Utiliza la cámara actual

        if recognizing:
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        ret, jpeg = cv2.imencode('.jpg', frame)
        data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recognition')
def toggle_recognition():
    global recognizing
    recognizing = not recognizing
    return {'status': 'success', 'recognizing': recognizing}

@app.route('/toggle_camera')
def toggle_camera():
    global current_cam
    current_cam = (current_cam + 1) % len(cameras)  # Cambia a la siguiente cámara en la lista
    return {'status': 'success', 'current_camera': current_cam}

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return {'status': 'error', 'message': 'No se seleccionó ninguna imagen'}
    
    file = request.files['image']

    if file.filename == '':
        return {'status': 'error', 'message': 'No se seleccionó ninguna imagen'}
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        sfr.load_encoding_images(app.config['UPLOAD_FOLDER'])
        return {'status': 'success', 'message': 'Imagen subida y cargada correctamente'}
    else:
        return {'status': 'error', 'message': 'Formato de archivo no permitido'}

if __name__ == '__main__':
    app.run(debug=True)
