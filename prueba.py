from flask import Flask, render_template, Response, request
import cv2
import os
from simple_facerec import SimpleFacerec

app = Flask(__name__)
sfr = SimpleFacerec()
sfr.load_encoding_images("imagenes/")  # Asegúrate de que la ruta sea correcta

# Intentar abrir varias cámaras
cameras = [cv2.VideoCapture(i) for i in range(4)]  # Puedes ajustar el rango según la cantidad de cámaras que quieras probar

for cam in cameras:
    if cam.isOpened():
        print(f"Se ha encontrado una cámara en el índice {cam.get(cv2.CAP_PROP_POS_FRAMES)}")
        break

if not cam.isOpened():
    print("No se pudo encontrar ninguna cámara. Saliendo.")
    exit()

recognizing = False

UPLOAD_FOLDER = 'imagenes'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    while True:
        ret, frame = cam.read()

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
