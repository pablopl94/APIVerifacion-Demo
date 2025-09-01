from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_from_directory
)
import face_recognition
import array as arr
import cv2
import os
from pathlib import Path
import imutils
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import uuid
from werkzeug.utils import secure_filename

# Create the application instance
app = Flask(__name__, template_folder="templates")

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create a URL route in our application for "/"
@app.route('/')
def index():
    return 'API FUNCIONANDO'

# New route to serve the KYC verification flow
@app.route('/test')
def kyc_verification():
    return render_template('kyc_verification.html')

# Upload endpoint - handles both images and videos
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check for files in any of these fields
        file = None
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
        elif 'video' in request.files and request.files['video'].filename != '':
            file = request.files['video']
        
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        if file and file.filename != '':
            if allowed_file(file.filename):
                # Generate unique filename
                filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                # Use forward slashes for URLs
                url_path = filepath.replace('\\', '/')
                
                return jsonify({
                    'success': True,
                    'path': url_path,
                    'filename': filename
                })
            else:
                return jsonify({'error': 'File type not allowed'}), 400
        
        return jsonify({'error': 'No file selected'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)



@app.route('/textImage')
def textImage():
    try:
        known = request.args.get('known')
        
        if not known:
            return ""
            
        text = pytesseract.image_to_string(Image.open(known))
        return text
        
    except Exception as e:
        return ""

@app.route('/faceImage')
def faceImage():
    try:
        known = request.args.get('known')
        unknown = request.args.get('unknown')

        if not known or not unknown:
            return str(False)

        picture_of_me = face_recognition.load_image_file(known)
        known_face_encodings = face_recognition.face_encodings(picture_of_me)
        
        if len(known_face_encodings) == 0:
            return str(False)
        
        my_face_encoding = known_face_encodings[0]

        unknown_picture = face_recognition.load_image_file(unknown)
        unknown_face_encodings = face_recognition.face_encodings(unknown_picture)
        
        if len(unknown_face_encodings) == 0:
            return str(False)
            
        unknown_face_encoding = unknown_face_encodings[0]

        # Compare the faces
        results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

        return str(results[0])
        
    except Exception as e:
        return str(False)

@app.route('/faceVideo')
def faceVideo():
    try:
        known = request.args.get('known')
        unknown = request.args.get('unknown')

        if not known or not unknown:
            return str(False)

        picture_of_me = face_recognition.load_image_file(known)
        face_encodings_known = face_recognition.face_encodings(picture_of_me)
        
        if len(face_encodings_known) == 0:
            return str(False)
            
        known_face_me = face_encodings_known[0]

        # Initialize some variables
        count = 0
        found = 0
        success = True

        vidcap = cv2.VideoCapture(unknown)
        #play first
        success, image = vidcap.read()

        while success and count < 100:  # Limit frames to prevent infinite loop
            try:
                image = imutils.rotate(image, -90)  # Use same rotation as play.py
                face_encodings_list = face_recognition.face_encodings(image)
                
                if len(face_encodings_list) > 0:
                    face_encodings = face_encodings_list[0]
                    matches = face_recognition.compare_faces([known_face_me], face_encodings)
                    
                    if True in matches:
                        found += 1
                        
                    if found > 10:
                        break
            except Exception as e:
                # Skip frame if face detection fails
                pass
                
            # Read next frame
            success, image = vidcap.read()
            count += 1

        vidcap.release()

        # Avoid division by zero
        resp = False
        if count > 0 and (found / count > 0.5):
            resp = True

        return str(resp)
        
    except Exception as e:
        return str(False)

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(debug=True)
