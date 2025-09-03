"""
📁 Upload Controller
Endpoints para subida y gestión de archivos
"""

from flask import Blueprint, request, jsonify, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename

# Importar configuración del servidor original
from app.utils import allowed_file, UPLOAD_FOLDER

# Crear blueprint
upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    📁 SUBIR ARCHIVOS - Endpoint de compatibilidad del servidor original
    Handles both images and videos
    """
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

@upload_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    📁 SERVIR ARCHIVOS SUBIDOS - Endpoint de compatibilidad
    """
    return send_from_directory(UPLOAD_FOLDER, filename)
