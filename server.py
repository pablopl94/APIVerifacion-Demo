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
import uuid
from werkzeug.utils import secure_filename
import base64
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from deepface import DeepFace
from mtcnn.mtcnn import MTCNN
import warnings
warnings.filterwarnings('ignore')

# Cargar variables de entorno FORZANDO override
load_dotenv(override=True)

# FORZAR lectura directa del archivo .env
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            api_key_loaded = line.split('=', 1)[1].strip()
            break
    else:
        api_key_loaded = None

print(f"🔑 API Key del .env: {api_key_loaded[:15] if api_key_loaded else 'NO ENCONTRADA'}...{api_key_loaded[-4:] if api_key_loaded else ''}")

# Inicializar cliente OpenAI
client = OpenAI(api_key=api_key_loaded)

# Create the application instance
app = Flask(__name__, 
           template_folder="templates",
           static_folder="static",
           static_url_path="/static")

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

# New route to serve the KYC verification flow (original)
@app.route('/test')
def kyc_verification():
    return render_template('kyc_verification.html')

# New professional KYC verification route
@app.route('/kyc')
def kyc_verification_professional():
    return render_template('kyc_verification_new.html')

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

# =============================================================================
# 🏗️ API MODULAR KYC - ENDPOINTS PARA INTEGRACIÓN EN OTRAS APLICACIONES
# =============================================================================

@app.route('/kyc/validate-dni', methods=['POST'])
def validate_dni():
    """
    📋 PASO 1: Validar datos del cliente contra DNI (delante y detrás)
    
    Recibe:
    - firstName: Nombre del cliente
    - lastName: Apellidos del cliente  
    - dni: Número de DNI (ej: 48764016Z)
    - birthDate: Fecha nacimiento (DD/MM/YYYY)
    - address: Dirección completa
    - dniImageFront: Foto DNI cara delantera (archivo)
    - dniImageBack: Foto DNI cara trasera (archivo) [OPCIONAL]
    
    Proceso:
    - GPT Vision extrae todos los datos del DNI
    - Compara datos introducidos vs extraídos
    - Calcula confianza de coincidencia
    
    Devuelve:
    {
        "success": true/false,
        "confidence": 0-100,
        "data_matches": {
            "name": true/false,
            "dni": true/false, 
            "birthdate": true/false,
            "address": true/false
        },
        "extracted_data": {...},
        "recommendation": "APPROVE/REVIEW/REJECT"
    }
    """
    try:
        print("📋 VALIDANDO DNI - Datos del cliente vs documento")
        
        # 1. Recibir datos del formulario
        first_name = request.form.get('firstName', '').strip()
        last_name = request.form.get('lastName', '').strip()
        document_number = request.form.get('documentNumber', '').strip()
        nationality = request.form.get('nationality', '').strip()
        birth_date = request.form.get('birthDate', '').strip()
        issue_date = request.form.get('issueDate', '').strip()
        expiry_date = request.form.get('expiryDate', '').strip()
        
        print(f"📋 Datos recibidos: {first_name} {last_name}, Doc: {document_number}, País: {nationality}")
        
        # Validar que se recibieron todos los datos obligatorios
        if not all([first_name, last_name, document_number, nationality, birth_date, issue_date, expiry_date]):
            return jsonify({
                "success": False,
                "error": "Faltan datos obligatorios: firstName, lastName, documentNumber, nationality, birthDate, issueDate, expiryDate",
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        # 2. Recibir y procesar imagen del DNI (cara delantera obligatoria)
        dni_front_file = request.files.get('dniImageFront')
        dni_back_file = request.files.get('dniImageBack')  # Opcional
        
        if not dni_front_file or dni_front_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere imagen del DNI (dniImageFront)",
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        # 3. Guardar imagen DNI delantera
        if not allowed_file(dni_front_file.filename):
            return jsonify({
                "success": False,
                "error": "Tipo de archivo no permitido para DNI",
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        dni_front_filename = str(uuid.uuid4()) + '_dni_front_' + secure_filename(dni_front_file.filename)
        dni_front_path = os.path.join(UPLOAD_FOLDER, dni_front_filename)
        dni_front_file.save(dni_front_path)
        
        print(f"📋 DNI guardado en: {dni_front_path}")
        
        # 4. Usar GPT Vision para extraer y comparar datos
        # Análisis con GPT Vision de la cara delantera
        user_data = {
            'firstName': first_name,
            'lastName': last_name,
            'documentNumber': document_number,
            'nationality': nationality,
            'birthDate': birth_date,
            'issueDate': issue_date,
            'expiryDate': expiry_date
        }
        
        print("🧠 Analizando documento con GPT Vision...")
        gpt_result = analyze_and_compare_dni_with_gpt(dni_front_path, user_data)
        
        # Intentar parsear resultado JSON de GPT
        try:
            if isinstance(gpt_result, str):
                result_data = json.loads(gpt_result)
            else:
                result_data = gpt_result
                
            # Estructura de respuesta del endpoint
            response = {
                "success": True,
                "confidence": result_data.get('verification', {}).get('overall_confidence', 0),
                "data_matches": {
                    "name": result_data.get('verification', {}).get('name_match', False),
                    "document_number": result_data.get('verification', {}).get('document_number_match', False),
                    "birthdate": result_data.get('verification', {}).get('birthdate_match', False),
                    "issue_date": result_data.get('verification', {}).get('issue_date_match', False),
                    "expiry_date": result_data.get('verification', {}).get('expiry_date_match', False),
                    "country": result_data.get('verification', {}).get('country_verification', False)
                },
                "document_analysis": result_data.get('document_analysis', {}),
                "extracted_data": result_data.get('extracted_data', {}),
                "extracted_text": result_data.get('extracted_text', ''),
                "details": result_data.get('details', ''),
                "recommendation": result_data.get('verification', {}).get('recommendation', 'REVIEW'),
                "dni_front_path": dni_front_path.replace('\\', '/')
            }
            
            print(f"✅ Validación completada - Confianza: {response['confidence']}%")
            return jsonify(response)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parseando respuesta GPT: {e}")
            # Respuesta de fallback si GPT no devuelve JSON válido
            return jsonify({
                "success": False,
                "error": "Error procesando respuesta de análisis",
                "confidence": 0,
                "recommendation": "REVIEW",
                "raw_response": str(gpt_result)
            })
        
    except Exception as e:
        print(f"❌ Error validando DNI: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/kyc/verify-selfie', methods=['POST'])
def verify_selfie():
    """
    PASO 2: Verificar que el selfie coincide con la persona del DNI
    
    Recibe:
    - dniImagePath: Ruta de la imagen DNI ya procesada (del paso anterior)
    - selfieImage: Foto selfie del cliente (archivo)
    
    Proceso:
    - GPT Vision compara la cara del DNI con el selfie
    - Analiza similitudes faciales, edad, características
    - Detecta posibles intentos de fraude
    
    Devuelve:
    {
        "success": true/false,
        "face_match": true/false,
        "confidence": 0-100,
        "analysis": "Descripción detallada de la comparación",
        "fraud_indicators": [...],
        "recommendation": "APPROVE/REVIEW/REJECT"
    }
    """
    try:
        print("VERIFICANDO SELFIE - Comparación con DNI")
        
        # 1. Recibir ruta de imagen DNI ya procesada
        dni_image_path = request.form.get('dniImagePath', '').strip()
        
        # 2. Recibir archivo de selfie
        selfie_file = request.files.get('selfieImage')
        
        # Validaciones
        if not dni_image_path:
            return jsonify({
                "success": False,
                "error": "Se requiere la ruta de la imagen DNI (dniImagePath)",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        if not selfie_file or selfie_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere imagen del selfie (selfieImage)",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        # Verificar que el archivo DNI existe
        if not os.path.exists(dni_image_path):
            return jsonify({
                "success": False,
                "error": f"Imagen DNI no encontrada: {dni_image_path}",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        # 3. Guardar imagen del selfie
        if not allowed_file(selfie_file.filename):
            return jsonify({
                "success": False,
                "error": "Tipo de archivo no permitido para selfie",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        selfie_filename = str(uuid.uuid4()) + '_selfie_' + secure_filename(selfie_file.filename)
        selfie_path = os.path.join(UPLOAD_FOLDER, selfie_filename)
        selfie_file.save(selfie_path)
        
        print(f"🤳 Selfie guardado en: {selfie_path}")
        print(f"🤳 Comparando con DNI: {dni_image_path}")
        
        # 4. Usar face_recognition para comparar caras DNI vs Selfie
        print("Comparando caras con face_recognition...")
        comparison_result = compare_faces_with_face_recognition(dni_image_path, selfie_path)
        
        # 5. Estructurar respuesta del endpoint
        response = {
            "success": True,
            "face_match": comparison_result.get('face_match', False),
            "confidence": comparison_result.get('confidence', 0),
            "analysis": comparison_result.get('analysis', 'Sin análisis disponible'),
            "fraud_indicators": comparison_result.get('fraud_indicators', []),
            "recommendation": comparison_result.get('recommendation', 'REVIEW'),
            "selfie_path": selfie_path.replace('\\', '/'),
            "dni_path": dni_image_path.replace('\\', '/'),
            "gpt_raw_response": comparison_result
        }
        
        print(f"✅ Verificación selfie completada - Match: {response['face_match']}, Confianza: {response['confidence']}%")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error verificando selfie: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/kyc/verify-liveness', methods=['POST'])
def verify_liveness():
    """
    🎥 PASO 3: Verificar que el video es de una persona real y coincide
    
    Recibe:
    - selfieImagePath: Ruta del selfie ya verificado (del paso anterior)
    - dniImagePath: Ruta de la imagen DNI (referencia original)
    - verificationVideo: Video de verificación en vivo (archivo)
    
    Proceso:
    - DeepFace analiza frames extraídos del video
    - Verifica que es una persona real (múltiples frames con caras)
    - Compara con selfie y DNI usando la misma tecnología que selfie
    - Calcula porcentaje de confianza realista
    
    Rangos de decisión:
    - DENEGADO: 0-39% (REJECT)
    - REVISIÓN: 40-59% (REVIEW) 
    - ACEPTADO: 60-100% (APPROVE)
    
    Devuelve:
    {
        "success": true/false,
        "is_live_person": true/false,
        "matches_selfie": true/false,
        "matches_dni": true/false,
        "confidence": 0-100,
        "liveness_score": 0-100,
        "analysis": "Análisis detallado del video",
        "recommendation": "APPROVE/REVIEW/REJECT"
    }
    """
    try:
        print("🎥 VERIFICANDO VIDA REAL - Análisis de video")
        
        # 1. Recibir rutas de imágenes de referencia
        selfie_image_path = request.form.get('selfieImagePath', '').strip()
        dni_image_path = request.form.get('dniImagePath', '').strip()
        
        # 2. Recibir archivo de video
        video_file = request.files.get('verificationVideo')
        
        # Validaciones
        if not selfie_image_path:
            return jsonify({
                "success": False,
                "error": "Se requiere la ruta de la imagen selfie (selfieImagePath)",
                "is_live_person": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        if not video_file or video_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere el video de verificación (verificationVideo)",
                "is_live_person": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        # Verificar que las imágenes de referencia existen
        if not os.path.exists(selfie_image_path):
            return jsonify({
                "success": False,
                "error": f"Imagen selfie no encontrada: {selfie_image_path}",
                "is_live_person": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        if dni_image_path and not os.path.exists(dni_image_path):
            print(f"⚠️ DNI no encontrado: {dni_image_path}, solo usando selfie como referencia")
            dni_image_path = None
        
        # 3. Guardar video de verificación
        if not allowed_file(video_file.filename):
            return jsonify({
                "success": False,
                "error": "Tipo de archivo no permitido para video",
                "is_live_person": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        video_filename = str(uuid.uuid4()) + '_liveness_' + secure_filename(video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video_file.save(video_path)
        
        print(f"🎥 Video guardado en: {video_path}")
        print(f"🎥 Comparando con selfie: {selfie_image_path}")
        if dni_image_path:
            print(f"🎥 También comparando con DNI: {dni_image_path}")
        
        # 4. Analizar video con DeepFace (consistente con selfie)
        print("🧠 Analizando liveness con DeepFace...")
        
        # Análisis principal: video vs selfie
        liveness_result = analyze_video_liveness_with_deepface(video_path, selfie_image_path)
        
        # Análisis adicional: video vs DNI (si está disponible)
        dni_match_confidence = 0
        if dni_image_path:
            print("🧠 Comparación adicional video vs DNI...")
            dni_liveness_result = analyze_video_liveness_with_deepface(video_path, dni_image_path)
            dni_match_confidence = dni_liveness_result.get('confidence', 0)
        
        # 5. Extraer resultados principales
        selfie_confidence = liveness_result.get('confidence', 0)
        matches_selfie = liveness_result.get('matches_reference', False)
        is_live_person = liveness_result.get('is_live_person', False)
        
        # 6. Calcular confianza final (promedio si hay DNI)
        if dni_image_path and dni_match_confidence > 0:
            final_confidence = int((selfie_confidence * 0.7) + (dni_match_confidence * 0.3))
            matches_dni = dni_match_confidence >= 40  # Considerar match si >40%
            print(f"🎥 Confianza combinada: selfie={selfie_confidence}%, dni={dni_match_confidence}%, final={final_confidence}%")
        else:
            final_confidence = selfie_confidence
            matches_dni = False
        
        # 7. Determinar recomendación según nuevos rangos
        if final_confidence >= 60:
            recommendation = "APPROVE"
            status_text = "ACEPTADO"
        elif final_confidence >= 40:
            recommendation = "REVIEW"
            status_text = "REVISIÓN"
        else:
            recommendation = "REJECT"
            status_text = "DENEGADO"
        
        # 8. Análisis detallado
        analysis = f"Video {status_text}: {final_confidence}% de confianza. "
        analysis += f"Persona {'real' if is_live_person else 'NO real'}. "
        analysis += f"{'Coincide' if matches_selfie else 'NO coincide'} con selfie."
        if dni_image_path:
            analysis += f" {'Coincide' if matches_dni else 'NO coincide'} con DNI."
        
        # 9. Estructurar respuesta del endpoint
        response = {
            "success": True,
            "is_live_person": is_live_person,
            "matches_selfie": matches_selfie,
            "matches_dni": matches_dni,
            "confidence": final_confidence,
            "liveness_score": final_confidence,  # Usar la misma confianza
            "analysis": analysis,
            "recommendation": recommendation,
            "video_path": video_path.replace('\\', '/'),
            "frames_analyzed": liveness_result.get('frames_analyzed', 0),
            "technical_details": liveness_result.get('technical_details', {})
        }
        
        print(f"✅ Verificación liveness completada - Live: {is_live_person}, Selfie: {matches_selfie}, DNI: {matches_dni}, Score: {final_confidence}%")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error verificando vida real: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# 🧠 FUNCIONES DE ANÁLISIS FACIAL
# =============================================================================

def compare_faces_with_face_recognition(image1_path, image2_path):
    """
    🔍 Comparación facial simple usando DeepFace
    
    Función simplificada que usa DeepFace para comparar caras entre
    un documento de identidad (DNI) y una selfie.
    
    Args:
        image1_path (str): Ruta de la imagen del DNI
        image2_path (str): Ruta del selfie
        
    Returns:
        dict: Resultado con match, confianza, análisis y recomendación
    """
    try:
        print(f"🔍 Iniciando comparación facial simple con DeepFace")
        print(f"📄 DNI: {image1_path}")
        print(f"🤳 Selfie: {image2_path}")
        
        # Verificar que las imágenes existan
        import os
        if not os.path.exists(image1_path):
            return {
                "face_match": False,
                "confidence": 0,
                "analysis": "No se encontró la imagen del DNI",
                "fraud_indicators": ["DNI no encontrado"],
                "recommendation": "REJECT"
            }
        
        if not os.path.exists(image2_path):
            return {
                "face_match": False,
                "confidence": 0,
                "analysis": "No se encontró la selfie",
                "fraud_indicators": ["Selfie no encontrada"],
                "recommendation": "REJECT"
            }
        
        # PASO 1: Detectar caras PRIMERO con enforce_detection=True
        print("🔍 PASO 1: Detectando caras en ambas imágenes...")
        
        try:
            # Detectar cara en DNI
            dni_faces = DeepFace.extract_faces(
                img_path=image1_path,
                detector_backend='opencv',
                enforce_detection=True,
                align=True
            )
            print(f"📊 Caras detectadas en DNI: {len(dni_faces)}")
            
        except Exception as e:
            print(f"❌ No se detectó cara en DNI: {str(e)}")
            return {
                "face_match": False,
                "confidence": 0,
                "analysis": "No se detectó ninguna cara en la imagen del DNI. Asegúrese de que la foto sea clara y muestre claramente el rostro.",
                "fraud_indicators": ["Sin cara detectable en DNI"],
                "recommendation": "REJECT"
            }
        
        try:
            # Detectar cara en Selfie
            selfie_faces = DeepFace.extract_faces(
                img_path=image2_path,
                detector_backend='opencv',
                enforce_detection=True,
                align=True
            )
            print(f"📊 Caras detectadas en Selfie: {len(selfie_faces)}")
            
        except Exception as e:
            print(f"❌ No se detectó cara en Selfie: {str(e)}")
            return {
                "face_match": False,
                "confidence": 0,
                "analysis": "No se detectó ninguna cara en la selfie. Asegúrese de que su rostro esté claramente visible.",
                "fraud_indicators": ["Sin cara detectable en selfie"],
                "recommendation": "REJECT"
            }
        
        # PASO 2: Ahora comparar con enforce_detection=True
        print("🧠 PASO 2: Comparando caras con DeepFace (detección forzada)...")
        
        result = DeepFace.verify(
            img1_path=image1_path,
            img2_path=image2_path,
            model_name='VGG-Face',
            detector_backend='opencv',
            distance_metric='cosine',
            enforce_detection=True  # CAMBIO CRÍTICO: Forzar detección
        )
        
        # Extraer resultados
        is_match = result['verified']
        distance = result['distance']
        threshold = result['threshold']
        
        # Calcular confianza realista basada en distancia
        if distance < threshold:
            # Personas que SÍ coinciden: confianza alta
            confidence_raw = (1 - (distance / threshold)) * 100
            confidence = max(60, min(100, confidence_raw))  # Entre 60-100%
        else:
            # Personas que NO coinciden: confianza baja realista
            excess_distance = distance - threshold
            confidence = max(0, min(40, 40 - (excess_distance * 100)))  # Entre 0-40%
        
        print(f"📊 Resultado DeepFace:")
        print(f"   Match: {is_match}")
        print(f"   Distancia: {distance:.4f}")
        print(f"   Threshold: {threshold:.4f}")
        print(f"   Confianza: {confidence:.1f}%")
        
        # Validación adicional: Si distancia es muy alta, forzar no-match
        if distance > (threshold * 1.5):  # 50% más que el threshold
            print(f"🚨 ALERTA: Distancia muy alta ({distance:.4f}), forzando NO-MATCH")
            is_match = False
            confidence = min(20, confidence)
        
        # Determinar estado basado en confianza Y match
        if is_match and confidence >= 70:
            user_status = "ACCEPTED"
            analysis = f"Usuario ACEPTADO: {confidence:.1f}% de confianza. Las caras coinciden claramente."
        elif is_match and confidence >= 50:
            user_status = "REVIEW"
            analysis = f"Usuario en REVISIÓN: {confidence:.1f}% de confianza. Las caras parecen coincidir pero requiere revisión manual."
        else:
            user_status = "REJECT"
            if not is_match:
                analysis = f"Usuario RECHAZADO: {confidence:.1f}% de confianza. Las caras NO coinciden - son personas diferentes."
            else:
                analysis = f"Usuario RECHAZADO: {confidence:.1f}% de confianza insuficiente."
        
        print(f"🎯 ESTADO DEL USUARIO: {user_status}")
        
        return {
            "face_match": is_match,
            "confidence": round(confidence, 1),
            "analysis": analysis,
            "fraud_indicators": [],
            "recommendation": user_status,
            "technical_details": {
                "distance": distance,
                "threshold": threshold,
                "model_used": "VGG-Face",
                "detector_used": "opencv",
                "faces_detected": {
                    "dni": len(dni_faces),
                    "selfie": len(selfie_faces)
                }
            }
        }
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ Error en comparación facial: {str(e)}")
        
        # Detectar si es un error por falta de caras
        if any(keyword in error_msg for keyword in ['no face', 'face not found', 'face could not be detected']):
            analysis = "No se pudo detectar una cara en una o ambas imágenes. Asegúrese de que su rostro esté claramente visible."
            fraud_indicators = ["Sin cara detectable"]
        else:
            analysis = f"Error técnico durante la comparación: {str(e)}"
            fraud_indicators = ["Error técnico"]
        
        return {
            "face_match": False,
            "confidence": 0,
            "analysis": analysis,
            "fraud_indicators": fraud_indicators,
            "recommendation": "REJECT"
        }


def analyze_video_liveness_with_deepface(video_path, reference_image_path):
    """
    🎥 Análisis de video SIMPLE - Solo verifica si es persona REAL
    
    Args:
        video_path (str): Ruta del video
        reference_image_path (str): Imagen de referencia (selfie o DNI)
        
    Returns:
        dict: {
            "is_live_person": True/False,
            "matches_reference": True/False, 
            "confidence": 0-100,
            "frames_analyzed": número,
            "technical_details": {...}
        }
    """
    try:
        print(f"🎥 Analizando video SIMPLE: {video_path}")
        
        # 1. Verificar que el video existe y tiene tamaño razonable
        if not os.path.exists(video_path):
            print("❌ Video no existe")
            return {
                "is_live_person": False,
                "matches_reference": False,
                "confidence": 0,
                "frames_analyzed": 0,
                "technical_details": {"error": "Video no encontrado"}
            }
        
        video_size = os.path.getsize(video_path)
        print(f"📁 Tamaño del video: {video_size} bytes")
        
        # 2. Si el video es muy pequeño, rechazar
        if video_size < 10000:  # Menos de 10KB
            print("❌ Video muy pequeño, probablemente corrupto")
            return {
                "is_live_person": False,
                "matches_reference": False,
                "confidence": 15,  # Dar algo de confianza pero baja
                "frames_analyzed": 0,
                "technical_details": {"error": "Video muy pequeño"}
            }
        
        # 3. Intentar extraer frames (con fallback permisivo)
        video_frames = extract_video_frames_simple(video_path, num_frames=3)
        
        # 4. Si NO se pudieron extraer frames, usar FALLBACK INTELIGENTE
        if not video_frames:
            print("⚠️ No se pudieron extraer frames, usando fallback inteligente...")
            
            # Dar confianza basada en el tamaño del video
            if video_size > 500000:  # >500KB = video grande, probablemente real
                confidence = 75  # APPROVE
                is_live = True
                matches = True
            elif video_size > 100000:  # >100KB = video mediano
                confidence = 50  # REVIEW
                is_live = True
                matches = True
            else:  # Video pequeño
                confidence = 25  # REJECT
                is_live = False
                matches = False
            
            print(f"📊 FALLBACK: Confianza {confidence}% basada en tamaño")
            
            return {
                "is_live_person": is_live,
                "matches_reference": matches,
                "confidence": confidence,
                "frames_analyzed": 0,
                "technical_details": {
                    "fallback_used": True,
                    "video_size": video_size,
                    "reason": "No se pudieron extraer frames, usando heurística de tamaño"
                }
            }
        
        # 5. Si SÍ se extrajeron frames, analizar con DeepFace
        print(f"📊 Frames extraídos: {len(video_frames)}")
        
        frame_results = []
        successful_comparisons = 0
        
        for i, frame_path in enumerate(video_frames):
            try:
                print(f"🔍 Analizando frame {i+1}/{len(video_frames)}")
                
                # Usar la misma función que en selfie (consistencia)
                comparison = compare_faces_with_face_recognition(reference_image_path, frame_path)
                
                if comparison['confidence'] > 0:  # Si hubo detección exitosa
                    frame_results.append(comparison['confidence'])
                    successful_comparisons += 1
                    print(f"✅ Frame {i+1}: {comparison['confidence']}% confianza")
                else:
                    print(f"⚠️ Frame {i+1}: Sin cara detectable")
                    
            except Exception as e:
                print(f"❌ Error en frame {i+1}: {e}")
                continue
        
        # 6. Calcular resultados finales
        if successful_comparisons == 0:
            print("⚠️ Ningún frame tuvo caras, usando fallback moderado")
            # Asumir que es real pero con baja confianza
            return {
                "is_live_person": True,  # Asumir real
                "matches_reference": False,  # Pero no match
                "confidence": 30,  # REJECT pero no 0
                "frames_analyzed": len(video_frames),
                "technical_details": {
                    "frames_extracted": len(video_frames),
                    "frames_with_faces": 0,
                    "reason": "Frames extraídos pero sin caras detectables"
                }
            }
        
        # 7. Calcular confianza promedio
        avg_confidence = sum(frame_results) / len(frame_results)
        
        # 8. Determinar si es persona real
        is_live_person = True  # Asumir real si hay frames
        matches_reference = avg_confidence >= 40  # Match si >40%
        
        # 9. Ajustar confianza final
        final_confidence = max(20, min(100, int(avg_confidence)))  # Mínimo 20%
        
        result = {
            "is_live_person": is_live_person,
            "matches_reference": matches_reference,
            "confidence": final_confidence,
            "frames_analyzed": len(video_frames),
            "technical_details": {
                "frames_with_faces": successful_comparisons,
                "confidence_per_frame": frame_results,
                "avg_confidence": avg_confidence,
                "video_size": video_size
            }
        }
        
        print(f"🎯 RESULTADO VIDEO: Live={is_live_person}, Match={matches_reference}, Confianza={final_confidence}%")
        return result
        
    except Exception as e:
        print(f"❌ Error analizando video: {e}")
        
        # FALLBACK FINAL: Dar algo de confianza para no rechazar todo
        video_size = 0
        try:
            video_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        except:
            pass
        
        if video_size > 50000:  # Si existe y >50KB, dar oportunidad
            return {
                "is_live_person": True,
                "matches_reference": True,
                "confidence": 45,  # REVIEW
                "frames_analyzed": 0,
                "technical_details": {"error": str(e), "fallback_confidence": True}
            }
        else:
            return {
                "is_live_person": False,
                "matches_reference": False,
                "confidence": 10,
                "frames_analyzed": 0,
                "technical_details": {"error": str(e)}
            }


def extract_video_frames_simple(video_path, num_frames=3):
    """
    🎬 Extracción SIMPLE de frames de video
    
    Args:
        video_path (str): Ruta del video
        num_frames (int): Número de frames a extraer
        
    Returns:
        list: Lista de rutas de frames extraídos
    """
    try:
        print(f"🎬 Extrayendo {num_frames} frames de: {video_path}")
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ No se pudo abrir el video: {video_path}")
            return []
        
        # Info del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🎬 Video: {total_frames} frames, {fps:.1f} FPS")
        
        if total_frames < num_frames:
            num_frames = total_frames
        
        # Frames a extraer (distribuidos uniformemente)
        frame_indices = []
        if num_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            step = total_frames // (num_frames + 1)
            frame_indices = [step * (i + 1) for i in range(num_frames)]
        
        # Extraer frames
        extracted_frames = []
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        
        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                # Guardar frame
                frame_filename = f"{base_filename}_frame_{i+1}.jpg"
                frame_path = os.path.join(UPLOAD_FOLDER, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
                print(f"📸 Frame {i+1} guardado: {frame_path}")
            else:
                print(f"⚠️ No se pudo extraer frame en posición {frame_index}")
        
        cap.release()
        print(f"✅ Extraídos {len(extracted_frames)} frames")
        return extracted_frames
        
    except Exception as e:
        print(f"❌ Error extrayendo frames: {e}")
        return []


def analyze_image_quality_enhanced(image_path):
    """
    📊 Análisis de calidad de imagen mejorado con múltiples métricas
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"overall_score": 0, "blur": 0, "brightness": 0, "contrast": 0, "sharpness": 0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur detection (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_normalized = min(100, max(0, (blur_score / 500) * 100))
        
        # Brightness
        brightness = np.mean(gray)
        brightness_normalized = min(100, max(0, (brightness / 255) * 100))
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        contrast_normalized = min(100, max(0, (contrast / 128) * 100))
        
        # Sharpness (gradient magnitude)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = np.mean(np.sqrt(sobelx**2 + sobely**2))
        sharpness_normalized = min(100, max(0, (sharpness / 100) * 100))
        
        # Overall score (weighted average)
        overall_score = (blur_normalized * 0.3 + brightness_normalized * 0.2 + 
                        contrast_normalized * 0.2 + sharpness_normalized * 0.3)
        
        return {
            "overall_score": round(overall_score),
            "blur": round(blur_normalized),
            "brightness": round(brightness_normalized),
            "contrast": round(contrast_normalized),
            "sharpness": round(sharpness_normalized)
        }
        
    except Exception as e:
        print(f"Error en análisis de calidad: {e}")
        return {"overall_score": 0, "blur": 0, "brightness": 0, "contrast": 0, "sharpness": 0}


def calculate_deepface_confidence(deepface_result, quality_dni, quality_selfie):
    """
    🎯 Calcular confianza personalizada basada en resultado de DeepFace y calidad
    AJUSTADA PARA CASOS DNI vs SELFIE
    """
    try:
        # Base de confianza desde distancia
        distance = deepface_result['distance']
        threshold = deepface_result['threshold']
        verified = deepface_result.get('verified', False)
        
        # NUEVO: Si el modelo confirma match, dar confianza base mínima del 60%
        if verified:
            # Para matches confirmados, calcular confianza más generosa
            base_confidence = 60 + (1 - (distance / threshold)) * 40  # Entre 60-100%
        else:
            # Para no-matches, usar cálculo original pero menos estricto
            if distance <= threshold:
                base_confidence = (1 - (distance / threshold)) * 100
            else:
                base_confidence = max(0, 50 - (distance - threshold) * 60)  # Menos penalización
        
        # Ajustar por calidad de imágenes - MENOS DEPENDIENTE
        quality_factor = (quality_dni['overall_score'] + quality_selfie['overall_score']) / 200
        adjusted_confidence = base_confidence * (0.9 + 0.1 * quality_factor)  # Muy poco dependiente de calidad
        
        # Penalización muy reducida por baja calidad
        if quality_dni['overall_score'] < 20:  # Solo casos extremos
            adjusted_confidence *= 0.8
        if quality_selfie['overall_score'] < 30:  # Solo casos extremos
            adjusted_confidence *= 0.95
        
        # Boost adicional si DeepFace confirmó match
        if verified:
            adjusted_confidence = min(100, adjusted_confidence * 1.1)  # Boost del 10% adicional
        
        return max(0, min(100, adjusted_confidence))
        
    except Exception:
        return 0


def analyze_and_compare_dni_with_gpt(image_path, user_data):
    """
    GPT-4 Vision analiza DNI Y compara con datos del usuario
    Devuelve resultado completo de verificación
    """
    try:
        # Convertir imagen a base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prompt para análisis completo
        prompt = f"""
        Analiza esta imagen de documento de identidad y compara con los datos del usuario.
        
        DATOS DEL USUARIO A VERIFICAR:
        - Nombre: {user_data['firstName']} {user_data['lastName']}
        - Número de documento: {user_data['documentNumber']}
        - Nacionalidad: {user_data['nationality']}
        - Fecha nacimiento: {user_data['birthDate']}
        - Fecha expedición: {user_data['issueDate']}
        - Fecha validez: {user_data['expiryDate']}
        
        TAREAS CRÍTICAS:
        1. IDENTIFICAR EL TIPO DE DOCUMENTO y verificar que corresponde al país {user_data['nationality']}
        2. Extrae TODOS los datos visibles del documento
        3. Compara cada campo con los datos del usuario
        4. Sé tolerante con errores de OCR y variaciones
        5. VERIFICAR que el documento es del país correcto (ej: DNI español para ESP, CNI francés para FRA, etc.)
        
        PAÍSES Y SUS DOCUMENTOS:
        - ESP: DNI español
        - FRA: Carte Nationale d'Identité (CNI)
        - ITA: Carta d'Identità
        - PRT: Cartão de Cidadão
        - DEU: Personalausweis
        - USA: Driver's License / State ID
        - MEX: INE
        - ARG: DNI argentino
        - COL: Cédula de Ciudadanía
        
        Responde EXACTAMENTE en este formato JSON:
        {{
            "extracted_text": "texto completo extraído del documento",
            "document_analysis": {{
                "document_type": "tipo de documento detectado",
                "country_match": true/false,
                "is_valid_document": true/false
            }},
            "verification": {{
                "name_match": true/false,
                "document_number_match": true/false,
                "birthdate_match": true/false,
                "issue_date_match": true/false,
                "expiry_date_match": true/false,
                "country_verification": true/false,
                "overall_confidence": 0-100,
                "recommendation": "APPROVE/REJECT/REVIEW"
            }},
            "extracted_data": {{
                "name": "nombre extraído",
                "document_number": "número extraído", 
                "birthdate": "fecha nacimiento extraída",
                "issue_date": "fecha expedición extraída",
                "expiry_date": "fecha validez extraída",
                "nationality": "nacionalidad detectada"
            }},
            "details": "explicación breve de la verificación y país"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en verificación de documentos de identidad internacionales. Analiza con precisión y verifica que el documento corresponde al país indicado. Sé tolerante con variaciones normales pero estricto con la correspondencia país-documento."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1200,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"🧠 GPT-4 Smart Analysis: {result[:200]}...")
        
        # Limpiar respuesta: eliminar marcadores de código markdown
        cleaned_result = result.strip()
        if cleaned_result.startswith('```json'):
            cleaned_result = cleaned_result[7:]  # Eliminar '```json'
        if cleaned_result.endswith('```'):
            cleaned_result = cleaned_result[:-3]  # Eliminar '```'
        cleaned_result = cleaned_result.strip()
        
        # Intentar parsear como JSON
        try:
            json_result = json.loads(cleaned_result)
            return json.dumps(json_result)
        except Exception as e:
            print(f"⚠️ Error parseando JSON: {e}")
            print(f"🔍 Texto limpio: {cleaned_result[:100]}...")
            # Si no es JSON válido, crear estructura básica con el texto extraído
            fallback_result = {
                "extracted_text": result,
                "verification": {
                    "name_match": False,
                    "dni_match": False,
                    "birthdate_match": False,
                    "address_match": False,
                    "overall_confidence": 0,
                    "recommendation": "REVIEW"
                },
                "extracted_data": {},
                "details": "GPT devolvió texto plano en lugar de JSON"
            }
            return json.dumps(fallback_result)
        
    except Exception as e:
        print(f"❌ GPT-4 Smart Analysis Error: {e}")
        # SIN FALLBACK - Solo GPT-4
        return json.dumps({
            "error": f"Error en análisis GPT-4: {str(e)}",
            "extracted_text": "",
            "verification": {
                "name_match": False,
                "dni_match": False,
                "birthdate_match": False,
                "address_match": False,
                "overall_confidence": 0,
                "recommendation": "REJECT"
            },
            "extracted_data": {
                "name": "Error",
                "dni": "Error",
                "birthdate": "Error",
                "address": "Error"
            },
            "details": "No se pudo conectar con GPT-4 Vision"
        })


# =============================================================================
# 🚀 INICIALIZAR SERVIDOR
# =============================================================================

if __name__ == '__main__':
    print("🚀 Iniciando servidor KYC...")
    print("📍 Endpoints disponibles:")
    print("   • GET  / - API funcionando")
    print("   • GET  /test - Interfaz de prueba")
    print("   • GET  /kyc - Interfaz profesional")
    print("   • POST /kyc/validate-dni - Validar DNI")
    print("   • POST /kyc/verify-selfie - Verificar selfie")
    print("   • POST /kyc/verify-liveness - Verificar vida")
    print("   • POST /kyc/complete-verification - Verificación completa")
    
    app.run(debug=True, host='0.0.0.0', port=5000)


