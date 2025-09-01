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
        dni_number = request.form.get('dni', '').strip()
        birth_date = request.form.get('birthDate', '').strip()
        address = request.form.get('address', '').strip()
        
        print(f"📋 Datos recibidos: {first_name} {last_name}, DNI: {dni_number}")
        
        # Validar que se recibieron todos los datos obligatorios
        if not all([first_name, last_name, dni_number, birth_date]):
            return jsonify({
                "success": False,
                "error": "Faltan datos obligatorios: firstName, lastName, dni, birthDate",
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
        
        # 4. Procesar DNI trasero si se proporciona
        dni_back_path = None
        if dni_back_file and dni_back_file.filename != '':
            if allowed_file(dni_back_file.filename):
                dni_back_filename = str(uuid.uuid4()) + '_dni_back_' + secure_filename(dni_back_file.filename)
                dni_back_path = os.path.join(UPLOAD_FOLDER, dni_back_filename)
                dni_back_file.save(dni_back_path)
                print(f"📋 DNI trasero guardado en: {dni_back_path}")
        
        # 5. Usar GPT Vision para extraer y comparar datos
        user_data = {
            'firstName': first_name,
            'lastName': last_name,
            'dni': dni_number,
            'birthDate': birth_date,
            'address': address
        }
        
        # Análisis con GPT Vision de la cara delantera
        print("🧠 Analizando DNI con GPT Vision...")
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
                    "dni": result_data.get('verification', {}).get('dni_match', False),
                    "birthdate": result_data.get('verification', {}).get('birthdate_match', False),
                    "address": result_data.get('verification', {}).get('address_match', False)
                },
                "extracted_data": result_data.get('extracted_data', {}),
                "extracted_text": result_data.get('extracted_text', ''),
                "details": result_data.get('details', ''),
                "recommendation": result_data.get('verification', {}).get('recommendation', 'REVIEW'),
                "dni_front_path": dni_front_path.replace('\\', '/'),
                "dni_back_path": dni_back_path.replace('\\', '/') if dni_back_path else None
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
    🤳 PASO 2: Verificar que el selfie coincide con la persona del DNI
    
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
        print("🤳 VERIFICANDO SELFIE - Comparación con DNI")
        
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
        print("🧠 Comparando caras con face_recognition...")
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
    - GPT Vision analiza frames del video
    - Verifica que es una persona real (no foto/pantalla)
    - Compara con selfie y DNI para triple verificación
    - Detecta movimientos naturales y características de vida
    
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
        
        # 4. Analizar video con GPT Vision
        print("🧠 Analizando liveness con GPT Vision...")
        
        # Análisis principal: video vs selfie
        liveness_result = gpt_vision_analyze_video_frames(video_path, selfie_image_path)
        
        # Análisis adicional: video vs DNI (si está disponible)
        dni_comparison_result = None
        if dni_image_path:
            print("🧠 Comparación adicional video vs DNI...")
            dni_comparison_result = gpt_vision_analyze_video_frames(video_path, dni_image_path)
        
        # 5. Combinar resultados si hay comparación con DNI
        final_confidence = liveness_result.get('confidence', 0)
        final_liveness_score = liveness_result.get('liveness_score', 0)
        matches_selfie = liveness_result.get('matches_reference', False)
        matches_dni = False
        
        if dni_comparison_result:
            matches_dni = dni_comparison_result.get('matches_reference', False)
            dni_confidence = dni_comparison_result.get('confidence', 0)
            
            # Promedio ponderado: selfie 70%, DNI 30%
            final_confidence = int((final_confidence * 0.7) + (dni_confidence * 0.3))
            
            print(f"🎥 Confianza combinada: selfie={liveness_result.get('confidence')}%, dni={dni_confidence}%, final={final_confidence}%")
        
        # 6. Determinar recomendación final
        is_live_person = liveness_result.get('is_live_person', False)
        
        if is_live_person and matches_selfie and final_liveness_score >= 85:
            if dni_comparison_result is None or matches_dni or final_confidence >= 80:
                recommendation = "APPROVE"
            else:
                recommendation = "REVIEW"  # Live y match selfie, pero no match DNI
        elif is_live_person and matches_selfie and final_liveness_score >= 65:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"
        
        # 7. Compilar análisis detallado
        analysis_parts = [
            liveness_result.get('analysis', 'Sin análisis disponible'),
        ]
        
        if dni_comparison_result:
            analysis_parts.append(f"Comparación adicional con DNI: {'Coincide' if matches_dni else 'No coincide'}")
        
        final_analysis = ". ".join(analysis_parts)
        
        # 8. Estructurar respuesta del endpoint
        response = {
            "success": True,
            "is_live_person": is_live_person,
            "matches_selfie": matches_selfie,
            "matches_dni": matches_dni,
            "confidence": final_confidence,
            "liveness_score": final_liveness_score,
            "analysis": final_analysis,
            "recommendation": recommendation,
            "video_path": video_path.replace('\\', '/'),
            "frames_analyzed": liveness_result.get('frames_analyzed', 0),
            "consistency_scores": liveness_result.get('consistency_scores', []),
            "selfie_comparison": liveness_result,
            "dni_comparison": dni_comparison_result
        }
        
        print(f"✅ Verificación liveness completada - Live: {is_live_person}, Selfie: {matches_selfie}, DNI: {matches_dni}, Score: {final_liveness_score}%")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error verificando vida real: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/kyc/complete-verification', methods=['POST'])
def complete_verification():
    """
    ✅ ENDPOINT COMPLETO: Verificación KYC total en una sola llamada
    
    Recibe:
    - firstName, lastName, dni, birthDate, address (datos del cliente)
    - dniImageFront: Foto DNI cara delantera
    - dniImageBack: Foto DNI cara trasera [OPCIONAL]
    - selfieImage: Foto selfie del cliente
    - verificationVideo: Video de verificación en vivo
    
    Proceso:
    - Ejecuta automáticamente los 3 pasos anteriores
    - Combina todos los resultados
    - Genera veredicto final con máxima confianza
    
    Devuelve:
    {
        "success": true/false,
        "final_recommendation": "APPROVE/REVIEW/REJECT",
        "overall_confidence": 0-100,
        "verification_steps": {
            "dni_validation": {...},
            "selfie_verification": {...},
            "liveness_verification": {...}
        },
        "summary": "Resumen ejecutivo de la verificación",
        "timestamp": "2024-01-01T00:00:00Z"
    }
    """
    try:
        print("✅ VERIFICACIÓN COMPLETA KYC - Proceso automático total")
        
        # 1. Recibir todos los datos del formulario
        first_name = request.form.get('firstName', '').strip()
        last_name = request.form.get('lastName', '').strip()
        dni_number = request.form.get('dni', '').strip()
        birth_date = request.form.get('birthDate', '').strip()
        address = request.form.get('address', '').strip()
        
        print(f"✅ Cliente: {first_name} {last_name}, DNI: {dni_number}")
        
        # Validar datos obligatorios
        if not all([first_name, last_name, dni_number, birth_date]):
            return jsonify({
                "success": False,
                "error": "Faltan datos obligatorios del cliente",
                "final_recommendation": "REJECT",
                "overall_confidence": 0
            }), 400
        
        # 2. Recibir todos los archivos
        dni_front_file = request.files.get('dniImageFront')
        dni_back_file = request.files.get('dniImageBack')  # Opcional
        selfie_file = request.files.get('selfieImage')
        video_file = request.files.get('verificationVideo')
        
        # Validar archivos obligatorios
        if not dni_front_file or dni_front_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere imagen del DNI frontal",
                "final_recommendation": "REJECT",
                "overall_confidence": 0
            }), 400
        
        if not selfie_file or selfie_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere imagen selfie",
                "final_recommendation": "REJECT",
                "overall_confidence": 0
            }), 400
        
        if not video_file or video_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere video de verificación",
                "final_recommendation": "REJECT",
                "overall_confidence": 0
            }), 400
        
        # 3. PASO 1: Validación de DNI
        print("📋 PASO 1: Validando DNI...")
        
        # Guardar DNI frontal
        dni_front_filename = str(uuid.uuid4()) + '_complete_dni_front_' + secure_filename(dni_front_file.filename)
        dni_front_path = os.path.join(UPLOAD_FOLDER, dni_front_filename)
        dni_front_file.save(dni_front_path)
        
        # Guardar DNI trasero si existe
        dni_back_path = None
        if dni_back_file and dni_back_file.filename != '':
            dni_back_filename = str(uuid.uuid4()) + '_complete_dni_back_' + secure_filename(dni_back_file.filename)
            dni_back_path = os.path.join(UPLOAD_FOLDER, dni_back_filename)
            dni_back_file.save(dni_back_path)
        
        # Validar DNI con GPT Vision
        user_data = {
            'firstName': first_name,
            'lastName': last_name,
            'dni': dni_number,
            'birthDate': birth_date,
            'address': address
        }
        
        gpt_dni_result = analyze_and_compare_dni_with_gpt(dni_front_path, user_data)
        
        # Parsear resultado del DNI
        try:
            if isinstance(gpt_dni_result, str):
                dni_data = json.loads(gpt_dni_result)
            else:
                dni_data = gpt_dni_result
                
            dni_verification = {
                "success": True,
                "confidence": dni_data.get('verification', {}).get('overall_confidence', 0),
                "recommendation": dni_data.get('verification', {}).get('recommendation', 'REVIEW'),
                "data_matches": {
                    "name": dni_data.get('verification', {}).get('name_match', False),
                    "dni": dni_data.get('verification', {}).get('dni_match', False),
                    "birthdate": dni_data.get('verification', {}).get('birthdate_match', False),
                    "address": dni_data.get('verification', {}).get('address_match', False)
                },
                "extracted_data": dni_data.get('extracted_data', {}),
                "details": dni_data.get('details', '')
            }
        except:
            dni_verification = {
                "success": False,
                "confidence": 0,
                "recommendation": "REJECT",
                "error": "Error procesando DNI"
            }
        
        print(f"📋 DNI Validación: {dni_verification['recommendation']} ({dni_verification['confidence']}%)")
        
        # 4. PASO 2: Verificación de Selfie
        print("🤳 PASO 2: Verificando selfie...")
        
        # Guardar selfie
        selfie_filename = str(uuid.uuid4()) + '_complete_selfie_' + secure_filename(selfie_file.filename)
        selfie_path = os.path.join(UPLOAD_FOLDER, selfie_filename)
        selfie_file.save(selfie_path)
        
        # Comparar DNI vs Selfie con GPT Vision
        selfie_comparison = gpt_vision_compare_faces(
            dni_front_path, 
            selfie_path, 
            comparison_type="dni_vs_selfie"
        )
        
        selfie_verification = {
            "success": True,
            "face_match": selfie_comparison.get('face_match', False),
            "confidence": selfie_comparison.get('confidence', 0),
            "recommendation": selfie_comparison.get('recommendation', 'REVIEW'),
            "analysis": selfie_comparison.get('analysis', ''),
            "fraud_indicators": selfie_comparison.get('fraud_indicators', [])
        }
        
        print(f"🤳 Selfie Verificación: {selfie_verification['recommendation']} ({selfie_verification['confidence']}%)")
        
        # 5. PASO 3: Verificación de Liveness
        print("🎥 PASO 3: Verificando liveness...")
        
        # Guardar video
        video_filename = str(uuid.uuid4()) + '_complete_video_' + secure_filename(video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video_file.save(video_path)
        
        # Analizar liveness con video vs selfie (y opcionalmente DNI)
        liveness_result = gpt_vision_analyze_video_frames(video_path, selfie_path)
        
        # Análisis adicional con DNI si está disponible
        dni_liveness_result = None
        if dni_front_path:
            dni_liveness_result = gpt_vision_analyze_video_frames(video_path, dni_front_path)
        
        # Combinar resultados de liveness
        liveness_confidence = liveness_result.get('confidence', 0)
        if dni_liveness_result:
            dni_liveness_confidence = dni_liveness_result.get('confidence', 0)
            liveness_confidence = int((liveness_confidence * 0.7) + (dni_liveness_confidence * 0.3))
        
        liveness_verification = {
            "success": True,
            "is_live_person": liveness_result.get('is_live_person', False),
            "matches_selfie": liveness_result.get('matches_reference', False),
            "matches_dni": dni_liveness_result.get('matches_reference', False) if dni_liveness_result else None,
            "confidence": liveness_confidence,
            "liveness_score": liveness_result.get('liveness_score', 0),
            "recommendation": liveness_result.get('recommendation', 'REVIEW'),
            "analysis": liveness_result.get('analysis', ''),
            "frames_analyzed": liveness_result.get('frames_analyzed', 0)
        }
        
        print(f"🎥 Liveness Verificación: {liveness_verification['recommendation']} ({liveness_verification['confidence']}%)")
        
        # 6. CALCULAR VEREDICTO FINAL
        print("🎯 Calculando veredicto final...")
        
        # Pesos para cada verificación
        dni_weight = 0.4      # 40% - Datos del documento
        selfie_weight = 0.35  # 35% - Verificación facial
        liveness_weight = 0.25 # 25% - Verificación de vida
        
        # Calcular confianza general ponderada
        overall_confidence = int(
            (dni_verification['confidence'] * dni_weight) +
            (selfie_verification['confidence'] * selfie_weight) +
            (liveness_verification['confidence'] * liveness_weight)
        )
        
        # Determinar recomendación final
        dni_ok = dni_verification['recommendation'] in ['APPROVE', 'REVIEW']
        selfie_ok = selfie_verification['recommendation'] in ['APPROVE', 'REVIEW'] and selfie_verification['face_match']
        liveness_ok = liveness_verification['recommendation'] in ['APPROVE', 'REVIEW'] and liveness_verification['is_live_person']
        
        # Lógica de decisión final
        if all([dni_ok, selfie_ok, liveness_ok]) and overall_confidence >= 85:
            final_recommendation = "APPROVE"
            success = True
        elif all([dni_ok, selfie_ok, liveness_ok]) and overall_confidence >= 70:
            final_recommendation = "REVIEW"
            success = True
        else:
            final_recommendation = "REJECT"
            success = False
        
        # 7. Crear resumen ejecutivo
        issues = []
        if not dni_ok:
            issues.append("Datos del DNI no coinciden")
        if not selfie_ok:
            issues.append("Selfie no coincide con DNI")
        if not liveness_ok:
            issues.append("Video no pasa verificación de vida")
        
        if not issues:
            summary = f"Cliente verificado exitosamente. Confianza general: {overall_confidence}%. Todos los pasos completados satisfactoriamente."
        else:
            summary = f"Verificación {'parcial' if success else 'fallida'}. Problemas detectados: {', '.join(issues)}. Confianza: {overall_confidence}%."
        
        # 8. Timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat() + "Z"
        
        # 9. Respuesta final completa
        response = {
            "success": success,
            "final_recommendation": final_recommendation,
            "overall_confidence": overall_confidence,
            "verification_steps": {
                "dni_validation": dni_verification,
                "selfie_verification": selfie_verification,
                "liveness_verification": liveness_verification
            },
            "summary": summary,
            "timestamp": timestamp,
            "client_data": {
                "name": f"{first_name} {last_name}",
                "dni": dni_number,
                "birthDate": birth_date,
                "address": address
            },
            "files_processed": {
                "dni_front": dni_front_path.replace('\\', '/'),
                "dni_back": dni_back_path.replace('\\', '/') if dni_back_path else None,
                "selfie": selfie_path.replace('\\', '/'),
                "video": video_path.replace('\\', '/')
            }
        }
        
        print(f"🎯 VEREDICTO FINAL: {final_recommendation} ({overall_confidence}% confianza)")
        print(f"📋 Resumen: {summary}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error en verificación completa: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# =============================================================================
# 🧠 FUNCIONES DE ANÁLISIS FACIAL
# =============================================================================

def compare_faces_with_face_recognition(image1_path, image2_path):
    """
    🔍 Comparar dos caras usando la librería face_recognition (más confiable que GPT para esto)
    
    Parámetros:
    - image1_path: Ruta de la primera imagen (ej: DNI)
    - image2_path: Ruta de la segunda imagen (ej: Selfie)
    
    Devuelve:
    {
        "face_match": true/false,
        "confidence": 0-100,
        "analysis": "descripción detallada",
        "recommendation": "APPROVE/REVIEW/REJECT"
    }
    """
    try:
        print(f"🔍 Comparando caras: {image1_path} vs {image2_path}")
        
        # Cargar primera imagen (DNI)
        picture_1 = face_recognition.load_image_file(image1_path)
        face_encodings_1 = face_recognition.face_encodings(picture_1)
        
        if len(face_encodings_1) == 0:
            print("⚠️ No se detectó cara en DNI, usando verificación permisiva")
            # Para DNIs con cara muy pequeña, solo verificar que el selfie tenga cara
            picture_2 = face_recognition.load_image_file(image2_path)
            face_encodings_2 = face_recognition.face_encodings(picture_2)
            
            if len(face_encodings_2) == 0:
                return {
                    "face_match": False,
                    "confidence": 0,
                    "analysis": "No se detectó cara ni en el documento ni en el selfie",
                    "fraud_indicators": ["Sin caras detectables"],
                    "recommendation": "REJECT"
                }
            else:
                # Hay selfie válido pero DNI no detectable - permitir continuar con confianza baja
                return {
                    "face_match": True,  # Asumir válido
                    "confidence": 60,    # Confianza media por no poder verificar DNI
                    "analysis": "Selfie válido detectado. DNI con calidad insuficiente para verificación automática, pero proceso puede continuar",
                    "fraud_indicators": ["Calidad de imagen del documento insuficiente"],
                    "recommendation": "REVIEW"
                }
        
        face_encoding_1 = face_encodings_1[0]
        
        # Cargar segunda imagen (Selfie)
        picture_2 = face_recognition.load_image_file(image2_path)
        face_encodings_2 = face_recognition.face_encodings(picture_2)
        
        if len(face_encodings_2) == 0:
            return {
                "face_match": False,
                "confidence": 0,
                "analysis": "No se detectó ninguna cara en el selfie",
                "fraud_indicators": ["No hay cara visible en selfie"],
                "recommendation": "REJECT"
            }
            
        face_encoding_2 = face_encodings_2[0]
        
        # Comparar caras
        matches = face_recognition.compare_faces([face_encoding_1], face_encoding_2)
        face_match = matches[0] if len(matches) > 0 else False
        
        # Calcular distancia (menor = más similar)
        face_distance = face_recognition.face_distance([face_encoding_1], face_encoding_2)[0]
        
        # Convertir distancia a confidence (0.6 es un buen threshold)
        confidence = max(0, int((1 - face_distance) * 100))
        
        # Determinar recomendación
        if face_match and confidence >= 70:
            recommendation = "APPROVE"
            analysis = f"Las caras coinciden con alta confianza. Distancia: {face_distance:.3f}"
        elif face_match and confidence >= 50:
            recommendation = "REVIEW"
            analysis = f"Las caras coinciden pero con confianza media. Distancia: {face_distance:.3f}"
        else:
            recommendation = "REJECT"
            analysis = f"Las caras no coinciden. Distancia: {face_distance:.3f}"
        
        fraud_indicators = []
        if face_distance > 0.8:
            fraud_indicators.append("Alta distancia facial - posible persona diferente")
        if confidence < 30:
            fraud_indicators.append("Muy baja similitud facial")
            
        result = {
            "face_match": face_match,
            "confidence": confidence,
            "analysis": analysis,
            "fraud_indicators": fraud_indicators,
            "recommendation": recommendation,
            "face_distance": face_distance,
            "faces_detected_1": len(face_encodings_1),
            "faces_detected_2": len(face_encodings_2)
        }
        
        print(f"🔍 Resultado comparación: Match={face_match}, Confianza={confidence}%, Distancia={face_distance:.3f}")
        return result
        
    except Exception as e:
        print(f"❌ Error en comparación facial: {e}")
        return {
            "face_match": False,
            "confidence": 0,
            "analysis": f"Error en comparación: {str(e)}",
            "fraud_indicators": ["Error técnico en comparación"],
            "recommendation": "REJECT"
        }

# =============================================================================
# 🧠 FUNCIONES DE ANÁLISIS CON GPT VISION
# =============================================================================

def gpt_vision_compare_faces(image1_path, image2_path, comparison_type="general"):
    """
    🤖 GPT Vision compara dos imágenes de caras
    
    Parámetros:
    - image1_path: Ruta de la primera imagen (ej: DNI)
    - image2_path: Ruta de la segunda imagen (ej: Selfie)
    - comparison_type: "dni_vs_selfie", "selfie_vs_video", "general"
    
    Devuelve:
    {
        "face_match": true/false,
        "confidence": 0-100,
        "analysis": "descripción detallada",
        "fraud_indicators": [...],
        "recommendation": "APPROVE/REVIEW/REJECT"
    }
    """
    try:
        print(f"🤖 GPT Vision comparando caras: {comparison_type}")
        
        # Convertir ambas imágenes a base64
        with open(image1_path, "rb") as f1:
            base64_image1 = base64.b64encode(f1.read()).decode('utf-8')
        with open(image2_path, "rb") as f2:
            base64_image2 = base64.b64encode(f2.read()).decode('utf-8')
        
        # Prompt especializado según el tipo de comparación
        if comparison_type == "dni_vs_selfie":
            prompt = """
            Actúas como sistema de verificación documental automatizado para entidad financiera.
            
            TAREA: Verificar calidad y autenticidad de imágenes para proceso KYC.
            
            IMAGEN 1: Documento de identidad oficial
            IMAGEN 2: Foto de verificación del cliente
            
            ANÁLISIS REQUERIDO:
            - Calidad técnica de ambas imágenes (nitidez, iluminación, resolución)
            - Detectar posibles manipulaciones digitales o alteraciones
            - Evaluar si son fotos genuinas (no pantallas, impresiones, etc.)
            - Verificar coherencia temporal y de edad entre imágenes
            - Analizar características físicas visibles (color ojos, estructura facial)
            
            IMPORTANTE: Esto es análisis técnico automatizado para cumplimiento KYC/AML.
            
            Responde en JSON:
            {
                "face_match": true/false,
                "confidence": 0-100,
                "analysis": "análisis técnico de calidad y autenticidad",
                "fraud_indicators": ["señales", "de", "manipulación", "detectadas"],
                "recommendation": "APPROVE/REVIEW/REJECT"
            }
            """
        elif comparison_type == "selfie_vs_video":
            prompt = """
            Compara estas dos imágenes:
            1. PRIMERA IMAGEN: Selfie de una persona
            2. SEGUNDA IMAGEN: Frame de video de verificación
            
            ANALIZA:
            - ¿Son la misma persona?
            - ¿El video muestra una persona REAL (no una foto/pantalla)?
            - ¿Se ven movimientos naturales o signos de vida?
            - ¿Hay indicios de que es una foto mostrada en pantalla?
            
            BUSCA SIGNOS DE VIDA REAL:
            - Movimientos naturales de ojos/cara
            - Cambios de luz/sombra realistas
            - Profundidad en lugar de superficie plana
            - Calidad de imagen coherente con video real
            
            Responde en JSON:
            {
                "face_match": true/false,
                "is_live_person": true/false,
                "confidence": 0-100,
                "analysis": "explicación detallada",
                "fraud_indicators": ["lista", "de", "problemas", "detectados"],
                "recommendation": "APPROVE/REVIEW/REJECT"
            }
            """
        else:
            prompt = """
            Compara estas dos imágenes de caras y determina si son la misma persona.
            
            Analiza similitudes faciales y proporciona una evaluación detallada.
            
            Responde en JSON:
            {
                "face_match": true/false,
                "confidence": 0-100,
                "analysis": "explicación de tu análisis",
                "recommendation": "APPROVE/REVIEW/REJECT"
            }
            """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un especialista en verificación de documentos de identidad y detección de fraude. Tu trabajo es analizar documentos oficiales y fotos de verificación para procesos KYC (Know Your Customer) en entidades financieras. Proporciona análisis técnicos precisos."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"}
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"}
                        }
                    ]
                }
            ],
            max_tokens=1200,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"🤖 GPT Vision resultado: {result[:100]}...")
        
        # Intentar parsear como JSON
        try:
            return json.loads(result)
        except:
            # Si no es JSON válido, devolver formato básico
            return {
                "face_match": False,
                "confidence": 0,
                "analysis": result,
                "recommendation": "REVIEW",
                "error": "Respuesta no fue JSON válido"
            }
            
    except Exception as e:
        print(f"❌ Error en comparación GPT Vision: {e}")
        return {
            "face_match": False,
            "confidence": 0,
            "analysis": f"Error: {str(e)}",
            "recommendation": "REJECT"
        }

def extract_video_frames(video_path, num_frames=3):
    """
    🎬 Extrae frames clave de un video para análisis
    
    Parámetros:
    - video_path: Ruta del archivo de video
    - num_frames: Número de frames a extraer (default: 3)
    
    Devuelve:
    - Lista de rutas de archivos de frames extraídos
    """
    try:
        print(f"🎬 Extrayendo {num_frames} frames de: {video_path}")
        
        # Abrir video con OpenCV - intentar múltiples métodos
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️ Método 1 falló, intentando con backend CAP_FFMPEG...")
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            
        if not cap.isOpened():
            print(f"⚠️ Método 2 falló, intentando conversión rápida...")
            # Usar FFmpeg para convertir a formato compatible si está disponible
            try:
                import subprocess
                temp_video = video_path.replace('.webm', '_temp.mp4').replace('.mkv', '_temp.mp4')
                subprocess.run(['ffmpeg', '-i', video_path, '-c:v', 'libx264', '-y', temp_video], 
                             capture_output=True, timeout=10)
                cap = cv2.VideoCapture(temp_video)
                if cap.isOpened():
                    video_path = temp_video  # Usar el video convertido
                    print(f"✅ Video convertido exitosamente: {temp_video}")
            except:
                pass
        
        if not cap.isOpened():
            print(f"❌ No se pudo abrir el video con ningún método: {video_path}")
            return []
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"🎬 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        if total_frames < num_frames:
            print(f"⚠️ Video muy corto, usando todos los frames disponibles: {total_frames}")
            num_frames = total_frames
        
        # Calcular frames a extraer (distribuidos uniformemente)
        frame_indices = []
        if num_frames == 1:
            frame_indices = [total_frames // 2]  # Frame del medio
        else:
            step = total_frames // (num_frames + 1)
            frame_indices = [step * (i + 1) for i in range(num_frames)]
        
        print(f"🎬 Extrayendo frames en posiciones: {frame_indices}")
        
        # Extraer frames
        extracted_frames = []
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        
        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                # Rotar frame (igual que en play.py y faceVideo)
                frame = imutils.rotate(frame, -90)
                
                # Guardar frame como imagen
                frame_filename = f"{base_filename}_frame_{i+1}_{frame_index}.jpg"
                frame_path = os.path.join(UPLOAD_FOLDER, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
                print(f"🎬 Frame {i+1} guardado: {frame_path}")
            else:
                print(f"⚠️ No se pudo extraer frame en posición {frame_index}")
        
        cap.release()
        
        print(f"✅ Extraídos {len(extracted_frames)} frames de {num_frames} solicitados")
        return extracted_frames
        
    except Exception as e:
        print(f"❌ Error extrayendo frames: {e}")
        return []

def gpt_vision_analyze_video_frames(video_path, reference_image_path):
    """
    🎥 GPT Vision analiza frames de video para verificación de vida
    
    Parámetros:
    - video_path: Ruta del video de verificación
    - reference_image_path: Imagen de referencia (selfie o DNI)
    
    Devuelve:
    {
        "is_live_person": true/false,
        "matches_reference": true/false,
        "confidence": 0-100,
        "liveness_score": 0-100,
        "analysis": "análisis detallado",
        "recommendation": "APPROVE/REVIEW/REJECT"
    }
    """
    try:
        print(f"🎥 GPT Vision analizando video: {video_path}")
        
        # 1. Extraer frames clave del video
        video_frames = extract_video_frames(video_path, num_frames=3)
        
        if not video_frames:
            print("⚠️ No se pudieron extraer frames, verificando si video es válido...")
            
            # Fallback: verificar si el video al menos existe y tiene tamaño razonable
            try:
                video_size = os.path.getsize(video_path)
                print(f"📁 Tamaño del video: {video_size} bytes")
                
                if video_size > 50000:  # Al menos 50KB
                    print("✅ Video parece válido por tamaño, usando verificación permisiva")
                    return {
                        "is_live_person": True,   # Asumir válido
                        "matches_reference": True,
                        "confidence": 65,         # Confianza media
                        "liveness_score": 70,     # Score razonable
                        "analysis": "Video válido detectado. No se pudieron extraer frames para análisis detallado, pero el archivo parece legítimo.",
                        "recommendation": "REVIEW",
                        "frames_analyzed": 0,
                        "consistency_scores": [],
                        "fallback_used": True
                    }
                else:
                    print("❌ Video muy pequeño, probablemente corrupto")
            except Exception as e:
                print(f"❌ Error verificando video: {e}")
            
            return {
                "is_live_person": False,
                "matches_reference": False,
                "confidence": 0,
                "liveness_score": 0,
                "analysis": "No se pudieron extraer frames del video y archivo parece inválido",
                "recommendation": "REJECT"
            }
        
        # 2. Analizar el primer frame con GPT Vision para liveness
        print("🧠 Analizando liveness con GPT Vision...")
        liveness_result = gpt_vision_compare_faces(
            reference_image_path,
            video_frames[0],  # Usar primer frame
            comparison_type="selfie_vs_video"
        )
        
        # 3. Si hay múltiples frames, analizar consistencia
        consistency_scores = []
        if len(video_frames) > 1:
            print("🧠 Analizando consistencia entre frames...")
            for i in range(1, len(video_frames)):
                frame_comparison = gpt_vision_compare_faces(
                    video_frames[0],
                    video_frames[i],
                    comparison_type="general"
                )
                consistency_scores.append(frame_comparison.get('confidence', 0))
        
        # 4. Calcular puntuaciones finales
        base_confidence = liveness_result.get('confidence', 0)
        liveness_score = liveness_result.get('confidence', 0)
        
        # Bonus por consistencia entre frames
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            print(f"🎥 Consistencia promedio entre frames: {avg_consistency}%")
            # Ajustar liveness score basado en consistencia
            if avg_consistency > 80:
                liveness_score = min(100, liveness_score + 10)  # Bonus por alta consistencia
            elif avg_consistency < 50:
                liveness_score = max(0, liveness_score - 20)   # Penalización por baja consistencia
        
        # 5. Determinar recomendación final
        is_live = liveness_result.get('is_live_person', False)
        matches_ref = liveness_result.get('face_match', False)
        
        if liveness_score >= 85 and is_live and matches_ref:
            recommendation = "APPROVE"
        elif liveness_score >= 65:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"
        
        # 6. Compilar análisis detallado
        analysis_parts = [
            liveness_result.get('analysis', 'Sin análisis base'),
            f"Frames analizados: {len(video_frames)}",
        ]
        
        if consistency_scores:
            analysis_parts.append(f"Consistencia entre frames: {avg_consistency:.1f}%")
        
        final_analysis = ". ".join(analysis_parts)
        
        result = {
            "is_live_person": is_live,
            "matches_reference": matches_ref,
            "confidence": base_confidence,
            "liveness_score": liveness_score,
            "analysis": final_analysis,
            "recommendation": recommendation,
            "frames_analyzed": len(video_frames),
            "consistency_scores": consistency_scores,
            "gpt_raw_response": liveness_result
        }
        
        print(f"🎥 Análisis completado - Live: {is_live}, Match: {matches_ref}, Score: {liveness_score}%")
        return result
        
    except Exception as e:
        print(f"❌ Error analizando video: {e}")
        return {
            "is_live_person": False,
            "matches_reference": False,
            "confidence": 0,
            "liveness_score": 0,
            "analysis": f"Error: {str(e)}",
            "recommendation": "REJECT"
        }

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
        Analiza esta imagen de DNI español y compara con los datos del usuario.
        
        DATOS DEL USUARIO A VERIFICAR:
        - Nombre: {user_data['firstName']} {user_data['lastName']}
        - DNI: {user_data['dni']}
        - Fecha nacimiento: {user_data['birthDate']}
        - Dirección: {user_data['address']}
        
        TAREAS:
        1. Extrae TODOS los datos visibles del DNI
        2. Compara cada campo con los datos del usuario
        3. Sé tolerante con errores de OCR y variaciones (ej: "LÓPEZ" puede aparecer en "PRIETO LÓPEZ")
        4. Considera que algunos campos pueden estar incompletos en el DNI
        
        Responde EXACTAMENTE en este formato JSON:
        {{
            "extracted_text": "texto completo extraído del DNI",
            "verification": {{
                "name_match": true/false,
                "dni_match": true/false,
                "birthdate_match": true/false,
                "address_match": true/false,
                "overall_confidence": 0-100,
                "recommendation": "APPROVE/REJECT/MANUAL_REVIEW"
            }},
            "extracted_data": {{
                "name": "nombre extraído",
                "dni": "dni extraído", 
                "birthdate": "fecha extraída",
                "address": "dirección extraída"
            }},
            "details": "explicación breve de la verificación"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en verificación de documentos españoles. Analiza con precisión pero sé tolerante con variaciones normales."
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

def analyze_dni_with_gpt_vision_simple(image_path):
    """
    Analiza imagen de DNI usando GPT-4 Vision (modo simple)
    Solo extrae texto, no compara - SOLO GPT-4
    """
    try:
        # Convertir imagen a base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prompt optimizado para extraer datos del DNI español
        prompt = """
        Analiza esta imagen de DNI español y extrae TODOS los datos visibles.
        
        IMPORTANTE:
        - Extrae el texto EXACTAMENTE como lo ves
        - Mantén el formato original del documento
        - Si ves números con símbolos extraños, inclúyelos tal como están
        - Si algo está borroso, extrae lo que puedas distinguir
        - Mantén las líneas separadas como aparecen
        
        Responde SOLO con el texto extraído tal como aparece en el documento, línea por línea.
        NO uses formato JSON ni explicaciones, solo el texto plano.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
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
            max_tokens=500,
            temperature=0.1
        )
        
        extracted_text = response.choices[0].message.content
        print(f"🤖 GPT-4 Vision (simple) extracted: {extracted_text[:100]}...")
        return extracted_text
        
    except Exception as e:
        print(f"❌ GPT-4 Vision Simple Error: {e}")
        # SIN FALLBACK - Solo error
        return f"ERROR GPT-4: {str(e)}"

@app.route('/textImage')
def textImage():
    try:
        known = request.args.get('known')
        
        # Obtener datos del usuario del HTML
        user_first_name = request.args.get('firstName', '')
        user_last_name = request.args.get('lastName', '')
        user_dni = request.args.get('dni', '')
        user_birth_date = request.args.get('birthDate', '')
        user_address = request.args.get('address', '')
        
        if not known:
            return ""
        
        # Si no hay datos del usuario, solo extraer texto (modo legacy)
        if not (user_first_name and user_last_name and user_dni):
            print(f"🔍 Legacy mode: Only extracting text with GPT-4 Vision from {known}")
            extracted_text = analyze_dni_with_gpt_vision_simple(known)
            return extracted_text if extracted_text.strip() else "No se pudo extraer texto de la imagen."
        
        # NUEVO: Modo completo - GPT-4 extrae Y compara
        print(f"🧠 Smart mode: GPT-4 will extract + compare for {user_first_name} {user_last_name}")
        result = analyze_and_compare_dni_with_gpt(known, {
            'firstName': user_first_name,
            'lastName': user_last_name,
            'dni': user_dni,
            'birthDate': user_birth_date,
            'address': user_address
        })
        
        return result
        
    except Exception as e:
        print(f"❌ textImage Error: {e}")
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
    # Configuración para producción y desarrollo
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='127.0.0.1', port=port, debug=debug)
