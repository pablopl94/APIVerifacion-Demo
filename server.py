"""
🚀 SERVIDOR PRINCIPAL KYC API
API completa para verificación KYC con frontend/backend separation
"""
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Importar blueprints de la arquitectura existente
from app.controllers.base_controller import base_bp
from app.controllers.kyc_controller import kyc_bp
from app.controllers.upload_controller import upload_bp
from app.database.connection import create_tables, test_connection

# Cargar variables de entorno
load_dotenv(override=True)

# Configurar OpenAI
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("⚠️ OPENAI_API_KEY no encontrada en .env")
    print("💡 Crea un archivo .env con: OPENAI_API_KEY=tu_api_key")
else:
    print(f"🔑 API Key cargada: {api_key[:15]}...{api_key[-4:]}")

client = OpenAI(api_key=api_key) if api_key else None

# Crear aplicación Flask
app = Flask(__name__, 
           template_folder="templates",
           static_folder="static",
           static_url_path="/static")

# Configurar CORS para permitir peticiones desde Flutter Web
CORS(app, 
     origins=['http://localhost:65014', 'http://localhost:3000', 'http://127.0.0.1:65014', 'http://localhost:59177', 'http://localhost:60553'],
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Importar utilidades compartidas
from app.utils import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

# Crear carpeta de uploads
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =============================================================================
# 🔧 FUNCIONES UTILITARIAS (que usan los controladores)
# =============================================================================

def allowed_file(filename):
    """✅ Validar tipo de archivo permitido"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_and_compare_dni_with_gpt(image_path, user_data):
    """
    🧠 GPT-4 Vision: Análisis y comparación de DNI
    Función que faltaba en el sistema - ahora implementada
    """
    if not client:
        return json.dumps({
            "error": "OpenAI API key no configurada",
            "verification": {
                "name_match": False,
                "document_number_match": False,
                "birthdate_match": False,
                "issue_date_match": False,
                "expiry_date_match": False,
                "country_verification": False,
                "overall_confidence": 0,
                "recommendation": "REJECT"
            },
            "extracted_data": {},
            "details": "Configurar OPENAI_API_KEY en .env"
        })
    
    try:
        print(f"🧠 Analizando DNI con GPT-4 Vision: {image_path}")
        
        # Convertir imagen a base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prompt optimizado para análisis completo
        prompt = f"""
        Analiza esta imagen de documento de identidad y compárala con los datos del usuario.
        
        DATOS DEL USUARIO A VERIFICAR:
        - Nombre: {user_data['firstName']} {user_data['lastName']}
        - Número documento: {user_data['documentNumber']}
        - Nacionalidad: {user_data['nationality']}
        - Fecha nacimiento: {user_data['birthDate']} (DD/MM/YYYY)
        - Fecha expedición: {user_data['issueDate']} (DD/MM/YYYY)
        - Fecha validez: {user_data['expiryDate']} (DD/MM/YYYY)
        
        INSTRUCCIONES:
        1. Identifica el tipo de documento (DNI, CNI, ID, etc.)
        2. Verifica que corresponde al país indicado ({user_data['nationality']})
        3. Extrae todos los datos visibles
        4. Compara cada campo (sé tolerante con errores de OCR menores)
        5. Calcula confianza global (0-100%)
        
        DOCUMENTOS POR PAÍS:
        - ESP: DNI español
        - FRA: Carte Nationale d'Identité
        - ITA: Carta d'Identità
        - DEU: Personalausweis
        - USA: Driver's License/State ID
        
        Responde SOLO en formato JSON válido:
        {{
            "extracted_text": "texto completo visible",
            "document_analysis": {{
                "document_type": "tipo de documento",
                "country_match": true/false,
                "is_valid_document": true/false
            }},
            "extracted_data": {{
                "name": "nombre completo extraído",
                "document_number": "número extraído",
                "birthdate": "DD/MM/YYYY",
                "issue_date": "DD/MM/YYYY", 
                "expiry_date": "DD/MM/YYYY",
                "nationality": "código país"
            }},
            "data_matches": {{
                "name": true/false,
                "document_number": true/false,
                "birthdate": true/false,
                "issue_date": true/false,
                "expiry_date": true/false,
                "country": true/false
            }},
            "verification": {{
                "overall_confidence": 0-100,
                "recommendation": "APPROVE/REVIEW/REJECT"
            }},
            "details": "explicación de la verificación"
        }}
        """
        
        # Llamada a GPT-4 Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "Eres un experto en verificación de documentos de identidad. Analiza con precisión y responde SOLO en JSON válido."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Limpiar respuesta (eliminar markdown si existe)
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        # Validar que es JSON válido
        try:
            json_result = json.loads(result_text)
            print(f"✅ GPT-4 análisis exitoso - Confianza: {json_result.get('verification', {}).get('overall_confidence', 0)}%")
            return json.dumps(json_result, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            print(f"⚠️ GPT devolvió texto no-JSON: {e}")
            # Crear respuesta de fallback estructurada
            fallback = {
                "extracted_text": result_text,
                "document_analysis": {
                    "document_type": "No identificado",
                    "country_match": False,
                    "is_valid_document": False
                },
                "extracted_data": {
                    "name": "",
                    "document_number": "",
                    "birthdate": "",
                    "issue_date": "",
                    "expiry_date": "",
                    "nationality": ""
                },
                "data_matches": {
                    "name": False,
                    "document_number": False, 
                    "birthdate": False,
                    "issue_date": False,
                    "expiry_date": False,
                    "country": False
                },
                "verification": {
                    "overall_confidence": 0,
                    "recommendation": "REVIEW"
                },
                "details": f"GPT devolvió formato inesperado: {str(e)}"
            }
            return json.dumps(fallback, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Error en GPT-4 Vision: {e}")
        error_response = {
            "error": str(e),
            "extracted_text": "",
            "document_analysis": {
                "document_type": "Error",
                "country_match": False,
                "is_valid_document": False
            },
            "extracted_data": {
                "name": "",
                "document_number": "",
                "birthdate": "",
                "issue_date": "",
                "expiry_date": "",
                "nationality": ""
            },
            "data_matches": {
                "name": False,
                "document_number": False,
                "birthdate": False,
                "issue_date": False,
                "expiry_date": False,
                "country": False
            },
            "verification": {
                "overall_confidence": 0,
                "recommendation": "REJECT"
            },
            "details": f"Error GPT-4: {str(e)}"
        }
        return json.dumps(error_response, ensure_ascii=False)

def compare_faces_with_face_recognition(dni_path, selfie_path):
    """
    🤳 Delegación a SelfieVerificationService
    Los controladores usan esta función, pero delega al servicio real
    """
    try:
        from app.services.selfie_verification_service import SelfieVerificationService
        service = SelfieVerificationService()
        return service._compare_faces_deepface(dni_path, selfie_path)
    except Exception as e:
        print(f"❌ Error en comparación facial: {e}")
        return {
            "face_match": False,
            "confidence": 0,
            "analysis": f"Error técnico: {str(e)}",
            "fraud_indicators": ["Error de sistema"],
            "recommendation": "REJECT"
        }

def analyze_video_liveness_with_deepface(video_path, reference_path):
    """
    🎥 Delegación a VideoVerificationService  
    Los controladores usan esta función, pero delega al servicio real
    """
    try:
        from app.services.video_verification_service import VideoVerificationService
        service = VideoVerificationService()
        return service._analyze_video_liveness(video_path, reference_path)
    except Exception as e:
        print(f"❌ Error en análisis de video: {e}")
        return {
            "is_live_person": False,
            "matches_reference": False,
            "confidence": 0,
            "frames_analyzed": 0,
            "technical_details": {"error": str(e)}
        }

# =============================================================================
# 📡 REGISTRAR BLUEPRINTS (CONTROLADORES)
# =============================================================================

# Registrar todos los controladores existentes
app.register_blueprint(base_bp)           # Rutas: /, /test
app.register_blueprint(kyc_bp)            # Rutas: /kyc/*
app.register_blueprint(upload_bp)         # Rutas: /upload, /uploads/<file>

# =============================================================================
# 🏥 ENDPOINTS DE SALUD Y ESTADO
# =============================================================================

@app.route('/health')
def health_check():
    """🏥 Endpoint de salud para verificar que todo funciona"""
    try:
        # Verificar conexión a BD
        db_status = test_connection()
        
        # Verificar configuración OpenAI
        openai_status = client is not None
        
        # Verificar carpeta uploads
        uploads_status = os.path.exists(UPLOAD_FOLDER)
        
        health_data = {
            "status": "healthy" if all([db_status, openai_status, uploads_status]) else "unhealthy",
            "timestamp": "2025-09-03T00:00:00Z",
            "checks": {
                "database": "connected" if db_status else "disconnected",
                "openai": "configured" if openai_status else "not_configured", 
                "uploads": "ready" if uploads_status else "not_ready"
            },
            "version": "2.0.0",
            "endpoints": {
                "dni_validation": "/kyc/validate-dni",
                "selfie_verification": "/kyc/verify-selfie", 
                "liveness_verification": "/kyc/verify-liveness",
                "user_status": "/kyc/user-status/<document>",
                "complete_verification": "/kyc/complete-verification/<document>"
            }
        }
        
        return jsonify(health_data), 200 if health_data["status"] == "healthy" else 503
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": "2025-09-03T00:00:00Z"
        }), 500

@app.route('/api/info')
def api_info():
    """📋 Información completa de la API"""
    return jsonify({
        "name": "KYC Verification API",
        "version": "2.0.0",
        "description": "API completa para verificación KYC con DNI, Selfie y Video Liveness",
        "architecture": "Flask + SQLAlchemy + GPT-4 Vision + DeepFace",
        "capabilities": [
            "DNI OCR con GPT-4 Vision",
            "Verificación facial con DeepFace", 
            "Análisis de video liveness",
            "Base de datos MySQL persistente",
            "Flujo completo KYC automatizado"
        ],
        "endpoints": {
            "POST /kyc/validate-dni": "Paso 1: Validar datos vs DNI",
            "POST /kyc/verify-selfie": "Paso 2: Verificar selfie vs DNI",
            "POST /kyc/verify-liveness": "Paso 3: Verificar video liveness",
            "GET /kyc/user-status/<doc>": "Consultar estado usuario",
            "GET /kyc/complete-verification/<doc>": "Resumen completo"
        },
        "frontend_integration": {
            "test_interface": "/test",
            "upload_files": "POST /upload",
            "serve_files": "GET /uploads/<filename>"
        }
    })

# =============================================================================
# 🚀 INICIALIZACIÓN
# =============================================================================

def initialize_app():
    """🚀 Inicializar aplicación completa"""
    print("🏗️ Inicializando sistema KYC...")
    
    # 1. Verificar base de datos
    print("🗄️ Verificando base de datos...")
    if test_connection():
        print("✅ Conexión a MySQL exitosa")
        create_tables()
        print("✅ Tablas verificadas/creadas")
    else:
        print("⚠️ Problema con base de datos - API funcionará en modo limitado")
    
    # 2. Verificar OpenAI
    if client:
        print("✅ OpenAI GPT-4 Vision configurado")
    else:
        print("⚠️ OpenAI no configurado - crear archivo .env con OPENAI_API_KEY")
    
    # 3. Verificar carpetas
    print(f"📁 Carpeta uploads: {UPLOAD_FOLDER}")
    
    print("✅ Inicialización completa")

if __name__ == '__main__':
    initialize_app()
    
    print("\n🚀 SERVIDOR KYC API INICIADO")
    print("="*50)
    print("📍 ENDPOINTS PRINCIPALES:")
    print("   🏠 GET  /                     - Estado de la API")
    print("   🧪 GET  /test                 - Interfaz de pruebas")
    print("   🏥 GET  /health               - Health check")
    print("   📋 GET  /api/info             - Info de la API")
    print("")
    print("📍 ENDPOINTS KYC:")
    print("   📄 POST /kyc/validate-dni     - Validar DNI (Paso 1)")
    print("   🤳 POST /kyc/verify-selfie    - Verificar selfie (Paso 2)") 
    print("   🎥 POST /kyc/verify-liveness  - Verificar liveness (Paso 3)")
    print("   📊 GET  /kyc/user-status/<doc> - Estado usuario")
    print("   🏆 GET  /kyc/complete-verification/<doc> - Resumen final")
    print("")
    print("📍 UTILIDADES:")
    print("   📤 POST /upload               - Subir archivos")
    print("   📥 GET  /uploads/<file>       - Descargar archivos")
    print("="*50)
    
    # Ejecutar servidor
    app.run(
        debug=True,
        host='0.0.0.0', 
        port=5000,
        use_reloader=False  # Evita doble inicialización
    )
