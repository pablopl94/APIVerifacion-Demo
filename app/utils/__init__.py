"""
🔧 UTILIDADES COMPARTIDAS KYC
Funciones que usan los controladores
"""
import os
import uuid
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(override=True)

# Configurar OpenAI
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key) if api_key else None

# Configuración de archivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    """✅ Validar tipo de archivo permitido"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_and_compare_dni_with_gpt(image_path, user_data):
    """
    🧠 GPT-4 Vision: Análisis y comparación de DNI
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
