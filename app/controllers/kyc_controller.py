"""
🏗️ KYC Controller
Endpoints para verificación KYC usando la nueva arquitectura por capas
"""

from flask import Blueprint, request, jsonify
import os
import json
import uuid
from werkzeug.utils import secure_filename

# Importar servicios de la nueva arquitectura
from app.database.connection import get_db
from app.services.user_service import UserService
from app.services.dni_verification_service import DNIVerificationService
from app.services.selfie_verification_service import SelfieVerificationService
from app.services.video_verification_service import VideoVerificationService

# Importar funciones utilitarias desde utils
from app.utils import (
    analyze_and_compare_dni_with_gpt,
    compare_faces_with_face_recognition,
    analyze_video_liveness_with_deepface,
    allowed_file,
    UPLOAD_FOLDER
)

# Crear blueprint
kyc_bp = Blueprint('kyc', __name__, url_prefix='/kyc')

@kyc_bp.route('/validate-dni', methods=['POST'])
def validate_dni():
    """
    📋 PASO 1: Validar datos del cliente contra DNI
    REFACTORIZADO: Usa DNIVerificationService + UserService
    """
    db = next(get_db())
    
    try:
        print("📋 VALIDANDO DNI - Versión refactorizada con BD")
        
        # 1. Recibir datos del formulario
        form_data = {
            'firstName': request.form.get('firstName', '').strip(),
            'lastName': request.form.get('lastName', '').strip(),
            'documentNumber': request.form.get('documentNumber', '').strip(),
            'nationality': request.form.get('nationality', '').strip(),
            'birthDate': request.form.get('birthDate', '').strip(),
            'issueDate': request.form.get('issueDate', '').strip(),
            'expiryDate': request.form.get('expiryDate', '').strip()
        }
        
        print(f"📋 Datos: {form_data['firstName']} {form_data['lastName']}, Doc: {form_data['documentNumber']}")
        
        # Validar datos obligatorios
        if not all(form_data.values()):
            return jsonify({
                "success": False,
                "error": "Faltan datos obligatorios",
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
        # 2. Recibir y guardar imagen DNI
        dni_front_file = request.files.get('dniImageFront')
        if not dni_front_file or dni_front_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere imagen del DNI (dniImageFront)",
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400
        
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
        
        print(f"📄 DNI guardado: {dni_front_path}")
        
        # 3. Analizar con GPT (reutilizar función original)
        print("🧠 Analizando con GPT...")
        gpt_result_str = analyze_and_compare_dni_with_gpt(dni_front_path, form_data)
        
        # Parsear resultado GPT
        try:
            gpt_result = json.loads(gpt_result_str)
        except json.JSONDecodeError:
            return jsonify({
                "success": False,
                "error": "Error procesando respuesta de análisis",
                "confidence": 0,
                "recommendation": "REVIEW"
            }), 500
        
        print(f"🧠 GPT OK, confianza: {gpt_result.get('verification', {}).get('overall_confidence', 0)}%")
        
        # 4. USAR DNIVerificationService para crear verificación en BD
        dni_verification = DNIVerificationService.create_dni_verification(
            db=db,
            form_data=form_data,
            extracted_data=gpt_result,
            gpt_analysis=gpt_result,
            dni_image_path=dni_front_path.replace('\\', '/')
        )
        
        if not dni_verification:
            return jsonify({
                "success": False,
                "error": "Error creando verificación DNI",
                "confidence": 0,
                "recommendation": "REJECT"
            }), 500
        
        # 5. Preparar respuesta compatible con el original
        response = {
            "success": True,
            "verification_id": dni_verification.id,
            "confidence": dni_verification.confidence,
            "data_matches": {
                "name": dni_verification.match_name,
                "document_number": dni_verification.match_document_number,
                "birthdate": dni_verification.match_birth_date,
                "issue_date": dni_verification.match_issue_date,
                "expiry_date": dni_verification.match_expiry_date,
                "country": dni_verification.match_nationality
            },
            "document_analysis": json.loads(dni_verification.gpt_analysis).get('document_analysis', {}),
            "extracted_data": dni_verification.extracted_data,
            "extracted_text": json.loads(dni_verification.gpt_analysis).get('extracted_text', ''),
            "details": dni_verification.details,
            "recommendation": "APPROVE" if dni_verification.is_approved else dni_verification.status.value,
            "dni_front_path": dni_front_path.replace('\\', '/'),
            "user_created": dni_verification.is_approved,
            "document_number": dni_verification.document_number
        }
        
        print(f"✅ DNI verificado: {dni_verification.status.value} ({dni_verification.confidence}%)")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error en validación DNI: {e}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}",
            "confidence": 0,
            "recommendation": "REJECT"
        }), 500
    finally:
        db.close()

@kyc_bp.route('/verify-selfie', methods=['POST'])
def verify_selfie():
    """
    🤳 PASO 2: Verificar selfie vs DNI  
    REFACTORIZADO: Usa SelfieVerificationService + arquitectura nueva
    """
    db = next(get_db())
    
    try:
        print("🤳 VERIFICANDO SELFIE - Versión refactorizada")
        
        # 1. Obtener número de documento
        document_number = request.form.get('documentNumber', '').strip()
        
        # COMPATIBILIDAD: También aceptar dniImagePath del original
        dni_image_path = request.form.get('dniImagePath', '').strip()
        
        if not document_number and not dni_image_path:
            return jsonify({
                "success": False,
                "error": "Se requiere número de documento (documentNumber) o ruta DNI (dniImagePath)",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400

        # 2. Si tenemos documentNumber, buscar usuario en BD
        user = None
        if document_number:
            user = UserService.get_user_by_document(db, document_number)
            if not user:
                return jsonify({
                    "success": False,
                    "error": "Usuario no encontrado. Primero debe completar la validación del DNI",
                    "face_match": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 400

            # Verificar que tiene DNI aprobado
            if not user.dni_verification or not user.dni_verification.is_approved:
                return jsonify({
                    "success": False,
                    "error": "El DNI debe estar aprobado antes de la verificación de selfie",
                    "face_match": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 400
            
            # Usar DNI del usuario
            dni_image_path = user.dni_verification.dni_image_path

        # 3. Si no tenemos usuario pero sí dniImagePath, usar modo compatibilidad
        if not user and dni_image_path:
            print("⚠️ Modo compatibilidad: usando dniImagePath sin usuario en BD")
            # Verificar que el archivo DNI existe
            if not os.path.exists(dni_image_path):
                return jsonify({
                    "success": False,
                    "error": f"Imagen DNI no encontrada: {dni_image_path}",
                    "face_match": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 400

        # 4. Recibir y guardar archivo de selfie
        selfie_file = request.files.get('selfieImage')
        
        if not selfie_file or selfie_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere imagen del selfie (selfieImage)",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400

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

        print(f"🤳 Selfie guardado: {selfie_path}")
        print(f"🤳 Comparando con DNI: {dni_image_path}")

        # 5. Si tenemos usuario, usar SelfieVerificationService
        if user:
            selfie_verification = SelfieVerificationService.create_selfie_verification(
                db=db,
                user=user,
                selfie_image_path=selfie_path.replace('\\', '/'),
                dni_image_path=dni_image_path
            )

            if not selfie_verification:
                return jsonify({
                    "success": False,
                    "error": "Error creando verificación de selfie",
                    "face_match": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 500

            # Respuesta con datos del servicio
            response = {
                "success": True,
                "verification_id": selfie_verification.id,
                "face_match": selfie_verification.match_dni,
                "confidence": selfie_verification.confidence,
                "analysis": selfie_verification.analysis_result,
                "fraud_indicators": selfie_verification.fraud_indicators,
                "recommendation": "APPROVE" if selfie_verification.is_approved else selfie_verification.status.value,
                "selfie_path": selfie_path.replace('\\', '/'),
                "dni_path": dni_image_path.replace('\\', '/'),
                "status": selfie_verification.status.value
            }
        else:
            # 6. Modo compatibilidad: usar función original directamente
            print("🔄 Usando función original para compatibilidad...")
            comparison_result = compare_faces_with_face_recognition(dni_image_path, selfie_path)
            
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
        
        print(f"✅ Selfie verificado - Match: {response['face_match']}, Confianza: {response['confidence']}%")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error verificando selfie: {e}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}",
            "face_match": False,
            "confidence": 0,
            "recommendation": "REJECT"
        }), 500
    finally:
        db.close()

@kyc_bp.route('/verify-liveness', methods=['POST'])
def verify_liveness():
    """
    🎥 PASO 3: Verificar video de liveness
    REFACTORIZADO: Usa VideoVerificationService + arquitectura nueva
    """
    db = next(get_db())
    
    try:
        print("🎥 VERIFICANDO LIVENESS - Versión refactorizada")
        
        # 1. Obtener parámetros
        document_number = request.form.get('documentNumber', '').strip()
        selfie_image_path = request.form.get('selfieImagePath', '').strip()
        dni_image_path = request.form.get('dniImagePath', '').strip()
        
        user = None
        
        # 2. Si tenemos documentNumber, buscar usuario en BD
        if document_number:
            user = UserService.get_user_by_document(db, document_number)
            if not user:
                return jsonify({
                    "success": False,
                    "error": "Usuario no encontrado. Debe completar los pasos anteriores",
                    "is_live_person": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 400

            # Verificar que tiene DNI y selfie aprobados
            if not user.dni_verification or not user.dni_verification.is_approved:
                return jsonify({
                    "success": False,
                    "error": "El DNI debe estar aprobado antes del video",
                    "is_live_person": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 400

            if not user.selfie_verification or not user.selfie_verification.is_approved:
                return jsonify({
                    "success": False,
                    "error": "El selfie debe estar aprobado antes del video",
                    "is_live_person": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 400
            
            # Usar rutas del usuario
            selfie_image_path = user.selfie_verification.selfie_image_path
            dni_image_path = user.dni_verification.dni_image_path

        # 3. Si no tenemos usuario, usar modo compatibilidad con rutas proporcionadas
        if not user:
            print("⚠️ Modo compatibilidad: usando rutas proporcionadas sin usuario en BD")
            
            if not selfie_image_path:
                return jsonify({
                    "success": False,
                    "error": "Se requiere la ruta de la imagen selfie (selfieImagePath)",
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

        # 4. Recibir y guardar video
        video_file = request.files.get('verificationVideo')
        
        if not video_file or video_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Se requiere video de verificación (verificationVideo)",
                "is_live_person": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }), 400

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

        print(f"🎥 Video guardado: {video_path}")

        # 5. Si tenemos usuario, usar VideoVerificationService
        if user:
            video_verification = VideoVerificationService.create_video_verification(
                db=db,
                user=user,
                video_path=video_path.replace('\\', '/'),
                reference_image_path=selfie_image_path
            )

            if not video_verification:
                return jsonify({
                    "success": False,
                    "error": "Error creando verificación de video",
                    "is_live_person": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }), 500

            # Respuesta con datos del servicio
            response = {
                "success": True,
                "verification_id": video_verification.id,
                "is_live_person": video_verification.is_live_person,
                "matches_selfie": video_verification.matches_reference,
                "matches_dni": video_verification.matches_reference,  # Simplificado
                "confidence": video_verification.confidence,
                "liveness_score": video_verification.confidence,
                "analysis": video_verification.analysis_result,
                "recommendation": "APPROVE" if video_verification.is_approved else video_verification.status.value,
                "video_path": video_path.replace('\\', '/'),
                "frames_analyzed": video_verification.frames_analyzed,
                "status": video_verification.status.value
            }
        else:
            # 6. Modo compatibilidad: usar función original
            print("🔄 Usando función original para compatibilidad...")
            
            # Análisis principal: video vs selfie
            liveness_result = analyze_video_liveness_with_deepface(video_path, selfie_image_path)
            
            # Análisis adicional: video vs DNI (si está disponible)
            dni_match_confidence = 0
            if dni_image_path:
                print("🧠 Comparación adicional video vs DNI...")
                dni_liveness_result = analyze_video_liveness_with_deepface(video_path, dni_image_path)
                dni_match_confidence = dni_liveness_result.get('confidence', 0)
            
            # Extraer resultados principales
            selfie_confidence = liveness_result.get('confidence', 0)
            matches_selfie = liveness_result.get('matches_reference', False)
            is_live_person = liveness_result.get('is_live_person', False)
            
            # Calcular confianza final (promedio si hay DNI)
            if dni_image_path and dni_match_confidence > 0:
                final_confidence = int((selfie_confidence * 0.7) + (dni_match_confidence * 0.3))
                matches_dni = dni_match_confidence >= 40
            else:
                final_confidence = selfie_confidence
                matches_dni = False
            
            # Determinar recomendación
            if final_confidence >= 60:
                recommendation = "APPROVE"
            elif final_confidence >= 40:
                recommendation = "REVIEW"
            else:
                recommendation = "REJECT"
            
            # Análisis detallado
            analysis = f"Video {recommendation}: {final_confidence}% de confianza. "
            analysis += f"Persona {'real' if is_live_person else 'NO real'}. "
            analysis += f"{'Coincide' if matches_selfie else 'NO coincide'} con selfie."
            if dni_image_path:
                analysis += f" {'Coincide' if matches_dni else 'NO coincide'} con DNI."
            
            response = {
                "success": True,
                "is_live_person": is_live_person,
                "matches_selfie": matches_selfie,
                "matches_dni": matches_dni,
                "confidence": final_confidence,
                "liveness_score": final_confidence,
                "analysis": analysis,
                "recommendation": recommendation,
                "video_path": video_path.replace('\\', '/'),
                "frames_analyzed": liveness_result.get('frames_analyzed', 0),
                "technical_details": liveness_result.get('technical_details', {})
            }
        
        print(f"✅ Liveness verificado - Live: {response['is_live_person']}, Score: {response['confidence']}%")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error verificando liveness: {e}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}",
            "is_live_person": False,
            "confidence": 0,
            "recommendation": "REJECT"
        }), 500
    finally:
        db.close()

@kyc_bp.route('/user-status/<document_number>', methods=['GET'])
def get_user_status(document_number: str):
    """
    📊 CONSULTAR ESTADO completo de verificación de un usuario
    """
    db = next(get_db())
    
    try:
        print(f"📊 Consultando estado de usuario: {document_number}")
        
        status_data = UserService.get_user_verification_status(db, document_number)
        
        return jsonify(status_data)
        
    except Exception as e:
        print(f"❌ Error consultando estado: {e}")
        return jsonify({
            "success": False,
            "error": f"Error consultando estado: {str(e)}"
        }), 500
    finally:
        db.close()

@kyc_bp.route('/complete-verification/<document_number>', methods=['GET'])
def get_complete_verification(document_number: str):
    """
    🏆 RESUMEN COMPLETO: Estado final de todas las verificaciones (MODULAR)
    """
    db = next(get_db())
    
    def _determine_verification_mode(steps_attempted):
        """🎯 Determinar modo de verificación usado"""
        if 'liveness' in steps_attempted:
            return "FULL_KYC"  # DNI + Selfie + Video
        elif 'selfie' in steps_attempted:
            return "BASIC_KYC"  # DNI + Selfie
        elif 'dni' in steps_attempted:
            return "DNI_ONLY"   # Solo DNI
        else:
            return "NO_VERIFICATION"
    
    try:
        print(f"🏆 Consultando verificación completa de: {document_number}")
        
        # Obtener estado completo del usuario
        status_data = UserService.get_user_verification_status(db, document_number)
        
        if not status_data.get('user_exists'):
            return jsonify({
                "success": False,
                "error": "Usuario no encontrado",
                "is_fully_verified": False
            }), 404
        
        user = UserService.get_user_by_document(db, document_number)
        
        # Determinar si está completamente verificado (MODULAR)
        is_fully_verified = user.is_fully_verified if user else False
        verification_progress = user.verification_progress if user else {}
        
        # Preparar resumen completo MODULAR
        summary = {
            "success": True,
            "document_number": document_number,
            "is_fully_verified": is_fully_verified,
            "verification_summary": status_data,
            "final_recommendation": "APPROVE" if is_fully_verified else "REVIEW",
            
            # 🎯 INFORMACIÓN MODULAR
            "modular_info": {
                "steps_attempted": verification_progress.get('steps_attempted', []),
                "steps_completed": verification_progress.get('steps_completed', []),
                "completion_rate": verification_progress.get('completion_rate', 0),
                "next_step": verification_progress.get('next_step'),
                "verification_mode": _determine_verification_mode(verification_progress.get('steps_attempted', []))
            },
            
            # Compatibilidad con formato anterior
            "steps_completed": {
                "dni": status_data.get('dni_verification', {}).get('completed', False),
                "selfie": status_data.get('selfie_verification', {}).get('completed', False),
                "liveness": status_data.get('video_verification', {}).get('completed', False)
            }
        }
        
        print(f"🏆 Verificación completa - Modo: {summary['modular_info']['verification_mode']}, Aprobado: {is_fully_verified}")
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"❌ Error en verificación completa: {e}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}",
            "is_fully_verified": False
        }), 500
    finally:
        db.close()
