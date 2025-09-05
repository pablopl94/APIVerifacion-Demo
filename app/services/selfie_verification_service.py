"""
🤳 Selfie Verification Service
Servicio para verificación de selfies contra documentos de identidad
"""

import os
import uuid
import json
from werkzeug.utils import secure_filename
from sqlalchemy.orm import Session
from app.models.selfie_verification import SelfieVerification
from app.models.user import User
from app.database.connection import get_db
import json
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')

class SelfieVerificationService:
    
    def __init__(self):
        self.upload_folder = 'uploads'
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        
        # Crear directorio uploads si no existe
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
    
    def verify_selfie_against_dni(self, user_document_number: str, dni_image_path: str, 
                                selfie_file, db: Session = None) -> dict:
        """
        🔍 Verificar selfie contra imagen DNI de un usuario existente
        
        Args:
            user_document_number: Número de documento del usuario
            dni_image_path: Ruta de la imagen DNI ya procesada
            selfie_file: Archivo de selfie subido
            db: Sesión de base de datos (opcional)
            
        Returns:
            dict: Resultado de la verificación con persistencia en BD
        """
        if not db:
            db = next(get_db())
        
        try:
            print(f"🤳 Iniciando verificación de selfie para usuario: {user_document_number}")
            
            # 1. Buscar usuario existente
            user = db.query(User).filter(User.document_number == user_document_number).first()
            if not user:
                return {
                    "success": False,
                    "error": f"Usuario no encontrado: {user_document_number}",
                    "face_match": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }
            
            # 2. Validaciones de entrada
            validation_result = self._validate_selfie_inputs(dni_image_path, selfie_file)
            if not validation_result["success"]:
                return validation_result
            
            # 3. Guardar archivo selfie
            selfie_path = self._save_selfie_file(selfie_file)
            if not selfie_path:
                return {
                    "success": False,
                    "error": "Error guardando archivo selfie",
                    "face_match": False,
                    "confidence": 0,
                    "recommendation": "REJECT"
                }
            
            print(f"🤳 Selfie guardado en: {selfie_path}")
            print(f"🤳 Comparando con DNI: {dni_image_path}")
            
            # 4. Realizar comparación facial con DeepFace
            comparison_result = self._compare_faces_deepface(dni_image_path, selfie_path)
            
            # 5. Crear registro en base de datos
            selfie_verification = SelfieVerification(
                user_id=user.document_number,
                dni_image_path=dni_image_path,
                selfie_image_path=selfie_path,
                face_match=comparison_result.get('face_match', False),
                confidence_score=comparison_result.get('confidence', 0),
                analysis_result=json.dumps(comparison_result),
                fraud_indicators=json.dumps(comparison_result.get('fraud_indicators', [])),
                recommendation=comparison_result.get('recommendation', 'REVIEW'),
                technical_details=json.dumps(comparison_result.get('technical_details', {}))
            )
            
            db.add(selfie_verification)
            db.commit()
            
            print(f"✅ Verificación selfie completada - ID: {selfie_verification.id}")
            
            # 6. Estructurar respuesta del endpoint
            response = {
                "success": True,
                "verification_id": selfie_verification.id,
                "user_document_number": user_document_number,
                "face_match": comparison_result.get('face_match', False),
                "confidence": comparison_result.get('confidence', 0),
                "analysis": comparison_result.get('analysis', 'Sin análisis disponible'),
                "fraud_indicators": selfie_verification.get_fraud_indicators() if selfie_verification else [],
                "recommendation": comparison_result.get('recommendation', 'REVIEW'),
                "selfie_path": selfie_path.replace('\\', '/'),
                "dni_path": dni_image_path.replace('\\', '/'),
                "created_at": selfie_verification.created_at.isoformat(),
                "technical_details": comparison_result.get('technical_details', {})
            }
            
            print(f"✅ Selfie verification completada - Match: {response['face_match']}, Confianza: {response['confidence']}%")
            return response
            
        except Exception as e:
            print(f"❌ Error en verificación de selfie: {e}")
            db.rollback()
            return {
                "success": False,
                "error": str(e),
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }
        finally:
            db.close()
    
    def get_selfie_verification_by_user(self, user_document_number: str, db: Session = None) -> dict:
        """
        📊 Obtener verificaciones de selfie de un usuario
        """
        if not db:
            db = next(get_db())
        
        try:
            user = db.query(User).filter(User.document_number == user_document_number).first()
            if not user:
                return {"success": False, "error": "Usuario no encontrado"}
            
            verifications = db.query(SelfieVerification).filter(
                SelfieVerification.document_number == user.document_number
            ).order_by(SelfieVerification.created_at.desc()).all()
            
            return {
                "success": True,
                "user_document_number": user_document_number,
                "verifications": [
                    {
                        "id": v.id,
                        "face_match": v.face_match,
                        "confidence_score": v.confidence_score,
                        "recommendation": v.recommendation,
                        "created_at": v.created_at.isoformat(),
                        "selfie_path": v.selfie_image_path
                    }
                    for v in verifications
                ]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            db.close()
    
    def _validate_selfie_inputs(self, dni_image_path: str, selfie_file) -> dict:
        """🔍 Validar entradas para verificación de selfie"""
        
        if not dni_image_path:
            return {
                "success": False,
                "error": "Se requiere la ruta de la imagen DNI (dniImagePath)",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }
        
        if not selfie_file or selfie_file.filename == '':
            return {
                "success": False,
                "error": "Se requiere imagen del selfie (selfieImage)",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }
        
        # Verificar que el archivo DNI existe
        if not os.path.exists(dni_image_path):
            return {
                "success": False,
                "error": f"Imagen DNI no encontrada: {dni_image_path}",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }
        
        # Verificar extensión del selfie
        if not self._allowed_file(selfie_file.filename):
            return {
                "success": False,
                "error": "Tipo de archivo no permitido para selfie",
                "face_match": False,
                "confidence": 0,
                "recommendation": "REJECT"
            }
        
        return {"success": True}
    
    def _save_selfie_file(self, selfie_file) -> str:
        """💾 Guardar archivo selfie en uploads"""
        try:
            selfie_filename = str(uuid.uuid4()) + '_selfie_' + secure_filename(selfie_file.filename)
            selfie_path = os.path.join(self.upload_folder, selfie_filename)
            selfie_file.save(selfie_path)
            return selfie_path
        except Exception as e:
            print(f"❌ Error guardando selfie: {e}")
            return None
    
    def _allowed_file(self, filename: str) -> bool:
        """🔍 Verificar si el archivo tiene extensión permitida"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _compare_faces_deepface(self, image1_path: str, image2_path: str) -> dict:
        """
        🔍 Comparación facial usando DeepFace - Versión refactorizada
        
        Esta función está basada en compare_faces_with_face_recognition del servidor original
        pero adaptada para usar en el servicio refactorizado.
        """
        try:
            print(f"🔍 Iniciando comparación facial con DeepFace")
            print(f"📄 DNI: {image1_path}")
            print(f"🤳 Selfie: {image2_path}")
            
            # Verificar que las imágenes existan
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
                user_status = "APPROVE"
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
    
    @classmethod
    def create_selfie_verification(cls, db: Session, user: User, selfie_image_path: str, dni_image_path: str) -> SelfieVerification:
        """
        🤳 Método principal para crear verificación de selfie (compatible con server_refactored.py)
        
        Args:
            db: Sesión de base de datos
            user: Usuario ya existente en BD
            selfie_image_path: Ruta del archivo selfie
            dni_image_path: Ruta del DNI de referencia
            
        Returns:
            SelfieVerification: Objeto de verificación creado
        """
        try:
            print(f"🤳 Creando verificación de selfie para usuario: {user.document_number}")
            
            # Importar funciones del servidor original para reutilizar lógica
            from app.utils import compare_faces_with_face_recognition
            
            # 1. Realizar comparación usando la función del servidor original
            comparison_result = compare_faces_with_face_recognition(dni_image_path, selfie_image_path)
            
            # 2. Determinar status basado en resultado
            confidence = comparison_result.get('confidence', 0)
            face_match = comparison_result.get('face_match', False)
            recommendation = comparison_result.get('recommendation', 'REJECT')
            
            # Mapear recommendation a VerificationStatus
            from app.models.selfie_verification import VerificationStatus
            if recommendation == 'APPROVE' or (face_match and confidence >= 70):
                status = VerificationStatus.APPROVED
            elif recommendation == 'REVIEW' or (face_match and confidence >= 50):
                status = VerificationStatus.PENDING
            else:
                status = VerificationStatus.REJECTED
            
            # 3. Crear objeto SelfieVerification
            selfie_verification = SelfieVerification(
                document_number=user.document_number,
                dni_image_path=dni_image_path,
                selfie_image_path=selfie_image_path,
                match_dni=face_match,
                confidence=int(confidence),
                analysis_result=comparison_result.get('analysis', ''),
                status=status,
                fraud_indicators=json.dumps(comparison_result.get('fraud_indicators', [])),
                details=f"Verificación selfie: {confidence}% confianza, {'Match' if face_match else 'No match'}",
            )
            
            # 4. Guardar en base de datos
            db.add(selfie_verification)
            db.commit()
            db.refresh(selfie_verification)
            
            print(f"✅ SelfieVerification creado - ID: {selfie_verification.id}, Status: {status.value}")
            
            return selfie_verification
            
        except Exception as e:
            print(f"❌ Error creando SelfieVerification: {e}")
            db.rollback()
            return None
