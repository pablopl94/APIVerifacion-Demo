"""
📄 SERVICIO VERIFICACIÓN DNI
Lógica de negocio para verificar datos formulario vs imagen DNI
"""
from sqlalchemy.orm import Session
from app.models import User, DNIVerification, VerificationStatus
from app.services.user_service import UserService
from typing import Dict, Any, Optional
import json
import uuid

class DNIVerificationService:
    """
    📄 SERVICIO DE VERIFICACIÓN DNI
    
    Gestiona el proceso completo de verificación:
    1. Recibe datos del formulario + imagen DNI
    2. Extrae datos de la imagen con GPT
    3. Compara formulario vs imagen extraída
    4. Si verificación OK → crea usuario
    5. Guarda resultado en base de datos
    """
    
    @staticmethod
    def create_dni_verification(
        db: Session,
        form_data: Dict[str, str],
        extracted_data: Dict[str, Any],
        gpt_analysis: Dict[str, Any],
        dni_image_path: str
    ) -> Optional[DNIVerification]:
        """
        📄 CREAR VERIFICACIÓN DNI
        
        Args:
            db: Sesión de base de datos
            form_data: Datos del formulario del usuario
            extracted_data: Datos extraídos de la imagen DNI
            gpt_analysis: Análisis completo de GPT
            dni_image_path: Ruta de la imagen DNI
            
        Returns:
            DNIVerification creada
        """
        try:
            # Determinar status basado en análisis GPT
            verification_data = gpt_analysis.get('verification', {})
            confidence = verification_data.get('overall_confidence', 0)
            gpt_recommendation = verification_data.get('recommendation', 'REVIEW')

            # Solo permitir PENDING, ACCEPTED, REJECTED para verificación de DNI
            # Considerar coincidencia de todos los campos requeridos
            required_fields = [
                'name', 'document_number', 'birthdate', 'issue_date', 'expiry_date', 'country'
            ]
            data_matches = gpt_analysis.get('data_matches', {})
            extracted = gpt_analysis.get('extracted_data', {})

            # Refuerzo: comparar explícitamente el número de documento
            form_document_number = form_data.get('documentNumber', '').strip()
            extracted_document_number = extracted.get('document_number', '').strip()
            match_document_number = (
                data_matches.get('document_number', False)
                and form_document_number == extracted_document_number
            )

            # Forzar el valor correcto en data_matches para el resto del flujo
            data_matches['document_number'] = match_document_number

            all_match = all(data_matches.get(field, False) for field in required_fields)

            if all_match and confidence >= 70 and gpt_recommendation == 'APPROVE':
                status = VerificationStatus.ACCEPTED
            else:
                status = VerificationStatus.REJECTED

            # --- NUEVO: Asegurar que el usuario existe antes de guardar la verificación ---
            from app.models.user import User, VerificationStatus as UserVerificationStatus
            document_number = form_data.get('documentNumber', '')
            user = db.query(User).filter(User.document_number == document_number).first()
            if not user:
                user = User(
                    document_number=document_number,
                    first_name=form_data.get('firstName', ''),
                    last_name=form_data.get('lastName', ''),
                    nationality=form_data.get('nationality', ''),
                    birth_date=form_data.get('birthDate', ''),
                    issue_date=form_data.get('issueDate', ''),
                    expiry_date=form_data.get('expiryDate', ''),
                    status=UserVerificationStatus.PENDING
                )
                db.add(user)
                db.commit()
                db.refresh(user)

            # --- FIN NUEVO ---

            # Extraer datos de la respuesta GPT
            extracted = gpt_analysis.get('extracted_data', {})
            data_matches = gpt_analysis.get('data_matches', {})
            document_analysis = gpt_analysis.get('document_analysis', {})

            # Crear registro de verificación DNI
            dni_verification = DNIVerification(
                id=str(uuid.uuid4()),
                document_number=form_data.get('documentNumber', ''),
                
                # Datos del formulario
                form_first_name=form_data.get('firstName', ''),
                form_last_name=form_data.get('lastName', ''),
                form_document_number=form_data.get('documentNumber', ''),
                form_nationality=form_data.get('nationality', ''),
                form_birth_date=form_data.get('birthDate', ''),
                form_issue_date=form_data.get('issueDate', ''),
                form_expiry_date=form_data.get('expiryDate', ''),
                
                # Datos extraídos de la imagen
                extracted_first_name=extracted.get('name', '').split(' ')[0] if extracted.get('name') else '',
                extracted_last_name=' '.join(extracted.get('name', '').split(' ')[1:]) if extracted.get('name') else '',
                extracted_document_number=extracted.get('document_number', ''),
                extracted_nationality=extracted.get('nationality', ''),
                extracted_birth_date=extracted.get('birthdate', ''),
                extracted_issue_date=extracted.get('issue_date', ''),
                extracted_expiry_date=extracted.get('expiry_date', ''),
                extracted_full_text=gpt_analysis.get('extracted_text', ''),
                
                # Resultados de comparación
                match_name=data_matches.get('name', False),
                match_document_number=data_matches.get('document_number', False),
                match_birth_date=data_matches.get('birthdate', False),
                match_issue_date=data_matches.get('issue_date', False),
                match_expiry_date=data_matches.get('expiry_date', False),
                match_nationality=data_matches.get('country', False),
                
                # Análisis del documento
                document_type=document_analysis.get('document_type', ''),
                is_valid_document=document_analysis.get('is_valid_document', False),
                document_country_match=document_analysis.get('country_match', False),
                
                # Resultado final
                confidence=confidence,
                status=status,
                details=gpt_analysis.get('details', ''),
                
                # Archivo
                dni_image_path=dni_image_path,
                
                # Raw data
                gpt_analysis=json.dumps(gpt_analysis, ensure_ascii=False, indent=2),
                gpt_raw_response=json.dumps(gpt_analysis, ensure_ascii=False, indent=2)
            )
            
            # Guardar verificación
            db.add(dni_verification)
            db.commit()
            db.refresh(dni_verification)

            # Si la verificación fue exitosa, actualizar el usuario con los datos extraídos
            if status == VerificationStatus.ACCEPTED:
                print("🚀 Verificación exitosa, actualizando usuario...")
                user.first_name = dni_verification.extracted_first_name
                user.last_name = dni_verification.extracted_last_name
                user.nationality = dni_verification.extracted_nationality
                user.birth_date = dni_verification.extracted_birth_date
                user.issue_date = dni_verification.extracted_issue_date
                user.expiry_date = dni_verification.extracted_expiry_date
                user.status = UserVerificationStatus.ACCEPTED
                db.commit()
                db.refresh(user)

            print(f"✅ DNI verification creada: {dni_verification.id} ({status.value}, {confidence}%)")
            return dni_verification
        except Exception as e:
            print(f"❌ Error creando verificación DNI: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_dni_verification(db: Session, document_number: str) -> Optional[DNIVerification]:
        """
        🔍 OBTENER VERIFICACIÓN DNI por número de documento
        """
        try:
            return db.query(DNIVerification).filter(
                DNIVerification.document_number == document_number
            ).first()
        except Exception as e:
            print(f"❌ Error obteniendo verificación DNI: {e}")
            return None
    
    @staticmethod
    def update_verification_status(
        db: Session,
        verification_id: str,
        new_status: VerificationStatus,
        details: str = None
    ) -> Optional[DNIVerification]:
        """
        🔄 ACTUALIZAR STATUS de verificación DNI
        """
        try:
            verification = db.query(DNIVerification).filter(
                DNIVerification.id == verification_id
            ).first()
            
            if not verification:
                return None
            
            verification.status = new_status
            if details:
                verification.details = details
            
            db.commit()
            db.refresh(verification)
            
            print(f"✅ Verificación DNI actualizada: {verification_id} → {new_status.value}")
            return verification
            
        except Exception as e:
            print(f"❌ Error actualizando verificación DNI: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_verification_summary(db: Session, document_number: str) -> Dict[str, Any]:
        """
        📊 RESUMEN COMPLETO de verificación DNI
        """
        try:
            verification = DNIVerificationService.get_dni_verification(db, document_number)
            
            if not verification:
                return {
                    'exists': False,
                    'message': 'Verificación DNI no encontrada'
                }
            
            return {
                'exists': True,
                'verification': verification.to_dict(),
                'summary': {
                    'document_number': verification.document_number,
                    'status': verification.status.value,
                    'confidence': verification.confidence,
                    'is_approved': verification.is_approved,
                    'matches_summary': {
                        'name': verification.name_match,
                        'document': verification.document_match,
                        'birthdate': verification.birthdate_match,
                        'dates': verification.issue_date_match and verification.expiry_date_match,
                        'country': verification.country_match
                    },
                    'document_info': {
                        'type': verification.document_type,
                        'valid': verification.is_valid_document,
                        'country_match': verification.document_country_match
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo resumen: {e}")
            return {
                'exists': False,
                'error': str(e)
            }
