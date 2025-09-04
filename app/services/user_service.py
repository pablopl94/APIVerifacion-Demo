"""
👤 SERVICIO USUARIOS KYC
Lógica de negocio para creación y gestión de usuarios
"""
from sqlalchemy.orm import Session
from app.models import User, VerificationStatus, DNIVerification
from app.database.connection import get_db
from typing import Optional, Dict, Any, List
from datetime import datetime

class UserService:
    """
    👤 SERVICIO DE USUARIOS
    
    Maneja toda la lógica relacionada con usuarios:
    - Crear usuario SOLO tras verificación DNI exitosa
    - Consultar estados de verificación
    - Actualizar estado del usuario
    """
    
    @staticmethod
    def create_user_after_dni_verification(
        db: Session,
        dni_verification: DNIVerification
    ) -> Optional[User]:
        """
        👤 CREAR USUARIO tras verificación DNI exitosa
        
        Solo se ejecuta cuando:
        - DNI verification status = ACCEPTED
        - Datos extraídos son válidos
        
        Args:
            db: Sesión de base de datos
            dni_verification: Verificación DNI exitosa
            
        Returns:
            User creado o None si falla
        """
        try:
            # Verificar que la verificación DNI fue exitosa
            if dni_verification.status != VerificationStatus.ACCEPTED:
                print(f"❌ No se puede crear usuario: DNI no verificado ({dni_verification.status})")
                return None
            
            # Verificar que no existe ya un usuario con ese documento
            existing_user = db.query(User).filter(
                User.document_number == dni_verification.extracted_document_number
            ).first()
            
            if existing_user:
                print(f"⚠️ Usuario ya existe: {dni_verification.extracted_document_number}")
                return existing_user
            
            # Crear nuevo usuario con datos EXTRAÍDOS del DNI (no del formulario)
            new_user = User(
                document_number=dni_verification.extracted_document_number,
                first_name=dni_verification.extracted_first_name,
                last_name=dni_verification.extracted_last_name,
                nationality=dni_verification.extracted_nationality,
                birth_date=dni_verification.extracted_birth_date,
                issue_date=dni_verification.extracted_issue_date,
                expiry_date=dni_verification.extracted_expiry_date,
                status=VerificationStatus.PENDING  # Aún faltan selfie y video
            )
            
            # Guardar en base de datos
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            print(f"✅ Usuario creado: {new_user.document_number} - {new_user.full_name}")
            return new_user
            
        except Exception as e:
            print(f"❌ Error creando usuario: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_user_by_document(db: Session, document_number: str) -> Optional[User]:
        """
        🔍 BUSCAR USUARIO por número de documento
        """
        try:
            return db.query(User).filter(User.document_number == document_number).first()
        except Exception as e:
            print(f"❌ Error buscando usuario: {e}")
            return None
    
    @staticmethod
    def update_user_status(
        db: Session, 
        document_number: str, 
        new_status: VerificationStatus
    ) -> Optional[User]:
        """
        🔄 ACTUALIZAR ESTADO del usuario
        """
        try:
            user = db.query(User).filter(User.document_number == document_number).first()
            if not user:
                return None
            
            user.status = new_status
            user.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(user)
            
            print(f"✅ Estado actualizado: {document_number} → {new_status.value}")
            return user
            
        except Exception as e:
            print(f"❌ Error actualizando estado: {e}")
            db.rollback()
            return None
    
    @staticmethod 
    def get_user_verification_status(db: Session, document_number: str) -> Dict[str, Any]:
        """
        📊 OBTENER ESTADO COMPLETO de verificación del usuario
        """
        try:
            user = UserService.get_user_by_document(db, document_number)
            if not user:
                return {
                    'exists': False,
                    'message': 'Usuario no encontrado'
                }
            
            # Obtener todas las verificaciones
            dni_verification = user.dni_verification
            selfie_verification = user.selfie_verification
            video_verification = user.video_verification
            
            status_data = {
                'exists': True,
                'user': user.to_dict(),
                'verification_status': {
                    'dni': {
                        'completed': dni_verification is not None,
                        'status': dni_verification.status.value if dni_verification else 'PENDING',
                        'confidence': dni_verification.confidence if dni_verification else 0
                    },
                    'selfie': {
                        'completed': selfie_verification is not None,
                        'status': selfie_verification.status.value if selfie_verification else 'PENDING',
                        'confidence': selfie_verification.confidence if selfie_verification else 0
                    },
                    'video': {
                        'completed': video_verification is not None,
                        'status': video_verification.status.value if video_verification else 'PENDING',
                        'confidence': video_verification.final_confidence if video_verification else 0
                    }
                },
                'overall': {
                    'is_fully_verified': user.is_fully_verified,
                    'status': user.status.value,
                    'next_step': UserService._get_next_verification_step(
                        dni_verification, selfie_verification, video_verification
                    )
                }
            }
            
            return status_data
            
        except Exception as e:
            print(f"❌ Error obteniendo estado: {e}")
            return {
                'exists': False,
                'error': str(e)
            }
    
    @staticmethod
    def _get_next_verification_step(dni_ver, selfie_ver, video_ver) -> str:
        """
        📋 DETERMINAR siguiente paso de verificación
        """
        if not dni_ver or dni_ver.status != VerificationStatus.ACCEPTED:
            return "dni_verification"
        elif not selfie_ver or selfie_ver.status != VerificationStatus.ACCEPTED:
            return "selfie_verification" 
        elif not video_ver or video_ver.status != VerificationStatus.ACCEPTED:
            return "video_verification"
        else:
            return "completed"
    
    @staticmethod
    def list_all_users(db: Session, limit: int = 100) -> List[User]:
        """
        📋 LISTAR TODOS los usuarios
        """
        try:
            return db.query(User).limit(limit).all()
        except Exception as e:
            print(f"❌ Error listando usuarios: {e}")
            return []
    
    @staticmethod
    def ensure_user_exists_from_form(db: Session, form_data: Dict[str, str]) -> User:
        """
        Crea el usuario si no existe, usando los datos del formulario (aunque sean incorrectos).
        Solo para permitir la inserción en dni_verifications (por la FK).
        """
        document_number = form_data.get('documentNumber', '')
        user = db.query(User).filter(User.document_number == document_number).first()
        if user:
            return user
        # Crear usuario con datos del formulario
        user = User(
            document_number=document_number,
            first_name=form_data.get('firstName', ''),
            last_name=form_data.get('lastName', ''),
            nationality=form_data.get('nationality', ''),
            birth_date=form_data.get('birthDate', ''),
            issue_date=form_data.get('issueDate', ''),
            expiry_date=form_data.get('expiryDate', ''),
            status=VerificationStatus.PENDING
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"👤 Usuario creado temporalmente para FK: {user.document_number}")
        return user
