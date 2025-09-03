from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey, func, Enum
from sqlalchemy.orm import relationship
import uuid
from app.database.connection import Base
from app.models.user import VerificationStatus

class SelfieVerification(Base):
    """
    🤳 VERIFICACIÓN DE SELFIE
    
    Almacena el resultado de la comparación facial entre el DNI y la selfie,
    usando DeepFace para determinar si es la misma persona.
    """
    __tablename__ = 'selfie_verifications'
    
    # ID único de la verificación (MySQL compatible)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con el usuario
    document_number = Column(String(50), ForeignKey('users.document_number'), nullable=False)
    user = relationship("User", back_populates="selfie_verification")
    
    # Relación con la verificación de DNI (referencia)
    dni_verification_id = Column(String(36), ForeignKey('dni_verifications.id'), nullable=False)
    dni_verification = relationship("DNIVerification", backref="selfie_verifications")
    
    # Ruta de la imagen del selfie
    selfie_image_path = Column(String(500), nullable=False)
    
    # Resultados de la verificación facial
    match_dni = Column(Boolean, nullable=False, default=False)
    confidence = Column(Integer, nullable=False, default=0)  # 0-100
    
    # Análisis detallado
    analysis_result = Column(Text, nullable=True)
    fraud_indicators = Column(Text, nullable=True)  # JSON array como string
    
    # Detalles técnicos de DeepFace
    deepface_distance = Column(Float, nullable=True)
    deepface_threshold = Column(Float, nullable=True)
    deepface_model = Column(String(20), nullable=True, default='VGG-Face')
    detector_backend = Column(String(20), nullable=True, default='opencv')
    
    # Calidad de las imágenes
    dni_image_quality_score = Column(Integer, nullable=True)  # 0-100
    selfie_image_quality_score = Column(Integer, nullable=True)  # 0-100
    
    # Caras detectadas
    faces_detected_dni = Column(Integer, nullable=True, default=0)
    faces_detected_selfie = Column(Integer, nullable=True, default=0)
    
    # Estado final
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)
    details = Column(Text, nullable=True)
    
    # Metadatos
    created_at = Column(DateTime, default=func.now())
    processing_time_seconds = Column(Float, nullable=True)
    
    @property
    def is_approved(self):
        """Verificación aprobada si status es ACCEPTED"""
        return self.status == VerificationStatus.ACCEPTED
    
    def __repr__(self):
        return f"<SelfieVerification(id='{self.id}', document='{self.document_number}', match={self.match_dni}, confidence={self.confidence}%, status='{self.status.value if self.status else 'None'}')>"
    
    def __repr__(self):
        return f"<SelfieVerification(id='{self.id}', document='{self.document_number}', match={self.match_dni}, confidence={self.confidence}%, status='{self.status.value if self.status else 'None'}')>"
    
    def to_dict(self):
        """Convertir a diccionario para JSON responses"""
        return {
            'id': self.id,
            'document_number': self.document_number,
            'dni_verification_id': self.dni_verification_id,
            'selfie_image_path': self.selfie_image_path,
            'match_dni': self.match_dni,
            'confidence': self.confidence,
            'status': self.status.value if self.status else None,
            'is_approved': self.is_approved,
            'analysis_result': self.analysis_result,
            'fraud_indicators': self.fraud_indicators,
            'technical_details': {
                'deepface_distance': self.deepface_distance,
                'deepface_threshold': self.deepface_threshold,
                'deepface_model': self.deepface_model,
                'detector_backend': self.detector_backend,
                'dni_image_quality': self.dni_image_quality_score,
                'selfie_image_quality': self.selfie_image_quality_score,
                'faces_detected': {
                    'dni': self.faces_detected_dni,
                    'selfie': self.faces_detected_selfie
                }
            },
            'details': self.details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processing_time_seconds': self.processing_time_seconds
        }
