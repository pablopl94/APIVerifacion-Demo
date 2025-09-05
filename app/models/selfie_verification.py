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
    
    # Ruta de la imagen del DNI
    dni_image_path = Column(String(255), nullable=True)
    
    # Ruta de la imagen del selfie
    selfie_image_path = Column(String(500), nullable=False)
    
    # Resultados de la verificación facial
    match_dni = Column(Boolean, nullable=False, default=False)
    confidence = Column(Integer, nullable=False, default=0)  # 0-100
    
    # Análisis detallado
    analysis_result = Column(Text, nullable=True)
    fraud_indicators = Column(Text, nullable=True)  # JSON array como string
    def get_fraud_indicators(self):
        import json
        try:
            return json.loads(self.fraud_indicators) if self.fraud_indicators else []
        except Exception:
            return []
    
    # Detalles técnicos de DeepFace
    
    # Calidad de las imágenes
    
    # Caras detectadas

    # Estado final
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)
    details = Column(Text, nullable=True)
    
    # Estado final
    created_at = Column(DateTime, default=func.now())
    def is_approved(self):
        """Verificación aprobada si status es ACCEPTED"""
    def __repr__(self):
        return f"<SelfieVerification(id='{self.id}', document='{self.document_number}', match={self.match_dni}, confidence={self.confidence}%, status='{self.status.value if self.status else 'None'}')>"
    
    def __repr__(self):
        return f"<SelfieVerification(id='{self.id}', document='{self.document_number}', match={self.match_dni}, confidence={self.confidence}%, status='{self.status.value if self.status else 'None'}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'document_number': self.document_number,
            'selfie_image_path': self.selfie_image_path,
            'match_dni': self.match_dni,
            'confidence': self.confidence,
            'status': self.status.value if self.status else None,
            'is_approved': self.is_approved,
            'analysis_result': self.analysis_result,
            'fraud_indicators': self.get_fraud_indicators(),
            'details': self.details,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
