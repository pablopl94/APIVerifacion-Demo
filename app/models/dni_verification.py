from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey, func, Enum
from sqlalchemy.orm import relationship
import uuid
from app.database.connection import Base
from app.models.user import VerificationStatus

class DNIVerification(Base):
    """
    📄 VERIFICACIÓN DE DNI/DOCUMENTO DE IDENTIDAD
    
    Almacena el resultado de la verificación del documento de identidad,
    incluyendo el análisis GPT-4 Vision y la comparación de datos.
    """
    __tablename__ = 'dni_verifications'
    
    # ID único de la verificación (MySQL compatible)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con el usuario (por número de documento)
    document_number = Column(String(50), ForeignKey('users.document_number'), nullable=False)
    user = relationship("User", back_populates="dni_verification")
    
    # Ruta de la imagen del DNI
    dni_image_path = Column(String(500), nullable=False)
    
    # Datos del formulario (entrada del usuario)
    form_first_name = Column(String(100), nullable=False)
    form_last_name = Column(String(100), nullable=False)
    form_document_number = Column(String(50), nullable=False)
    form_nationality = Column(String(10), nullable=False)
    form_birth_date = Column(String(20), nullable=False)
    form_issue_date = Column(String(20), nullable=False)
    form_expiry_date = Column(String(20), nullable=False)
    
    # Datos extraídos del DNI por GPT-4 Vision
    extracted_first_name = Column(String(100), nullable=True)
    extracted_last_name = Column(String(100), nullable=True)
    extracted_document_number = Column(String(50), nullable=True)
    extracted_nationality = Column(String(10), nullable=True)
    extracted_birth_date = Column(String(20), nullable=True)
    extracted_issue_date = Column(String(20), nullable=True)
    extracted_expiry_date = Column(String(20), nullable=True)
    extracted_full_text = Column(Text, nullable=True)
    
    # Resultados de comparación (formulario vs extraído)
    match_name = Column(Boolean, nullable=False, default=False)
    match_document_number = Column(Boolean, nullable=False, default=False)
    match_birth_date = Column(Boolean, nullable=False, default=False)
    match_issue_date = Column(Boolean, nullable=False, default=False)
    match_expiry_date = Column(Boolean, nullable=False, default=False)
    match_nationality = Column(Boolean, nullable=False, default=False)
    
    # Análisis del documento
    document_type = Column(String(100), nullable=True)
    is_valid_document = Column(Boolean, nullable=False, default=False)
    document_country_match = Column(Boolean, nullable=False, default=False)
    
    # Resultado final
    confidence = Column(Integer, nullable=False, default=0)  # 0-100
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)
    details = Column(Text, nullable=True)
    
    # Datos raw de GPT
    gpt_analysis = Column(Text, nullable=True)  # JSON string del análisis completo
    gpt_raw_response = Column(Text, nullable=True)
    
    # Metadatos
    created_at = Column(DateTime, default=func.now())
    processing_time_seconds = Column(Float, nullable=True)
    
    @property
    def is_approved(self):
        """Verificación aprobada si status es ACCEPTED"""
        return self.status == VerificationStatus.ACCEPTED
    
    def __repr__(self):
        return f"<DNIVerification(id='{self.id}', document='{self.document_number}', confidence={self.confidence}%, status='{self.status.value if self.status else 'None'}')>"
    
    def to_dict(self):
        """Convertir a diccionario para JSON responses"""
        return {
            'id': self.id,
            'document_number': self.document_number,
            'dni_image_path': self.dni_image_path,
            'confidence': self.confidence,
            'status': self.status.value if self.status else None,
            'is_approved': self.is_approved,
            'form_data': {
                'first_name': self.form_first_name,
                'last_name': self.form_last_name,
                'document_number': self.form_document_number,
                'nationality': self.form_nationality,
                'birth_date': self.form_birth_date,
                'issue_date': self.form_issue_date,
                'expiry_date': self.form_expiry_date
            },
            'extracted_data': {
                'first_name': self.extracted_first_name,
                'last_name': self.extracted_last_name,
                'document_number': self.extracted_document_number,
                'nationality': self.extracted_nationality,
                'birth_date': self.extracted_birth_date,
                'issue_date': self.extracted_issue_date,
                'expiry_date': self.extracted_expiry_date
            },
            'matches': {
                'name': self.match_name,
                'document_number': self.match_document_number,
                'birth_date': self.match_birth_date,
                'issue_date': self.match_issue_date,
                'expiry_date': self.match_expiry_date,
                'nationality': self.match_nationality
            },
            'document_analysis': {
                'type': self.document_type,
                'is_valid': self.is_valid_document,
                'country_match': self.document_country_match
            },
            'details': self.details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processing_time_seconds': self.processing_time_seconds
        }
