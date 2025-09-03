from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey, func, Enum
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from app.database.connection import Base
from app.models.user import VerificationStatus

class VideoVerification(Base):
    """
    🎥 VERIFICACIÓN DE VIDEO/LIVENESS
    
    Almacena el resultado del análisis de liveness mediante video,
    comparando frames del video con imágenes de referencia.
    """
    __tablename__ = 'video_verifications'
    
    # ID único de la verificación (MySQL compatible)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con el usuario
    document_number = Column(String(50), ForeignKey('users.document_number'), nullable=False)
    user = relationship("User", back_populates="video_verification")
    
    # Rutas de archivos
    video_path = Column(String(500), nullable=False)
    reference_image_path = Column(String(500), nullable=False)  # Selfie principalmente
    
    # Resultados de liveness
    is_live_person = Column(Boolean, nullable=False, default=False)
    matches_reference = Column(Boolean, nullable=False, default=False)
    confidence = Column(Integer, nullable=False, default=0)  # 0-100
    
    # Análisis de frames
    frames_analyzed = Column(Integer, nullable=False, default=0)
    frames_with_faces = Column(Integer, nullable=False, default=0)
    detection_rate = Column(Float, nullable=True)  # Porcentaje de detección
    
    # Análisis detallado
    analysis_result = Column(Text, nullable=True)
    
    # Detalles técnicos
    technical_details = Column(Text, nullable=True)  # JSON con detalles técnicos
    video_size = Column(Integer, nullable=True)
    video_duration = Column(Float, nullable=True)
    
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
        return f"<VideoVerification(id='{self.id}', document='{self.document_number}', live={self.is_live_person}, matches={self.matches_reference}, confidence={self.confidence}%, status='{self.status.value if self.status else 'None'}')>"
    
    def to_dict(self):
        """Convertir a diccionario para JSON responses"""
        return {
            'id': self.id,
            'document_number': self.document_number,
            'video_path': self.video_path,
            'reference_image_path': self.reference_image_path,
            'is_live_person': self.is_live_person,
            'matches_reference': self.matches_reference,
            'confidence': self.confidence,
            'status': self.status.value if self.status else None,
            'is_approved': self.is_approved,
            'frames_analyzed': self.frames_analyzed,
            'frames_with_faces': self.frames_with_faces,
            'detection_rate': self.detection_rate,
            'analysis_result': self.analysis_result,
            'video_info': {
                'size': self.video_size,
                'duration': self.video_duration
            },
            'technical_details': self.technical_details,
            'details': self.details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processing_time_seconds': self.processing_time_seconds
        }
    status = Column(Enum(VerificationStatus), nullable=False)
    analysis = Column(Text)  # Análisis detallado
    recommendation = Column(String(20))  # APPROVE/REVIEW/REJECT
    
    # === ARCHIVOS DE REFERENCIA ===
    video_path = Column(String(500), nullable=False)
    selfie_reference_path = Column(String(500))
    dni_reference_path = Column(String(500))
    
    # === METADATOS ===
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_method = Column(String(100))  # Método usado para extraer frames
    technical_details = Column(Text)  # JSON con detalles técnicos completos
    
    # Relación
    user = relationship("User", back_populates="video_verification")
    
    def __str__(self):
        return f"VideoVerification({self.document_number}: {self.status.value}, {self.final_confidence}%)"
    
    @property
    def is_approved(self):
        return self.status == VerificationStatus.ACCEPTED
    
    @property
    def detection_sufficient(self):
        """¿La tasa de detección es suficiente? (≥70%)"""
        return self.detection_rate >= 70.0
    
    def to_dict(self):
        return {
            'id': self.id,
            'document_number': self.document_number,
            'liveness_analysis': {
                'is_live_person': self.is_live_person,
                'liveness_score': self.liveness_score
            },
            'facial_comparison': {
                'matches_selfie': self.matches_selfie,
                'matches_dni': self.matches_dni,
                'selfie_confidence': self.selfie_confidence,
                'dni_confidence': self.dni_confidence,
                'final_confidence': self.final_confidence
            },
            'frame_analysis': {
                'frames_extracted': self.frames_extracted,
                'frames_analyzed': self.frames_analyzed,
                'frames_with_faces': self.frames_with_faces,
                'detection_rate': self.detection_rate,
                'detection_sufficient': self.detection_sufficient
            },
            'video_quality': {
                'duration': self.video_duration,
                'size': self.video_size,
                'codec': self.video_codec
            },
            'scoring': {
                'consistency_bonus': self.consistency_bonus,
                'detection_penalty': self.detection_penalty
            },
            'verification': {
                'status': self.status.value if self.status else None,
                'recommendation': self.recommendation,
                'analysis': self.analysis
            },
            'files': {
                'video_path': self.video_path,
                'selfie_reference_path': self.selfie_reference_path,
                'dni_reference_path': self.dni_reference_path
            },
            'processing': {
                'method': self.processing_method,
                'technical_details': self.technical_details
            },
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
