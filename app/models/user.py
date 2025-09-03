"""
👤 MODELO USUARIO KYC
Solo se guarda DESPUÉS de verificación exitosa del DNI
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Enum
from sqlalchemy.orm import relationship
from app.database.connection import Base
from datetime import datetime
import enum

class VerificationStatus(enum.Enum):
    PENDING = "PENDING"      # Verificación pendiente
    ACCEPTED = "ACCEPTED"    # Verificación aceptada
    REVIEW = "REVIEW"        # Requiere revisión manual
    REJECTED = "REJECTED"    # Verificación rechazada

class User(Base):
    """
    � USUARIO KYC - Se crea SOLO tras verificación DNI exitosa
    
    Flujo:
    1. Datos llegan por parámetros (NO se guardan)
    2. Se verifica con imagen DNI
    3. SOLO si verificación OK → se crea el usuario
    """
    __tablename__ = "users"
    
    # PK: Número de documento (único)
    document_number = Column(String(50), primary_key=True, index=True)
    
    # Datos básicos (extraídos de DNI verificado)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    nationality = Column(String(10), nullable=False)  # ESP, FRA, ITA, etc.
    birth_date = Column(String(20), nullable=False)   # DD/MM/YYYY
    issue_date = Column(String(20), nullable=False)   # DD/MM/YYYY
    expiry_date = Column(String(20), nullable=False)  # DD/MM/YYYY
    
    # Metadatos
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Estado general del usuario
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)
    
    # Relaciones
    dni_verification = relationship("DNIVerification", back_populates="user", uselist=False)
    selfie_verification = relationship("SelfieVerification", back_populates="user", uselist=False)
    video_verification = relationship("VideoVerification", back_populates="user", uselist=False)
    
    def __str__(self):
        return f"User({self.document_number}: {self.first_name} {self.last_name})"
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_fully_verified(self):
        """
        🎯 VERIFICACIÓN MODULAR INTELIGENTE
        
        Determina si está "completamente verificado" basado en lo que se ha usado:
        - Solo DNI: DNI debe estar aprobado
        - DNI + Selfie: ambos deben estar aprobados
        - DNI + Selfie + Video: los 3 deben estar aprobados
        """
        # 1. DNI siempre es obligatorio (base del sistema)
        dni_ok = self.dni_verification and self.dni_verification.status == VerificationStatus.ACCEPTED
        if not dni_ok:
            return False
        
        # 2. Si no hay selfie ni video → Solo DNI = Verificado ✅
        if not self.selfie_verification and not self.video_verification:
            return True
            
        # 3. Si hay selfie pero no video → DNI + Selfie = Verificado ✅
        if self.selfie_verification and not self.video_verification:
            return self.selfie_verification.status == VerificationStatus.ACCEPTED
            
        # 4. Si hay video → Todos los pasos usados deben estar aprobados ✅
        if self.video_verification:
            selfie_ok = self.selfie_verification and self.selfie_verification.status == VerificationStatus.ACCEPTED
            video_ok = self.video_verification.status == VerificationStatus.ACCEPTED
            return selfie_ok and video_ok
            
        # 5. Solo selfie sin DNI (caso raro, no debería pasar)
        return False
    
    @property
    def verification_progress(self):
        """
        📊 PROGRESO DE VERIFICACIÓN MODULAR
        Devuelve información detallada de qué pasos se han usado y su estado
        """
        progress = {
            'steps_attempted': [],
            'steps_completed': [],
            'completion_rate': 0,
            'next_step': None,
            'is_complete': False
        }
        
        # DNI
        if self.dni_verification:
            progress['steps_attempted'].append('dni')
            if self.dni_verification.status == VerificationStatus.ACCEPTED:
                progress['steps_completed'].append('dni')
        
        # Selfie
        if self.selfie_verification:
            progress['steps_attempted'].append('selfie')
            if self.selfie_verification.status == VerificationStatus.ACCEPTED:
                progress['steps_completed'].append('selfie')
        
        # Video
        if self.video_verification:
            progress['steps_attempted'].append('liveness')
            if self.video_verification.status == VerificationStatus.ACCEPTED:
                progress['steps_completed'].append('liveness')
        
        # Calcular progreso
        if progress['steps_attempted']:
            progress['completion_rate'] = int(
                (len(progress['steps_completed']) / len(progress['steps_attempted'])) * 100
            )
        
        # Determinar siguiente paso
        if 'dni' not in progress['steps_attempted']:
            progress['next_step'] = 'dni'
        elif 'dni' in progress['steps_completed'] and 'selfie' not in progress['steps_attempted']:
            progress['next_step'] = 'selfie'
        elif 'selfie' in progress['steps_completed'] and 'liveness' not in progress['steps_attempted']:
            progress['next_step'] = 'liveness'
        else:
            progress['next_step'] = None
        
        # ¿Está completo según lo que se ha intentado?
        progress['is_complete'] = self.is_fully_verified
        
        return progress
    
    def to_dict(self):
        """Convertir a diccionario para JSON responses"""
        return {
            'document_number': self.document_number,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'nationality': self.nationality,
            'birth_date': self.birth_date,
            'issue_date': self.issue_date,
            'expiry_date': self.expiry_date,
            'status': self.status.value if self.status else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_fully_verified': self.is_fully_verified
        }
