"""
🎮 Controllers Package
Capa de controladores - Endpoints organizados por funcionalidad
"""

from .base_controller import base_bp
from .upload_controller import upload_bp  
from .kyc_controller import kyc_bp

__all__ = ['base_bp', 'upload_bp', 'kyc_bp']
