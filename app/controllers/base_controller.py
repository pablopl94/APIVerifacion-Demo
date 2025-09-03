"""
🏠 Base Controller  
Endpoints básicos y páginas de inicio
"""

from flask import Blueprint, render_template

# Crear blueprint
base_bp = Blueprint('base', __name__)


@base_bp.route('/')
def kyc_verification():
    """
    🧪 INTERFAZ DE PRUEBA - Página HTML para probar el sistema
    """
    return render_template('kyc_verification.html')
